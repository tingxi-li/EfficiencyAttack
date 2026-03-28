import os
import time
import json
import torch
import random
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from . import utilities as U


class BasePipeline(ABC):
    def __init__(self, config, device=None):
        self.device = device
        self.config = config

        self.num_iterations = config["num_iterations"]
        self.dataset_name = config["dataset_name"]
        self.conf_threshold_1 = config["conf_threshold_1"]
        self.conf_threshold_2 = config["conf_threshold_2"]
        self.hf_model_name_1 = config["hf_model_name_1"]
        self.hf_model_name_2 = config["hf_model_name_2"]
        self.num_queries_1 = int(config.get("num_queries_1", 25))
        self.num_queries_2 = int(config.get("num_queries_2", 25))
        self.adv_patch_xywh = config["adv_patch_xywh"]
        self.target_labels = config["target_labels"]
        self.lr = config["learning_rate"]
        self.budget = config["budget"]
        self.model2_batch_size = int(config.get("model2_batch_size", 8))

        self.cls_loss_weight_1 = float(config.get("cls_loss_weight_1", 1.0))
        self.cls_loss_weight_2 = float(config.get("cls_loss_weight_2", 1.0))
        weight_sum = self.cls_loss_weight_1 + self.cls_loss_weight_2
        if weight_sum <= 0:
            self.cls_loss_weight_1 = 0.5
            self.cls_loss_weight_2 = 0.5
        else:
            self.cls_loss_weight_1 /= weight_sum
            self.cls_loss_weight_2 /= weight_sum

        self.log_dict = {}
        self.log_dir = config.get("log_dir", "./log")

        self.randomseed()

    def randomseed(self):
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def load_dataset(self):
        if self.dataset_name == "coco":
            coco_data = load_dataset("detection-datasets/coco", split="val")
            test_size = self.config.get("test_size")
            if test_size is not None:
                self.test_size = test_size
            else:
                self.test_size = len(coco_data)
            random_indices = random.sample(range(len(coco_data)), self.test_size)
            return coco_data.select(random_indices)
        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def load_model(self):
        self.model_1 = RTDetrForObjectDetection.from_pretrained(self.hf_model_name_1).to(self.device)
        self.model_2 = RTDetrForObjectDetection.from_pretrained(self.hf_model_name_2).to(self.device)
        self.processor_1 = RTDetrImageProcessor.from_pretrained(self.hf_model_name_1)
        self.processor_2 = RTDetrImageProcessor.from_pretrained(self.hf_model_name_2)
        self.model_1.config.num_queries = self.num_queries_1
        self.model_2.config.num_queries = self.num_queries_2
        self.model_1.eval()
        self.model_2.eval()

    def get_mask(self, tensor_shape):
        bs, _, H, W = tensor_shape
        if not isinstance(self.adv_patch_xywh, list):
            raise ValueError("random placement is not supported yet")
        cx, cy, pw, ph = self.adv_patch_xywh
        cx = random.uniform(0.0, 1.0) if cx == -1 else cx
        cy = random.uniform(0.0, 1.0) if cy == -1 else cy
        pw = random.uniform(0.0, 1.0) if pw == -1 else pw
        ph = random.uniform(0.0, 1.0) if ph == -1 else ph
        x1 = int((cx - 0.5 * pw) * W)
        y1 = int((cy - 0.5 * ph) * H)
        x2 = int((cx + 0.5 * pw) * W + 0.9999)
        y2 = int((cy + 0.5 * ph) * H + 0.9999)
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))
        if x2 <= x1:
            x2 = min(W, x1 + 1)
        if y2 <= y1:
            y2 = min(H, y1 + 1)
        mask = torch.zeros((bs, 1, H, W), device=self.device, dtype=torch.float32)
        mask[:, :, y1:y2, x1:x2] = 1.0
        return mask

    def calc_norm_loss(self, order=None):
        if order is None:
            order = ["linf"]
        total = torch.tensor(0.0, device=self.device)
        if "linf" in order:
            total = total + torch.norm(self.patch * self.mask, p=float("inf"))
        if "l2" in order:
            total = total + torch.norm(self.patch * self.mask, p=2)
        if "l1" in order:
            total = total + torch.norm(self.patch * self.mask, p=1)
        return total

    def update_log(self, image_id, iteration, **metrics):
        if not bool(self.config.get("output_log", True)):
            return
        entry = {"iteration": iteration}
        for k, v in metrics.items():
            entry[k] = self.unhook(v) if isinstance(v, torch.Tensor) else v
        self.log_dict[image_id].append(entry)

    def write_log(self, log_path=None):
        if not bool(self.config.get("output_log", True)):
            return
        if log_path is None:
            log_path = self.log_dir
        os.makedirs(log_path, exist_ok=True)
        timestamp = getattr(self, "run_timestamp", None)
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.run_timestamp = timestamp
        stemname = type(self).__module__.split(".")[-1]
        log_file = os.path.join(log_path, f"log_{stemname}_{timestamp}.json")
        config_file = os.path.join(log_path, f"log_{stemname}_{timestamp}_configs.json")
        with open(log_file, "w") as f:
            json.dump(self.log_dict, f, indent=4)
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=4)
        print(f"Logs written to {log_file}")
        print(f"Config written to {config_file}")
        self.log_dict = {}

    def unhook(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        return x

    def run_attack(self, dataset):
        if not hasattr(self, "run_timestamp") or self.run_timestamp is None:
            self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        for _, example in tqdm(enumerate(dataset), total=len(dataset)):
            image_id = example["image_id"]
            image = example["image"].convert("RGB")
            image_tensor = self.processor_1(images=image, return_tensors="pt")["pixel_values"].to(self.device)
            self.patch = torch.zeros_like(image_tensor).requires_grad_(True).to(self.device)
            self.mask = self.get_mask(image_tensor.shape).to(self.device)
            self.log_dict[image_id] = []
            self._init_image_state()
            for i in range(self.num_iterations):
                obj_count_1, obj_count_2, extra = self._attack_step(i, image_tensor)
                self.update_log(image_id, i, obj_count_1=obj_count_1, obj_count_2=obj_count_2, **extra)
            try:
                stemname = type(self).__module__.split(".")[-1]
                out_dir = os.path.join("output", f"{stemname}_{self.run_timestamp}")
                os.makedirs(out_dir, exist_ok=True)
                perturbed = (image_tensor + self.patch * self.mask).clamp(0.0, 1.0)
                img_np = perturbed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                plt.imsave(os.path.join(out_dir, f"{image_id}.png"), img_np)
            except Exception as e:
                print(f"[warn] failed to save perturbed image for {image_id}: {e}")
        self.write_log()

    @abstractmethod
    def _init_image_state(self):
        """Reset per-image state: phase flags, plateau detectors, ALM multipliers, etc."""

    @abstractmethod
    def _attack_step(self, i: int, image_tensor) -> tuple:
        """
        One optimization step.
        Returns: (obj_count_1: int, obj_count_2: int, extra_log_fields: dict)
        Uses self.patch, self.mask, self.model_1, self.model_2, self.lr, self.budget.
        """
