import os
import time
import json
import torch
import random
import numpy as np
from . import utilities as U
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import load_dataset
from collections import defaultdict
from .base_pipeline import BasePipeline
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


class Pipeline(BasePipeline):
    def __init__(self, config, device=None):
        self.device = device

        self.config = config
        self.num_iterations = config["num_iterations"]
        self.dataset_name = config["dataset_name"]
        self.conf_threshold_1 = config["conf_threshold_1"]
        self.conf_threshold_2 = config["conf_threshold_2"]
        self.hf_model_name_1 = config["hf_model_name_1"]
        self.hf_model_name_2 = config["hf_model_name_2"]
        self.num_queries_1 = config["num_queries_1"]
        self.num_queries_2 = config["num_queries_2"]
        self.adv_patch_xywh = config["adv_patch_xywh"]
        self.target_labels = config["target_labels"]
        self.lr = config["learning_rate"]
        self.budget = config["budget"]
        # control memory use when batching model_2 inputs
        self.model2_batch_size = config.get("model2_batch_size", 8)

        # We reuse the same scheduling knobs for B-primary phase detection
        self.b_window = config.get("a_phase_window", 20)
        self.b_min_delta = float(config.get("a_phase_min_delta", 0.0))
        self.b_patience = int(config.get("a_phase_patience", 3))
        self.b_min_iters = int(config.get("a_phase_min_iters", 10))
        try:
            n_total = int(self.num_iterations)
            win_frac = config.get("a_phase_window_frac", None)
            if win_frac is not None:
                self.b_window = max(5, int(round(float(win_frac) * n_total)))
            pat_frac = config.get("a_phase_patience_frac", None)
            if pat_frac is not None:
                self.b_patience = max(1, int(round(float(pat_frac) * self.b_window)))
            min_iters_frac = config.get("a_phase_min_iters_frac", None)
            if min_iters_frac is not None:
                self.b_min_iters = max(1, int(round(float(min_iters_frac) * n_total)))
        except Exception:
            pass
        self.b_min_rel = float(self.config.get("a_phase_min_rel", 0.0))
        self.b_slope_thresh = float(self.config.get("a_phase_slope_thresh", 0.0))
        self.b_use_window_best = bool(self.config.get("a_phase_use_window_best", True))
        self.b_min_drop = float(self.config.get("a_phase_min_drop", 0.0))
        self.b_min_drop_rel = float(self.config.get("a_phase_min_drop_rel", 0.0))
        self.b_plateau = U.LossPlateauDetector(
            self.b_window, self.b_min_delta, self.b_patience,
            min_rel=self.b_min_rel,
            slope_thresh=self.b_slope_thresh,
            use_window_best=self.b_use_window_best,
        )
        self.in_phase2 = False
        self.log_dict = {}

        self.cls_loss_weight_1 = float(config.get("cls_loss_weight_1", 1.0))
        self.cls_loss_weight_2 = float(config.get("cls_loss_weight_2", 1.0))
        weight_sum = self.cls_loss_weight_1 + self.cls_loss_weight_2
        if weight_sum <= 0:
            self.cls_loss_weight_1 = 0.5
            self.cls_loss_weight_2 = 0.5
        else:
            self.cls_loss_weight_1 /= weight_sum
            self.cls_loss_weight_2 /= weight_sum

        # projection settings
        self.projection_eps = float(config.get("projection_eps", 1e-12))
        self.b_grad_scale = float(config.get("b_grad_scale", 1.0))

        # record resolved values for reproducibility in logs
        try:
            self.config["b_phase_window_resolved"] = int(self.b_window)
            self.config["b_phase_patience_resolved"] = int(self.b_patience)
            self.config["b_phase_min_iters_resolved"] = int(self.b_min_iters)
        except Exception:
            pass

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
            if self.config["test_size"] is not None:
                self.test_size = self.config["test_size"]
            else:
                self.test_size = len(coco_data)
            random_indices = random.sample(range(len(coco_data)), self.test_size)
            return coco_data.select(random_indices)
        else:
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

    def run_attack(self, dataset):
        if not hasattr(self, "run_timestamp") or self.run_timestamp is None:
            self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")

        for index, example in tqdm(enumerate(dataset), total=dataset.__len__()):
            image_id = example["image_id"]
            image = example["image"].convert("RGB")

            image_tensor = self.processor_1(images=image, return_tensors="pt")["pixel_values"].to(self.device)
            self.bx = torch.zeros_like(image_tensor).requires_grad_(True).to(self.device)
            self.mask = self.get_mask(image_tensor.shape).to(self.device)

            self.log_dict[image_id] = []
            # reset phase state and plateau detector per-sample
            self.in_phase2 = False
            self.b_plateau = U.LossPlateauDetector(
                self.b_window, self.b_min_delta, self.b_patience,
                min_rel=self.b_min_rel,
                slope_thresh=self.b_slope_thresh,
                use_window_best=self.b_use_window_best,
            )
            b_initial_loss = None
            b_best_loss = None

            for i in range(self.num_iterations):
                # Forward both models on the current perturbed image
                img_pert = image_tensor + self.bx * self.mask

                # Primary model (B)
                model_2_outputs = self.model_2(img_pert, output_hidden_states=True)
                probs2 = F.sigmoid(model_2_outputs.logits)
                # form combined tensor for ROI generation
                combined2 = torch.cat(
                    [model_2_outputs.pred_boxes, probs2.max(dim=2)[0].unsqueeze(-1), model_2_outputs.logits], dim=-1
                )
                _labels2 = self.target_labels[1] if isinstance(self.target_labels[1], (list, tuple)) else []
                _probs2 = probs2[..., _labels2] if len(_labels2) > 0 else probs2
                obj_count_2 = (_probs2 > self.conf_threshold_2).sum().item()

                # Build ROI masks from B boxes (primary)
                roi_masks_list = U.masks_from_boxes(image_tensor, combined2, self.conf_threshold_2)
                flat_masks = []
                for masks in roi_masks_list:
                    flat_masks.extend(masks)

                # Secondary model (A) on full image for its gradient
                model_1_outputs = self.model_1(img_pert, output_hidden_states=True)
                probs1 = F.sigmoid(model_1_outputs.logits)
                cls_loss_1 = U.calc_cls_loss(probs1, self.target_labels[0])
                norm_loss = self.calc_norm_loss(order=[""])
                weighted_cls_loss_1 = self.cls_loss_weight_1 * cls_loss_1
                A_loss = weighted_cls_loss_1 + norm_loss
                _labels1 = self.target_labels[0] if isinstance(self.target_labels[0], (list, tuple)) else []
                _probs1 = probs1[..., _labels1] if len(_labels1) > 0 else probs1
                obj_count_1 = (_probs1 > self.conf_threshold_1).sum().item()

                # Compute B loss and grad over ROIs (average over masks)
                cls_loss_2_total = torch.tensor(0.0, device=self.device)
                grad_B = torch.zeros_like(self.bx.view(-1))
                num_masks = len(flat_masks)
                if num_masks > 0:
                    full_img = img_pert  # [1,C,H,W]
                    Bf, C, H, W = full_img.shape
                    min_tokens = max(1, (H // 32) * (W // 32))
                    safe_queries = max(1, min(self.num_queries_2, min_tokens))
                    if self.model_2.config.num_queries != safe_queries:
                        self.model_2.config.num_queries = safe_queries
                    N = len(flat_masks)
                    bs2 = max(1, int(self.model2_batch_size))
                    for start in range(0, N, bs2):
                        end = min(start + bs2, N)
                        batch_pixel_masks = torch.stack(flat_masks[start:end], dim=0)
                        batch_images = full_img.expand(end - start, -1, -1, -1).contiguous()
                        _m = batch_pixel_masks.to(batch_images.dtype).unsqueeze(1).expand(-1, 3, -1, -1)
                        outputs2 = self.model_2(batch_images * _m, output_hidden_states=True)
                        probs_2 = F.sigmoid(outputs2.logits)
                        loss_B = U.calc_cls_loss(probs_2, self.target_labels[1])
                        weighted_loss_B = self.cls_loss_weight_2 * loss_B
                        grad_B_micro = U.grad_wrt(self.bx, weighted_loss_B, retain_graph=False, create_graph=False)
                        grad_B = grad_B + grad_B_micro
                        cls_loss_2_total = cls_loss_2_total + loss_B.detach()
                    denom = float(max(1, num_masks))
                    grad_B = grad_B / denom
                    cls_loss_2_total = cls_loss_2_total / denom
                else:
                    cls_loss_2_total = torch.tensor(0.0, device=self.device)

                # B-phase scheduling on B loss value
                B_loss_value = (self.cls_loss_weight_2 * cls_loss_2_total + norm_loss).detach().item()
                if b_initial_loss is None:
                    b_initial_loss = B_loss_value
                    b_best_loss = B_loss_value
                else:
                    b_best_loss = min(b_best_loss, B_loss_value)
                drop_abs = (b_initial_loss - b_best_loss) if b_initial_loss is not None else 0.0
                drop_rel = (drop_abs / abs(b_initial_loss)) if (b_initial_loss is not None and abs(b_initial_loss) > 1e-12) else 0.0
                drop_ok = True
                if self.b_min_drop > 0.0 and drop_abs < self.b_min_drop:
                    drop_ok = False
                if self.b_min_drop_rel > 0.0 and drop_rel < self.b_min_drop_rel:
                    drop_ok = False
                plateau = self.b_plateau.update(B_loss_value)
                if (not self.in_phase2) and (i >= self.b_min_iters) and plateau and drop_ok:
                    self.in_phase2 = True

                # Compute grad_A and projection onto B
                grad_A = U.grad_wrt(self.bx, A_loss, retain_graph=False, create_graph=False)
                eps = self.projection_eps
                norm_B = grad_B.norm()
                if norm_B > eps:
                    dot_AB = torch.dot(grad_A, grad_B)
                    proj_coeff = dot_AB / (norm_B * norm_B + eps)
                    grad_A_on_B = proj_coeff * grad_B
                else:
                    grad_A_on_B = torch.zeros_like(grad_A)

                # Phase 1: only A projected on B; Phase 2: A_on_B + B
                if not self.in_phase2:
                    grad_final = grad_A_on_B
                else:
                    grad_final = grad_A_on_B + self.b_grad_scale * grad_B

                U.assign_flattened_grad(self.bx, grad_final.detach())

                with torch.no_grad():
                    self.bx.add_(-self.lr * self.bx.grad)
                    self.bx.clamp_(-self.budget, self.budget)

                self.bx = self.bx.detach().requires_grad_(True)

                self.update_log(
                    image_id=image_id,
                    iteration=i,
                    cls_loss_1=cls_loss_1,
                    cls_loss_2=cls_loss_2_total,
                    norm_loss_1=norm_loss,
                    obj_count_1=obj_count_1,
                    obj_count_2=obj_count_2,
                    grad_A_norm=grad_A.norm() if isinstance(grad_A, torch.Tensor) else None,
                    grad_B_norm=grad_B.norm() if isinstance(grad_B, torch.Tensor) else None,
                    grad_final_norm=grad_final.norm() if isinstance(grad_final, torch.Tensor) else None,
                    grad_B_eff_norm=grad_A_on_B.norm() if isinstance(grad_A_on_B, torch.Tensor) else None,
                    phase_flag=int(self.in_phase2),
                    a_drop_abs=drop_abs,
                    a_drop_rel=drop_rel,
                )

            # save final perturbed image for this sample
            try:
                out_root = "output"
                stemname = Path(__file__).stem
                out_dir = os.path.join(out_root, f"{stemname}_{self.run_timestamp}")
                os.makedirs(out_dir, exist_ok=True)
                perturbed = (image_tensor + self.bx * self.mask).clamp(0.0, 1.0)
                img_np = perturbed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                out_path = os.path.join(out_dir, f"{image_id}.png")
                plt.imsave(out_path, img_np)
            except Exception as e:
                print(f"[warn] failed to save perturbed image for {image_id}: {e}")

        self.write_log()

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

    def calc_norm_loss(self, order=["linf"]):
        total_norm_loss = torch.tensor(0.0, device=self.device)
        if "linf" in order:
            l_inf = torch.norm(self.bx * self.mask, p=float("inf"))
            total_norm_loss = total_norm_loss + l_inf
        if "l2" in order:
            l_2 = torch.norm(self.bx * self.mask, p=2)
            total_norm_loss = total_norm_loss + l_2
        if "l1" in order:
            l_1 = torch.norm(self.bx * self.mask, p=1)
            total_norm_loss = total_norm_loss + l_1
        return total_norm_loss

    def update_log(
        self,
        image_id,
        iteration,
        cls_loss_1=None,
        cls_loss_2=None,
        norm_loss_1=None,
        norm_loss_2=None,
        obj_count_1=None,
        obj_count_2=None,
        total_loss=None,
        grad_A_norm=None,
        grad_B_norm=None,
        grad_final_norm=None,
        grad_B_eff_norm=None,
        grad_cosine=None,
        phase_flag=None,
        a_drop_abs=None,
        a_drop_rel=None,
    ):
        if self.config["output_log"] is not True:
            return
        log_entry = {
            "iteration": iteration,
            "obj_count_1": obj_count_1,
            "obj_count_2": obj_count_2,
            "cls_loss_1": self.unhook(cls_loss_1),
            "cls_loss_2": self.unhook(cls_loss_2),
            "norm_loss_1": self.unhook(norm_loss_1),
            "norm_loss_2": self.unhook(norm_loss_2),
            "grad_A_norm": self.unhook(grad_A_norm),
            "grad_B_norm": self.unhook(grad_B_norm),
            "grad_final_norm": self.unhook(grad_final_norm),
        }
        if grad_B_eff_norm is not None:
            log_entry["grad_B_eff_norm"] = self.unhook(grad_B_eff_norm)
        if grad_cosine is not None:
            if isinstance(grad_cosine, torch.Tensor):
                log_entry["grad_cosine"] = self.unhook(grad_cosine)
            else:
                log_entry["grad_cosine"] = float(grad_cosine)
        if phase_flag is not None:
            log_entry["phase_flag"] = int(phase_flag)
        if a_drop_abs is not None:
            log_entry["a_drop_abs"] = float(a_drop_abs)
        if a_drop_rel is not None:
            log_entry["a_drop_rel"] = float(a_drop_rel)
        self.log_dict[image_id].append(log_entry)

    def write_log(self, log_path=None):
        if self.config["output_log"] is not True:
            return
        if log_path is None:
            log_path = "./logs"
        os.makedirs(log_path, exist_ok=True)
        timestamp = getattr(self, "run_timestamp", None)
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.run_timestamp = timestamp
        stemname = Path(__file__).stem
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
        return x.detach().cpu().item()

    def get_logs(self):
        return self.log_dict

