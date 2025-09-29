import os
import pdb
import sys
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
        # A-phase stabilization settings
        self.a_window = config.get("a_phase_window", 20)
        self.a_min_delta = float(config.get("a_phase_min_delta", 0.0))
        self.a_patience = int(config.get("a_phase_patience", 3))
        self.a_min_iters = int(config.get("a_phase_min_iters", 10))
        self.a_min_rel = float(self.config.get("a_phase_min_rel", 0.0))
        self.a_slope_thresh = float(self.config.get("a_phase_slope_thresh", 0.0))
        self.a_use_window_best = bool(self.config.get("a_phase_use_window_best", True))
        self.a_min_drop = float(self.config.get("a_phase_min_drop", 0.0))
        self.a_min_drop_rel = float(self.config.get("a_phase_min_drop_rel", 0.0))
        self.a_plateau = U.LossPlateauDetector(
            self.a_window, self.a_min_delta, self.a_patience,
            min_rel=self.a_min_rel,
            slope_thresh=self.a_slope_thresh,
            use_window_best=self.a_use_window_best,
        )
        self.in_b_phase = False
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

        # gradient projection behaviour
        self.projection_conflict_threshold = float(config.get("projection_conflict_threshold", 0.0))
        self.projection_match_norm = bool(config.get("projection_match_norm", True))
        self.b_grad_scale = float(config.get("b_grad_scale", 1.0))
        self.projection_eps = float(config.get("projection_eps", 1e-12))
        
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
        # establish a run-level timestamp to sync saved assets and logs
        if not hasattr(self, "run_timestamp") or self.run_timestamp is None:
            self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        for index, example in tqdm(enumerate(dataset), total=dataset.__len__()):
            image_id = example["image_id"]
            image = example["image"].convert("RGB")

            image_tensor = self.processor_1(images=image, return_tensors="pt")["pixel_values"].to(self.device)
            self.bx = torch.zeros_like(image_tensor).requires_grad_(True).to(self.device)
            self.mask =  self.get_mask(image_tensor.shape).to(self.device)

            self.log_dict[image_id] = []
            # reset phase state and plateau detector per-sample
            self.in_b_phase = False
            self.a_plateau = U.LossPlateauDetector(
                self.a_window, self.a_min_delta, self.a_patience,
                min_rel=self.a_min_rel,
                slope_thresh=self.a_slope_thresh,
                use_window_best=self.a_use_window_best,
            )
            a_initial_loss = None
            a_best_loss = None
            for i in range(self.num_iterations):
                model_1_outputs = self.model_1(image_tensor + self.bx * self.mask, output_hidden_states=True)
                probs = F.sigmoid(model_1_outputs.logits)

                cls_loss_1  = U.calc_cls_loss(probs, self.target_labels[0])
                norm_loss_1 = self.calc_norm_loss(order=[""])
                weighted_cls_loss_1 = self.cls_loss_weight_1 * cls_loss_1

                combined = torch.cat([model_1_outputs.pred_boxes, probs.max(dim=2)[0].unsqueeze(-1), model_1_outputs.logits], dim=-1)
                # [batch_size, num_queries, xywh + conf + num_classes]
                
                # drawn = U.debug_image_with_boxes(image_tensor + self.bx * self.mask, combined, self.conf_threshold_1)

                _labels1 = self.target_labels[0] if isinstance(self.target_labels[0], (list, tuple)) else []
                _probs1 = probs[..., _labels1] if len(_labels1) > 0 else probs
                obj_count_1 = (_probs1 > self.conf_threshold_1).sum().item()

                # Build ROI pixel masks for boxes and batch model_2 over the full-size image
                roi_masks_list = U.masks_from_boxes(image_tensor, combined, self.conf_threshold_1)
                # flatten masks across batch (usually B=1)
                flat_masks = []
                for masks in roi_masks_list:
                    flat_masks.extend(masks)

                # Phase scheduling: optimize A first until plateau, then optimize B with projection
                A_loss = weighted_cls_loss_1 + norm_loss_1
                A_loss_value = A_loss.detach().item()
                if a_initial_loss is None:
                    a_initial_loss = A_loss_value
                    a_best_loss = A_loss_value
                else:
                    a_best_loss = min(a_best_loss, A_loss_value)

                drop_abs = (a_initial_loss - a_best_loss) if a_initial_loss is not None else 0.0
                if a_initial_loss is None or abs(a_initial_loss) <= 1e-12:
                    drop_rel = 0.0
                else:
                    drop_rel = drop_abs / abs(a_initial_loss)

                drop_ok = True
                if self.a_min_drop > 0.0 and drop_abs < self.a_min_drop:
                    drop_ok = False
                if self.a_min_drop_rel > 0.0 and drop_rel < self.a_min_drop_rel:
                    drop_ok = False

                plateau = self.a_plateau.update(A_loss_value)

                if (not self.in_b_phase) and (i >= self.a_min_iters) and plateau and drop_ok:
                    self.in_b_phase = True

                grad_B_eff_norm = None
                grad_cosine = None
                if not self.in_b_phase:
                    # Only optimize A
                    grad_A = U.grad_wrt(self.bx, A_loss, retain_graph=False, create_graph=False)
                    U.assign_flattened_grad(self.bx, grad_A.detach())
                    grad_B = torch.tensor(0.0, device=self.device)
                    grad_final = grad_A
                else:
                    # Compute B only when needed to save compute/memory
                    cls_loss_2_total = torch.tensor(0.0, device=self.device)
                    obj_count_2 = 0
                    grad_A = U.grad_wrt(self.bx, A_loss, retain_graph=False, create_graph=False)
                    grad_B = torch.zeros_like(grad_A)
                    num_masks = len(flat_masks)
                    if num_masks > 0:
                        # replicate the perturbed full image for each mask
                        full_img = (image_tensor + self.bx * self.mask)  # [1,C,H,W]
                        Bf, C, H, W = full_img.shape
                        # Keep num_queries safe; tokens derived from full image size
                        min_tokens = max(1, (H // 32) * (W // 32))
                        safe_queries = max(1, min(self.num_queries_2, min_tokens))
                        if self.model_2.config.num_queries != safe_queries:
                            self.model_2.config.num_queries = safe_queries

                        # micro-batch to control VRAM
                        N = len(flat_masks)
                        bs2 = max(1, int(self.model2_batch_size))
                        for start in range(0, N, bs2):
                            end = min(start + bs2, N)
                            batch_pixel_masks = torch.stack(flat_masks[start:end], dim=0)  # [n,H,W]
                            batch_images = full_img.expand(end - start, -1, -1, -1).contiguous()
                            _m = batch_pixel_masks.to(batch_images.dtype).unsqueeze(1).expand(-1, 3, -1, -1)
                            outputs = self.model_2(batch_images * _m, output_hidden_states=True)

                            probs_2 = F.sigmoid(outputs.logits)
                            loss_B = U.calc_cls_loss(probs_2, self.target_labels[1])
                            weighted_loss_B = self.cls_loss_weight_2 * loss_B
                            grad_B_micro = U.grad_wrt(self.bx, weighted_loss_B, retain_graph=False, create_graph=False)
                            grad_B = grad_B + grad_B_micro
                            cls_loss_2_total = cls_loss_2_total + loss_B.detach()
                            _labels2 = self.target_labels[1] if isinstance(self.target_labels[1], (list, tuple)) else []
                            _probs2 = probs_2[..., _labels2] if len(_labels2) > 0 else probs_2
                            obj_count_2 += (_probs2 > self.conf_threshold_2).sum().item()
                        denom = float(max(1, num_masks))
                        grad_B = grad_B / denom
                        cls_loss_2_total = cls_loss_2_total / denom
                    else:
                        cls_loss_2_total = torch.tensor(0.0, device=self.device)
                        obj_count_2 = 0
                    eps = self.projection_eps
                    grad_final = grad_A.clone()
                    grad_cosine = None
                    grad_B_effective = grad_B
                    grad_B_eff_norm = None
                    norm_A = grad_A.norm()
                    norm_B = grad_B.norm()
                    if norm_B > eps:
                        if norm_A > eps:
                            grad_cosine = torch.dot(grad_A, grad_B) / (norm_A * norm_B + eps)
                        if (norm_A > eps) and (grad_cosine is not None) and (grad_cosine <= self.projection_conflict_threshold):
                            grad_B_effective = U.project_onto_orthogonal_complement(grad_B, [grad_A], eps=eps)
                            norm_proj = grad_B_effective.norm()
                            if self.projection_match_norm and norm_proj > eps:
                                scale = norm_B / (norm_proj + eps)
                                grad_B_effective = grad_B_effective * scale
                        grad_B_effective = grad_B_effective * self.b_grad_scale
                        grad_final = grad_A + grad_B_effective
                        grad_B_eff_norm = grad_B_effective.norm()
                    else:
                        grad_B_effective = torch.zeros_like(grad_A)
                        grad_B_eff_norm = grad_B_effective.norm()
                        grad_final = grad_A
                    U.assign_flattened_grad(self.bx, grad_final.detach())
                    cls_loss_2 = cls_loss_2_total

                # total_loss = cls_loss_1 + norm_loss_1 + cls_loss_2 
                # total_loss.backward(retain_graph=False)

                with torch.no_grad():
                    self.bx.add_(-self.lr * self.bx.grad)
                    self.bx.clamp_(-self.budget, self.budget)  # budget is a scalar in JSON
                    

                # prepare for next iteration
                self.bx = self.bx.detach().requires_grad_(True)

                self.update_log(
                    image_id=image_id,
                    iteration=i,
                    cls_loss_1=cls_loss_1,
                    norm_loss_1=norm_loss_1,
                    obj_count_1=obj_count_1,
                    cls_loss_2=(cls_loss_2 if self.in_b_phase else torch.tensor(0.0, device=self.device)),
                    obj_count_2=(obj_count_2 if self.in_b_phase else 0),
                    grad_A_norm=grad_A.norm() if isinstance(grad_A, torch.Tensor) else None,
                    grad_B_norm=(grad_B.norm() if isinstance(grad_B, torch.Tensor) and hasattr(grad_B, 'norm') else None),
                    grad_final_norm=grad_final.norm() if isinstance(grad_final, torch.Tensor) else None,
                    grad_B_eff_norm=(grad_B_eff_norm if isinstance(grad_B_eff_norm, torch.Tensor) else None),
                    grad_cosine=grad_cosine,
                    phase_flag=int(self.in_b_phase),
                    a_drop_abs=drop_abs,
                    a_drop_rel=drop_rel,
                )

            # save final perturbed image for this sample (timestamp aligned with logs)
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
        if isinstance(self.adv_patch_xywh, list):
            pass
        else:
            # falls back to random size and position
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
        
        if x2 <= x1: x2 = min(W, x1 + 1)
        if y2 <= y1: y2 = min(H, y1 + 1)

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
        if self.config["output_log"] is not True: return
        log_entry = {
            "iteration": iteration,
            "obj_count_1": obj_count_1,
            "obj_count_2": obj_count_2,  
            "cls_loss_1":  self.unhook(cls_loss_1),
            "cls_loss_2":  self.unhook(cls_loss_2),
            "norm_loss_1": self.unhook(norm_loss_1),
            "norm_loss_2": self.unhook(norm_loss_2),
            # "total_loss":  self.unhook(total_loss),
            "grad_A_norm":     self.unhook(grad_A_norm),
            "grad_B_norm":     self.unhook(grad_B_norm),
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
            log_entry["phase"] = float(phase_flag)
        if a_drop_abs is not None:
            log_entry["a_drop_abs"] = float(a_drop_abs)
        if a_drop_rel is not None:
            log_entry["a_drop_rel"] = float(a_drop_rel)

        self.log_dict[image_id].append(log_entry)
        
        
    def write_log(self, log_path=None):
        if self.config["output_log"] is not True: return
        if log_path is None:
            log_path = "./logs"
        os.makedirs(log_path, exist_ok=True)
        
        # use the same timestamp as run_attack for consistency
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
    
    
def show_image(image_tensor, title=None):
    processed_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    plt.imsave('debug_image.png', processed_img)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    import argparse
    parser = argparse.ArgumentParser(description="dynamic deep learning pipeline system")
    parser.add_argument("--config_file", type=str, default="./config/test.json", help="Path to the config.json file")
    args = parser.parse_args()
    
    with open(args.config_file, "r") as f:
        config = json.load(f)
        
    pipeline = Pipeline(config, device)
    pipeline.load_model()
    dataset = pipeline.load_dataset()
    pipeline.run_attack(dataset)
