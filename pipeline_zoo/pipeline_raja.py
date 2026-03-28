import math
import os
import time
import json
import torch
import random
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.ops import roi_align, nms

from . import utilities as U
from .base_pipeline import BasePipeline


class Pipeline(BasePipeline):
    """
    RAJA: ROI-Align-based Joint Attack.
    Overrides run_attack entirely. Uses roi_align to crop proposal patches
    and runs PGD with optional momentum.
    Inherits: randomseed, load_dataset, load_model, write_log from BasePipeline.
    """

    def __init__(self, config, device=None):
        # Raja uses different hyperparameter names so we don't call super().__init__().
        # We set the attributes that inherited methods (load_dataset, load_model,
        # randomseed, write_log) need directly.
        self.device = device
        self.config = config

        self.dataset_name = config["dataset_name"]
        self.hf_model_name_1 = config["hf_model_name_1"]
        self.hf_model_name_2 = config["hf_model_name_2"]
        self.num_queries_1 = int(config.get("num_queries_1", 25))
        self.num_queries_2 = int(config.get("num_queries_2", 25))
        self.model2_batch_size = int(config.get("model2_batch_size", 8))

        targets = config.get("target_labels", [[], []])
        if len(targets) < 2:
            targets = list(targets) + [[]]
        self.target_labels_stage1 = targets[0]
        self.target_labels_stage2 = targets[1]
        self.target_labels = targets  # required by load_model

        self.raja_norm = str(config.get("raja_norm", "linf")).lower()
        self.raja_steps = int(config.get("raja_steps", config.get("num_iterations", 100)))
        self.raja_eps = float(config.get("raja_eps", 0.01))
        self.raja_step_size = float(
            config.get("raja_step_size", self.raja_eps / max(1, self.raja_steps))
        )
        self.raja_momentum = float(config.get("raja_momentum", 0.0))
        self.raja_topk = int(config.get("raja_topk", 200))
        self.raja_score_thresh = float(config.get("raja_score_thresh", 0.05))
        self.raja_nms_iou = float(config.get("raja_nms_iou", 0.6))
        self.raja_roi_size = int(config.get("raja_roi_size", 224))
        self.raja_roi_bleed = float(config.get("raja_roi_bleed", 0.0))
        self.raja_eot_k = int(config.get("raja_eot_k", 1))
        self.raja_jitter_center = float(config.get("raja_jitter_center", 0.0))
        self.raja_jitter_size = float(config.get("raja_jitter_size", 0.0))
        self.raja_log_grad_decomp = bool(config.get("raja_log_grad_decomp", False))
        self.raja_save_images = bool(config.get("raja_save_images", False))
        self.raja_hinge_weight_stage1 = float(config.get("raja_hinge_weight_stage1", 0.0))
        self.raja_hinge_weight_stage2 = float(config.get("raja_hinge_weight_stage2", 0.0))
        self.raja_hinge_topk = int(config.get("raja_hinge_topk", 0))

        self.conf_threshold_1 = float(config.get("conf_threshold_1", self.raja_score_thresh))
        self.conf_threshold_2 = float(config.get("conf_threshold_2", self.raja_score_thresh))

        self.log_dict = {}
        self.log_dir = config.get("log_dir", "./log")

        self.randomseed()

    # ---- ABC stubs (never called — run_attack is overridden) ----

    def _init_image_state(self):
        pass

    def _attack_step(self, i, image_tensor):
        raise NotImplementedError("PipelineRaja overrides run_attack entirely")

    # ---- Attack loop ----

    def run_attack(self, dataset):
        if not hasattr(self, "run_timestamp") or self.run_timestamp is None:
            self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")

        for _, example in tqdm(enumerate(dataset), total=len(dataset)):
            image_id = example["image_id"]
            image = example["image"].convert("RGB")
            clean_inputs = self.processor_1(images=image, return_tensors="pt")
            x_clean = clean_inputs["pixel_values"].to(self.device)

            delta = torch.zeros_like(x_clean, device=self.device)
            momentum = torch.zeros_like(x_clean, device=self.device)
            self.log_dict[image_id] = []

            for step_idx in range(self.raja_steps):
                x_adv = (x_clean + delta).clamp(0.0, 1.0).detach()
                x_adv.requires_grad_(True)

                model1_outputs = self.model_1(x_adv, output_hidden_states=True)
                probs_stage1 = torch.sigmoid(model1_outputs.logits)
                stage1_cls = U.calc_cls_loss(probs_stage1, self.target_labels_stage1)
                stage1_hinge = self._hinge_penalty(
                    probs_stage1, self.target_labels_stage1,
                    self.conf_threshold_1, self.raja_hinge_weight_stage1,
                )
                stage1_total = stage1_cls + stage1_hinge

                route = self._route_boxes(
                    model1_outputs.pred_boxes.detach(),
                    probs_stage1.detach(),
                    x_adv.shape,
                )
                stage2_cls, stage2_hinge, false_pos_stage2 = self._forward_stage2(
                    x_adv, route["boxes_norm"]
                )
                stage2_total = stage2_cls + stage2_hinge
                total_loss = stage1_total + stage2_total

                grad_stage1 = None
                grad_stage2 = None
                if self.raja_log_grad_decomp:
                    retain_for_stage2 = stage2_total.requires_grad
                    if stage1_total.requires_grad:
                        grad_stage1 = torch.autograd.grad(
                            stage1_total, x_adv,
                            retain_graph=retain_for_stage2, allow_unused=True,
                        )[0]
                    if stage2_total.requires_grad:
                        grad_stage2 = torch.autograd.grad(
                            stage2_total, x_adv,
                            retain_graph=False, allow_unused=True,
                        )[0]
                    grad = self._combine_grads(grad_stage1, grad_stage2)
                else:
                    grad = torch.autograd.grad(
                        total_loss, x_adv, retain_graph=False, allow_unused=True
                    )[0]

                if grad is None:
                    break

                delta, momentum = self._pgd_step(x_clean, delta, grad, momentum)

                num_boxes = int(route["boxes_norm"].shape[0])
                false_pos_stage1 = self._count_false_pos(
                    probs_stage1, self.target_labels_stage1, self.conf_threshold_1
                )
                log_entry = {
                    "step": step_idx,
                    "loss_stage1": float(stage1_total.detach().cpu().item()),
                    "loss_stage2": float(stage2_total.detach().cpu().item()),
                    "loss_total": float(total_loss.detach().cpu().item()),
                    "grad_norm": float(self._grad_norm(grad)),
                    "false_pos_stage1": int(false_pos_stage1),
                    "false_pos_stage2": float(false_pos_stage2),
                    "num_boxes": num_boxes,
                }
                if self.raja_log_grad_decomp:
                    roi_energy_ratio = self._roi_energy_ratio(
                        grad.detach(), route["boxes_xyxy"], x_adv.shape[-2], x_adv.shape[-1],
                    )
                    cosine, sign_agree = self._grad_alignment_metrics(grad_stage1, grad_stage2)
                    log_entry.update({
                        "cosine": cosine,
                        "sign_agreement": sign_agree,
                        "norm_g1": float(self._grad_norm(grad_stage1)),
                        "norm_g2": float(self._grad_norm(grad_stage2)),
                        "roi_energy_ratio": roi_energy_ratio,
                    })
                self.log_dict[image_id].append(log_entry)

            final_adv = (x_clean + delta).clamp(0.0, 1.0)
            if self.raja_save_images:
                self._save_image(final_adv, image_id)

        self.write_log()

    # ---- Helper methods ----

    def _route_boxes(self, pred_boxes, probs, image_shape):
        B, _, _ = pred_boxes.shape
        H, W = image_shape[-2], image_shape[-1]
        if B != 1:
            raise NotImplementedError("RAJA pipeline currently supports batch size 1.")
        boxes = pred_boxes[0]
        scores = self._extract_scores(probs[0], self.target_labels_stage1)
        if boxes.numel() == 0 or scores.numel() == 0:
            return {
                "boxes_norm": boxes.new_zeros((0, 4)),
                "boxes_xyxy": boxes.new_zeros((0, 4)),
                "scores": scores.new_zeros((0,)),
            }
        mask = scores > self.raja_score_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        if boxes.numel() == 0:
            return {
                "boxes_norm": boxes.new_zeros((0, 4)),
                "boxes_xyxy": boxes.new_zeros((0, 4)),
                "scores": scores.new_zeros((0,)),
            }
        boxes_xyxy = self._cxcywh_to_xyxy(boxes, W, H)
        keep = nms(boxes_xyxy, scores, self.raja_nms_iou)
        boxes = boxes[keep]
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        if self.raja_topk > 0 and boxes.shape[0] > self.raja_topk:
            values, indices = scores.topk(self.raja_topk)
            boxes = boxes[indices]
            boxes_xyxy = boxes_xyxy[indices]
            scores = values
        return {"boxes_norm": boxes, "boxes_xyxy": boxes_xyxy, "scores": scores}

    def _forward_stage2(self, x_adv, boxes_norm):
        if boxes_norm.numel() == 0:
            zero = x_adv.new_zeros(())
            return zero, zero, 0
        eot_runs = max(1, self.raja_eot_k)
        H, W = x_adv.shape[-2], x_adv.shape[-1]
        accumulated_cls = x_adv.new_zeros(())
        accumulated_hinge = x_adv.new_zeros(())
        accumulated_fp = 0.0
        for _ in range(eot_runs):
            jittered = self._jitter_boxes(boxes_norm)
            jittered_xyxy = self._cxcywh_to_xyxy(jittered, W, H, bleed=self.raja_roi_bleed)
            rois = self._build_roi_tensor(jittered_xyxy, device=x_adv.device)
            if rois.numel() == 0:
                continue
            patches = roi_align(x_adv, rois, output_size=self.raja_roi_size, aligned=True)
            roi_count = max(1, patches.shape[0])
            step_cls = x_adv.new_zeros(())
            step_hinge = x_adv.new_zeros(())
            step_fp = 0.0
            for start in range(0, patches.shape[0], self.model2_batch_size):
                end = min(start + self.model2_batch_size, patches.shape[0])
                batch_patches = patches[start:end]
                outputs2 = self.model_2(batch_patches, output_hidden_states=True)
                probs_stage2 = torch.sigmoid(outputs2.logits)
                chunk_loss = U.calc_cls_loss(probs_stage2, self.target_labels_stage2)
                chunk_hinge = self._hinge_penalty(
                    probs_stage2, self.target_labels_stage2,
                    self.conf_threshold_2, self.raja_hinge_weight_stage2,
                )
                step_cls = step_cls + chunk_loss
                step_hinge = step_hinge + chunk_hinge
                step_fp += self._count_false_pos(
                    probs_stage2, self.target_labels_stage2, self.conf_threshold_2
                )
            accumulated_cls = accumulated_cls + step_cls / roi_count
            accumulated_hinge = accumulated_hinge + step_hinge / roi_count
            accumulated_fp += step_fp
        accumulated_cls = accumulated_cls / eot_runs
        accumulated_hinge = accumulated_hinge / eot_runs
        accumulated_fp = accumulated_fp / eot_runs
        return accumulated_cls, accumulated_hinge, accumulated_fp

    def _pgd_step(self, x_clean, delta, grad, momentum):
        grad = grad.detach()
        momentum = momentum.detach()
        if self.raja_norm == "linf":
            normalized_grad = grad.sign()
        elif self.raja_norm == "l2":
            flat = grad.view(grad.shape[0], -1)
            norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            broadcast_shape = (grad.shape[0],) + (1,) * (grad.dim() - 1)
            normalized_grad = grad / norm.view(broadcast_shape)
        else:
            raise ValueError(f"Unsupported RAJA norm: {self.raja_norm}")
        momentum = self.raja_momentum * momentum + normalized_grad
        step_direction = momentum if self.raja_momentum != 0 else normalized_grad
        delta = delta + self.raja_step_size * step_direction
        if self.raja_norm == "linf":
            delta = delta.clamp(-self.raja_eps, self.raja_eps)
        else:
            flat = delta.view(delta.shape[0], -1)
            norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.minimum(torch.ones_like(norm), self.raja_eps / norm)
            delta = (flat * scale).view_as(delta)
        x_adv = (x_clean + delta).clamp(0.0, 1.0)
        delta = x_adv - x_clean
        return delta.detach(), momentum.detach()

    def _jitter_boxes(self, boxes_norm):
        if boxes_norm.numel() == 0:
            return boxes_norm
        boxes = boxes_norm.clone()
        if self.raja_jitter_center > 0:
            boxes[..., 0:2] = boxes[..., 0:2] + torch.randn_like(boxes[..., 0:2]) * self.raja_jitter_center
        if self.raja_jitter_size > 0:
            boxes[..., 2:4] = boxes[..., 2:4] * (1.0 + torch.randn_like(boxes[..., 2:4]) * self.raja_jitter_size)
        cx, cy = boxes[..., 0], boxes[..., 1]
        w = boxes[..., 2].clamp_min(1e-4)
        h = boxes[..., 3].clamp_min(1e-4)
        x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
        w = (x2 - x1).clamp_min(1e-4)
        h = (y2 - y1).clamp_min(1e-4)
        return torch.stack([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dim=-1)

    def _build_roi_tensor(self, boxes_xyxy, device):
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0, 5), device=device)
        batch_idx = torch.zeros((boxes_xyxy.shape[0], 1), device=device)
        return torch.cat([batch_idx, boxes_xyxy], dim=1)

    def _cxcywh_to_xyxy(self, boxes, width, height, bleed=0.0):
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        if bleed > 0:
            w = w * (1.0 + bleed)
            h = h * (1.0 + bleed)
        coords = torch.stack(
            [(cx - 0.5 * w) * width, (cy - 0.5 * h) * height,
             (cx + 0.5 * w) * width, (cy + 0.5 * h) * height], dim=-1
        )
        coords[..., 0::2] = coords[..., 0::2].clamp(0.0, width)
        coords[..., 1::2] = coords[..., 1::2].clamp(0.0, height)
        return coords

    def _extract_scores(self, probs, target_labels):
        if target_labels is None or len(target_labels) == 0:
            return probs.max(dim=-1).values
        idx = torch.tensor(target_labels, device=probs.device, dtype=torch.long)
        return probs.index_select(dim=-1, index=idx).max(dim=-1).values

    def _hinge_penalty(self, probs, target_labels, threshold, weight):
        if weight <= 0 or self.raja_hinge_topk <= 0:
            return probs.new_zeros(())
        scores = self._extract_scores(probs, target_labels)
        if scores.numel() == 0:
            return probs.new_zeros(())
        scores_flat = scores.reshape(-1)
        topk = min(scores_flat.numel(), int(self.raja_hinge_topk))
        if topk <= 0:
            return probs.new_zeros(())
        top_scores, _ = scores_flat.topk(topk)
        return weight * torch.clamp(top_scores - threshold, min=0.0).mean()

    def _combine_grads(self, g1, g2):
        grad = None
        if g1 is not None:
            grad = g1.clone()
        if g2 is not None:
            grad = g2.clone() if grad is None else grad + g2
        return grad

    def _grad_norm(self, grad):
        if grad is None:
            return 0.0
        return float(grad.view(grad.shape[0], -1).norm(p=2, dim=1).mean().item())

    def _grad_alignment_metrics(self, g1, g2):
        if g1 is None or g2 is None:
            return None, None
        flat1 = g1.view(g1.shape[0], -1)
        flat2 = g2.view(g2.shape[0], -1)
        dot = (flat1 * flat2).sum(dim=1)
        norm1 = flat1.norm(p=2, dim=1).clamp_min(1e-12)
        norm2 = flat2.norm(p=2, dim=1).clamp_min(1e-12)
        cosine = (dot / (norm1 * norm2)).mean().item()
        sign_match = torch.eq(torch.sign(flat1), torch.sign(flat2)).float().mean().item()
        return float(cosine), float(sign_match)

    def _count_false_pos(self, probs, target_labels, threshold):
        scores = self._extract_scores(probs, target_labels)
        if scores.numel() == 0:
            return 0
        return int((scores > threshold).sum().item())

    def _roi_energy_ratio(self, grad, boxes_xyxy, height, width):
        if grad is None or boxes_xyxy.numel() == 0:
            return None
        mask = self._boxes_to_mask(boxes_xyxy, height, width, grad.device, grad.shape[1])
        grad_sq = grad.pow(2)
        total_energy = grad_sq.sum()
        if total_energy.item() <= 0:
            return 0.0
        return float((grad_sq * mask).sum() / total_energy)

    def _boxes_to_mask(self, boxes_xyxy, height, width, device, channels):
        mask = torch.zeros((1, 1, height, width), device=device)
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, min(width, int(math.floor(x1))))
            y1 = max(0, min(height, int(math.floor(y1))))
            x2 = max(0, min(width, int(math.ceil(x2))))
            y2 = max(0, min(height, int(math.ceil(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[:, :, y1:y2, x1:x2] = 1.0
        return mask.expand(-1, channels, -1, -1)

    def _save_image(self, tensor, image_id):
        stemname = type(self).__module__.split(".")[-1]
        out_dir = os.path.join("output", f"{stemname}_{self.run_timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        img_np = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        plt.imsave(os.path.join(out_dir, f"{image_id}.png"), img_np)
