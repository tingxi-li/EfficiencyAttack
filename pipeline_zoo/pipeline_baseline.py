import torch
from . import utilities as U
from .base_pipeline import BasePipeline


class Pipeline(BasePipeline):
    """Baseline dual-model attack: optimize model_1 always, model_2 ROI after b_start_frac of iterations."""

    def __init__(self, config, device=None):
        super().__init__(config, device)
        self.b_start_frac = float(config.get("b_start_frac", 0.5))
        self.compute_grad_metrics = bool(config.get("compute_grad_metrics", False))

    def _init_image_state(self):
        pass

    def _attack_step(self, i, image_tensor):
        outputs_1 = self.model_1(image_tensor + self.patch * self.mask, output_hidden_states=True)
        probs = torch.sigmoid(outputs_1.logits)
        cls_loss_1 = U.calc_cls_loss(probs, self.target_labels[0])
        norm_loss_1 = self.calc_norm_loss(order=[""])

        detections = torch.cat(
            [outputs_1.pred_boxes, probs.max(dim=2)[0].unsqueeze(-1), outputs_1.logits], dim=-1
        )
        _labels1 = self.target_labels[0] if isinstance(self.target_labels[0], (list, tuple)) else []
        _probs1 = probs[..., _labels1] if len(_labels1) > 0 else probs
        obj_count_1 = (_probs1 > self.conf_threshold_1).sum().item()

        if self.patch.grad is not None:
            self.patch.grad.zero_()
        (self.cls_loss_weight_1 * cls_loss_1 + norm_loss_1).backward(retain_graph=False)

        cls_loss_2_total = torch.tensor(0.0, device=self.device)
        obj_count_2 = 0
        enable_b = (i + 1) / max(1, self.num_iterations) >= self.b_start_frac

        if enable_b:
            roi_masks_list = U.masks_from_boxes(image_tensor, detections, self.conf_threshold_1)
            flat_masks = [m for masks in roi_masks_list for m in masks]
            if flat_masks:
                Bf, C, H, W = image_tensor.shape
                min_tokens = max(1, (H // U.GRID_SIZE) * (W // U.GRID_SIZE))
                safe_queries = max(1, min(self.num_queries_2, min_tokens))
                if self.model_2.config.num_queries != safe_queries:
                    self.model_2.config.num_queries = safe_queries
                bs2 = max(1, self.model2_batch_size)
                for start in range(0, len(flat_masks), bs2):
                    end = min(start + bs2, len(flat_masks))
                    batch_pixel_masks = torch.stack(flat_masks[start:end], dim=0)
                    full_img = image_tensor + self.patch * self.mask
                    batch_images = full_img.expand(end - start, -1, -1, -1).contiguous()
                    _m = batch_pixel_masks.to(batch_images.dtype).unsqueeze(1).expand(-1, 3, -1, -1)
                    outputs = self.model_2(batch_images * _m, output_hidden_states=True)
                    probs_2 = torch.sigmoid(outputs.logits)
                    raw_loss_B = U.calc_cls_loss(probs_2, self.target_labels[1])
                    (self.cls_loss_weight_2 * raw_loss_B).backward(retain_graph=False)
                    cls_loss_2_total = cls_loss_2_total + raw_loss_B.detach()
                    _labels2 = self.target_labels[1] if isinstance(self.target_labels[1], (list, tuple)) else []
                    _probs2 = probs_2[..., _labels2] if len(_labels2) > 0 else probs_2
                    obj_count_2 += (_probs2 > self.conf_threshold_2).sum().item()

        with torch.no_grad():
            self.patch.add_(-self.lr * self.patch.grad)
            self.patch.clamp_(-self.budget, self.budget)
        self.patch = self.patch.detach().requires_grad_(True)

        total_loss = (
            self.cls_loss_weight_1 * cls_loss_1
            + norm_loss_1
            + self.cls_loss_weight_2 * cls_loss_2_total
        )
        return obj_count_1, obj_count_2, {
            "cls_loss_1": cls_loss_1,
            "cls_loss_2": cls_loss_2_total,
            "norm_loss_1": norm_loss_1,
            "total_loss": total_loss,
        }
