import torch
from . import utilities as U
from .base_pipeline import BasePipeline


class Pipeline(BasePipeline):
    """
    Two-phase plateau-detection attack with gradient projection.

    primary_model=1 (A-primary): optimize model_1 until plateau, then add model_2 gradient
        projected onto the orthogonal complement of model_1's gradient.
    primary_model=2 (B-primary): project model_1 gradient onto model_2's direction,
        then after model_2 plateaus, add model_2 gradient as well.
    """

    def __init__(self, config, device=None):
        super().__init__(config, device)
        self.primary_model = int(config.get("primary_model", 1))

        # Plateau detection settings (shared config keys for both primary modes)
        self.a_window = config.get("a_phase_window", 20)
        self.a_min_delta = float(config.get("a_phase_min_delta", 0.0))
        self.a_patience = int(config.get("a_phase_patience", 3))
        self.a_min_iters = int(config.get("a_phase_min_iters", 10))
        try:
            n = int(self.num_iterations)
            win_frac = config.get("a_phase_window_frac")
            if win_frac is not None:
                self.a_window = max(5, int(round(float(win_frac) * n)))
            pat_frac = config.get("a_phase_patience_frac")
            if pat_frac is not None:
                self.a_patience = max(1, int(round(float(pat_frac) * self.a_window)))
            min_iters_frac = config.get("a_phase_min_iters_frac")
            if min_iters_frac is not None:
                self.a_min_iters = max(1, int(round(float(min_iters_frac) * n)))
        except Exception:
            pass
        self.a_min_rel = float(config.get("a_phase_min_rel", 0.0))
        self.a_slope_thresh = float(config.get("a_phase_slope_thresh", 0.0))
        self.a_use_window_best = bool(config.get("a_phase_use_window_best", True))
        self.a_min_drop = float(config.get("a_phase_min_drop", 0.0))
        self.a_min_drop_rel = float(config.get("a_phase_min_drop_rel", 0.0))

        # Gradient projection settings
        self.projection_conflict_threshold = float(config.get("projection_conflict_threshold", 0.0))
        self.projection_match_norm = bool(config.get("projection_match_norm", True))
        self.b_grad_scale = float(config.get("b_grad_scale", 1.0))
        self.projection_eps = float(config.get("projection_eps", 1e-12))

    def _init_image_state(self):
        self.in_b_phase = False
        self.primary_plateau = U.LossPlateauDetector(
            self.a_window, self.a_min_delta, self.a_patience,
            min_rel=self.a_min_rel,
            slope_thresh=self.a_slope_thresh,
            use_window_best=self.a_use_window_best,
        )
        self.primary_initial_loss = None
        self.primary_best_loss = None

    def _attack_step(self, i, image_tensor):
        img_pert = image_tensor + self.patch * self.mask
        eps = self.projection_eps

        # Model 1 forward — always needed for grad_A and obj_count_1
        outputs_1 = self.model_1(img_pert, output_hidden_states=True)
        probs_1 = torch.sigmoid(outputs_1.logits)
        cls_loss_1 = U.calc_cls_loss(probs_1, self.target_labels[0])
        norm_loss_1 = self.calc_norm_loss(order=[""])
        A_loss = self.cls_loss_weight_1 * cls_loss_1 + norm_loss_1
        _labels1 = self.target_labels[0] if isinstance(self.target_labels[0], (list, tuple)) else []
        obj_count_1 = (
            (probs_1[..., _labels1] if len(_labels1) > 0 else probs_1) > self.conf_threshold_1
        ).sum().item()

        # ROI masks: from model_1 boxes for A-primary, from model_2 boxes for B-primary
        if self.primary_model == 1:
            detections = torch.cat(
                [outputs_1.pred_boxes, probs_1.max(dim=2)[0].unsqueeze(-1), outputs_1.logits], dim=-1
            )
            roi_masks_list = U.masks_from_boxes(image_tensor, detections, self.conf_threshold_1)
        else:
            with torch.no_grad():
                outputs_2_full = self.model_2(img_pert, output_hidden_states=False)
                probs_2_full = torch.sigmoid(outputs_2_full.logits)
            det2 = torch.cat(
                [outputs_2_full.pred_boxes, probs_2_full.max(dim=2)[0].unsqueeze(-1), outputs_2_full.logits], dim=-1
            )
            roi_masks_list = U.masks_from_boxes(image_tensor, det2, self.conf_threshold_2)
        flat_masks = [m for masks in roi_masks_list for m in masks]

        # Compute grad_A first — must happen before any B backward that shares the graph
        grad_A = U.grad_wrt(self.patch, A_loss, retain_graph=False)

        # Compute grad_B: always for B-primary; only in phase B for A-primary.
        # B micro-batches recompute (image_tensor + self.patch * self.mask) fresh so their
        # graphs are independent from A_loss (which was already freed above).
        cls_loss_2_total = torch.tensor(0.0, device=self.device)
        obj_count_2 = 0
        grad_B = torch.zeros_like(grad_A)
        need_b_grad = (self.primary_model == 2) or self.in_b_phase

        if need_b_grad and flat_masks:
            Bf, C, H, W = image_tensor.shape
            min_tokens = max(1, (H // U.GRID_SIZE) * (W // U.GRID_SIZE))
            safe_queries = max(1, min(self.num_queries_2, min_tokens))
            if self.model_2.config.num_queries != safe_queries:
                self.model_2.config.num_queries = safe_queries
            bs2 = max(1, self.model2_batch_size)
            _labels2 = self.target_labels[1] if isinstance(self.target_labels[1], (list, tuple)) else []
            for start in range(0, len(flat_masks), bs2):
                end = min(start + bs2, len(flat_masks))
                batch_pixel_masks = torch.stack(flat_masks[start:end], dim=0)
                # Recompute fresh — independent graph from A_loss
                full_img = image_tensor + self.patch * self.mask
                batch_images = full_img.expand(end - start, -1, -1, -1).contiguous()
                _m = batch_pixel_masks.to(batch_images.dtype).unsqueeze(1).expand(-1, 3, -1, -1)
                outputs = self.model_2(batch_images * _m, output_hidden_states=True)
                probs_2 = torch.sigmoid(outputs.logits)
                loss_B = U.calc_cls_loss(probs_2, self.target_labels[1])
                grad_B_micro = U.grad_wrt(
                    self.patch, self.cls_loss_weight_2 * loss_B, retain_graph=False
                )
                grad_B = grad_B + grad_B_micro
                cls_loss_2_total = cls_loss_2_total + loss_B.detach()
                obj_count_2 += (
                    (probs_2[..., _labels2] if len(_labels2) > 0 else probs_2) > self.conf_threshold_2
                ).sum().item()
            denom = float(max(1, len(flat_masks)))
            grad_B = grad_B / denom
            cls_loss_2_total = cls_loss_2_total / denom

        # Plateau detection on primary model's loss
        if self.primary_model == 1:
            tracked_loss = A_loss.detach().item()
        else:
            tracked_loss = (self.cls_loss_weight_2 * cls_loss_2_total + norm_loss_1).detach().item()

        if self.primary_initial_loss is None:
            self.primary_initial_loss = tracked_loss
            self.primary_best_loss = tracked_loss
        else:
            self.primary_best_loss = min(self.primary_best_loss, tracked_loss)

        drop_abs = self.primary_initial_loss - self.primary_best_loss
        drop_rel = (
            drop_abs / abs(self.primary_initial_loss)
            if abs(self.primary_initial_loss) > 1e-12
            else 0.0
        )
        drop_ok = (self.a_min_drop <= 0.0 or drop_abs >= self.a_min_drop) and (
            self.a_min_drop_rel <= 0.0 or drop_rel >= self.a_min_drop_rel
        )
        plateau = self.primary_plateau.update(tracked_loss)
        if not self.in_b_phase and i >= self.a_min_iters and plateau and drop_ok:
            self.in_b_phase = True

        if self.primary_model == 1:
            if not self.in_b_phase:
                grad_final = grad_A
            else:
                norm_B = grad_B.norm()
                if norm_B > eps:
                    grad_B_eff = U.project_onto_orthogonal_complement(grad_B, [grad_A], eps=eps)
                    norm_proj = grad_B_eff.norm()
                    if self.projection_match_norm and norm_proj > eps:
                        grad_B_eff = grad_B_eff * (norm_B / (norm_proj + eps))
                    grad_final = grad_A + self.b_grad_scale * grad_B_eff
                else:
                    grad_final = grad_A
        else:
            # B-primary: project grad_A onto grad_B direction
            norm_B = grad_B.norm()
            if norm_B > eps:
                proj_coeff = torch.dot(grad_A, grad_B) / (norm_B * norm_B + eps)
                grad_A_on_B = proj_coeff * grad_B
            else:
                grad_A_on_B = torch.zeros_like(grad_A)
            if not self.in_b_phase:
                grad_final = grad_A_on_B
            else:
                grad_final = grad_A_on_B + self.b_grad_scale * grad_B

        U.assign_flattened_grad(self.patch, grad_final.detach())

        with torch.no_grad():
            self.patch.add_(-self.lr * self.patch.grad)
            self.patch.clamp_(-self.budget, self.budget)
        self.patch = self.patch.detach().requires_grad_(True)

        return obj_count_1, obj_count_2, {
            "cls_loss_1": cls_loss_1,
            "cls_loss_2": cls_loss_2_total,
            "norm_loss_1": norm_loss_1,
            "phase_flag": int(self.in_b_phase),
            "a_drop_abs": drop_abs,
            "a_drop_rel": drop_rel,
        }
