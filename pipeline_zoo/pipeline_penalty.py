import torch
from . import utilities as U
from .base_pipeline import BasePipeline


class Pipeline(BasePipeline):
    """
    Two-phase ALM penalty attack.
    Phase A: optimize model_1 until plateau.
    Phase B: optimize model_2 ROI with PCGrad-style projection, plus ALM penalty to
             guard the phase-A loss from increasing.
    """

    def __init__(self, config, device=None):
        super().__init__(config, device)

        # Plateau detection
        self.a_window = config.get("a_phase_window", 20)
        self.a_min_delta = float(config.get("a_phase_min_delta", 0.0))
        self.a_patience = int(config.get("a_phase_patience", 3))
        self.a_min_iters = int(config.get("a_phase_min_iters", 10))
        self.a_margin = float(config.get("a_phase_margin", 0.0))
        self.a_min_rel = float(config.get("a_phase_min_rel", 0.0))
        self.a_slope_thresh = float(config.get("a_phase_slope_thresh", 0.0))
        self.a_use_window_best = bool(config.get("a_phase_use_window_best", True))
        self.a_min_drop = float(config.get("a_phase_min_drop", 0.0))
        self.a_min_drop_rel = float(config.get("a_phase_min_drop_rel", 0.0))

        # ALM parameters
        self.alm_lambda_init = float(config.get("alm_lambda_init", 0.0))
        self.alm_rho = float(config.get("alm_rho_init", 1.0))
        self.alm_penalty_step_scale = float(config.get("alm_penalty_step_scale", 0.25))
        self.alm_grad_clip = float(config.get("alm_grad_clip", 5.0))
        self.alm_g_tolerance = float(config.get("alm_g_tolerance", 1e-4))
        self.alm_focus_A_when_violated = bool(config.get("alm_focus_A_when_violated", True))
        self.alm_lambda_step = float(config.get("alm_lambda_step", self.alm_rho))
        self.alm_lambda_max = float(config.get("alm_lambda_max", 1e4))

        # Gradient projection (PCGrad-style)
        self.projection_conflict_threshold = float(config.get("projection_conflict_threshold", 0.0))
        self.projection_match_norm = bool(config.get("projection_match_norm", True))
        self.b_grad_scale = float(config.get("b_grad_scale", 1.0))
        self.projection_eps = float(config.get("projection_eps", 1e-12))

    def _init_image_state(self):
        self.in_b_phase = False
        self.a_plateau = U.LossPlateauDetector(
            self.a_window, self.a_min_delta, self.a_patience,
            min_rel=self.a_min_rel,
            slope_thresh=self.a_slope_thresh,
            use_window_best=self.a_use_window_best,
        )
        self.a_initial_loss = None
        self.a_best_loss = None
        self.alm_lambda = self.alm_lambda_init
        self.a_target = None

    def _attack_step(self, i, image_tensor):
        outputs_1 = self.model_1(image_tensor + self.patch * self.mask, output_hidden_states=True)
        probs = torch.sigmoid(outputs_1.logits)
        cls_loss_1 = U.calc_cls_loss(probs, self.target_labels[0])
        norm_loss_1 = self.calc_norm_loss(order=[""])
        A_loss = self.cls_loss_weight_1 * cls_loss_1 + norm_loss_1

        detections = torch.cat(
            [outputs_1.pred_boxes, probs.max(dim=2)[0].unsqueeze(-1), outputs_1.logits], dim=-1
        )
        _labels1 = self.target_labels[0] if isinstance(self.target_labels[0], (list, tuple)) else []
        obj_count_1 = (
            (probs[..., _labels1] if len(_labels1) > 0 else probs) > self.conf_threshold_1
        ).sum().item()

        roi_masks_list = U.masks_from_boxes(image_tensor, detections, self.conf_threshold_1)
        flat_masks = [m for masks in roi_masks_list for m in masks]

        # Plateau scheduling on A loss
        A_loss_value = A_loss.detach().item()
        if self.a_initial_loss is None:
            self.a_initial_loss = A_loss_value
            self.a_best_loss = A_loss_value
        else:
            self.a_best_loss = min(self.a_best_loss, A_loss_value)

        drop_abs = self.a_initial_loss - self.a_best_loss
        drop_rel = drop_abs / abs(self.a_initial_loss) if abs(self.a_initial_loss) > 1e-12 else 0.0
        drop_ok = (self.a_min_drop <= 0.0 or drop_abs >= self.a_min_drop) and (
            self.a_min_drop_rel <= 0.0 or drop_rel >= self.a_min_drop_rel
        )
        plateau = self.a_plateau.update(A_loss_value)
        if not self.in_b_phase and i >= self.a_min_iters and plateau and drop_ok:
            self.in_b_phase = True
            self.a_target = (self.a_best_loss if self.a_best_loss is not None else A_loss_value) + self.a_margin

        eps = self.projection_eps
        cls_loss_2 = torch.tensor(0.0, device=self.device)
        obj_count_2 = 0

        # Compute grad_A first (before any B backward that could free A's graph)
        grad_A = U.grad_wrt(self.patch, A_loss, retain_graph=False)

        if not self.in_b_phase:
            grad_final = grad_A
            total_loss = A_loss
        else:
            grad_B = torch.zeros_like(grad_A)
            cls_loss_2_total = torch.tensor(0.0, device=self.device)

            if flat_masks:
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
            cls_loss_2 = cls_loss_2_total

            # PCGrad-style projection: remove grad_A component from grad_B if conflicting
            norm_A = grad_A.norm()
            norm_B = grad_B.norm()
            grad_B_effective = torch.zeros_like(grad_A)
            if norm_B > eps:
                grad_cosine = None
                if norm_A > eps:
                    grad_cosine = torch.dot(grad_A, grad_B) / (norm_A * norm_B + eps)
                grad_B_effective = grad_B.clone()
                if grad_cosine is not None and grad_cosine <= self.projection_conflict_threshold:
                    grad_B_effective = U.project_onto_orthogonal_complement(grad_B, [grad_A], eps=eps)
                    norm_proj = grad_B_effective.norm()
                    if self.projection_match_norm and norm_proj > eps:
                        grad_B_effective = grad_B_effective * (norm_B / (norm_proj + eps))
                grad_B_effective = grad_B_effective * self.b_grad_scale

            # ALM penalty: guard A loss from exceeding a_target
            g_val = A_loss_value - self.a_target
            g_pos = max(0.0, g_val)
            indicator = 1.0 if g_pos > 0.0 else 0.0
            scale = (self.alm_lambda + self.alm_rho * g_pos) * indicator
            grad_penalty = self.alm_penalty_step_scale * scale * grad_A

            if self.alm_focus_A_when_violated and g_pos > self.alm_g_tolerance:
                grad_B_effective = torch.zeros_like(grad_B_effective)

            grad_final = grad_A + grad_B_effective + grad_penalty
            grad_final_norm = grad_final.norm().item()
            if self.alm_grad_clip > 0.0 and grad_final_norm > self.alm_grad_clip:
                grad_final = grad_final * (self.alm_grad_clip / (grad_final_norm + 1e-12))

            # Update ALM multiplier and target
            with torch.no_grad():
                if g_pos > self.alm_g_tolerance:
                    self.alm_lambda = max(
                        0.0, min(self.alm_lambda_max, self.alm_lambda + self.alm_lambda_step * g_pos)
                    )
                if A_loss_value + 1e-12 < (self.a_target - self.a_min_delta):
                    self.a_target = A_loss_value + self.a_margin

            penalty_value = self.alm_lambda * g_pos + 0.5 * self.alm_rho * g_pos * g_pos
            total_loss = self.cls_loss_weight_2 * cls_loss_2_total + torch.tensor(
                penalty_value, device=self.device
            )

        U.assign_flattened_grad(self.patch, grad_final.detach())

        with torch.no_grad():
            self.patch.add_(-self.lr * self.patch.grad)
            self.patch.clamp_(-self.budget, self.budget)
        self.patch = self.patch.detach().requires_grad_(True)

        extra = {
            "cls_loss_1": cls_loss_1,
            "cls_loss_2": cls_loss_2,
            "norm_loss_1": norm_loss_1,
            "total_loss": total_loss,
            "phase_flag": int(self.in_b_phase),
            "a_drop_abs": drop_abs,
            "a_drop_rel": drop_rel,
        }
        if self.in_b_phase:
            extra["alm_lambda"] = self.alm_lambda
            extra["a_target"] = self.a_target
        return obj_count_1, obj_count_2, extra
