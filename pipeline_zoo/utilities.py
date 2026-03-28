import torch
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch

GRID_SIZE = 32  # RT-DETR backbone token grid size (used for safe num_queries computation)

############################################
# Visualization helpers (boxes, debug)
############################################

def draw_boxes(image_tensor, combined, thres):
    # pred_boxes are in [B, box_num, 4], [cx,cy,w,h] normalized to [0,1]
    # image_tensor is in [B,C,H,W] and in [0,1]
    imgs = image_tensor.clone().detach()
    B, C, H, W = imgs.shape
    pred_boxes = combined[..., :4]   # [B, box_num, 4]
    confidence = combined[..., 4]        # [B, box_num]
    for i in range(B):
        for box, conf in zip(pred_boxes[i], confidence[i]):
            if conf < thres:
                continue
            cx, cy, w, h = box.tolist()

            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            if C == 3:
                imgs[i, 0, y1:y2, x1] = 1.0  # left
                imgs[i, 1, y1:y2, x1] = 0.0
                imgs[i, 2, y1:y2, x1] = 0.0

                imgs[i, 0, y1:y2, x2] = 1.0  # right
                imgs[i, 1, y1:y2, x2] = 0.0
                imgs[i, 2, y1:y2, x2] = 0.0

                imgs[i, 0, y1, x1:x2] = 1.0  # top
                imgs[i, 1, y1, x1:x2] = 0.0
                imgs[i, 2, y1, x1:x2] = 0.0

                imgs[i, 0, y2, x1:x2] = 1.0  # bottom
                imgs[i, 1, y2, x1:x2] = 0.0
                imgs[i, 2, y2, x1:x2] = 0.0
            else:
                imgs[i, 0, y1:y2, x1] = 1.0
                imgs[i, 0, y1:y2, x2] = 1.0
                imgs[i, 0, y1, x1:x2] = 1.0
                imgs[i, 0, y2, x1:x2] = 1.0

    return imgs



def cutout(image_tensor, pred_boxes, thres):
    # pred_boxes are in [cx,cy,w,h] normalized to [0,1]
    # image_tensor is in [B,C,H,W] and in [0,1]
    B, C, H, W = image_tensor.shape
    # 确保只取前 5 个元素：cx, cy, w, h, score
    pred_boxes = pred_boxes[..., :5]      # [B, box_num, 5]
    results = []

    def _pad_to_multiple_32(patch: torch.Tensor, value: float = 0.0):
        # patch: [C,h,w]
        _, h, w = patch.shape
        hn = (h + 31) // 32 * 32
        wn = (w + 31) // 32 * 32
        if hn == h and wn == w:
            return patch
        # pad = (left, right, top, bottom)
        pad = (0, wn - w, 0, hn - h)
        return F.pad(patch, pad, mode='constant', value=value)

    for b in range(B):
        boxes = pred_boxes[b]
        crops = []
        # 用 .tolist() 简化循环（数据量不大时足够；需要高性能可再向量化）
        for cx, cy, w, h, score in boxes.tolist():
            if score < thres:
                continue

            # 归一化 -> 像素；用 floor/ceil，右/下边界取独占
            x1 = math.floor((cx - w / 2.0) * W)
            y1 = math.floor((cy - h / 2.0) * H)
            x2 = math.ceil ((cx + w / 2.0) * W)
            y2 = math.ceil ((cy + h / 2.0) * H)

            # clamp 到有效范围（上界独占）
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W,     x2))
            y2 = max(0, min(H,     y2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_tensor[b, :, y1:y2, x1:x2]  # [C, h, w]
            crop = _pad_to_multiple_32(crop, value=0.0)
            crops.append(crop)
        results.append(crops)

    return results


def masks_from_boxes(image_tensor, pred_boxes, thres):
    """
    Build boolean pixel masks for each box above threshold.
    - image_tensor: [B,C,H,W]
    - pred_boxes: [..., >=5] where first 4 are [cx,cy,w,h] in [0,1], 5th is score
    Returns: list over batch; each inner list contains masks shaped [H,W] (bool)
    """
    B, C, H, W = image_tensor.shape
    pred_boxes = pred_boxes[..., :5]  # [B, box_num, 5]
    results = []

    for b in range(B):
        boxes = pred_boxes[b]
        masks = []
        for cx, cy, w, h, score in boxes.tolist():
            if score < thres:
                continue
            # to pixel coordinates (right/bottom exclusive)
            x1 = math.floor((cx - w / 2.0) * W)
            y1 = math.floor((cy - h / 2.0) * H)
            x2 = math.ceil ((cx + w / 2.0) * W)
            y2 = math.ceil ((cy + h / 2.0) * H)

            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W,     x2))
            y2 = max(0, min(H,     y2))
            if x2 <= x1 or y2 <= y1:
                continue

            m = torch.zeros((H, W), dtype=torch.bool, device=image_tensor.device)
            m[y1:y2, x1:x2] = True
            masks.append(m)
        results.append(masks)

    return results


def calc_cls_loss(probs, target_labels):
    batch_size, num_of_query, number_of_cls = probs.shape
    target_tensor = torch.zeros_like(probs)
    if len(target_labels) > 0:
        for i in target_labels:
            target_tensor[:, :, i] = 1.0
    else:
        target_tensor = torch.ones_like(probs)
    cls_loss = F.mse_loss(probs, target_tensor, reduction='sum') / (target_tensor.sum() + 1)
    return cls_loss
    
############################################
# Gradient helpers and optimization utilities
############################################

def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1)


def grad_wrt(param: torch.Tensor, loss: torch.Tensor, retain_graph: bool = False, create_graph: bool = False) -> torch.Tensor:
    """Compute gradient of loss w.r.t param and return a flattened vector.
    If the loss is not connected to param, returns a zero vector of matching size.
    The function does not set param.grad; it returns the gradient tensor.
    """
    if param.grad is not None:
        param.grad = None
    if loss is None or (isinstance(loss, torch.Tensor) and not loss.requires_grad):
        return torch.zeros_like(param).view(-1)
    g = torch.autograd.grad(loss, param, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)[0]
    if g is None:
        return torch.zeros_like(param).view(-1)
    return _flatten(g)


def project_onto_orthogonal_complement(g: torch.Tensor, basis_list: list[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """Project g onto the orthogonal complement of the span of basis_list.
    All vectors are assumed flattened. Returns the residual vector.
    """
    r = g
    for b in basis_list:
        if b is None:
            continue
        bn = b.norm().clamp_min(eps)
        bhat = b / bn
        coeff = torch.dot(r, bhat)
        r = r - coeff * bhat
    return r


def assign_flattened_grad(param: torch.Tensor, flat_grad: torch.Tensor) -> None:
    """Assign a flattened gradient vector back to param.grad with correct shape."""
    param.grad = flat_grad.view_as(param).clone()


class LossPlateauDetector:
    """Moving-window plateau detector for a scalar loss with optional slope/relative checks.
    - window: number of recent points to monitor
    - min_delta: minimum absolute improvement to reset patience (against chosen baseline)
    - patience: number of consecutive windows meeting plateau condition before True
    - min_rel: minimum relative improvement (fraction) to reset patience
    - slope_thresh: if >0, require decreasing slope magnitude larger than this to avoid plateau
    - use_window_best: compare to best within the current window instead of global best
    """
    def __init__(
        self,
        window: int = 20,
        min_delta: float = 1e-4,
        patience: int = 3,
        *,
        min_rel: float = 0.0,
        slope_thresh: float = 0.0,
        use_window_best: bool = False,
    ):
        self.window = max(1, int(window))
        self.min_delta = float(min_delta)
        self.min_rel = float(min_rel)
        self.slope_thresh = float(slope_thresh)
        self.use_window_best = bool(use_window_best)
        self.patience = max(1, int(patience))
        self.buffer: list[float] = []
        self.best = float('inf')
        self.bad_windows = 0

    def _slope(self) -> float:
        # simple least-squares slope over the window
        n = len(self.buffer)
        if n < 2:
            return 0.0
        x = torch.arange(n, dtype=torch.float32)
        y = torch.tensor(self.buffer, dtype=torch.float32)
        xm = x.mean()
        ym = y.mean()
        denom = ((x - xm) ** 2).sum().item()
        if denom <= 0:
            return 0.0
        slope = (((x - xm) * (y - ym)).sum().item()) / denom
        return float(slope)

    def update(self, value: float) -> bool:
        """Add a loss value. Returns True if plateau detected."""
        self.buffer.append(float(value))
        if len(self.buffer) < self.window:
            return False
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

        current = self.buffer[-1]
        baseline = (min(self.buffer) if self.use_window_best else self.best)
        abs_improve = max(0.0, baseline - current)
        rel_improve = 0.0
        ref = self.buffer[0]
        if abs(ref) > 1e-12:
            rel_improve = max(0.0, (ref - current) / abs(ref))

        slope = self._slope()

        # plateau if: small absolute AND small relative improvement AND slope not meaningfully negative
        is_plateau = (abs_improve <= self.min_delta) and (rel_improve <= self.min_rel) and (slope >= -self.slope_thresh)

        # update global best tracking
        if current < self.best - self.min_delta:
            self.best = current

        if is_plateau:
            self.bad_windows += 1
        else:
            self.bad_windows = 0
        return self.bad_windows >= self.patience

def debug_image_with_boxes(image_tensor, combined, thres):
    # show the image tensor with boxes drawen for debugging
    # image_tensor is in [B,C,H,W] and in [0,1]
    drawn = draw_boxes(image_tensor, combined, thres)
    B, C, H, W = drawn.shape
    for i in range(B):
        img = drawn[i].permute(1, 2, 0).cpu().numpy()  # [H,W,C]
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"image_with_boxes_{i}.png", bbox_inches='tight', pad_inches=0)
