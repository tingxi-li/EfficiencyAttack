import os
import pdb
import sys
import math
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset

class BasePipeline:
    def __init__(self, config, device=None):
        self.device = device
        self.config = config
        self.logs = []
        
    def load_dataset(self):
        pass
    
    def load_model(self):
        pass



    def calc_iou(self, pred_boxes):
        if self.adv_patch_xywh is None:
            return torch.tensor([1.0], device=self.device)
        else:
            assert isinstance(self.adv_patch_xywh, list), \
                "self.adv_patch_xywh must be [patch_cx, patch_cy, patch_w, patch_h] normalized to [0,1]"
            patch_cx, patch_cy, patch_w, patch_h = self.adv_patch_xywh
            
            orig_shape = pred_boxes.shape
            if pred_boxes.dim() == 1:          # [4]
                pred_boxes = pred_boxes.unsqueeze(0).unsqueeze(0)  # [1,1,4]
            elif pred_boxes.dim() == 2:        # [N,4]
                pred_boxes = pred_boxes.unsqueeze(0)               # [1,N,4]
            elif pred_boxes.dim() == 3:        # [B,N,4]
                pass
            else:
                raise ValueError(f"pred_boxes shape not supported: {orig_shape}")
                
        px1 = torch.tensor(patch_cx - patch_w/2.0, device=self.device)
        py1 = torch.tensor(patch_cy - patch_h/2.0, device=self.device)
        px2 = torch.tensor(patch_cx + patch_w/2.0, device=self.device)
        py2 = torch.tensor(patch_cy + patch_h/2.0, device=self.device)

        # clamp to [0,1] 
        px1 = px1.clamp(0.0, 1.0)
        py1 = py1.clamp(0.0, 1.0)
        px2 = px2.clamp(0.0, 1.0)
        py2 = py2.clamp(0.0, 1.0)

        # broadcast 
        B, N, _ = pred_boxes.shape
        patch_xyxy = torch.stack([px1, py1, px2, py2])                     # [4]
        patch_xyxy = patch_xyxy.view(1, 1, 4).expand(B, N, 4).contiguous() # [B,N,4]

        # [cx,cy,w,h] -> [x1,y1,x2,y2]
        cx, cy, w, h = pred_boxes.unbind(-1)
        x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # [B,N,4]

        # IOU
        ix1 = torch.maximum(boxes_xyxy[..., 0], patch_xyxy[..., 0])
        iy1 = torch.maximum(boxes_xyxy[..., 1], patch_xyxy[..., 1])
        ix2 = torch.minimum(boxes_xyxy[..., 2], patch_xyxy[..., 2])
        iy2 = torch.minimum(boxes_xyxy[..., 3], patch_xyxy[..., 3])

        inter_w = (ix2 - ix1).clamp(min=0.0)
        inter_h = (iy2 - iy1).clamp(min=0.0)
        inter = inter_w * inter_h

        area_boxes = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        area_patch = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
        union = area_boxes + area_patch - inter

        eps = 1e-6
        iou = inter / (union + eps)  # [B,N]

        # reshape to original shape
        if len(orig_shape) == 1:       # input [4]
            return iou.view(()).to(self.device)  # scalar (single box)
        elif len(orig_shape) == 2:     # input [N,4]
            return iou.squeeze(0)      # [N]
        else:                          # input [B,N,4]
            return iou
