import argparse
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


DEFAULT_MODEL = "PekingU/rtdetr_r18vd"
STAGES = ["encoder", "selector", "decoder", "head"]

#TODO：current results have ~200 boxes, box area is ideally large, no detections on the second layer
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def grad_norm_on_delta(loss, delta):
    if delta.grad is not None:
        delta.grad.zero_()
    loss.backward(retain_graph=True)
    return delta.grad.norm().item()

def grad_vec_on_delta(loss, delta):
    g = torch.autograd.grad(loss, delta, retain_graph=True, create_graph=False)[0]
    return g

def run_attack(
    model: RTDetrForObjectDetection,
    clean_tensor: torch.Tensor,
    target_label: int,
    steps: int,
    lr: float,
    epsilon: float,
    image_processor: RTDetrImageProcessor,
    target_sizes: torch.Tensor,
    threshold: float,
    top_k: int,
    capture_every: int = 5,
):
    delta = torch.zeros_like(clean_tensor, device=clean_tensor.device, requires_grad=True)
    target_mask = None
    snapshots = []
    target_sizes_cpu = target_sizes.detach().cpu()

    for step in range(steps):
        delta.requires_grad = True
        adv = (clean_tensor + delta).clamp(0.0, 1.0)
        layer1_outputs = model(adv)
        probs = torch.sigmoid(layer1_outputs.logits)
        pred_boxes = layer1_outputs.pred_boxes
        
        target_probs = probs[:, :, target_label]
        topk_values, topk_indices = torch.topk(target_probs, k=top_k, dim=1)
        
        # pred_boxes: [B, num_queries, 4]
        topk_boxes = torch.gather(
            pred_boxes, 
            dim=1, 
            index=topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        )  # [B, top_k, 4]
        
        # import pdb; pdb.set_trace()

        if target_mask is None or target_mask.shape != probs.shape:
            target_mask = torch.zeros_like(probs)
            target_mask[:, :, target_label] = 1.0

        layer1_cls_loss = F.mse_loss(probs, target_mask, reduction="sum") / (target_mask.sum() + 1e-6)
        layer1_area_loss = torch.mean(topk_boxes[..., 2] * topk_boxes[..., 3])
        norm_loss   = torch.norm(delta, p=float("inf"))

        total_loss  = layer1_cls_loss + norm_loss - layer1_area_loss

        # ---- compute gradient vectors wrt delta ----
        g_cls = torch.autograd.grad(layer1_cls_loss, delta, retain_graph=True, create_graph=False)[0]
        g_area = torch.autograd.grad(layer1_area_loss, delta, retain_graph=True, create_graph=False)[0]
        g_norm = torch.autograd.grad(norm_loss, delta, retain_graph=True, create_graph=False)[0]

        λ1 = g_cls.norm().detach()
        λ2 = g_area.norm().detach()
        w1 = 1 / (λ1 + 1e-8)
        w2 = 1 / (λ2 + 1e-8)
        grad = w1 * g_cls - w2 * g_area + g_norm
        
        with torch.no_grad():
            delta.add_(-lr * grad)
            delta.clamp_(-epsilon, epsilon)
        if delta.grad is not None:
            delta.grad.zero_()
            
        # import pdb; pdb.set_trace()

        if step % capture_every == 0 or step == steps - 1:
            with torch.no_grad():

                layer1_detections = image_processor.post_process_object_detection(
                    layer1_outputs, target_sizes=target_sizes_cpu, threshold=threshold
                )
                snapshots.append(
                    {
                        "step": step,
                        "image": adv.detach().cpu(),
                        "detections": layer1_detections[0],
                    }
                )

                print(f"num boxes at step {step}: {len(layer1_detections[0]['boxes'])}")
                print(f"Step {step}: Total Loss={total_loss.item():.4f}, L_cls={layer1_cls_loss.item():.4f}, L_area={layer1_area_loss.item():.4f}, L_norm={norm_loss.item():.4f}")
                print(f"∥g_cls∥={g_cls.norm():.3e}, ∥g_area∥={g_area.norm():.3e}, ∥g_norm∥={g_norm.norm():.3e}")


            
    adv_tensor = (clean_tensor + delta).clamp(0.0, 1.0).detach()

    B, C, H, W = adv_tensor.shape
    masks = []

    for det in layer1_detections:
        for box in det["boxes"][:25]:
            x1, y1, x2, y2 = box #xyxy
            mask = torch.zeros((H, W), device=adv_tensor.device)
            mask [int(y1):int(y2), int(x1):int(x2)] = 1.0
            masks.append(mask)
    
    masked = adv_tensor * torch.stack(masks).unsqueeze(1)
    
    for i in range(1):
        adv_tensor = delta.requires_grad_(False) + clean_tensor
        masked = adv_tensor * torch.stack(masks).unsqueeze(1)
        layer2_outputs = model(masked)
        probs2 = torch.sigmoid(layer2_outputs.logits)
        pred_boxes2 = layer2_outputs.pred_boxes
        
        
    layer2_detections = image_processor.post_process_object_detection(
        layer2_outputs, target_sizes=torch.tensor([[H, W]]).repeat(layer2_outputs.logits.shape[0],1), threshold=threshold
    )
    
    for det in layer2_detections:
        print(f"num boxes in stage 2: {len(det['boxes'])}")
        
    for m in masked:
        snapshots.append(
            {
                "step": "stage2",
                "image": m.detach().cpu(),
                "detections": layer2_detections[0],
            }
        )

    import pdb; pdb.set_trace()
    
    return adv_tensor, snapshots


def _tensor_to_image(tensor: torch.Tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(img, 0.0, 1.0)


def _draw_detections(ax, image_np, detections, top_k):
    ax.imshow(image_np)
    ax.axis("off")
    scores = detections["scores"]
    boxes = detections["boxes"]
    labels = detections["labels"]
    if scores.numel() == 0:
        return
    order = torch.argsort(scores, descending=True)
    limit = min(top_k, order.numel()) if top_k > 0 else order.numel()
    for idx in order[:limit]:
        score = scores[idx].item()
        box = boxes[idx].tolist()
        label = labels[idx].item()
        xmin, ymin, xmax, ymax = box
        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            max(ymin - 2, 0),
            f"{label}:{score:.2f}",
            fontsize=6,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
        )


def save_snapshot_grid(snapshots, out_path: Path, top_k: int):
    if not snapshots:
        return
    cols = min(4, len(snapshots))
    rows = math.ceil(len(snapshots) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, snap in zip(axes, snapshots):
        image_np = _tensor_to_image(snap["image"])
        _draw_detections(ax, image_np, snap["detections"], top_k)
        ax.set_title(f"step {snap['step']}")
    for ax in axes[len(snapshots):]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)




def main():
    parser = argparse.ArgumentParser(description="Compare clean vs perturbed activations stage-by-stage.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset_split", type=str, default="val")
    parser.add_argument("--dataset_name", type=str, default="detection-datasets/coco")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to analyze.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1.5)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--target_label", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RTDetrForObjectDetection.from_pretrained(args.model_name).to(device).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(args.model_name)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    subset = dataset.select(list(range(min(args.num_samples, len(dataset)))))
    for sample in tqdm(subset):
        image: Image.Image = sample["image"].convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        adv_tensor, snapshots = run_attack(
            model,
            pixel_values.clone(),
            target_label=args.target_label,
            steps=args.attack_steps,
            lr=args.learning_rate,
            epsilon=args.epsilon,
            image_processor=image_processor,
            target_sizes=torch.tensor([[image.height, image.width]], device=device),
            threshold=args.threshold,
            top_k=args.top_k,
            capture_every=int(args.attack_steps * 0.05),
        )

        out_dir = Path("exp-vis")
        filename = f"{sample['image_id']}_steps_weighted_sum.png"
        save_snapshot_grid(snapshots, out_dir / filename, -1)


if __name__ == "__main__":
    main()
