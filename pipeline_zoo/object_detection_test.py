import argparse
import json
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


DEFAULT_MODEL = "PekingU/rtdetr_r18vd"
STAGES = ["encoder", "selector", "decoder", "head"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StageProbe:
    """Utility for running RT-DETR stage-by-stage so activations can be swapped."""

    def __init__(self, model: RTDetrForObjectDetection):
        self.model = model.eval()
        self.core = model.model
        self.config = model.config

    def run(self, pixel_values: torch.Tensor, overrides=None):
        overrides = overrides or {}
        encoder_state = overrides.get("encoder")
        if encoder_state is None:
            encoder_state = self._run_encoder(pixel_values)

        selector_state = overrides.get("selector")
        if selector_state is None:
            selector_state = self._run_selector(encoder_state)

        decoder_state = overrides.get("decoder")
        if decoder_state is None:
            decoder_state = self._run_decoder(encoder_state, selector_state)

        head_state = overrides.get("head")
        if head_state is None:
            head_state = self._run_head(decoder_state)

        return {
            "encoder": encoder_state,
            "selector": selector_state,
            "decoder": decoder_state,
            "head": head_state,
        }

    def _run_encoder(self, pixel_values: torch.Tensor):
        batch_size, _, height, width = pixel_values.shape
        pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
        features = self.core.backbone(pixel_values, pixel_mask)
        proj_feats = [self.core.encoder_input_proj[level](source) for level, (source, _) in enumerate(features)]

        encoder_outputs = self.core.encoder(
            proj_feats,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        sources = []
        encoded = encoder_outputs.last_hidden_state
        for level, source in enumerate(encoded):
            sources.append(self.core.decoder_input_proj[level](source))

        if self.core.config.num_feature_levels > len(sources):
            len_sources = len(sources)
            sources.append(self.core.decoder_input_proj[len_sources](encoded[-1]))
            for i in range(len_sources + 1, self.core.config.num_feature_levels):
                sources.append(self.core.decoder_input_proj[i](encoded[-1]))

        source_flatten = []
        spatial_shapes_list = []
        for source in sources:
            _, _, h, w = source.shape
            spatial_shapes_list.append((h, w))
            source_flatten.append(source.flatten(2).transpose(1, 2))
        source_flatten = torch.cat(source_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        if self.core.training or self.core.config.anchor_image_size is None:
            anchors, valid_mask = self.core.generate_anchors(
                tuple(spatial_shapes_list), device=source_flatten.device, dtype=source_flatten.dtype
            )
        else:
            anchors = self.core.anchors.to(source_flatten.device, source_flatten.dtype)
            valid_mask = self.core.valid_mask.to(source_flatten.device, source_flatten.dtype)

        memory = valid_mask.to(source_flatten.dtype) * source_flatten
        output_memory = self.core.enc_output(memory)
        enc_outputs_class = self.core.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.core.enc_bbox_head(output_memory) + anchors

        return {
            "pixel_mask": pixel_mask,
            "features": features,
            "proj_feats": proj_feats,
            "encoder_outputs": encoder_outputs,
            "sources": sources,
            "source_flatten": source_flatten,
            "spatial_shapes": spatial_shapes,
            "spatial_shapes_list": spatial_shapes_list,
            "level_start_index": level_start_index,
            "anchors": anchors,
            "valid_mask": valid_mask,
            "memory": memory,
            "output_memory": output_memory,
            "enc_outputs_class": enc_outputs_class,
            "enc_outputs_coord_logits": enc_outputs_coord_logits,
        }

    def _run_selector(self, encoder_state):
        enc_outputs_class = encoder_state["enc_outputs_class"]
        enc_outputs_coord_logits = encoder_state["enc_outputs_coord_logits"]
        output_memory = encoder_state["output_memory"]

        max_scores = enc_outputs_class.max(-1).values
        _, topk_ind = torch.topk(max_scores, self.config.num_queries, dim=1)
        gather_idx_classes = topk_ind.unsqueeze(-1).expand(-1, -1, enc_outputs_class.shape[-1])
        gather_idx_boxes = topk_ind.unsqueeze(-1).expand(-1, -1, enc_outputs_coord_logits.shape[-1])

        reference_points_unact = enc_outputs_coord_logits.gather(1, gather_idx_boxes)
        enc_topk_bboxes = torch.sigmoid(reference_points_unact)

        if self.core.config.learn_initial_query:
            target = self.core.weight_embedding.tile([enc_outputs_class.shape[0], 1, 1])
        else:
            gather_idx_memory = topk_ind.unsqueeze(-1).expand(-1, -1, output_memory.shape[-1])
            target = output_memory.gather(1, gather_idx_memory).detach()

        enc_topk_logits = enc_outputs_class.gather(1, gather_idx_classes)
        init_reference_points = reference_points_unact.detach()

        return {
            "topk_ind": topk_ind,
            "target": target,
            "reference_points_unact": reference_points_unact,
            "init_reference_points": init_reference_points,
            "enc_topk_logits": enc_topk_logits,
            "enc_topk_bboxes": enc_topk_bboxes,
            "attention_mask": None,
        }

    def _run_decoder(self, encoder_state, selector_state):
        decoder_outputs = self.core.decoder(
            inputs_embeds=selector_state["target"],
            encoder_hidden_states=encoder_state["source_flatten"],
            encoder_attention_mask=selector_state["attention_mask"],
            reference_points=selector_state["init_reference_points"],
            spatial_shapes=encoder_state["spatial_shapes"],
            spatial_shapes_list=encoder_state["spatial_shapes_list"],
            level_start_index=encoder_state["level_start_index"],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return {"decoder_outputs": decoder_outputs}

    def _run_head(self, decoder_state):
        decoder_outputs = decoder_state["decoder_outputs"]
        logits = decoder_outputs.intermediate_logits[:, -1]
        pred_boxes = decoder_outputs.intermediate_reference_points[:, -1]
        return {
            "logits": logits,
            "pred_boxes": pred_boxes,
            "intermediate_logits": decoder_outputs.intermediate_logits,
            "intermediate_reference_points": decoder_outputs.intermediate_reference_points,
        }

    def decode_from_selector(self, encoder_state, selector_state):
        decoder_state = self._run_decoder(encoder_state, selector_state)
        return self._run_head(decoder_state)


def craft_adversarial_sample(
    model: RTDetrForObjectDetection,
    clean_tensor: torch.Tensor,
    target_label: int,
    steps: int,
    lr: float,
    epsilon: float,
):
    delta = torch.zeros_like(clean_tensor, device=clean_tensor.device, requires_grad=True)
    target_mask = None

    for _ in range(steps):
        adv = (clean_tensor + delta).clamp(0.0, 1.0)
        outputs = model(adv)
        probs = torch.sigmoid(outputs.logits)

        if target_mask is None or target_mask.shape != probs.shape:
            target_mask = torch.zeros_like(probs)
            target_mask[:, :, target_label] = 1.0

        class_loss = F.mse_loss(probs, target_mask, reduction="sum") / (target_mask.sum() + 1e-6)
        norm_loss = torch.norm(delta, p=float("inf"))
        total_loss = class_loss + norm_loss
        total_loss.backward()

        with torch.no_grad():
            delta.add_(-lr * delta.grad)
            delta.clamp_(-epsilon, epsilon)
        if delta.grad is not None:
            delta.grad.zero_()

    adv_tensor = (clean_tensor + delta).clamp(0.0, 1.0).detach()
    return adv_tensor


def clone_tensor_dict(data: dict):
    clone = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            clone[key] = value.detach().clone()
        else:
            clone[key] = value
    return clone


def gather_target_from_memory(memory_state, topk_ind):
    output_memory = memory_state["output_memory"]
    gather_idx = topk_ind.unsqueeze(-1).expand(-1, -1, output_memory.shape[-1])
    return output_memory.gather(1, gather_idx).detach()


def summarize_outputs(
    image_processor: RTDetrImageProcessor,
    head_state: dict,
    target_sizes: torch.Tensor,
    target_label: int,
    threshold: float,
):
    logits = head_state["logits"]
    probs = torch.sigmoid(logits)
    target_prob = probs[..., target_label].mean().item()
    output_obj = SimpleNamespace(
        logits=logits.detach().cpu(),
        pred_boxes=head_state["pred_boxes"].detach().cpu(),
    )
    results = image_processor.post_process_object_detection(
        output_obj, threshold=threshold, target_sizes=target_sizes.cpu()
    )
    counts = [len(entry["scores"]) for entry in results]
    avg_count = float(np.mean(counts)) if counts else 0.0
    return {"avg_count": avg_count, "target_prob": target_prob}


def run_stage_swap_trial(
    probe: StageProbe,
    image_processor: RTDetrImageProcessor,
    clean_tensor: torch.Tensor,
    adv_tensor: torch.Tensor,
    target_sizes: torch.Tensor,
    target_label: int,
    threshold: float,
):
    with torch.no_grad():
        clean_bundle = probe.run(clean_tensor)
        adv_bundle = probe.run(adv_tensor)

    summaries = {
        "clean": summarize_outputs(image_processor, clean_bundle["head"], target_sizes, target_label, threshold),
        "perturbed": summarize_outputs(image_processor, adv_bundle["head"], target_sizes, target_label, threshold),
    }

    for stage in STAGES:
        overrides = {stage: clean_bundle[stage]}
        with torch.no_grad():
            swapped_bundle = probe.run(adv_tensor, overrides=overrides)
        label = f"swap_{stage}"
        summaries[label] = summarize_outputs(
            image_processor, swapped_bundle["head"], target_sizes, target_label, threshold
        )

    return clean_bundle, adv_bundle, summaries


def experiment_memory_swap_fixed_selector(
    probe: StageProbe,
    image_processor: RTDetrImageProcessor,
    clean_bundle,
    adv_bundle,
    target_sizes,
    target_label,
    threshold,
):
    adv_selector = adv_bundle["selector"]
    adv_encoder = adv_bundle["encoder"]
    custom_selector = dict(adv_selector)
    custom_selector["target"] = gather_target_from_memory(clean_bundle["encoder"], adv_selector["topk_ind"])
    head_state = probe.decode_from_selector(adv_encoder, custom_selector)
    stats = summarize_outputs(image_processor, head_state, target_sizes, target_label, threshold)
    return {"swap_memory_fixed_selector": stats}


def experiment_score_swap(
    probe: StageProbe,
    image_processor: RTDetrImageProcessor,
    clean_bundle,
    adv_bundle,
    target_sizes,
    target_label,
    threshold,
):
    clean_enc = clean_bundle["encoder"]
    adv_enc = adv_bundle["encoder"]
    clean_scores = clean_enc["enc_outputs_class"]
    max_scores = clean_scores.max(-1).values
    _, topk_ind = torch.topk(max_scores, probe.config.num_queries, dim=1)
    gather_idx_boxes = topk_ind.unsqueeze(-1).expand(-1, -1, adv_enc["enc_outputs_coord_logits"].shape[-1])
    reference_points_unact = adv_enc["enc_outputs_coord_logits"].gather(1, gather_idx_boxes)
    selector_state = {
        "topk_ind": topk_ind,
        "target": gather_target_from_memory(adv_enc, topk_ind),
        "reference_points_unact": reference_points_unact,
        "init_reference_points": reference_points_unact.detach(),
        "enc_topk_logits": clean_scores.gather(
            1, topk_ind.unsqueeze(-1).expand(-1, -1, clean_scores.shape[-1])
        ),
        "enc_topk_bboxes": torch.sigmoid(reference_points_unact),
        "attention_mask": None,
    }
    head_state = probe.decode_from_selector(adv_enc, selector_state)
    stats = summarize_outputs(image_processor, head_state, target_sizes, target_label, threshold)
    return {"swap_scores_clean_logits": stats}


def experiment_memory_interpolation(
    probe: StageProbe,
    image_processor: RTDetrImageProcessor,
    clean_bundle,
    adv_bundle,
    target_sizes,
    target_label,
    threshold,
    alphas,
):
    adv_selector = adv_bundle["selector"]
    adv_encoder = adv_bundle["encoder"]
    clean_memory = clean_bundle["encoder"]["output_memory"]
    adv_memory = adv_encoder["output_memory"]
    results = {}
    for alpha in alphas:
        mix = (1 - alpha) * clean_memory + alpha * adv_memory
        temp_encoder = dict(adv_encoder)
        temp_encoder["output_memory"] = mix
        custom_selector = dict(adv_selector)
        custom_selector["target"] = gather_target_from_memory(temp_encoder, adv_selector["topk_ind"])
        head_state = probe.decode_from_selector(temp_encoder, custom_selector)
        label = f"interp_memory_alpha_{alpha:.2f}"
        results[label] = summarize_outputs(image_processor, head_state, target_sizes, target_label, threshold)
    return results


def experiment_grad_norms(probe: StageProbe, model, pixel_values, target_label):
    encoder_state = probe._run_encoder(pixel_values)
    encoder_state["output_memory"].retain_grad()
    selector_state = probe._run_selector(encoder_state)
    decoder_state = probe._run_decoder(encoder_state, selector_state)
    decoder_state["decoder_outputs"].last_hidden_state.retain_grad()
    head_state = probe._run_head(decoder_state)
    probs = torch.sigmoid(head_state["logits"])
    target_mask = torch.zeros_like(probs)
    target_mask[:, :, target_label] = 1.0
    loss = F.mse_loss(probs, target_mask, reduction="sum") / (target_mask.sum() + 1e-6)
    loss.backward()
    enc_norm = encoder_state["output_memory"].grad.norm().item()
    dec_norm = decoder_state["decoder_outputs"].last_hidden_state.grad.norm().item()
    model.zero_grad(set_to_none=True)
    return {"encoder_grad_norm": enc_norm, "decoder_grad_norm": dec_norm}


def experiment_frozen_encoder_attack(
    probe: StageProbe,
    image_processor: RTDetrImageProcessor,
    clean_tensor,
    target_sizes,
    target_label,
    threshold,
    steps,
    lr,
    epsilon,
):
    with torch.no_grad():
        clean_bundle = probe.run(clean_tensor)
    frozen_encoder = clone_tensor_dict(clean_bundle["encoder"])

    delta = torch.zeros_like(clean_tensor, device=clean_tensor.device, requires_grad=True)
    target_mask = None
    for _ in range(steps):
        adv = (clean_tensor + delta).clamp(0.0, 1.0)
        bundle = probe.run(adv, overrides={"encoder": frozen_encoder})
        probs = torch.sigmoid(bundle["head"]["logits"])
        if target_mask is None or target_mask.shape != probs.shape:
            target_mask = torch.zeros_like(probs)
            target_mask[:, :, target_label] = 1.0
        loss = F.mse_loss(probs, target_mask, reduction="sum") / (target_mask.sum() + 1e-6)
        loss.backward()
        with torch.no_grad():
            delta.add_(-lr * delta.grad)
            delta.clamp_(-epsilon, epsilon)
        if delta.grad is not None:
            delta.grad.zero_()

    adv_tensor = (clean_tensor + delta).clamp(0.0, 1.0).detach()
    bundle = probe.run(adv_tensor, overrides={"encoder": frozen_encoder})
    return summarize_outputs(image_processor, bundle["head"], target_sizes, target_label, threshold)


def format_summary_table(image_id, summaries):
    header = f"Stage swap summary for image {image_id}"
    lines = [header, "-" * len(header)]
    lines.append(f"{'scenario':<15} | {'avg_count':>9} | {'target_prob':>12}")
    lines.append("-" * 45)
    for key, stats in summaries.items():
        lines.append(f"{key:<15} | {stats['avg_count']:>9.2f} | {stats['target_prob']:>12.4f}")
    return "\n".join(lines)


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
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--experiments",
        type=str,
        default="stage_swap",
        help="Comma-separated list of experiments: stage_swap,swap_memory,swap_scores,interp_memory,grad_norm,frozen_encoder",
    )
    parser.add_argument("--interp_alphas", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--frozen_attack_steps", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RTDetrForObjectDetection.from_pretrained(args.model_name).to(device).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(args.model_name)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    probe = StageProbe(model)
    summaries_log = []
    experiment_set = {exp.strip() for exp in args.experiments.split(",") if exp.strip()}
    interp_alphas = [float(v) for v in args.interp_alphas.split(",") if v]

    subset = dataset.select(list(range(min(args.num_samples, len(dataset)))))
    for sample in tqdm(subset):
        image: Image.Image = sample["image"].convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        adv_tensor = craft_adversarial_sample(
            model,
            pixel_values.clone(),
            target_label=args.target_label,
            steps=args.attack_steps,
            lr=args.learning_rate,
            epsilon=args.epsilon,
        )

        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        record = {"image_id": sample["image_id"], "summaries": {}}

        clean_bundle = adv_bundle = None

        if "stage_swap" in experiment_set:
            clean_bundle, adv_bundle, swap_summary = run_stage_swap_trial(
                probe,
                image_processor,
                pixel_values,
                adv_tensor,
                target_sizes,
                target_label=args.target_label,
                threshold=args.threshold,
            )
            record["summaries"].update(swap_summary)
        else:
            with torch.no_grad():
                clean_bundle = probe.run(pixel_values)
                adv_bundle = probe.run(adv_tensor)

        if "swap_memory" in experiment_set:
            stats = experiment_memory_swap_fixed_selector(
                probe,
                image_processor,
                clean_bundle,
                adv_bundle,
                target_sizes,
                args.target_label,
                args.threshold,
            )
            record["summaries"].update(stats)

        if "swap_scores" in experiment_set:
            stats = experiment_score_swap(
                probe,
                image_processor,
                clean_bundle,
                adv_bundle,
                target_sizes,
                args.target_label,
                args.threshold,
            )
            record["summaries"].update(stats)

        if "interp_memory" in experiment_set:
            stats = experiment_memory_interpolation(
                probe,
                image_processor,
                clean_bundle,
                adv_bundle,
                target_sizes,
                args.target_label,
                args.threshold,
                interp_alphas,
            )
            record["summaries"].update(stats)

        if "grad_norm" in experiment_set:
            norms = experiment_grad_norms(probe, model, pixel_values.clone().detach(), args.target_label)
            record["grad_norms"] = norms

        if "frozen_encoder" in experiment_set:
            stats = experiment_frozen_encoder_attack(
                probe,
                image_processor,
                pixel_values,
                target_sizes,
                args.target_label,
                args.threshold,
                steps=args.frozen_attack_steps,
                lr=args.learning_rate,
                epsilon=args.epsilon,
            )
            record["frozen_encoder_attack"] = stats

        if record["summaries"]:
            print(format_summary_table(sample["image_id"], record["summaries"]))
        else:
            print(f"No summary stats computed for image {sample['image_id']}.")

        summaries_log.append(record)

    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(summaries_log, handle, indent=2)


if __name__ == "__main__":
    main()
