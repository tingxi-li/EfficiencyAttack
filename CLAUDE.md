# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EfficiencyAttack is a research framework for adversarial patch attacks on object detection models (RT-DETR). The goal is efficient dual-model adversarial optimization — attacking a primary model while testing transferability to a secondary model.

## Common Commands

**Run a single pipeline:**
```bash
python main.py --config_file ./config/test.json --pipeline_id 3
```

**Run a loss weight sweep:**
```bash
python scripts/run_loss_sweep.py --config_file config/loss_sweep.json --pipeline_id 3 --ngpus 2
```

**Batch execution scripts:**
```bash
bash baselines.sh        # Run baseline pipelines
bash parallel.sh         # Run multiple pipelines in parallel
bash sweep.sh            # Run experimental sweeps
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Architecture

### Entry Point
`main.py` orchestrates execution: reads a JSON config, selects a pipeline by ID, and either runs single-process or shards the dataset across multiple GPUs (auto-detected). Logs from parallel workers are merged and saved to `./log/`.

### Pipeline System
Pipelines live in `pipeline_zoo/` and are registered in `pipeline_zoo/zoo.py` (IDs 0–10). Each inherits from `base_pipeline.py` which provides:
- `load_dataset()` / `load_model()` stubs
- `calc_iou()` for adversarial patch IoU computation

The canonical baseline is **pipeline 3** (`dual_object_detection.py`). Key variants:
- **4/5** (`phase_a`, `phase_b`): Split two-phase optimization into separate pipelines
- **6/8** (`gradient_projection`): Phase B uses gradient projection to preserve Model_1 gains while optimizing Model_2
- **7** (`penalty`): Penalty-based multi-model coordination instead of projection
- **9** (`sweep`): Parameterized for loss weight grid search
- **10** (`raja`): Extended pipeline with additional experimental features

### Attack Loop (per image)
1. Load image from COCO validation split
2. Initialize adversarial patch `bx` and spatial mask
3. **Phase A**: Optimize against Model_1 until plateau (via `LossPlateauDetector`)
4. **Phase B**: Apply gradient projection to optimize Model_2 while preserving Model_1 loss
5. Log per-iteration metrics to `./log/`

### Key Shared Utilities (`pipeline_zoo/utilities.py`)
- `calc_cls_loss()` — classification loss for detection suppression
- `masks_from_boxes()` — extract ROI masks from detection bounding boxes
- `LossPlateauDetector` — sliding window convergence detection for phase transitions
- `draw_boxes()`, `cutout()` — visualization helpers

### Configuration
JSON configs in `config/`. Key fields:
```json
{
  "seed": 42,
  "parallel": true,
  "num_iterations": 200,
  "dataset_name": "coco",
  "hf_model_name_1": "PekingU/rtdetr_r18vd",
  "hf_model_name_2": "PekingU/rtdetr_r18vd",
  "learning_rate": 1.5,
  "cls_loss_weight_1": 19.0,
  "cls_loss_weight_2": 32.333,
  "budget": 0.08,
  "adv_patch_xywh": [0.5, 0.5, 0.2, 0.2]
}
```
`budget` controls the L∞ perturbation budget. `adv_patch_xywh` is the normalized patch location/size.

### Output
- Logs: `./log/log_[pipeline_name]_[timestamp]_configs.json`
- Visualizations: `./vis/` and `./exp-vis/`
- Analysis scripts: `scripts/analyze_sweep_results.py`, `scripts/run_loss_sweep.py`

## Models
Models are loaded from Hugging Face Hub (RT-DETR family, e.g. `PekingU/rtdetr_r18vd`). Requires internet access or cached models on first run.

## Dataset
Uses COCO validation split loaded via HuggingFace `datasets`. Subset size is configurable; default configs use 100 images.
