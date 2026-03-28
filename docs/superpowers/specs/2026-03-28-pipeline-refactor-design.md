# Pipeline Refactor Design

**Date**: 2026-03-28
**Status**: Approved

## Context

The pipeline zoo has 11 pipeline files averaging 300–500 lines each, with 80–90% of each file being identical copy-paste (dataset loading, model loading, seed setting, mask generation, norm loss, logging). Three files are dead debug artifacts. The attack loop logic — the only thing that differs between pipelines — is buried in 130–200 line `run_attack()` methods alongside all the shared boilerplate. The goal is to make each pipeline's unique logic immediately visible and eliminate the maintenance risk of 11 copies of the same helpers.

## Approach: Template Method

`BasePipeline` owns the full attack loop and all shared infrastructure. Concrete pipelines implement 2 abstract hooks that contain only their unique logic.

---

## File Changes

### Delete (4 files)
- `pipeline_zoo/dual_object_detection_clean.py` — all losses zeroed, superseded by phase_a
- `pipeline_zoo/dual_object_detection_clean_rand.py` — debug artifact
- `pipeline_zoo/dual_object_detection_clean_randn.py` — debug artifact, has duplicate method bug
- `pipeline_zoo/dual_object_detection_sweep.py` — sweep expansion already in `scripts/run_loss_sweep.py`

### Rename to 6 concrete pipelines
| New file | From | Pipeline ID |
|---|---|---|
| `pipeline_zoo/pipeline_baseline.py` | `dual_object_detection.py` | 0 |
| `pipeline_zoo/pipeline_phase_a.py` | `dual_object_detection_phase_a.py` | 1 |
| `pipeline_zoo/pipeline_phase_b.py` | `dual_object_detection_phase_b.py` | 2 |
| `pipeline_zoo/pipeline_plateau_projection.py` | `dual_object_detection_gradient_projection.py` + `dual_object_detection_gradient_projection_b_primary.py` | 3 |
| `pipeline_zoo/pipeline_penalty.py` | `dual_object_detection_penalty.py` | 4 |
| `pipeline_zoo/pipeline_raja.py` | `dual_object_detection_raja.py` | 5 |

### Modify
- `pipeline_zoo/base_pipeline.py` — major expansion (see below)
- `pipeline_zoo/utilities.py` — add `GRID_SIZE` constant, unify `calc_cls_loss`
- `pipeline_zoo/zoo.py` — update registry to map 0–5 to new classes
- `main.py` — no changes needed

---

## BasePipeline Design

```python
class BasePipeline(ABC):
    # Shared __init__: parses all common config fields, normalizes loss weights
    def __init__(self, config, device): ...

    # Shared infrastructure (not overridden by subclasses)
    def randomseed(self)
    def load_dataset(self) -> Dataset
    def load_model(self)
    def get_mask(self, tensor_shape) -> Tensor
    def calc_norm_loss(self, order=["linf"]) -> Tensor
    def update_log(self, image_id, iteration, **metrics)
    def write_log(self, log_path=None)
    def unhook(self, x) -> float | list

    # Owns the attack loop
    def run_attack(self, dataset):
        for each image in dataset:
            image_tensor = preprocess(image)
            self.patch = torch.zeros_like(image_tensor).requires_grad_(True)
            self.mask = self.get_mask(image_tensor.shape)
            self._init_image_state()
            for i in range(self.num_iterations):
                count1, count2, extra = self._attack_step(i, image_tensor)
                self.update_log(image_id, i, obj_count_1=count1, obj_count_2=count2, **extra)
            save_perturbed_image(image_tensor)
        self.write_log()

    # Hook points (subclasses implement these only)
    @abstractmethod
    def _init_image_state(self):
        """Reset per-image state: phase flags, plateau detectors, ALM multipliers, etc."""

    @abstractmethod
    def _attack_step(self, i: int, image_tensor: Tensor) -> tuple[int, int, dict]:
        """
        One optimization step.
        Returns: (obj_count_1, obj_count_2, extra_log_fields)
        Responsible for: forward passes, loss computation, gradient step, patch update.
        Uses self.patch, self.mask, self.model_1, self.model_2, self.lr, self.budget.
        """
```

`self.bx` is renamed to `self.patch` throughout for clarity.

---

## Concrete Pipeline Designs

### PipelineBaseline (~50 lines of unique logic)
- `_init_image_state`: no-op
- `_attack_step`: model1 forward → cls_loss_1 + norm_loss → if `i/num_iters >= b_start_frac`: model2 ROI micro-batch forward → accumulate cls_loss_2 → combined weighted loss → backward → update patch

### PipelinePhaseA (~45 lines)
- `_init_image_state`: no-op
- `_attack_step`: model1 forward → loss → backward → model2 in `torch.no_grad()` for obj_count only → update patch

### PipelinePhaseB (~50 lines)
- `_init_image_state`: no-op
- `_attack_step`: model2-primary ROI forward → loss → backward → model1 in `torch.no_grad()` for obj_count only → update patch

### PipelinePlateauProjection (~90 lines)
- `_init_image_state`: reset `LossPlateauDetector`, set `self.in_b_phase = False`, `self.a_best_loss = None`
- `_attack_step`:
  - Phase A: model1 forward → simple loss → backward (grad_A)
  - Plateau check → transition to phase B when plateau detected and min_iters reached
  - Phase B: explicit `grad_wrt(model1)` + `grad_wrt(model2 ROI micro-batches)` → `project_onto_orthogonal_complement(grad_B, grad_A)` → update patch
  - `primary_model` config field (1 or 2) determines which gradient is preserved in the projection (merges the A-primary and B-primary variants)

### PipelinePenalty (~90 lines)
- `_init_image_state`: reset `LossPlateauDetector`, `self.alm_lambda = initial_lambda`, `self.in_b_phase = False`
- `_attack_step`: same two-phase structure as PlateauProjection but uses ALM penalty (`alm_lambda * constraint_violation`) instead of gradient projection; updates `alm_lambda` on phase transition and plateau

### PipelineRaja (~150 lines)
- Overrides `run_attack` entirely (different outer loop structure: box routing + `roi_align` PGD)
- Inherits all shared infra: `load_dataset`, `load_model`, `write_log`, `get_mask`, `calc_norm_loss`
- `_init_image_state` and `_attack_step` are implemented but called from the overridden `run_attack`

---

## Additional Fixes

### `utilities.py`
- Add `GRID_SIZE = 32` constant; replace all hardcoded `32` references (currently in 4 places)
- Remove duplicate `calc_cls_loss` — pipeline-local copies all deleted; unified version in utilities uses `target_tensor.sum() + 1` as denominator throughout. **Behavior note**: pipeline_baseline previously used `len(probs.squeeze()) + 1` (number of queries) as denominator — this was inconsistent with utilities and is corrected here. Loss values for that pipeline will differ numerically but the optimization direction is unchanged.

### Variable renames (throughout)
- `self.bx` → `self.patch`
- `combined` → `detections` (model output tensor)

### Dead code removal
- `calc_iou()` in `BasePipeline` — never called by any subclass; delete

---

## Verification

After refactoring, run the baseline pipeline on a small config and confirm log output matches pre-refactor:

```bash
# Smoke test: 5 images, 10 iterations
python main.py --config_file ./config/test.json --pipeline_id 0

# Check log written to ./log/
ls -lt ./log/ | head -5

# Confirm object counts are in expected range (not all zeros, not all max)
python -c "
import json, glob, os
f = max(glob.glob('log/*.json'), key=os.path.getmtime)
d = json.load(open(f))
print({k: d[k][-1] for k in list(d)[:3]})
"
```

Run all 6 pipelines (0–5) at least once on `test_size: 5` to confirm no import or attribute errors.
