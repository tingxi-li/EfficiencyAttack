#!/usr/bin/env bash
set -euo pipefail

# Run selected pipelines with two configs: weights and no-weights
# - Main entry for most: main.py --config_file <cfg> --pipeline_id <id>
# - Teaspoon runs its own __main__ with --config_file

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

CFG_WEIGHTS="config/parallel_weights.json"
CFG_NO_WEIGHTS="config/parallel_no_weights.json"

if [[ ! -f "$CFG_WEIGHTS" || ! -f "$CFG_NO_WEIGHTS" ]]; then
  echo "Missing config files: $CFG_WEIGHTS or $CFG_NO_WEIGHTS" >&2
  exit 1
fi

run_main() {
  local cfg="$1"; shift
  local pid="$1"; shift
  echo "=== Running main.py pipeline_id=$pid with $cfg ==="
  python main.py --config_file "$cfg" --pipeline_id "$pid"
}



# Pipelines to run via main.py (IDs from pipeline_zoo/zoo.py)
# 3: dual_object_detection
# 4: phase_a
# 5: phase_b
# 6: gradient_projection
# 7: penalty
MAIN_PIPELINES=(3 4 5 6 7)

for cfg in "$CFG_WEIGHTS"; do
  for pid in "${MAIN_PIPELINES[@]}"; do
    run_main "$cfg" "$pid"
  done
done

echo "All runs completed."
