#!/usr/bin/env bash
set -euo pipefail

# 在 GPU 0 和 GPU 1 上同时运行所有管道任务，无日志

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

pipelines=(
  "dual_object_detection.py"
  # "dual_object_detection_clean.py"
  # "dual_object_detection_clean_rand.py"
  # "dual_object_detection_clean_randn.py"
  "dual_object_detection_gradient_projection.py"
  "dual_object_detection_penalty.py"
  "dual_object_detection_phase_a.py"
  "dual_object_detection_phase_b.py"
)

# 计算每个 GPU 分配的任务数量
NUM_PIPELINES=${#pipelines[@]}
SPLIT_POINT=$(( (NUM_PIPELINES + 1) / 2 ))

# 将任务列表分成两部分
pipelines_gpu0=("${pipelines[@]:0:$SPLIT_POINT}")
pipelines_gpu1=("${pipelines[@]:$SPLIT_POINT}")

# 定义一个函数，用于在后台运行一个GPU的任务
run_pipelines_on_gpu() {
  local gpu="$1"
  shift
  local pipelines_to_run=("$@")

  for script in "${pipelines_to_run[@]}"; do
    # 在子shell中运行，确保环境变量只在当前进程中生效
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python "$BASE_DIR/pipeline_zoo/$script"
    ) || echo "WARNING: $script on GPU $gpu failed." >&2
  done
}

# 启动两个后台任务，每个任务负责一个GPU
run_pipelines_on_gpu 0 "${pipelines_gpu0[@]}" &
run_pipelines_on_gpu 1 "${pipelines_gpu1[@]}" &

# 等待所有后台任务完成
wait

echo "All tasks finished."