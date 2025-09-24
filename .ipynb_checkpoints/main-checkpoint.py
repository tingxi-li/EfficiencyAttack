import json
import time
import copy
import queue
import torch
import argparse
import numpy as np
import torch.multiprocessing as mp
from collections import OrderedDict
from torch.utils.data import Subset

from pipeline_zoo.zoo import Pipeline_dict
import base_utils


def get_pipeline(pipeline_id):
    if pipeline_id == -1:
        raise ValueError("Pipeline ID must be provided.")
    try:
        return Pipeline_dict[pipeline_id]
    except KeyError as exc:
        raise KeyError("invalid pipeline id, please refer to ./pipeline_zoo/zoo.py") from exc


def _run_single_process(config: dict) -> None:
    """Run the selected pipeline on a single device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline_cls = get_pipeline(config["pipeline_id"])
    pipeline = pipeline_cls(config, device)
    dataset = pipeline.load_dataset()
    pipeline.load_model()
    pipeline.run_attack(dataset)


def _override_write_log(pipeline, buffer_holder):
    """Replace pipeline.write_log to capture logs instead of writing files."""

    def _capture(log_path=None):
        buffer_holder["data"] = [
            (key, copy.deepcopy(value))
            for key, value in pipeline.log_dict.items()
        ]
        pipeline.log_dict = {}

    return _capture


def _worker(rank: int, world_size: int, config: dict, run_timestamp: str, result_queue) -> None:
    """Worker process: runs a shard of the dataset and reports logs."""
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        seed = config.get("seed", 42)
        base_utils.set_all_seeds(seed)

        pipeline_cls = get_pipeline(config["pipeline_id"])
        local_config = copy.deepcopy(config)
        local_config["parallel"] = False

        pipeline = pipeline_cls(local_config, device)
        pipeline.run_timestamp = run_timestamp

        captured = {"data": []}
        pipeline.write_log = _override_write_log(pipeline, captured)

        dataset = pipeline.load_dataset()
        indices = list(range(len(dataset)))
        chunks = np.array_split(indices, world_size)
        shard = chunks[rank] if len(chunks) > rank else []
        subset = Subset(dataset, [int(i) for i in shard])

        pipeline.load_model()
        pipeline.run_attack(subset)

        result_queue.put(("log", rank, captured["data"]))
    except Exception as exc:
        result_queue.put(("error", rank, repr(exc)))
        raise
    finally:
        result_queue.put(("done", rank, None))


def _merge_and_write_logs(config: dict, run_timestamp: str, shard_logs: dict) -> None:
    """Merge shard logs and write identical outputs as the single-process case."""
    pipeline_cls = get_pipeline(config["pipeline_id"])
    writer_config = copy.deepcopy(config)
    writer_config["parallel"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = pipeline_cls(writer_config, device)
    writer.run_timestamp = run_timestamp

    combined = OrderedDict()
    for rank in sorted(shard_logs.keys()):
        for key, value in shard_logs[rank]:
            combined[key] = value

    writer.log_dict = combined
    writer.write_log()


def _run_parallel(config: dict) -> None:
    """Run the selected pipeline across multiple GPUs using data parallelism."""
    world_size = config.get("ngpus")
    if not world_size:
        world_size = torch.cuda.device_count()
        config["ngpus"] = world_size

    world_size = int(world_size)

    if world_size <= 0:
        raise RuntimeError("No GPUs available for parallel execution.")

    manager = mp.Manager()
    result_queue = manager.Queue()
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    spawn_args = (world_size, config, run_timestamp, result_queue)
    mp.spawn(_worker, args=spawn_args, nprocs=world_size, join=True)

    shard_logs = {}
    errors = {}
    completed = 0

    try:
        while completed < world_size:
            try:
                kind, rank, payload = result_queue.get(timeout=60)
            except queue.Empty:
                continue

            if kind == "log":
                shard_logs[rank] = payload
            elif kind == "error":
                errors[rank] = payload
            elif kind == "done":
                completed += 1
                shard_logs.setdefault(rank, [])

        if errors:
            raise RuntimeError(f"Parallel workers failed: {errors}")

        _merge_and_write_logs(config, run_timestamp, shard_logs)
    finally:
        manager.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic deep learning pipeline system")
    parser.add_argument("--config_file", type=str, default="./config/test.json", help="Path to the config.json file")
    parser.add_argument("--pipeline_id", type=int, default=-1, help="specify the pipeline used by numbers")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    config["pipeline_id"] = args.pipeline_id

    if config.get("parallel"):
        _run_parallel(config)
    else:
        _run_single_process(config)
