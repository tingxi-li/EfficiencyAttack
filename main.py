import os
import sys
import pdb
import json
import torch
import argparse
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset

import base_utils


def get_pipeline(pipeline_id):
    if pipeline_id is None:
        raise ValueError("Pipeline ID must be provided.")
    
    if pipeline_id == 0:
        from pipeline_zoo import dual_object_detection
        return dual_object_detection.Pipeline
    else:
        raise ValueError(f"Invalid pipeline ID: {pipeline_id}.")


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_utils.set_all_seeds(42)

    p = get_pipeline(config.pipeline_id)

    pipeline = p(config, device)
    dataset = pipeline.load_dataset()
    
    pipeline.load_model()
    pipeline.run_attack(dataset)
    
    logs = pipeline.get_logs()
    with open(config.output_log, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"Logs saved to {config.output_log}")


def worker(gpu, config):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    base_utils.set_all_seeds(42)

    p = get_pipeline(config.pipeline_id)
    pipeline = p(config, device)
    dataset = pipeline.load_dataset()
    chunks = np.array_split(idx, config.ngpus)
    subset = Subset(full, [int(i) for i in chunks[gpu]])
    
    pipeline.load_model()
    pipeline.run_attack(subset)

    logs = pipeline.get_logs()
    out = os.path.join(config.output_dir, f"RANK_{gpu}.json")
    with open(out, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"[GPU {gpu}] done, results saved to {out}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic deep learning pipeline system")

    parser.add_argument("config_file", type=str, help="Path to the config.json file")
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join(config.output_dir, timestamp)

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=4)

    if config.p:
        config.ngpus = torch.cuda.device_count()
        mp.spawn(worker, nprocs=config.ngpus, args=(config,))
    else:
        main(config)


    
    