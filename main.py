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
from pipeline_zoo.zoo import Pipeline_dict

import base_utils


def get_pipeline(pipeline_id):
    if pipeline_id == -1:
        raise ValueError("Pipeline ID must be provided.")
    else:
        try:
            return Pipeline_dict[pipeline_id]
        except:
            raise KeyError("invalid pipeline id, please refer to ./pipeline_zoo/zoo.py")



def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = get_pipeline(config["pipeline_id"])

    pipeline = p(config, device)
    dataset = pipeline.load_dataset()
    
    pdb.set_trace()  # Debugging breakpoint

    pipeline.load_model()
    pipeline.run_attack(dataset)



def worker(gpu, config):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    base_utils.set_all_seeds(42)

    p = get_pipeline(config.pipeline_id)
    pipeline = p(config, device)
    dataset = pipeline.load_dataset()
    indices = list(range(len(dataset)))
    chunks = np.array_split(indices, config.ngpus)
    subset = Subset(dataset, [int(i) for i in chunks[gpu]])
    
    pipeline.load_model()
    pipeline.run_attack(subset)

    logs = pipeline.get_logs()
    out = os.path.join(config.output_dir, f"RANK_{gpu}.json")
    with open(out, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"[GPU {gpu}] done, results saved to {out}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic deep learning pipeline system")
    parser.add_argument("--config_file", type=str, default="./config/test.json", help="Path to the config.json file")
    parser.add_argument("--pipeline_id", type=int, default=-1, help="specify the pipeline used by numbers")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)
        
    config["pipeline_id"] = args.pipeline_id
    
    if config.parallel:
        config.ngpus = torch.cuda.device_count()
        mp.spawn(worker, nprocs=config.ngpus, args=(config,))
    else:
        main(config)


    
    