import os
import sys
import pdb
import json
import math
import torch
import random
import argparse
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
import torch.multiprocessing as mp
from torch.utils.data import Subset
from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_ZOO = {
    0: "PekingU/rtdetr_r50vd",
    1: "PekingU/rtdetr_r50vd_coco_o365",
    2: "PekingU/rtdetr_v2_r50vd"
}


def show_image(image_tensor, title=None):
    processed_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    plt.imsave('debug_image.png', processed_img)
    

def set_all_seeds(seed=42):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
    os.environ['PYTHONHASHSEED'] = str(seed)    
