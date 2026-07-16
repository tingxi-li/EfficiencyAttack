import pandas as pd
import torch
import numpy as np
import os
import random
import sys
import json

path = "../traffic/profile"
path_var = "../traffic/profile_var"

def calculate_flops(pipeline, num_list):
    img_num, person_num, car_num, oven_num, girrafe_num = num_list
    
    # flops of img streaming
    flops_1 = 0.0
    
    # flops of object detection
    flops_2 = img_num * 136e9
    
    # flops of face recognition
    flops_3 = person_num * 48.5156e9
    
    # flops of license plate recognition
    flops_4 = car_num * 588.722e9
    
    # flops of cap
    flops_5 = img_num * 1204224
    
    # flops of kr
    flops_6 = person_num * 2562537
    
    if pipeline == 0:
        ret = flops_1 + flops_2 + flops_3 + flops_4 + flops_5 + flops_6
    if pipeline == 1:
        ret = flops_1 + flops_2 + flops_3 + flops_4 + flops_6
    if pipeline == 2:
        ret = flops_1 + flops_2 + flops_5
        
    return ret

class model_0:
    def __init__(self):
        self.teaspoon_tgt_68 = [7.40422e+05, 4.52900e+03, 1.34100e+03, 7.27431e+05, 2.00000e+01]
        self.phantom_tgt_none = [1.8169e+04, 7.1080e+03, 6.7600e+02, 1.4000e+01, 9.2000e+01]
        self.teaspoon_tgt_0_2 = [9.70439e+05, 1.87931e+05, 7.77190e+05, 0.00000e+00, 1.88000e+02]
        self.teaspoon_tgt_2 = [9.56858e+05, 1.79700e+03, 9.44718e+05, 9.00000e+00, 1.80000e+01]
        self.teaspoon_tgt_23 = [833254.,   1446.,   1008.,      0., 828748.]
        self.teaspoon_tgt_none = [103163.,   4107.,   6089.,      0.,      0.]
        self.slowtrack_tgt_none = [103871.,   4060.,   5929.,      0.,      0.]
        self.teaspoon_tgt_0 = [8.85682e+05, 8.80408e+05, 6.47000e+02, 0.00000e+00, 1.80000e+01]
        self.overload_tgt_none = [6.850e+04, 4.285e+03, 4.721e+03, 0.000e+00, 1.500e+01]
    
class model_1:
    def __init__(self):
        self.teaspoon_tgt_68 = [5.52218e+05, 2.17800e+03, 1.80000e+01, 5.46807e+05, 2.00000e+00]
        self.phantom_tgt_none = [1.9126e+04, 6.2330e+03, 6.0600e+02, 1.6000e+01, 1.0200e+02]
        self.teaspoon_tgt_0_2 = [9.62400e+05, 4.91269e+05, 4.66219e+05, 5.00000e+00, 5.20000e+01]
        self.teaspoon_tgt_2 = [9.09855e+05, 2.51500e+03, 9.02924e+05, 7.00000e+00, 3.00000e+00]
        self.teaspoon_tgt_23 = [6.93051e+05, 1.41400e+03, 9.80000e+01, 1.00000e+00, 6.89178e+05]
        self.teaspoon_tgt_none = [2.36695e+05, 6.26000e+03, 3.82500e+03, 0.00000e+00, 1.00000e+00]
        self.slowtrack_tgt_none = [2.25602e+05, 6.21400e+03, 3.90900e+03, 1.00000e+00, 1.00000e+00]
        self.teaspoon_tgt_0 = [8.59192e+05, 8.55277e+05, 2.11000e+02, 0.00000e+00, 7.20000e+01]
        self.overload_tgt_none = [8.0686e+04, 5.5550e+03, 1.6630e+03, 0.0000e+00, 5.3000e+01]
        
class model_2:
    def __init__(self):
        self.teaspoon_tgt_68 = [7.92531e+05, 4.06600e+03, 4.58000e+02, 7.78614e+05, 7.00000e+00]
        self.phantom_tgt_none = [4.4173e+04, 1.6104e+04, 1.5700e+03, 1.5000e+01, 3.1500e+02]
        self.teaspoon_tgt_0_2 = [9.97647e+05, 4.36270e+05, 5.57113e+05, 0.00000e+00, 3.10000e+01]
        self.teaspoon_tgt_2 = [9.35002e+05, 2.70700e+03, 9.25418e+05, 0.00000e+00, 2.50000e+01]
        self.teaspoon_tgt_23 = [8.88380e+05, 2.16800e+03, 1.06000e+03, 1.00000e+00, 8.81277e+05]
        self.teaspoon_tgt_none = [4.79648e+05, 1.68300e+04, 1.81670e+04, 1.60000e+01, 6.40000e+01]
        self.slowtrack_tgt_none = [4.68557e+05, 1.69380e+04, 1.81520e+04, 2.00000e+01, 5.00000e+01]
        self.teaspoon_tgt_0 = [9.02324e+05, 8.95824e+05, 1.80500e+03, 0.00000e+00, 2.99000e+02]
        self.overload_tgt_none = [3.16431e+05, 1.62320e+04, 9.91300e+03, 8.00000e+00, 5.40000e+01]
        
        
if __name__ == "__main__":
    data_model_0 = model_0()
    data_model_1 = model_1()
    data_model_2 = model_2()
    attrs = {k: v for k, v in vars(data_model_0).items() if not callable(v)}
    print("=" * 80)
    print("\n" + "pipeline: model 0" + "\n")
    flops_values = {}
    for k, v in attrs.items():
        flops_values[k] = calculate_flops(1, v)
    max_flops = max(flops_values.values())
    for k, flops in flops_values.items():
        if flops == max_flops:
            print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
        else:
            print(f"{k:<19}: "  f"{str(flops):>30}")
    print("\n" + "=" * 80 + "\n")

    attrs = {k: v for k, v in vars(data_model_1).items() if not callable(v)}
    print("=" * 80)
    print("\n" + "pipeline: model 1" + "\n")
    flops_values = {}
    for k, v in attrs.items():
        flops_values[k] = calculate_flops(1, v)
    max_flops = max(flops_values.values())
    for k, flops in flops_values.items():
        if flops == max_flops:
            print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
        else:
            print(f"{k:<19}: "  f"{str(flops):>30}")
    print("\n" + "=" * 80 + "\n")
    
    attrs = {k: v for k, v in vars(data_model_2).items() if not callable(v)}
    print("=" * 80)
    print("\n" + "pipeline: model 2" + "\n")
    flops_values = {}
    for k, v in attrs.items():
        flops_values[k] = calculate_flops(1, v)
    max_flops = max(flops_values.values())
    for k, flops in flops_values.items():
        if flops == max_flops:
            print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
        else:
            print(f"{k:<19}: "  f"{str(flops):>30}")
    print("\n" + "=" * 80 + "\n")
    
    attrs = {k: v for k, v in vars(data_model_0).items() if not callable(v)}
    print("=" * 80)
    print("\n" + "pipeline variation 1: model 0" + "\n")
    flops_values = {}
    for k, v in attrs.items():
        flops_values[k] = calculate_flops(1, v)
    max_flops = max(flops_values.values())
    for k, flops in flops_values.items():
        if flops == max_flops:
            print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
        else:
            print(f"{k:<19}: "  f"{str(flops):>30}")
    print("\n" + "=" * 80 + "\n")

    attrs = {k: v for k, v in vars(data_model_1).items() if not callable(v)}
    print("=" * 80)
    print("\n" + "pipeline variation 1: model 1" + "\n")
    flops_values = {}
    for k, v in attrs.items():
        flops_values[k] = calculate_flops(1, v)
    max_flops = max(flops_values.values())
    for k, flops in flops_values.items():
        if flops == max_flops:
            print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
        else:
            print(f"{k:<19}: "  f"{str(flops):>30}")
    print("\n" + "=" * 80 + "\n")
    
    attrs = {k: v for k, v in vars(data_model_2).items() if not callable(v)}
    print("=" * 80)
    print("\n" + "pipeline variation 1: model 2" + "\n")
    flops_values = {}
    for k, v in attrs.items():
        flops_values[k] = calculate_flops(1, v)
    max_flops = max(flops_values.values())
    for k, flops in flops_values.items():
        if flops == max_flops:
            print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
        else:
            print(f"{k:<19}: "  f"{str(flops):>30}")
    print("\n" + "=" * 80 + "\n")