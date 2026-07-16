import sys
sys.path.append("../")
from tqdm import tqdm
from multiprocessing import Process, Queue, Event
import torch
import glob
import time
from legacy_flops import FLOPs_DECORATOR, write_profile, write_profile_lt
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import multiprocessing as mp
import logging
import random
import gc


import pynvml
import json


random.seed(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_gaussian_noise(image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

class imgStream(Process):
    def __init__(self, img2od_queue, device=None):
        super().__init__(name="ImageStream")
        self.img2od_queue = img2od_queue
        self.stop_event = Event()
        self.device = device
        
    def set_config(self, src_folder_path=None, fps=30, profile_save_path=None, eval_size=50, buffer_limit=None,
                   input_defense_name=None, input_defense_kwargs=None,
                   mix_mode=False, clean_path=None, attack_path=None,
                   mix_ratio=None, shuffle_seed=0, eval_index_start=50,
                   clean_eval_index_start=None, attack_eval_index_start=None):
        self.src_folder_path = src_folder_path
        self.fps = fps
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
        self.eval_size = eval_size
        self.buffer_limit = buffer_limit
        self.drop_count = 0
        # input-defense + mixing config (defense object built inside child process)
        self.input_defense_name = input_defense_name
        self.input_defense_kwargs = input_defense_kwargs or {}
        self.mix_mode = bool(mix_mode)
        self.clean_path = clean_path
        self.attack_path = attack_path
        self.mix_ratio = mix_ratio
        self.shuffle_seed = int(shuffle_seed)
        self.clean_eval_index_start = int(eval_index_start if clean_eval_index_start is None else clean_eval_index_start)
        self.attack_eval_index_start = int(eval_index_start if attack_eval_index_start is None else attack_eval_index_start)
        self.input_drop_total = 0
        self.input_drop_clean = 0
        self.input_drop_attack = 0
        self.source_labels = {}

    def _buffered_put(self, queue, item):
        if self.buffer_limit is not None and queue.qsize() >= self.buffer_limit:
            self.drop_count += 1
            return
        while queue.full():
            time.sleep(0.01)
        queue.put(item)

    def enable_profile(self, flag):
        self.flag = flag
        logger.info(f"{self.__class__.__name__:<12} : internal profiling set to {self.flag}")

    def run(self):
        if self.flag:
            self.profile_run()
        else:
            self._run()
        
    def profile_run(self):    
        from torch.profiler import profile, record_function, ProfilerActivity
        from torch.profiler import schedule
        # torch.cuda.synchronize()   
        torch.cuda.empty_cache() 
        gc.collect()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True,
                     profile_memory=False,
                     record_shapes=False
                     ) as prof:
            with record_function(f"{self.__class__.__name__}"):
                # torch.cuda.synchronize()
                self._run()
                # torch.cuda.synchronize()   
        # torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        write_profile(prof, self.profile_save_path)
        
    def _build_image_list(self):
        """Return list[(path, source_label)] selected for this run."""
        if self.mix_mode:
            rng = random.Random(self.shuffle_seed)
            n_attack = int(round(self.eval_size * float(self.mix_ratio)))
            n_clean = self.eval_size - n_attack

            clean_pool = sorted(glob.glob(f"{self.clean_path}/*.pt"))
            attack_pool = sorted(glob.glob(f"{self.attack_path}/*.pt"))
            clean_eval = clean_pool[self.clean_eval_index_start:]
            attack_eval = attack_pool[self.attack_eval_index_start:]

            if len(clean_eval) < n_clean:
                raise RuntimeError(
                    f"clean eval pool too small: need {n_clean}, have {len(clean_eval)} "
                    f"(after skipping first {self.clean_eval_index_start})"
                )
            if len(attack_eval) < n_attack:
                raise RuntimeError(
                    f"attack eval pool too small: need {n_attack}, have {len(attack_eval)} "
                    f"(after skipping first {self.attack_eval_index_start})"
                )

            picks = [(p, "clean") for p in rng.sample(clean_eval, n_clean)]
            picks += [(p, "attack") for p in rng.sample(attack_eval, n_attack)]
            rng.shuffle(picks)
            return picks

        # single-source mode (preserve existing behavior)
        _paths = sorted(glob.glob(f"{self.src_folder_path}/*.pt"))
        paths = random.sample(_paths, min(self.eval_size, len(_paths)))
        # caller can't know per-image origin in single-source mode
        return [(p, "unknown") for p in paths]

    # @FLOPs_DECORATOR
    def _run(self):
        try:
            self.count = 0.0
            self.start_time = time.perf_counter()
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

            # build defense inside child process (avoids pickling sklearn objects across spawn)
            input_defense = None
            if self.input_defense_name not in (None, "none", ""):
                from input_defense import build_defense
                input_defense = build_defense(self.input_defense_name, **self.input_defense_kwargs)
                logger.info(f"{self.__class__.__name__:<12} : input defense = {self.input_defense_name}")

            picks = self._build_image_list()
            if self.mix_mode:
                logger.info(
                    f"{self.__class__.__name__:<12} : mix mode ratio={self.mix_ratio} "
                    f"seed={self.shuffle_seed} -> {sum(1 for _,l in picks if l=='clean')} clean / "
                    f"{sum(1 for _,l in picks if l=='attack')} attack"
                )
            else:
                logger.info(f"{self.__class__.__name__:<12} : sampled {len(picks)} files from {self.src_folder_path}")
            logger.info(f"{self.__class__.__name__:<12} : started")

            for img_id, (p, label) in enumerate(picks):
                if self.stop_event.is_set():
                    break

                self.source_labels[img_id] = label
                data_nparray = self.ultra_fast_load(p, device=None)

                if input_defense is not None:
                    data_nparray, status = input_defense.apply(data_nparray)
                    if data_nparray is None:
                        self.input_drop_total += 1
                        if label == "clean":
                            self.input_drop_clean += 1
                        elif label == "attack":
                            self.input_drop_attack += 1
                        self.count += 1
                        time.sleep(1 / self.fps)
                        continue

                self._buffered_put(self.img2od_queue, (img_id, data_nparray))
                self.count += 1

                time.sleep(1 / self.fps)

                torch.cuda.empty_cache()
                gc.collect()
                del data_nparray

            logger.info(f"{self.__class__.__name__:<12} : completed, sending END signal")
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.error(f"{self.__class__.__name__:<12} : Interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown")

            self.end_time = time.perf_counter()
            self.time_elapsed = self.end_time - self.start_time
            self.power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.energy = ( self.power * self.time_elapsed ) / (1e6)
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy,
                "drop_count" : self.drop_count,
                "input_drop_total" : self.input_drop_total,
                "input_drop_clean" : self.input_drop_clean,
                "input_drop_attack" : self.input_drop_attack,
                "source_labels" : {str(k): v for k, v in self.source_labels.items()},
                "mix_mode" : self.mix_mode,
                "mix_ratio" : self.mix_ratio,
                "shuffle_seed" : self.shuffle_seed,
                "input_defense" : self.input_defense_name,
            }
            with open(self.profile_save_path + ".json", "w") as f:
                json.dump(content, f, indent=4)

            self.shutdown()
                
    def shutdown(self):
        while self.img2od_queue.full():
            time.sleep(0.01)
        self.img2od_queue.put(None)
        # self.img2od_queue.close()
        self.stop_event.set()
        
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    # torch.cuda.synchronize()
                except:
                    pass
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in shutting down {str(e)}")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown successful")
        
                        
    def ultra_fast_load(self, filepath, device=None):
        with open(filepath, 'rb') as f:
            # Step 1: read number of shape dimensions (1 byte)
            shape_dim = np.frombuffer(f.read(1), dtype=np.int8)[0]

            # Step 2: read the shape (8 bytes per dimension)
            shape = np.frombuffer(f.read(8 * shape_dim), dtype=np.int64)

            # Step 3: read dtype string length (1 byte)
            dtype_len = np.frombuffer(f.read(1), dtype=np.int8)[0]

            # Step 4: read the dtype string
            dtype_str = f.read(dtype_len).decode('ascii')  # e.g. '<f4'

            # Step 5: interpret the rest of the file as data
            np_dtype = np.dtype(dtype_str)
            tensor_data = f.read()

            np_array = np.frombuffer(tensor_data, dtype=np_dtype).copy().reshape(shape)
            
            # return np_array
        
            tensor = torch.from_numpy(np_array)

            # Optional: move to device
            if device is not None:
                tensor = tensor.to(device)
                return tensor
            else:
                return np_array

            
if __name__ == "__main__":
    # tmp_queue = mp.Queue()
    # instance = imgStream(tmp_queue, device="cuda")
    # instance.set_config(src_folder_path="../test_src", fps=30, profile_save_path="../profile", eval_size=10)
    # instance.enable_profile(False)
    # instance.run()
    # instance.shutdown()

    
    import pynvml
    import time
    import torch
    def measure_gpu(device):
        pynvml.nvmlInit()
        device_id = 0 if device.index is None else device.index
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        t1 = time.time()
        power_list = []
        tmp_queue = mp.Queue()
        instance = imgStream(tmp_queue, device=device)
        instance.set_config(src_folder_path="../test_src", fps=30, profile_save_path="../profile", eval_size=10)
        instance.enable_profile(False)
        instance.run()
        instance.shutdown()
        del tmp_queue
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_list.append(power)
        t2 = time.time()
        latency = t2 - t1
        s_energy = sum(power_list) / len(power_list) * latency
        energy = s_energy / (10 ** 6)
        pynvml.nvmlShutdown()
        return latency, energy
    
        count = 0.0
        start_time = time.perf_counter()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
        
        end_time = time.perf_counter()
        time_elapsed = end_time - start_time
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        energy = ( power * time_elapsed ) / (1e6)
        content = {
            "count" : self.count,
            "time" : self.time_elapsed,
            "energy" : energy
        }
    
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    latency, energy = measure_gpu(device) # watts
    print(latency, energy)