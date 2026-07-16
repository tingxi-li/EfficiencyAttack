from multiprocessing import Process, Queue, Event
from queue import Empty
import multiprocessing as mp
import glob
import time
import torch
import numpy as np
import sys
sys.path.append("../")
import logging
import os
from torchvision import transforms
from transformers import AutoModelForCausalLM # microsoft/git-base
from transformers import AutoProcessor
from legacy_flops import FLOPs_DECORATOR, write_profile, write_profile_lt
from PIL import Image
import gc

import pynvml
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)



class capStream(Process):
    def __init__(self, od2cap_queue, cap2lm_queue, device=None):
        super().__init__(name="CAPStream")
        self.od2cap_queue = od2cap_queue
        self.cap2lm_queue = cap2lm_queue
        self.stop_event = Event()
        self.device = device         
        self.model_id = None
    
    def set_config(self, model_id="microsoft/git-base", profile_save_path=None, batch_size=1, buffer_limit=None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
        self.batch_size = batch_size
        self.buffer_limit = buffer_limit
        self.drop_count = 0
        self.per_image_log = {}
        self.flops_per_call = 0

    def _drain_queue(self, queue, max_items):
        items = []
        sentinel = False
        for _ in range(max_items):
            try:
                data = queue.get(block=False)
                if data is None:
                    sentinel = True
                    break
                items.append(data)
            except Empty:
                break
        return items, sentinel

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
        
    # @FLOPs_DECORATOR
    def _run(self):
        try:
            self.count = 0.0
            self.start_time = time.perf_counter()
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            self.resize_transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
            
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device) 
            self.model.eval()
            logger.info(f"{self.__class__.__name__:<12} : CAP model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            
            while not self.stop_event.is_set():
                items, sentinel = self._drain_queue(self.od2cap_queue, self.batch_size)

                if sentinel and len(items) == 0:
                    break

                for tagged_item in items:
                    img_id, data = tagged_item
                    if isinstance(data, np.ndarray):
                        data_tensor = torch.from_numpy(data).to(self.device)
                    elif isinstance(data, torch.Tensor):
                        data_tensor = data.to(self.device)

                    item_start = time.perf_counter()

                    with torch.no_grad():
                        if self.flops_per_call == 0:
                            from torch.profiler import profile, ProfilerActivity
                            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                                caption = self.inference(data_tensor)
                            for evt in prof.key_averages():
                                self.flops_per_call += evt.flops
                        else:
                            caption = self.inference(data_tensor)

                    item_end = time.perf_counter()

                    if img_id not in self.per_image_log:
                        self.per_image_log[img_id] = {"start": item_start, "end": item_end, "count": 0}
                    self.per_image_log[img_id]["end"] = item_end
                    self.per_image_log[img_id]["count"] += 1

                    self._buffered_put(self.cap2lm_queue, (img_id, caption))
                    self.count += 1

                    del data, data_tensor, caption
                    torch.cuda.empty_cache()
                    gc.collect()

                if sentinel:
                    break
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received END signal, shutting down")
            self.end_time = time.perf_counter()
            self.time_elapsed = self.end_time - self.start_time
            self.power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.energy = ( self.power * self.time_elapsed ) / (1e6)
            content = {
                "count": self.count,
                "time": self.time_elapsed,
                "energy": self.energy,
                "drop_count": self.drop_count,
                "flops_per_call": self.flops_per_call,
                "total_flops": self.flops_per_call * int(self.count),
                "per_image": {str(k): v for k, v in self.per_image_log.items()}
            }
            with open(self.profile_save_path + ".json", "w") as f:
                json.dump(content, f, indent=4)
            self.shutdown()

    def shutdown(self):
        while self.cap2lm_queue.full():
            time.sleep(0.01)
        self.cap2lm_queue.put(None)
        # self.cap2lm_queue.close()
        self.stop_event.set()
        
        try:
            if hasattr(self, 'model'):
                try:
                    self.model = self.model.to("cpu")
                except:
                    pass
                del self.model
                self.model = None
            if hasattr(self, 'processor'):
                del self.processor
                self.processor = None
                
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
            
    def inference(self, data_tensor):
        if isinstance(data_tensor, Image.Image):
            # Process PIL Image
            inputs = self.processor(images=data_tensor, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values
        elif isinstance(data_tensor, torch.Tensor):
            # Process torch Tensor directly
            pixel_values = data_tensor.clone()
        else:
            raise TypeError("Input to the captioning model must be a PIL Image or a torch.Tensor")
        
        pixel_values = self.resize_transform(pixel_values)
        
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption
        