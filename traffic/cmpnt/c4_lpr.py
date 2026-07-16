from multiprocessing import Process, Event, Queue
from queue import Empty
import multiprocessing as mp
import torch
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import numpy as np
import os
import glob
import logging
import sys
import time
from PIL import Image
sys.path.append("../")
from legacy_flops import FLOPs_DECORATOR, write_profile, write_profile_lt
# from model import create_model # https://github.com/dbpprt/pytorch-licenseplate-segmentation/tree/master
from fast_plate_ocr import LicensePlateRecognizer as ONNXPlateRecognizer # https://github.com/ankandrew/fast-plate-ocr
import onnxruntime as ort
import gc

import pynvml
import json


options = ort.SessionOptions()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)


class lprStream(Process):
    def __init__(self, od2lpr_queue, lpr2kr_queue, device=None):
        super().__init__(name="LPRStream")
        self.od2lpr_queue = od2lpr_queue
        self.lpr2kr_queue = lpr2kr_queue
        self.stop_event = Event()
        self.device = device         
        self.model_id = None
        
    def set_config(self, model_id="model_v2.pth", profile_save_path=None, batch_size=1, buffer_limit=None):
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

    def shutdown(self):
        while self.lpr2kr_queue.full():
            time.sleep(0.01)
        self.lpr2kr_queue.put(None)
        # self.lpr2kr_queue.close()
        self.stop_event.set()
        
        try:
            if hasattr(self, 'deeplabv3'):
                try:
                    self.deeplabv3 = self.deeplabv3.to("cpu")
                except:
                    pass
                del self.deeplabv3
                self.deeplabv3 = None
            if hasattr(self, 'onnx_lp_ocr'):
                del self.onnx_lp_ocr
                self.onnx_lp_ocr = None
            if hasattr(self, 'checkpoint'):
                del self.checkpoint
                self.checkpoint = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    # torch.cuda.synchronize()
                except:
                    pass
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in shutting down {str(e)}")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown successful")
            
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
        write_profile(prof, self.profile_save_path)
        
    # @FLOPs_DECORATOR
    def _run(self):
        try:

            self.count = 0.0
            self.start_time = time.perf_counter()
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            self.deeplabv3 = self.create_model()
            self.checkpoint = torch.load(self.model_id, map_location='cpu')
            self.deeplabv3.load_state_dict(self.checkpoint['model'])
            self.deeplabv3.eval().to(self.device) 
            self.onnx_lp_ocr = ONNXPlateRecognizer('argentinian-plates-cnn-model', 
                                                    providers=['CPUExecutionProvider'])
            logger.info(f"{self.__class__.__name__:<12} : LPR model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            while not self.stop_event.is_set():
                items, sentinel = self._drain_queue(self.od2lpr_queue, self.batch_size)

                if sentinel and len(items) == 0:
                    break

                for tagged_item in items:
                    try:
                        img_id, data = tagged_item
                        data_tensor = torch.from_numpy(data).to(self.device)

                        item_start = time.perf_counter()

                        with torch.no_grad():
                            if self.flops_per_call == 0:
                                from torch.profiler import profile, ProfilerActivity
                                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                                    pred = self.segmentation(data_tensor, self.deeplabv3)
                                for evt in prof.key_averages():
                                    self.flops_per_call += evt.flops
                            else:
                                pred = self.segmentation(data_tensor, self.deeplabv3)

                            plate_tensor = self.post_process(pred, data_tensor.detach().clone())
                            plate_result = self.ocr(plate_tensor)[0]
                            plate_text = plate_result.plate if hasattr(plate_result, 'plate') else str(plate_result)

                        item_end = time.perf_counter()

                        if img_id not in self.per_image_log:
                            self.per_image_log[img_id] = {"start": item_start, "end": item_end, "count": 0}
                        self.per_image_log[img_id]["end"] = item_end
                        self.per_image_log[img_id]["count"] += 1

                        self._buffered_put(self.lpr2kr_queue, (img_id, plate_text))
                        self.count += 1
                        torch.cuda.empty_cache()

                    except Exception as e:
                        logger.warning(f"{self.__class__.__name__:<12} : error processing item: {str(e)}")
                        self.count += 1
                        continue

                    del data, data_tensor, pred, plate_tensor, plate_text

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

    def segmentation(self, image, model):
        """Run prediction on the segmentation model"""

        if image.dim() == 3:
            image = image.unsqueeze(0)
        output = model(image)['out'][0]
        return output
    
    def post_process(self, output, request, threshold=0.1):
        """Extract the license plate region based on segmentation output"""
        output = (output > threshold).type(torch.IntTensor)
        output = output.cpu().numpy()[0]
        result = np.where(output > 0)
        coords = list(zip(result[0], result[1]))
        
        if coords:
            y_coords, x_coords = zip(*coords)
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            if request.dim() == 4:
                request = request.squeeze(0)
            if isinstance(request, torch.Tensor):
                return request[:, min_y:max_y+1, min_x:max_x+1]
            
            return torch.from_numpy(request[:, min_y:max_y+1, min_x:max_x+1])
        else:
            return request
            
    def ocr(self, plate_tensor):
        uint8_tensor = plate_tensor.type(torch.uint8)
        uint8_tensor = plate_tensor.clone().detach()
        if uint8_tensor.dim() == 4:
            uint8_tensor.squeeze_(0)
        
        gray_tensor = 0.299 * uint8_tensor[0] + 0.587 * uint8_tensor[1] + 0.114 * uint8_tensor[2]
        gray_tensor = gray_tensor.to(torch.uint8)  # Convert back to uint8 after grayscale calculation
        gray_tensor.unsqueeze_(-1)
            
        plate_array = gray_tensor.cpu().numpy()
        
        if plate_array.shape[0] == 0 or plate_array.shape[1] == 0:
            return ["UNKNOWN"]
        
        result = self.onnx_lp_ocr.run(plate_array)
        return result
    
    def create_model(self, outputchannels=1, aux_loss=True, freeze_backbone=False):
        model = models.segmentation.deeplabv3_resnet101(
            weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
            progress=True, 
            aux_loss=aux_loss)

        if freeze_backbone is True:
            for p in model.parameters():
                p.requires_grad = False

        model.classifier = DeepLabHead(2048, outputchannels)

        return model
    