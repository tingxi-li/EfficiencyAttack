from multiprocessing import Process, Event
import os
import sys
import glob
import time
import torch
import numpy as np
import logging
from queue import Empty
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sqlite3
sys.path.append("../")
from legacy_flops import FLOPs_DECORATOR, write_profile, write_profile_lt
import gc

import pynvml
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

class krStream(Process):
    def __init__(self, fr2kr_queue, lpr2kr_queue, kr2lm_queue, device=None):
        super().__init__(name="KRStream")
        self.fr2kr_queue = fr2kr_queue
        self.lpr2kr_queue = lpr2kr_queue
        self.kr2lm_queue = kr2lm_queue
        self.stop_event = Event()
        self.device = device         
        
    def set_config(self, embedding_path, profile_save_path=None, batch_size=1, buffer_limit=None):
        self.embedding_path = embedding_path
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
        self.batch_size = batch_size
        self.buffer_limit = buffer_limit
        self.drop_count = 0
        self.per_image_log = {}

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
        while self.kr2lm_queue.full():
            time.sleep(0.01)
        self.kr2lm_queue.put(None)
        # self.kr2lm_queue.close()
        self.stop_event.set()
        
        try:
            if hasattr(self, "stored_embeddings"):
                del self.stored_embeddings
                self.stored_embeddings = None
            
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
            
            embed_num = self.load_face_embeddings()
            logger.info(f"{self.__class__.__name__:<12} : face embeddings loaded, found {embed_num} embeddings")
            logger.info(f"{self.__class__.__name__:<12} : started")
            
            fr_end_received = False
            lpr_end_received = False
            
            while not self.stop_event.is_set():
                if fr_end_received and lpr_end_received:
                    break

                if not fr_end_received:
                    fr_items, fr_sentinel = self._drain_queue(self.fr2kr_queue, self.batch_size)
                    if fr_sentinel:
                        fr_end_received = True
                        logger.info(f"{self.__class__.__name__:<12} : Received end signal from FR")
                    for tagged_item in fr_items:
                        img_id, data = tagged_item
                        item_start = time.perf_counter()
                        result = self.process_fr_data(data)
                        if result is not None:
                            self._buffered_put(self.kr2lm_queue, (img_id, result))
                        item_end = time.perf_counter()
                        if img_id not in self.per_image_log:
                            self.per_image_log[img_id] = {"start": item_start, "end": item_end, "count": 0}
                        self.per_image_log[img_id]["end"] = item_end
                        self.per_image_log[img_id]["count"] += 1
                        self.count += 1
                        del tagged_item, data

                if not lpr_end_received:
                    lpr_items, lpr_sentinel = self._drain_queue(self.lpr2kr_queue, self.batch_size)
                    if lpr_sentinel:
                        lpr_end_received = True
                        logger.info(f"{self.__class__.__name__:<12} : Received end signal from LPR")
                    for tagged_item in lpr_items:
                        img_id, data = tagged_item
                        item_start = time.perf_counter()
                        result = self.process_lpr_data(data)
                        if result is not None:
                            self._buffered_put(self.kr2lm_queue, (img_id, result))
                        item_end = time.perf_counter()
                        if img_id not in self.per_image_log:
                            self.per_image_log[img_id] = {"start": item_start, "end": item_end, "count": 0}
                        self.per_image_log[img_id]["end"] = item_end
                        self.per_image_log[img_id]["count"] += 1
                        self.count += 1
                        del tagged_item, data

                torch.cuda.empty_cache()
                gc.collect()
                
                
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : Interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received BOTH END signal, shutting down")
            self.end_time = time.perf_counter()
            self.time_elapsed = self.end_time - self.start_time
            self.power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.energy = ( self.power * self.time_elapsed ) / (1e6)
            content = {
                "count": self.count,
                "time": self.time_elapsed,
                "energy": self.energy,
                "drop_count": self.drop_count,
                "per_image": {str(k): v for k, v in self.per_image_log.items()}
            }
            with open(self.profile_save_path + ".json", "w") as f:
                json.dump(content, f, indent=4)
            self.shutdown()
            
            
    def load_face_embeddings(self):
        """Load face embeddings from disk"""
        self.stored_embeddings = {}
        embeddings_path = "./face_embeddings"
        
        for embedding_file in glob.glob(os.path.join(embeddings_path, "*.npy")):
            basename = os.path.basename(embedding_file)
            name = basename.rsplit(".", 1)[0]  # Remove the .npy extension
                
            # Load the embedding
            embedding = np.load(embedding_file)
            self.stored_embeddings[name] = embedding
            
        return len(self.stored_embeddings)
            
        
    def process_fr_data(self, data):
        """Process face recognition data"""
        with torch.no_grad():
            best_match, similarity = self.find_most_similar(data)
            result_string = f"{best_match}|{float(similarity):.4f}"

            # Clean up
            del best_match, similarity
            torch.cuda.empty_cache()
            gc.collect()
            return result_string

    
    def process_lpr_data(self, data):
        """Process license plate recognition data"""
        with torch.no_grad():
            query_result = self.db_query(data)

            # Clean up
            torch.cuda.empty_cache()
            gc.collect()
            return query_result
        
        
    def find_most_similar(self, current_embedding):
        """
        Find the most similar face embedding from the stored embeddings.
        
        Args:
            current_embedding: numpy array of the current face embedding
            
        Returns:
            tuple: (name of the most similar face, similarity score)
        """
        best_match = None
        best_similarity = -1.0
        
        # Flatten the current embedding if needed
        if current_embedding.ndim > 1:
            current_embedding = current_embedding.flatten()
        
        # Normalize the current embedding
        current_embedding_norm = current_embedding / np.linalg.norm(current_embedding)
        
        # Compare with all stored embeddings
        for name, stored_embedding in self.stored_embeddings.items():
            # Flatten the stored embedding if needed
            if stored_embedding.ndim > 1:
                stored_embedding = stored_embedding.flatten()
            
            # Normalize the stored embedding
            stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(current_embedding_norm, stored_embedding_norm)
            
            # Update best match if this is more similar
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        return best_match, best_similarity
    
    
    def db_query(self, plate):
        conn = sqlite3.connect("us_license_plates.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT plate, state FROM license_plates WHERE plate = ?", (plate,))
        result = cursor.fetchone()
        
        if result:
            conn.close()
            return f"Plate found: {result[0]}, state: {result[1]}"
        else:
            return "Plate not found"
            cursor.execute("INSERT INTO license_plates (plate, state) VALUES (?, ?)", (plate, "TX"))
            conn.commit()
            conn.close()
            return "Plate not found, inserted into database"