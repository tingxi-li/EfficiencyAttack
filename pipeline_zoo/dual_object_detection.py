import os
import pdb
import sys
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

class Pipeline:
    def __init__(self, config, device=None):
        self.device = device
        
        self.config = config
        self.logs = []
        self.num_iterations = config["num_iterations"]
        self.dataset_name = config["dataset_name"]
        self.conf_threshold_1 = config["conf_threshold_1"]
        self.conf_threshold_2 = config["conf_threshold_2"]
        self.hf_model_name_1 = config["hf_model_name_1"]
        self.hf_model_name_2 = config["hf_model_name_2"]
        self.num_queries_1 = config["num_queries_1"]
        self.num_queries_2 = config["num_queries_2"]
        self.adv_patch_shape = config["adv_patch_shape"]
        self.adv_patch_placement = config["adv_patch_placement"]
        self.target_labels = config["target_labels"]
        self.lr = config["learning_rate"]
        self.budget = config["budget"]

        self.log_dict = {}

    def load_dataset(self):
        if self.dataset_name == "coco":
            coco_data = load_dataset("detection-datasets/coco", split="val")
            if self.config["test_size"] is not None:
                self.test_size = self.config["test_size"]
            else:
                self.test_size = len(coco_data)
            random_indices = random.sample(range(len(coco_data)), self.test_size)
            return coco_data.select(random_indices)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        
    def load_model(self):
        self.model_1 = RTDetrForObjectDetection.from_pretrained(self.hf_model_name_1).to(self.device)
        self.model_2 = RTDetrForObjectDetection.from_pretrained(self.hf_model_name_2).to(self.device)
        self.processor_1 = RTDetrImageProcessor.from_pretrained(self.hf_model_name_1)
        self.processor_2 = RTDetrImageProcessor.from_pretrained(self.hf_model_name_2)
        
        self.model_1.num_queries = self.num_queries_1
        self.model_2.num_queries = self.num_queries_2
                  
        self.model_1.eval()
        self.model_2.eval()
        
    
    def run_attack(self, dataset):
        for index, example in tqdm(enumerate(dataset), total=dataset.__len__()):
            image_id = example["image_id"]
            image = example["image"].convert("RGB")

            image_tensor = self.processor_1(images=image, return_tensors="pt")["pixel_values"].to(self.device)
            self.bx = torch.zeros_like(image_tensor).requires_grad_(True).to(self.device)
            self.mask =  self.get_mask(image_tensor).to(self.device)

            for i in range(self.num_iterations):
                pdb.set_trace()
                model_1_outputs = self.model_1(image_tensor + self.bx * self.mask.detach(), output_hidden_states=True)
                probs = F.sigmoid(model_1_outputs.logits)
                cls_loss_1  = self.calc_cls_loss(probs, self.target_labels[0])
                norm_loss = self.calc_norm_loss()

                total_loss = cls_loss_1.clone()
                total_loss.backward(retain_graph=True)
                
                with torch.no_grad():
                    self.bx.data -= self.bx.grad * self.lr
                    self.bx.grad.zero_()

                self.bx.data.clamp_( - self.budget["linf"], self.budget["linf"])
                self.bx.requires_grad_(True)


    def get_mask(self, image_tensor):
        if self.adv_patch_shape is None:
            return torch.ones_like(image_tensor)
        elif self.adv_patch_placement == "center":
            h, w = image_tensor.shape[2:]
            ph = pw = self.adv_patch_shape
            sh, sw = (h - ph) // 2, (w - pw) // 2
            mask = torch.zeros_like(image_tensor)
            mask[:, :, sh:sh+ph, sw:sw+pw] = 1.0
            return mask
        elif self.adv_patch_placement == "random":
            raise ValueError("Random patch placement is not supported in this attack.")


    def calc_cls_loss(self, probs, target_labels):
        target_tensor = torch.zeros_like(probs)
        for i in target_labels:
            target_tensor[:, :, i] = 1.0
        cls_loss = F.mse_loss(probs, target_tensor, reduction='sum') / (len(probs.squeeze()) + 1)
        return cls_loss
    
    
    def calc_norm_loss(self):
        l_1_loss = torch.norm(self.bx * self.mask.detach(), p=1) / (self.target_size[0][0] * self.target_size[0][1])  # normalize by target size
        l_2_loss = torch.norm(self.bx * self.mask.detach(), p=2) / 50.0
        norm_loss = l_1_loss + l_2_loss 
        return norm_loss
    
    def log(self):
        log_entry = {
            "image_id": self.img_id,
            "cls_loss_1": self.cls_loss_1.detach().cpu().item(),
            "total_loss": self.total_loss.detach().cpu().item(),
        }
        self.logs.append(log_entry)
    
    
    def get_logs(self):
        return self.logs
    
    
def show_image(image_tensor, title=None):
    processed_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    plt.imsave('debug_image.png', processed_img)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_config_path = "./config/test.json"
    with open(test_config_path, "r") as f:
        config = json.load(f)
        
    pipeline = Pipeline(config, device)
    pipeline.load_model()
    dataset = pipeline.load_dataset()
    pipeline.run_attack(dataset)

