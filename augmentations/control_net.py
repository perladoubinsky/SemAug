from augmentations.generative_augmentation import GenerativeAugmentation
from PIL import Image, ImageOps

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
from PIL import Image

import einops
import random
from pytorch_lightning import seed_everything

import sys
sys.path.append('../ControlNet')
from share import *
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import HWC3

class ControlNet(GenerativeAugmentation):

    def __init__(self,
                 model_config: str,
                 model_path: str,
                 prompt: str = None,
                 strength: float = 1.0, 
                 guidance_scale: float = 9.0,
                 steps: int = 20,
                 guess_mode: bool = False,
                 n_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                 a_prompt: str = "best quality, extremely detailed",
                 eta: float = 0.0,
                 resize: int = 512,
                 **kwargs):

        super(ControlNet, self).__init__()

        self.prompt = prompt

        self.model = create_model(model_config).cpu()
        self.model.load_state_dict(load_state_dict(model_path, location='cuda'))
        self.model.cuda()
        
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = steps
        self.strength = strength
        self.scale = guidance_scale
        self.eta = eta
        
        self.a_prompt = a_prompt
        self.n_prompt = n_prompt
        
        self.guess_mode = guess_mode
        
        self.resize = resize
        
    def density_to_rgb(self, density):
        density = (density - np.min(density)) / (np.max(density) - np.min(density))
        if self.resize is not None:
            density = cv2.resize(density, (self.resize, self.resize))
        density = (density * 255).astype(np.uint8)
        density_rgb = cv2.applyColorMap(density, cv2.COLORMAP_JET)
        density_rgb = cv2.cvtColor(density_rgb, cv2.COLOR_BGR2RGB)
        return density_rgb
    
    def normalize_density(self, density):
        density = (density - np.min(density)) / (np.max(density) - np.min(density))
        return density
    
    def forward(self, image, density, metadata):
        
        prompt = metadata.get("prompt", "")
        
        seed = random.randint(-1, 2147483647)
        
        with torch.no_grad():

            detected_map = HWC3((self.normalize_density(density)*255).astype(np.uint8))            
            detected_map = cv2.resize(detected_map, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
        
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
            H, W, C = detected_map.shape
    
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            cond = {"c_concat": [control], 
                    "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt])]}
            un_cond = {"c_concat": None if self.guess_mode else [control], 
                       "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt])]}
            shape = (4, H // 8, W // 8)

            self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, 1,
                                                              shape, cond, verbose=False,
                                                              eta=self.eta,
                                                              unconditional_guidance_scale=self.scale,
                                                              unconditional_conditioning=un_cond)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            canvas = Image.fromarray(x_samples[0]).resize((density.shape[1],density.shape[0]), Image.BILINEAR)

        return canvas, None