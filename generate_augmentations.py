import os
import argparse
import random
import numpy as np
from itertools import product
from tqdm import tqdm

import torch

from augmentations.real_guidance import RealGuidance
from augmentations.control_net import ControlNet
from augmentations.control_net_img2img import ControlNetImg2Img

from fsc147_dataset import FSC147Dataset


DATASETS = {
    "fsc147": FSC147Dataset,
}

AUGMENTATIONS = {
    "real-guidance": RealGuidance,
    "control-net": ControlNet,
    "control-net-img2img": ControlNetImg2Img,
}

DEFAULT_SYNTHETIC_DIR = "aug/{dataset}-{aug}-{seed}"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate Augmentations")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--data_path", type=str, help='path to data')
    parser.add_argument("--dataset", type=str, help='dataset type', default='fsc147')   
    parser.add_argument("--aug", type=str, help='augmentation type', default='control-net')     
    parser.add_argument("--model_path", type=str, help='path to generative model')
    parser.add_argument("--model_config", type=str, help='path to config for control-net', default=None)
    parser.add_argument("--num_synthetic", type=int, help='num of synthetic samples per image', default=0)
    parser.add_argument("--synthetic_dir", type=str, help='path to save synthetic images', default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--guidance_scale", type=float, help='guidance scale for classifier-free guidance', default=7.5)
    parser.add_argument("--steps", type=int, help='number of denoising steps', default=20)
    parser.add_argument("--t0", type=float, help='level of noise for img2img generation', default=0.5)
    parser.add_argument("--strength", type=float, help='strength of conditioning for control-net', default=1.0)
    parser.add_argument("--captions", type=str, help='path to dict with captions', default=None)
    parser.add_argument("--captions_sim", type=str, help='path to dict with captions similarities', default=None)
    parser.add_argument("--prompt_template", type=str, help='template prompt for real-guidance', default=None)
    parser.add_argument("--swap_caption_prob", type=float, help='probability to swap captions', default=0.0)
    parser.add_argument("--threshold", type=float, default=0.7, help="threshold for captions similarities")
    
    args = parser.parse_args()
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    dataset = DATASETS[args.dataset](
        args.data_path,
        args.captions,
        args.captions_sim,
    )
    
    aug = AUGMENTATIONS[args.aug](
            model_config=args.model_config,
            model_path=args.model_path,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            t0=args.t0,        
            strength=args.strength,
            prompt=args.prompt_template
    )
    
    synthetic_dir = args.synthetic_dir.format(**vars(args))
    os.makedirs(synthetic_dir, exist_ok=True)
        
    options = product(range(len(dataset)), range(args.num_synthetic))
        
    for idx, num in tqdm(list(options)):
        
#        filename = 'aug-{}-{}'.format(idx, num)
        data = dataset.__getitem__(idx)
        metadata = dataset.get_metadata(data['filename'])
        filename = 'aug-{}-{}'.format(data['filename'].split('.jpg')[0], num)
        
        if np.random.uniform() < args.swap_caption_prob:
            # retrieve captions more similar than some threshold
            list_captions = dataset.get_similar_captions(metadata['prompt'], args.threshold)
            if len(list_captions) > 0:
                # random choice among similar captions
                np.random.shuffle(list_captions)
                # update prompt 
                metadata['prompt'] = np.random.choice(list_captions)
                # keep track of the new prompt
                filename += '-caption-{}'.format(dataset.file_to_caption[metadata['prompt']][:-4])
                
        image, _ = aug(data['image'], data['density'], metadata)
        
        if synthetic_dir is not None:
            pil_image, image = image, os.path.join(synthetic_dir, filename + '.jpg')
            pil_image.save(image)