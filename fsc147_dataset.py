import os
import json
import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset

class FSC147Dataset(Dataset):
    def __init__(self, 
                 data_path,
                 captions=None,
                 captions_sim=None,                 
                 split='train',
                ):
        
        anno_file = os.path.join(data_path, 'annotation_FSC147_384.json')
        data_split_file = os.path.join(data_path, 'Train_Test_Val_FSC_147.json')
        im_dir = os.path.join(data_path, 'images_384_VarV2')
        gt_dir = os.path.join(data_path, 'gt_density_map_adaptive_384_VarV2')
         
        with open(anno_file) as f:
            annotations = json.load(f)
            
        with open(data_split_file) as f:
            data_split = json.load(f)
        
        self.im_dir = im_dir
        self.gt_dir = gt_dir 
        self.im_ids = data_split[split]
        self.annotations = annotations
        
        # Load BLIP2 captions and pre-computed captions similarities
        if captions is not None:
            self.captions = np.load(captions, allow_pickle=True)[()]
            self.file_to_caption = {v:k for k,v in self.captions.items()}
        else: self.captions = None
        
        if captions_sim is not None:
            self.captions_sim = np.load(captions_sim, allow_pickle=True)[()]
        else: self.captions_sim = None                              
            
    def __getitem__(self, idx):
        
        im_id = self.im_ids[idx]
        image = cv2.imread(os.path.join(self.im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        density_path = self.gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')
            
        return {'filename': im_id,
                'image': image, 
                'density': density
               }
    
    def get_similar_captions(self, caption, threshold=0.7):
        assert self.captions_sim is not None
        list_captions = [k for k,v in self.captions_sim[caption].items() if v > threshold]
        if caption in list_captions:
            list(filter(lambda x: x != caption, list_captions))
        return list_captions
    
    def get_metadata(self, filename) -> dict:
        return dict(prompt=self.captions[filename] if self.captions is not None else None)
            
    def __len__(self):
        return len(self.im_ids)   