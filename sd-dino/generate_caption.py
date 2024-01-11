import torch
from PIL import Image
import os
from tqdm import tqdm, trange
from lavis.models import load_model_and_preprocess
import numpy as np
import cv2

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

dir = '../data/cosod_data'
datasets = ['DUTS']

# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device)

for dataset in datasets:
    dataset_dir = os.path.join(dir, dataset)
    
    caption_dir = dataset_dir + '/' + 'blip2-caption'
    if not os.path.exists(caption_dir):
        os.makedirs(caption_dir)
    
    groups = os.listdir(dataset_dir + '/' + 'img')
    groups = sorted(groups)    
    
    for i in trange(len(groups)):    
        group = groups[i]
        caption_group_dir = caption_dir + '/' +group
        if not os.path.exists(caption_group_dir):
            os.makedirs(caption_group_dir)
        
        img_files = os.listdir(os.path.join(dataset_dir,'img',group))
        img_files = sorted(img_files)
        
        for img_file in img_files:
            print(img_file)
            raw_image = Image.open(os.path.join(dataset_dir,'img',group,img_file)).convert("RGB")
            raw_mask = Image.open(os.path.join(dataset_dir,'gt',group,img_file[:-4]+'.png')).convert("RGB")
            
            raw_image_np, raw_mask_np = np.array(raw_image), np.array(raw_mask)
            #print(raw_image_np.shape, raw_mask_np.shape, np.max(raw_mask_np), np.max(raw_image_np))
            raw_mask_np[raw_mask_np<=120] = 0
            raw_mask_np[raw_mask_np>120] = 1
            mask_image_np = raw_mask_np * raw_image_np 

            mask_image = Image.fromarray(mask_image_np)

            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            with open(caption_group_dir+'/'+img_file[:-4]+'.txt','w') as f: 
                caption = model.generate({"image": image})[0]
                print(caption)
                f.write(caption)
            
