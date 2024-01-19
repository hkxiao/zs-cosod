from segment_anything.build_sam_baseline import sam_model_registry_baseline
from segment_anything.predictor import SamPredictor
import os
import torch
import cv2
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import argparse
from process_feat import process_feat
from cosal import Cosal_Module
from kmeans_pytorch import kmeans
from utils import mkdir

def show_point(img, color, coord):
    radius = 8  # 点的半径
    thickness = -1  # 填充点的厚度，-1 表示填充
    cv2.circle(img, coord, radius, color, thickness)

def show_correlation(correlation_maps,save_path,name_list,tag=''):
    N,k,H,W = correlation_maps.shape            
    correlation_maps = torch.mean(correlation_maps, dim=1).flatten(-2)
    min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
    max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
    correlation_maps = (correlation_maps - min_value) / (max_value - min_value)  # shape=[N, HW]
    correlation_maps = correlation_maps.view(N,1,H, W) 
    correlation_maps[correlation_maps>0.5]=1
    correlation_maps[correlation_maps<0.5]=0
    correlation_maps = F.interpolate(correlation_maps,size=(256,256),mode='bilinear',align_corners=False) * 255
    
    for correlation_map,name in zip(correlation_maps,name_list):
        cv2.imwrite(os.path.join(save_path,name[:-4]+tag),correlation_map[0].cpu().numpy())
        
    return correlation_maps
      
def save(examples, outputs, update=False, tag=''):    
    for example,output in zip(examples,outputs):
        #print(output.keys())
        masks, low_res_logits, iou_predictions = output['masks'],output['low_res_logits'], output['iou_predictions']
        masks = torch.nn.functional.interpolate(masks*255.0, size=(224,224), mode='bilinear',align_corners=True)
        masks_np = masks[0,0,...].cpu().numpy()

        cv2.imwrite(example['save_dir']+'/'+example['name']+tag+'.png', masks_np) 
        img_np = example['image'].permute(1,2,0).cpu().numpy() 
        
        if 'point_coords' in example.keys():
            for point_coord,point_label in zip(example['point_coords'][0],example['point_labels'][0]):
                color = (255,0,0) if point_label.item() == 1 else (0,0,255)
                if args.only_show_positive and color == (0,0,255):
                    continue
                show_point(img_np, color, tuple(point_coord.to(torch.int).tolist()))
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(example['save_dir']+'/'+example['name']+'_prompt.png', img_np) 
        if update:
            masks = torch.nn.functional.interpolate(masks, size=(256,256), mode='bilinear',align_corners=True)
            example['mask_inputs'] = masks

        
def get_point(correlation_map, topk=3): # N k H W
    correlation_max = torch.max(correlation_map, dim=1)[0] # N H W
    ranged_index = torch.argsort(torch.flatten(correlation_max, -2), 1, descending=True) #N HW
    coords = torch.stack([ranged_index[:,:32]%60,ranged_index[:,:32]/60],-1) #N 32 2
    centers = []
    for k in range(coords.shape[0]):
        center = kmeans(coords[k],K=topk, max_iters=20) #2 2
        centers.append(center)
    max_centers = torch.stack(centers,dim=0) #N k 2
    
    return max_centers


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",type=str,nargs='+',default=['CoCA', 'CoSal2015', 'CoSOD3k'])
    parser.add_argument("--sam_type",type=str,default='vit_b')
    parser.add_argument("--sam_checkpoint",type=str,default='ckpt/sam_vit_b_01ec64.pth')
    parser.add_argument("--batch",type=int,default=10)
    parser.add_argument("--data_root",type=str,default='data/')
    parser.add_argument("--save_root",type=str,default='pred/')
    parser.add_argument("--sod_root",type=str,default='PoolNet/results/')
    parser.add_argument("--gpu",type=int,default=1)
    parser.add_argument("--iter",type=int,default=1)
    parser.add_argument("--kiter",type=int,default=10)
    parser.add_argument("--only_show_positive", action='store_true')
    global args
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    sam = sam_model_registry_baseline[args.sam_type](checkpoint=args.sam_checkpoint)
    sam.cuda()
    Cosal = Cosal_Module()
    mkdir(args.save_root)
    examples = []
    
    for i,dataset in enumerate(tqdm(args.datasets)):
        dataset_path = os.path.join(args.data_root,dataset)       
        groups = sorted(os.listdir(dataset_path+'/img'),reverse=False)
        mkdir(dataset_path.replace(args.data_root, args.save_root))
        topk = 3 if dataset != 'CoCA' else 2
        
        for j, group in enumerate(tqdm(groups)):
            save_group_path = os.path.join(args.save_root,dataset,group)
            mkdir(save_group_path)
                
            feat_path = os.path.join(dataset_path, 'sd_raw+dino_feat', group)
            files = sorted(os.listdir(feat_path))
            features, sisms = [], []
            
            for k,file in enumerate(files):
                all_feat = torch.load(feat_path+'/'+file, map_location='cuda')
                sd_feat, dino_feat = all_feat['sd_feat'], all_feat['dino_feat']
                feature = process_feat(sd_feat,dino_feat)
                features.append(feature)

                # print(args.sod_root+'/'+dataset+'/'+group+'/'+file[:-4]+'.png')
                sism = cv2.imread(args.data_root+'/'+dataset+'/sism/'+group+'/'+file[:-4]+'.png').astype(np.float32)
                sism = cv2.cvtColor(sism, cv2.COLOR_BGR2GRAY)
                sism = cv2.resize(sism, (60,60))
                sism_torch = torch.from_numpy(sism).cuda() 
                sisms.append(sism_torch.unsqueeze(0).unsqueeze(0))
                
            features, sisms = torch.cat(features), torch.cat(sisms) / 255.0

            correlation, correlation_b = Cosal(features,sisms), Cosal(features, (1-sisms))    # N k H W
            sisms = show_correlation(correlation, save_group_path, files, tag='_co_'+str(k)+'.png')
            sisms_b = show_correlation(correlation_b, save_group_path, files, tag='_co_'+str(k)+'_b_.png')    
                
            p_coords, n_coords = get_point(correlation,topk=topk), get_point(correlation_b,topk=topk)
            p_coords, n_coords = p_coords*1024/60, n_coords*1024/60
                        
            img_path = os.path.join(dataset_path, 'img', group)
            files = sorted(os.listdir(img_path))
            
            for k,file in enumerate(files):
                img = cv2.imread(img_path+'/'+file).astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (1024,1024))
                img_torch = torch.from_numpy(img).permute(2,0,1).cuda()
                
                example = {}
                example['image'] = img_torch
                example['point_coords'] = torch.cat([p_coords[k],n_coords[k]]).unsqueeze(0)
                example['point_labels'] = torch.cat([torch.ones(p_coords[k].shape[0]),torch.zeros(n_coords[k].shape[0])]).unsqueeze(0).cuda().to(torch.float32)                
                example['original_size'] = (1024, 1024)
                example['save_dir'] = os.path.join(args.save_root,dataset,group) 
                example['name'] = file[:-4]
                
                #print(example['point_labels'].shape, example['point_coords'].shape)
                examples.append(example)
                
                if len(examples) !=args.batch and (i != len(args.datasets)-1 or j != len(groups)-1 or k != len(files)-1):
                    continue
                                   
                with torch.no_grad():    
                    outputs = sam(examples, multimask_output=False)
                    save(examples, outputs, update=True)
                
                    outputs = sam(examples, multimask_output=False)
                    save(examples, outputs, update=False, tag='_refine')

                examples = []