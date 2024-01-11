import torch
from torch.nn import functional as F
 
def process_feat(imgs_desc, imgs_desc_dino):
    PCA_DIMS = [256, 256, 256]
    WEIGHT = [1,1,1,1,1]
    RAW = False
    CO_PCA = False 
    CO_PCA_DINO = False
    ONLY_DINO = False
    num_patches = 60

    if CO_PCA:
        imgs_desc = co_pca(imgs_desc, PCA_DIMS)
    
    if CO_PCA_DINO:
        print('CO_PCA_DINO')
        cat_desc_dino = torch.cat((img1_desc_dino, img2_desc_dino), dim=2).squeeze() # (1, 1, num_patches**2, dim)
        mean = torch.mean(cat_desc_dino, dim=0, keepdim=True)
        centered_features = cat_desc_dino - mean
        U, S, V = torch.pca_lowrank(centered_features, q=CO_PCA_DINO)
        reduced_features = torch.matmul(centered_features, V[:, :CO_PCA_DINO]) # (t_x+t_y)x(d)
        processed_co_features = reduced_features.unsqueeze(0).unsqueeze(0)
        img1_desc_dino = processed_co_features[:, :, :img1_desc_dino.shape[2], :]
        img2_desc_dino = processed_co_features[:, :, img1_desc_dino.shape[2]:, :]
    
    imgs_desc_dino = imgs_desc_dino.permute(0,1,3,2).reshape(-1, imgs_desc_dino.shape[-1], num_patches, num_patches)  
    
    imgs_desc['s5'] = F.interpolate(imgs_desc['s5'], size=(imgs_desc['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    imgs_desc['s5'] = torch.cat([imgs_desc['s4'], imgs_desc['s5']], dim=1)
    imgs_desc['s4'] = imgs_desc['s3']
    imgs_desc.pop('s3')
    imgs_desc = torch.cat([imgs_desc['s4'], F.interpolate(imgs_desc['s5'], size=(imgs_desc['s4'].shape[-2:]), mode='bilinear')], dim=1)
    
    imgs_desc = imgs_desc / imgs_desc.norm(dim=1, keepdim=True)
    imgs_desc_dino = imgs_desc_dino / imgs_desc_dino.norm(dim=1, keepdim=True)
    
    # return  imgs_desc_dino
    return torch.cat([imgs_desc,  imgs_desc_dino], 1)
        
def co_pca(features, dim=[128,128,128]):
    processed_features = {}
    B = features['s5'].shape[0]
    
    s5_size = features['s5'].shape[-1]
    s4_size = features['s4'].shape[-1]
    s3_size = features['s3'].shape[-1]
    # Get the feature tensors
    s5 = features['s5'].reshape(features['s5'].shape[0], features['s5'].shape[1], -1) #B*C*H*W -> B*C*HW 
    s4 = features['s4'].reshape(features['s4'].shape[0], features['s4'].shape[1], -1)
    s3 = features['s3'].reshape(features['s3'].shape[0], features['s3'].shape[1], -1)

    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensors in zip(['s5', 's4', 's3'], [s5, s4, s3]):
        target_dim = target_dims[name]

        # Concatenate the features        
        tensors = tensors.permute(0,2,1).contiguous().reshape(-1,tensors.shape[1]) #B*C*HW -> BHW*C
        origin_feature_nums = tensors.shape[0]
        if origin_feature_nums< target_dim:
            tensors = torch.cat([tensors,tensors.clone()],0)
        
        #print(tensors.shape)
        # equivalent to the above, pytorch implementation
        mean = torch.mean(tensors, dim=0, keepdim=True)
        centered_features = tensors - mean
        
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
        # print(reduced_features.shape)
        # raise NameError
        
        if origin_feature_nums< target_dim:
            reduced_features = reduced_features[:reduced_features.shape[0]//2,...]
        processed_features[name] = reduced_features.reshape(B,-1,target_dim).permute(0,2,1) # BHW*C -> B*HW*C -> B*C*HW

    # reshape the features
    # print(processed_features['s5'].shape)
    processed_features['s5']=processed_features['s5'].reshape(processed_features['s5'].shape[0], -1, s5_size, s5_size)
    processed_features['s4']=processed_features['s4'].reshape(processed_features['s4'].shape[0], -1, s4_size, s4_size)
    processed_features['s3']=processed_features['s3'].reshape(processed_features['s3'].shape[0], -1, s3_size, s3_size)
    
    # Upsample s5 spatially by a factor of 2
    processed_features['s5'] = F.interpolate(processed_features['s5'], size=(processed_features['s4'].shape[-2:]), mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features['s5'] = torch.cat([processed_features['s4'], processed_features['s5']], dim=1)

    # Set s3 as the new s4
    processed_features['s4'] = processed_features['s3']

    # Remove s3 from the features dictionary
    processed_features.pop('s3')

    # current order are layer 8, 5, 2
    features_gether_s4_s5 = torch.cat([processed_features['s4'], F.interpolate(processed_features['s5'], size=(processed_features['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features_gether_s4_s5
