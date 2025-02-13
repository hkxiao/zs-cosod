import torch
from torch.nn import functional as F
 
def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

class Cosal_Module():
    def __init__(self):
        pass

    def __call__(self, feats, SISMs=None):
        if SISMs == None:
            SISMs = torch.ones_like(feats).cuda()
        
        # print(SISMs.min(),SISMs.max())
        # raise NameError
        SISMs_thd = SISMs.clone()
        SISMs_thd[SISMs_thd<=0.5] = 0
        SISMs_thd[SISMs_thd>0.5] = 1
        
        N, C, H, W = feats.shape
        HW = H * W

        # Resize SISMs to the same size as the input feats.
        SISMs = resize(SISMs, [H, W])  # shape=[N, 1, H, W], SISMs are the saliency maps generated by saliency head.
        SISMs_thd = resize(SISMs_thd, [H, W])  
        
        # NFs: L2-normalized features.
        NFs = F.normalize(feats, dim=1)  # shape=[N, C, H, W]

        # Co_attention_maps are utilized to filter more background noise.
        def get_co_maps(co_proxy, NFs):
            correlation_maps = F.conv2d(NFs, weight=co_proxy)  # shape=[N, N, H, W]

            # Normalize correlation maps.
            correlation_maps = F.normalize(correlation_maps.reshape(N, N, HW), dim=2)  # shape=[N, N, HW]
            co_attention_maps = torch.sum(correlation_maps , dim=1)  # shape=[N, HW]

            # Max-min normalize co-attention maps.
            min_value = torch.min(co_attention_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(co_attention_maps, dim=1, keepdim=True)[0]
            co_attention_maps = (co_attention_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[N, HW]
            co_attention_maps = co_attention_maps.view(N, 1, H, W)  # shape=[N, 1, H, W]
            return co_attention_maps

        # Use co-representation to obtain co-saliency features.
        def get_CoFs(NFs, co_rep):
            SCFs = F.conv2d(NFs, weight=co_rep)
            return SCFs

        # Find the co-representation proxy.
        co_proxy = F.normalize((NFs * SISMs_thd).mean(dim=3).mean(dim=2), dim=1).view(N, C, 1, 1)  # shape=[N, C, 1, 1]

        # Reshape the co-representation proxy to compute correlations between all pixel embeddings and the proxy.
        r_co_proxy = F.normalize((NFs * SISMs_thd).mean(dim=3).mean(dim=2).mean(dim=0), dim=0)
        r_co_proxy = r_co_proxy.view(1, C)
        all_pixels = NFs.reshape(N, C, HW).permute(0, 2, 1).reshape(N*HW, C)
        correlation_index = torch.matmul(all_pixels, r_co_proxy.permute(1, 0))

        # Employ top-K pixel embeddings with high correlation as co-representation.
        ranged_index = torch.argsort(correlation_index, dim=0, descending=True).repeat(1, C)
        co_representation = torch.gather(all_pixels, dim=0, index=ranged_index)[:32, :].view(32, C, 1, 1)

        co_attention_maps = get_co_maps(co_proxy, NFs)  # shape=[N, 1, H, W]
        CoFs = get_CoFs(NFs, co_representation)  # shape=[N, K, H, W]
        # raise NameError
        
        ret = CoFs*co_attention_maps*SISMs_thd
        return ret
