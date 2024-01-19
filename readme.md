# **ICASSP204 : Zero-Shot Co-salient Object Detection Framework**

This repository is the official PyTorch implementation of our zero-shot cosod framework. [[**arXiv**](https://arxiv.org/abs/2309.05499)]

<div align=center><img width="550" height="190" src=assets/intro.PNG/></div>

## **Abstract**

Co-salient Object Detection (CoSOD) endeavors to replicate the human visual system’s capacity to recognize common and salient objects within a collection of images. Despite recent advancements in deep learning models, these models still rely on training with well-annotated CoSOD datasets. The exploration of training-free zero-shot CoSOD frameworks has been limited. In this paper, taking inspiration from the zero-shot transfer capabilities of foundational computer vision models, we introduce the first zero-shot CoSOD framework that harnesses these models without any training process. To achieve this, we introduce two novel components in our proposed framework: the group prompt generation (GPG) module and the co-saliency map generation (CMP) module. We evaluate the framework’s performance on widely-used datasets and observe impressive results. Our approach surpasses existing unsupervised methods and even outperforms fully supervised methods developed before 2020, while remaining competitive with some fully supervised methods developed before 2022.

## **Framework Overview**

<div align=center><img width="750" height="330" src=assets/framework.PNG/></div>

## **Results**

<!-- The predicted results of our model trained by COCO9k only is available at [google-drive](https://drive.google.com/file/d/1YWxLQhe26bvFXfXzXIFw19mx69ESs1Lq/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/19sDWXHk0D04IlNdeGhdKDw) (fetch code: 7lmh) -->
+ quantitative results
<div align=center><img width="800" height="220" src=./assets/quantitative.PNG/></div>

+ qualitative results
<div align=center><img width="800" height="400" src=./assets/qualitative.PNG/></div>

## **Usage**
1. **Environment**

    ```
   conda create -n zscosod python=3.9
   conda activate zscosod 
   pip install -e .
   pip install -r requirements.txt
    ```

2. **Datasets preparation**

    Download all the test datasets from my [google-drive](https://drive.google.com/file/d/1knhq7KYhaX-fLH7VYrfJhjoKiBnK3KAM/view?usp=drive_link) or [BaiduYun](https://pan.baidu.com/s/19NLkiRQz3BPrUrk7M1dfZw) (fetch code: qwt8). The file directory structure is as follows:
    ```
    +-- zs-cosod
    |   +-- data 
    |       +-- CoSal2015 (Testing Dataset)
    |           +-- img (Image Groups)  
    |           +-- gt (Ground Truth Groups)
    |           +-- blip2-caption (Image Caption)
    |       +-- CoCA (Testing Dataset)  
    |       +-- CoSOD3k (Testing Dataset)   
    |   ... 
    ```
 3. **Test and evalutation**
 
       Download the ckeckpoints of TSDN and SAM from [google-drive](https://drive.google.com/file/d/1YsvhQtqQyfjf-OMsA36uPefc2qAZnHxV/view?usp=drive_link) | [BaiduYun](https://pan.baidu.com/s/1mp8byGsBb3MpFdap-JEIig) (fetch code: be34). Place the **ckpt** folder in the main directory. Here is a command example of testing our model (test CoSal2015 with vit-base backbone).
    ```
    1. sh sd-dino/extract_feat.sh (Feature Extraction by StableDiffusion-1.5 and DINOv2-base)
    2. sh A2S-v2/inference_sod.sh (Saliency Map Generation by Unsupervised TSDN)
    3. sh inference_cosod.sh (CoSaliency Map Generation) 
    ```
    
    Run the following command to evaluate your prediction results，the metrics include **max F-measure**, **S-measure**, and **MAE**.
    
    ```
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --pred_root results --datasets CoSal2015
    ```
    For more metrics, CoSOD evaluation toolbox [eval-co-sod](https://github.com/zzhanghub/eval-co-sod) is strongly recommended.
    
 
 ## Citation
  ```
   @article{DBLP:journals/corr/abs-2309-05499,
   author       = {Haoke Xiao and
                     Lv Tang and
                     Bo Li and
                     Zhiming Luo and
                     Shaozi Li},
   title        = {Zero-Shot Co-salient Object Detection Framework},
   journal      = {CoRR},
   volume       = {abs/2309.05499},
   year         = {2023}
   }
  ```
 
## Acknowledgement

Our code is largely based on the following open-source projects: [ODISE](https://github.com/NVlabs/ODISE), [dino-vit-features (official implementation)](https://github.com/ShirAmir/dino-vit-features), [dino-vit-features (Kamal Gupta's implementation)](https://github.com/kampta/dino-vit-features), [SAM](https://github.com/facebookresearch/segment-anything), and [TSDN](https://github.com/moothes/A2S-v2). Our heartfelt gratitude goes to the developers of these resources!

 ## Contact
   
Feel free to leave issues here or send me e-mails (hk.xiao.me@gmail.com).