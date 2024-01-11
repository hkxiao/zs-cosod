# **ICASSP204 : Zero-Shot Co-salient Object Detection Framework**

This repository is the official PyTorch implementation of our zero-shot cosod framework. [[**arXiv**](https://arxiv.org/abs/2309.05499)]

<div align=center><img width="550" height="230" src=assets/intro.png/></div>

## **Abstract**

Co-salient Object Detection (CoSOD) endeavors to replicate the human visual system’s capacity to recognize common and salient objects within a collection of images. Despite recent advancements in deep learning models, these models still rely on training with well-annotated CoSOD datasets. The exploration of training-free zero-shot CoSOD frameworks has been limited. In this paper, taking inspiration from the zero-shot transfer capabilities of foundational computer vision models, we introduce the first zero-shot CoSOD framework that harnesses these models without any training process. To achieve this, we introduce two novel components in our proposed framework: the group prompt generation (GPG) module and the co-saliency map generation (CMP) module. We evaluate the framework’s performance on widely-used datasets and observe impressive results. Our approach surpasses existing unsupervised methods and even outperforms fully supervised methods developed before 2020, while remaining competitive with some fully supervised methods developed before 2022.

## **Framework Overview**

<div align=center><img width="750" height="330" src=assets/framework.png/></div>

## **Results**

The predicted results of our model trained by COCO9k only is available at [google-drive](https://drive.google.com/file/d/1YWxLQhe26bvFXfXzXIFw19mx69ESs1Lq/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/19sDWXHk0D04IlNdeGhdKDw) (fetch code: 7lmh)
+ quantitative results
<div align=center><img width="800" height="280" src=./assets/quantitative.png/></div>

+ qualitative results
<div align=center><img width="800" height="500" src=./assets/qualitative.png/></div>

## **Usage**
1. **Environment**

    ```
    Python==3.8.5
    opencv-python==4.5.3.56
    torch==1.9.0
    ```

2. **Datasets preparation**

    Download all the train/test datasets from my [google-drive](https://drive.google.com/file/d/1xD9BfxFnBl6vw0X97GXqLd8yBVR1tc3S/view?usp=sharing) and [google-drive](https://drive.google.com/file/d/1LAPmlWhnND9tBO3n_RaW2_ZIY0Jy1BGJ/view?usp=sharing), or [BaiduYun](https://pan.baidu.com/s/1npN6__inOd6uwKwza2TdZQ) (fetch code: s5m4). The file directory structure is as follows:
    ```
    +-- CoRP
    |   +-- Dataset
    |       +-- COCO9213  (Training Dataset for co-saliency branch)
    |       +-- Jigsaw_DUTS (Training Dataset for co-saliency branch)   
    |       +-- DUTS-TR (Training Dataset for saliency head)   
    |       +-- COCOSAL (Training Dataset for saliency head)  
    |       +-- CoSal2015 (Testing Dataset)   
    |       +-- CoCA (Testing Dataset)  
    |       +-- CoSOD3k (Testing Dataset)   
    |   +-- ckpt (The root for saving your checkpoint)
    |   ... 
    ```
 3. **Test and evalutation**
 
       Download the ckeckpoints of our model from [google-drive](https://drive.google.com/file/d/1viHjcuH0Ski67_zkgsQxAEhKL0Yf8Av8/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1YkWOjFNbtPjZs0VhROpZTA) (fetch code: utef). Place the **ckpt** folder in the main directory. Here is a command example of testing our model (trained by COCO9k with vgg16 backbone).
    ```
    CUDA_VISIBLE_DEVICES=0 python test.py --backbone vgg16 --ckpt_path './ckpt/vgg16_COCO9k/checkpoint.pth' --pred_root './Predictions/pred_vgg_coco/pred' 
    ```
    
    Run the following command to evaluate your prediction results，the metrics include **max F-measure**, **S-measure**, and **MAE**.
    
    ```
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --pred_root './Predictions/pred_vgg_coco/pred'
    ```
    For more metrics, CoSOD evaluation toolbox [eval-co-sod](https://github.com/zzhanghub/eval-co-sod) is strongly recommended.
    
 4. **Train your own model**

    Our CoRP can be trained with various backbones and training datasets.  
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --backbone <vgg16 or resnet50> --cosal_set <COCO9k or DUTS>  --sal_set <COCO9k or DUTS> --ckpt_root <Path for saving your checkpoint>
    ```
 
 ## Citation
  ```
  @ARTICLE{10008072,
  author={Zhu, Ziyue and Zhang, Zhao and Lin, Zheng and Sun, Xing and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Co-Salient Object Detection with Co-Representation Purification}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPAMI.2023.3234586}}
  ```
 
 ## Contact
   
Feel free to leave issues here or send me e-mails (zhuziyue@mail.nankai.edu.cn).