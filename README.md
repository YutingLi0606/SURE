# (CVPR 2024) SURE
Pytorch implementation of paper "SURE: SUrvey REcipes for building reliable and robust deep networks"

[[Project Page]](https://yutingli0606.github.io/SURE/)
[[arXiv]](https://arxiv.org/pdf/2403.00543.pdf) 
[[Google Drive]](https://drive.google.com/drive/folders/1xT-cX22_I8h5yAYT1WNJmhSLrQFZZ5t1?usp=sharing)

<p align="center">
<img src="img/Teaser.png" width="1000px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing :
```
@inproceedings{li2024sure,
 title={SURE: SUrvey REcipes for building reliable and robust deep networks},
 author={Li, Yuting and Chen, Yingyi and Yu, Xuanlong and Chen, Dexiong and Shen, Xi},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
 year={2024}}
```

## Table of Content
* [1. Overview of recipes](#1-overview-of-recipes)
* [2. Visual Results](#2-visual-results)
* [3. Installation](#3-installation)
* [4. Quick Start](#4-quick-start)
* [5. Acknowledgement](#5-acknowledgement)

## 1. Overview of recipes
<p align="center">
<img src="img/recipes.png" width="1000px" alt="method">
</p>

## 2. Visual Results
<p align="center">
<img src="img/confidence.png" width="1000px" alt="method">
</p>

## 3. Installation

### 3.1. Environment


Our model can be learnt in a **single GPU RTX-4090 24G**

```bash
conda env create -f environment.yml
conda activate u
```

The code was tested on Python 3.9 and PyTorch 1.13.0.


### 3.2. Datasets
#### 3.2.1 CIFAR and Tiny-ImageNet
* Using **CIFAR10, CIFAR100 and Tiny-ImageNet** for failure prediction(also known as misclassification detection).
* We keep **10%** of training samples as a validation dataset for failure prediction. 
* Download datasets to ./data/ and split into train/val/test.
Take CIFAR10 for an example:
```
cd data
bash download_cifar.sh
```
The structure of the file should be:
```
./data/CIFAR10/
├── train
├── val
└── test
```
* We have already split Tiny-imagenet, you can download it from [here.](https://drive.google.com/drive/folders/1xT-cX22_I8h5yAYT1WNJmhSLrQFZZ5t1?usp=sharing)
#### 3.2.2 Animal-10N and Food-101N
* Using **Animal-10N and Food-101N** for learning with noisy label.
* To download Animal-10N dataset [[Song et al., 2019]](https://proceedings.mlr.press/v97/song19b/song19b.pdf), please refer to [here](https://dm.kaist.ac.kr/datasets/animal-10n/). The structure of the file should be:
```
./data/Animal10N/
├── train
└── test
```
* To download Food-101N dataset [[Lee et al., 2018]](https://arxiv.org/pdf/1711.07131.pdf), please refer to [here](https://kuanghuei.github.io/Food-101N/). The structure of the file should be:
```
./data/Food-101N/
├── train
└── test
```
#### 3.2.3 CIFAR-LT
* Using **CIFAR-LT** with imbalance factor(10, 50, 100) for long-tailed classification.
* Rename the original CIFAR10 and CIFAR100 (do not split into validation set) to 'CIFAR10_LT' and 'CIFAR100_LT' respectively.
* The structure of the file should be:
```
./data/CIFAR10_LT/
├── train
└── test
```
#### 3.2.4 CIFAR10-C
* Using **CIFAR10-C** to test robustness under data corrputions.
* To download CIFAR10-C dataset [[Hendrycks et al., 2019]](https://arxiv.org/pdf/1903.12261.pdf), please refer to [here](https://github.com/hendrycks/robustness?tab=readme-ov-file). The structure of the file should be:
```
./data/CIFAR-10-C/
├── brightness.npy
├── contrast.npy
├── defocus_blur.npy
...
```

## 4. Quick Start
* Our model checkpoints are saved [here.](https://drive.google.com/drive/folders/1xT-cX22_I8h5yAYT1WNJmhSLrQFZZ5t1?usp=sharing)
* All results are saved in test_results.csv.
### 4.1 Failure Prediction
* We provide convenient and comprehensive commands in ./run/ to train and test different backbones across different datasets to help researchers reproducing the results of the paper.

<details>
<summary>
Take a example in run/CIFAR10/wideresnet.sh:

</summary>
  <details>
   <summary>
    MSP
   </summary>
    
      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name baseline \
      --crl-weight 0 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name baseline \
      --crl-weight 0 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10
  </details>

  <details>
   <summary>
    RegMixup
   </summary>
    

      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name baseline \
      --crl-weight 0 \
      --mixup-weight 0.5 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name baseline \
      --crl-weight 0 \
      --mixup-weight 0.5 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10

  </details>
  <details>
   <summary>
    CRL
   </summary>
    
      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name baseline \
      --crl-weight 0.5 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name baseline \
      --crl-weight 0.5 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10

  </details>
  <details>
   <summary>
    SAM
   </summary>
    

      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name sam \
      --crl-weight 0 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name sam \
      --crl-weight 0 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10

  </details>
  <details>
   <summary>
    SWA
   </summary>
    

      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name swa \
      --crl-weight 0 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name swa \
      --crl-weight 0 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10

  </details>
  <details>
   <summary>
    FMFP
   </summary>
    

      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10

  </details>
  <details>
   <summary>
    SURE
   </summary>
    

      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name fmfp \
      --crl-weight 0.5 \
      --mixup-weight 0.5 \
      --mixup-beta 10 \
      --use-cosine \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name wrn \
      --optim-name fmfp \
      --crl-weight 0.5 \
      --mixup-weight 0.5 \
      --use-cosine \
      --save-dir ./CIFAR10_out/wrn_out \
      Cifar10

  </details>
</details>

<details>
<summary>
Take a example in run/CIFAR10/deit.sh:

</summary>
  <details>
   <summary>
    MSP
   </summary>
    
      python3 main.py \
      --batch-size 64 \
      --gpu 5 \
      --epochs 50 \
      --lr 0.01 \
      --weight-decay 5e-5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name baseline \
      --crl-weight 0 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10
      
      python3 test.py \
      --batch-size 64 \
      --gpu 5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name baseline \
      --crl-weight 0 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10
  </details>

  <details>
   <summary>
    RegMixup
   </summary>
    

      python3 main.py \
      --batch-size 64 \
      --gpu 5 \
      --epochs 50 \
      --lr 0.01 \
      --weight-decay 5e-5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name baseline \
      --crl-weight 0 \
      --mixup-weight 0.2 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10
      
      python3 test.py \
      --batch-size 64 \
      --gpu 5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name baseline \
      --crl-weight 0 \
      --mixup-weight 0.2 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10

  </details>
  <details>
   <summary>
    CRL
   </summary>
    


     python3 main.py \
     --batch-size 64 \
     --gpu 5 \
     --epochs 50 \
     --lr 0.01 \
     --weight-decay 5e-5 \
     --nb-run 3 \
     --model-name deit \
     --optim-name baseline \
     --crl-weight 0.2 \
     --mixup-weight 0 \
     --mixup-beta 10 \
     --save-dir ./CIFAR10_out/deit_out \
     Cifar10
     
     python3 test.py \
     --batch-size 64 \
     --gpu 5 \
     --nb-run 3 \
     --model-name deit \
     --optim-name baseline \
     --crl-weight 0.2 \
     --mixup-weight 0 \
     --save-dir ./CIFAR10_out/deit_out \
     Cifar10

  </details>
  <details>
   <summary>
    SAM
   </summary>
    

      python3 main.py \
      --batch-size 64 \
      --gpu 5 \
      --epochs 50 \
      --lr 0.01 \
      --weight-decay 5e-5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name sam \
      --crl-weight 0 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10
      
      python3 test.py \
      --batch-size 64 \
      --gpu 5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name sam \
      --crl-weight 0 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10

  </details>
  <details>
   <summary>
    SWA
   </summary>
    

      python3 main.py \
      --batch-size 64 \
      --gpu 5 \
      --epochs 50 \
      --lr 0.01 \
      --weight-decay 5e-5 \
      --swa-epoch-start 0 \
      --swa-lr 0.004 \
      --nb-run 3 \
      --model-name deit \
      --optim-name swa \
      --crl-weight 0 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10
      
      python3 test.py \
      --batch-size 64 \
      --gpu 5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name swa \
      --crl-weight 0 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10

  </details>
  <details>
   <summary>
    FMFP
   </summary>
    

      python3 main.py \
      --batch-size 64 \
      --gpu 5 \
      --epochs 50 \
      --lr 0.01 \
      --weight-decay 5e-5 \
      --swa-epoch-start 0 \
      --swa-lr 0.004 \
      --nb-run 3 \
      --model-name deit \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 0 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10
      
      python3 test.py \
      --batch-size 64 \
      --gpu 5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 0 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10


  </details>
  <details>
   <summary>
    SURE
   </summary>
    

      python3 main.py \
      --batch-size 64 \
      --gpu 5 \
      --epochs 50 \
      --lr 0.01 \
      --weight-decay 5e-5 \
      --swa-epoch-start 0 \
      --swa-lr 0.004 \
      --nb-run 3 \
      --model-name deit \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 0.2 \
      --mixup-beta 10 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10
      
      python3 test.py \
      --batch-size 64 \
      --gpu 5 \
      --nb-run 3 \
      --model-name deit \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 0.2 \
      --save-dir ./CIFAR10_out/deit_out \
      Cifar10
  </details>
</details>


<details>
<summary>
The results of failure prediction.
</summary>
<p align="center">
<img src="img/main_results.jpeg" width="1000px" alt="method">
</p>
</details>


### 4.2 Long-tailed classification
* We provide convenient and comprehensive commands in ./run/CIFAR10_LT and ./run/CIFAR100_LT to train and test our method under long-tailed distribution.

<details>
<summary>
Take a example in run/CIFAR10_LT/resnet32.sh:

</summary>
  <details>
   <summary>
    Imbalance factor=10
   </summary>
    
      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name resnet32 \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 1 \
      --mixup-beta 10 \
      --use-cosine \
      --save-dir ./CIFAR10_LT/res32_out \
      Cifar10_LT
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name resnet32 \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 1 \
      --use-cosine \
      --save-dir ./CIFAR10_LT/res32_out \
      Cifar10_LT
  </details>

  <details>
   <summary>
    Imbalance factor = 50
   </summary>
    
      python3 main.py \
      --batch-size 128 \
      --gpu 0 \
      --epochs 200 \
      --nb-run 3 \
      --model-name resnet32 \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 1 \
      --mixup-beta 10 \
      --use-cosine \
      --save-dir ./CIFAR10_LT_50/res32_out \
      Cifar10_LT_50
      
      python3 test.py \
      --batch-size 128 \
      --gpu 0 \
      --nb-run 3 \
      --model-name resnet32 \
      --optim-name fmfp \
      --crl-weight 0 \
      --mixup-weight 1 \
      --use-cosine \
      --save-dir ./CIFAR10_LT_50/res32_out \
      Cifar10_LT_50
      
  </details>
  
  <details>
   <summary>
    Imbalance factor = 100
   </summary>
   
    python3 main.py \
    --batch-size 128 \
    --gpu 0 \
    --epochs 200 \
    --nb-run 3 \
    --model-name resnet32 \
    --optim-name fmfp \
    --crl-weight 0 \
    --mixup-weight 1 \
    --mixup-beta 10 \
    --use-cosine \
    --save-dir ./CIFAR10_LT_100/res32_out \
    Cifar10_LT_100
    
    python3 test.py \
    --batch-size 128 \
    --gpu 0 \
    --nb-run 3 \
    --model-name resnet32 \
    --optim-name fmfp \
    --crl-weight 0 \
    --mixup-weight 1 \
    --use-cosine \
    --save-dir ./CIFAR10_LT_100/res32_out \
    Cifar10_LT_100
  </details>
</details>

You can conduct second stage uncertainty-aware re-weighting by :
```
python3 finetune.py \
--batch-size 128 \
--gpu 5 \
--nb-run 1 \
--model-name resnet32 \
--optim-name fmfp \
--fine-tune-lr 0.005 \
--reweighting-type exp \
--t 1 \
--crl-weight 0 \
--mixup-weight 1 \
--mixup-beta 10 \
--fine-tune-epochs 50 \
--use-cosine \
--save-dir ./CIFAR100LT_100_out/51.60 \
Cifar100_LT_100
```

<details>
<summary>
The results of long-tailed classification.
</summary>
<p align="center">
<img src="img/long-tail.jpeg" width="600px" alt="method">
</p>
</details>

### 4.3 Learning with noisy labels
* We provide convenient and comprehensive commands in ./run/animal10N and ./run/Food101N to train and test our method with noisy labels.

<details>
   
   <summary>
    Animal-10N
   </summary>  
   
     python3 main.py \
     --batch-size 128 \
     --gpu 0 \
     --epochs 200 \
     --nb-run 1 \
     --model-name vgg19bn \
     --optim-name fmfp \
     --crl-weight 0.2 \
     --mixup-weight 1 \
     --mixup-beta 10 \
     --use-cosine \
     --save-dir ./Animal10N_out/vgg19bn_out \
     Animal10N
     
     python3 test.py \
     --batch-size 128 \
     --gpu 0 \
     --nb-run 1 \
     --model-name vgg19bn \
     --optim-name baseline \
     --crl-weight 0.2 \
     --mixup-weight 1 \
     --use-cosine \
     --save-dir ./Animal10N_out/vgg19bn_out \
     Animal10N

  </details>
  <details>
   <summary>
    Food-101N
   </summary>
    

     python3 main.py \
     --batch-size 64 \
     --gpu 0 \
     --epochs 30 \
     --nb-run 1 \
     --model-name resnet50 \
     --optim-name fmfp \
     --crl-weight 0.2 \
     --mixup-weight 1 \
     --mixup-beta 10 \
     --lr 0.01 \
     --swa-lr 0.005 \
     --swa-epoch-start 22 \
     --use-cosine True \
     --save-dir ./Food101N_out/resnet50_out \
     Food101N
     
     python3 test.py \
     --batch-size 64 \
     --gpu 0 \
     --nb-run 1 \
     --model-name resnet50 \
     --optim-name fmfp \
     --crl-weight 0.2 \
     --mixup-weight 1 \
     --use-cosine True \
     --save-dir ./Food101N_out/resnet50_out \
     Food101N

  </details>

  
<details>
<summary>
The results of learning with noisy labels.
</summary>
<p align="center">
<img src="img/label-noise.jpeg" width="600px" alt="method">
</p>
</details> 


### 4.4 Robustness under data corruption
* You can test on CIFAR10-C by the following code in test.py:
```
if args.data_name == 'cifar10':
    cor_results_storage = test_cifar10c_corruptions(net, args.corruption_dir, transform_test,
                                                    args.batch_size, metrics, logger)
    cor_results = {corruption: {
                   severity: {
                   metric: cor_results_storage[corruption][severity][metric][0] for metric in metrics} for severity
                   in range(1, 6)} for corruption in data.CIFAR10C.CIFAR10C.cifarc_subsets}
    cor_results_all_models[f"model_{r + 1}"] = cor_results
``` 
* The results are saved in cifar10c_results.csv.
* Testing on CIFAR10-C takes a while. If you don't need the results, just comment out this code.

<details>
<summary>
The results of failure prediction under distribution shift.
</summary>
<p align="center">
<img src="img/data-corruption.jpeg" width="1000px" alt="method">
</p>
</details>

## 5. Acknowledgement

We appreciate helps from public code like [FMFP](https://github.com/Impression2805/FMFP) and [OpenMix](https://github.com/Impression2805/OpenMix).  



