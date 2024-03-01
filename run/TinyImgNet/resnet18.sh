########## run ResNet18

## Ours
## Remember change the crl-weight and mixup weight from your ablations!!
## Different datasets should have different best weights!

## Baseline
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

## SAM
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

## swa
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

## FMFP
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

## Ours
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 2 \
--mixup-weight 1 \
--mixup-beta 10 \
--use-cosine \
--cos-temp 16 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 2 \
--mixup-weight 1 \
--use-cosine \
--cos-temp 16 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

# RegMixup
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 1 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 1 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

## CRL
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 2 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 2 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/res_out \
TinyImgNet
