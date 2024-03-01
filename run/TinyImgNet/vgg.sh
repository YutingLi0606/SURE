########## run VGG-16
# Confirm the best CRL and mixup weight from ablations
# Priority: ours -> fmfp
# If ours results better than fmfp：baseline -> swa -> sam

## Ours
## Remember change the crl-weight and mixup weight from your ablations!!
## Different datasets should have different best weights!

## Baseline
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name vgg \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name vgg \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

## SAM
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name vgg \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name vgg \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

## swa
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name vgg \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name vgg \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

## FMFP
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name vgg \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name vgg \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

## Ours
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name vgg \
--optim-name fmfp \
--crl-weight 0.2 \
--mixup-weight 0.2 \
--mixup-beta 10 \
--use-cosine \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name vgg \
--optim-name fmfp \
--crl-weight 0.2 \
--mixup-weight 0.2 \
--use-cosine \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

# RegMixup
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name vgg \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0.2 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name vgg \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0.2 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

## CRL
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name vgg \
--optim-name baseline \
--crl-weight 0.2 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name vgg \
--optim-name baseline \
--crl-weight 0.2 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/vgg_out \
TinyImgNet