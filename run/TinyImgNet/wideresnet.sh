########## run WideResNet
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
--model-name wrn \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name wrn \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

## SAM
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
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name wrn \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

## swa
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
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name wrn \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

## FMFP
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
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name wrn \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

## Ours
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name wrn \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0.5 \
--mixup-beta 10 \
--use-cosine \
--cos-temp 16 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name wrn \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0.5 \
--use-cosine \
--cos-temp 16 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

# RegMixup
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name wrn \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 2 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name wrn \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 2 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

## CRL
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name wrn \
--optim-name baseline \
--crl-weight 2 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name wrn \
--optim-name baseline \
--crl-weight 2 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/wrn_out \
TinyImgNet
