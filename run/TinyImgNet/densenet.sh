########## run DenseNet on Tiny-ImageNet ###########

## Baseline
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

## SAM
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

## swa
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

## FMFP
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

# RegMixup
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0.5 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0.5 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

## CRL
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 2 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 2 \
--mixup-weight 0 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

## Ours
python3 main.py \
--batch-size 128 \
--gpu 4 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0.5 \
--mixup-beta 10 \
--use-cosine \
--cos-temp 16 \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet

python3 test.py \
--batch-size 128 \
--gpu 4 \
--nb-run 3 \
--model-name densenet \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0.5 \
--cos-temp 16 \
--use-cosine \
--save-dir ./TinyImgNet_out/dense_out \
TinyImgNet