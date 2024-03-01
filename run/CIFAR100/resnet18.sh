########## run ResNet-18
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
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./CIFAR100_out/res_out \
Cifar100

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/res_out \
Cifar100

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
--save-dir ./CIFAR100_out/res_out \
Cifar100

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 1 \
--save-dir ./CIFAR100_out/res_out \
Cifar100

## CRL
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 1 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./CIFAR100_out/res_out \
Cifar100

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 1 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/res_out \
Cifar100

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
--save-dir ./CIFAR100_out/res_out \
Cifar100

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/res_out \
Cifar100

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
--save-dir ./CIFAR100_out/res_out \
Cifar100

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/res_out \
Cifar100

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
--save-dir ./CIFAR100_out/res_out \
Cifar100

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/res_out \
Cifar100

## Ours
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 1 \
--mixup-weight 1 \
--mixup-beta 10 \
--use-cosine \
--save-dir ./CIFAR100_out/res_out \
Cifar100

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 1 \
--mixup-weight 1 \
--use-cosine \
--save-dir ./CIFAR100_out/res_out \
Cifar100