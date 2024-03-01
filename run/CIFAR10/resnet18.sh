########## run ResNet-18

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
--save-dir ./CIFAR10_out/res_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/res_out \
Cifar10

# RegMixup
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0.5 \
--mixup-beta 10 \
--save-dir ./CIFAR10_out/res_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0.5 \
--save-dir ./CIFAR10_out/res_out \
Cifar10

## CRL
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0.5 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./CIFAR10_out/res_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name baseline \
--crl-weight 0.5 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/res_out \
Cifar10

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
--save-dir ./CIFAR10_out/res_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/res_out \
Cifar10

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
--save-dir ./CIFAR10_out/res_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/res_out \
Cifar10

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
--save-dir ./CIFAR10_out/res_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/res_out \
Cifar10

## Ours
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 0.5 \
--mixup-weight 0.5 \
--mixup-beta 10 \
--use-cosine \
--save-dir ./CIFAR10_out/res_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet18 \
--optim-name fmfp \
--crl-weight 0.5 \
--mixup-weight 0.5 \
--use-cosine \
--save-dir ./CIFAR10_out/res_out \
Cifar10