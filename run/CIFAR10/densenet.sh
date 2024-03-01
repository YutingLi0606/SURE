########## run DenseNet
## Ours
## Remember change the crl-weight and mixup weight from your ablations!!
## Different datasets should have different best weights!

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
--save-dir ./CIFAR10_out/dense_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/dense_out \
Cifar10

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
--save-dir ./CIFAR10_out/dense_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0.5 \
--save-dir ./CIFAR10_out/dense_out \
Cifar10

## CRL
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 0.5 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./CIFAR10_out/dense_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name baseline \
--crl-weight 0.5 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/dense_out \
Cifar10

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
--save-dir ./CIFAR10_out/dense_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/dense_out \
Cifar10

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
--save-dir ./CIFAR10_out/dense_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/dense_out \
Cifar10

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
--save-dir ./CIFAR10_out/dense_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR10_out/dense_out \
Cifar10

## Ours
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 3 \
--model-name densenet \
--optim-name fmfp \
--crl-weight 0.5 \
--mixup-weight 0.5 \
--mixup-beta 10 \
--use-cosine \
--save-dir ./CIFAR10_out/dense_out \
Cifar10

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name densenet \
--optim-name fmfp \
--crl-weight 0.5 \
--mixup-weight 0.5 \
--use-cosine \
--save-dir ./CIFAR10_out/dense_out \
Cifar10