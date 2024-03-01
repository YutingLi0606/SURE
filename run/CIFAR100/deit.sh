## Baseline
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
--save-dir ./CIFAR100_out/deit_out \
Cifar100

python3 test.py \
--batch-size 64 \
--gpu 5 \
--nb-run 3 \
--model-name deit \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/deit_out \
Cifar100

## CRL
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
--save-dir ./CIFAR100_out/deit_out \
Cifar100

python3 test.py \
--batch-size 64 \
--gpu 5 \
--nb-run 3 \
--model-name deit \
--optim-name baseline \
--crl-weight 0.2 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/deit_out \
Cifar100

## RegMixup
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
--save-dir ./CIFAR100_out/deit_out \
Cifar100

python3 test.py \
--batch-size 64 \
--gpu 5 \
--nb-run 3 \
--model-name deit \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0.2 \
--save-dir ./CIFAR100_out/deit_out \
Cifar100

## SAM
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
--save-dir ./CIFAR100_out/deit_out \
Cifar100

python3 test.py \
--batch-size 64 \
--gpu 5 \
--nb-run 3 \
--model-name deit \
--optim-name sam \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/deit_out \
Cifar100

# swa
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
--save-dir ./CIFAR100_out/deit_out \
Cifar100

python3 test.py \
--batch-size 64 \
--gpu 5 \
--nb-run 3 \
--model-name deit \
--optim-name swa \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/deit_out \
Cifar100

## FMFP
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
--save-dir ./CIFAR100_out/deit_out \
Cifar100

python3 test.py \
--batch-size 64 \
--gpu 5 \
--nb-run 3 \
--model-name deit \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./CIFAR100_out/deit_out \
Cifar100

## SURE
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
--save-dir ./CIFAR100_out/deit_out \
Cifar100

python3 test.py \
--batch-size 64 \
--gpu 5 \
--nb-run 3 \
--model-name deit \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 0.2 \
--save-dir ./CIFAR100_out/deit_out \
Cifar100

