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
--save-dir ./CIFAR100_LT_out/res32_out \
Cifar100_LT

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet32 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 1 \
--use-cosine \
--save-dir ./CIFAR100_LT_out/res32_out \
Cifar100_LT

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
--save-dir ./CIFAR100_LT_50_out/res32_out \
Cifar100_LT_50

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet32 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 1 \
--use-cosine \
--save-dir ./CIFAR100_LT_50_out/res32_out \
Cifar100_LT_50

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
--save-dir ./CIFAR100_LT_100_out/res32_out \
Cifar100_LT_100

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 3 \
--model-name resnet32 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 1 \
--use-cosine \
--save-dir ./CIFAR100_LT_100_out/res32_out \
Cifar100_LT_100