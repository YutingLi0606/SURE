python3 main.py \
--batch-size 32 \
--gpu 8 \
--epochs 200 \
--train-size 224 \
--lr 0.01 \
--weight-decay 5e-5 \
--deit-path ./data/deit_base_patch16_224-b5f2ef4d.pth \
--swa-epoch-start 50 \
--swa-lr 0.004 \
--nb-run 1 \
--model-name deit \
--optim-name fmfp \
--crl-weight 0 \
--use-cosine \
--mixup-weight 0.5 \
--mixup-beta 10 \
--save-dir ./CARS/deit_cos_deitps16_swa_start50_lr001 \
CARS

## this will give you top-1 acc. 93.0%