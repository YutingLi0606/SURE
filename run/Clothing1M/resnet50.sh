python3 main.py \
--batch-size 64 \
--gpu 0 \
--epochs 20 \
--nb-run 1 \
--model-name resnet50 \
--optim-name fmfp \
--crl-weight 0.2 \
--mixup-weight 1 \
--mixup-beta 10 \
--lr 0.01 \
--swa-lr 0.005 \
--swa-epoch-start 12 \
--use-cosine True \
--resume-path /user/leuven/334/vsc33476/data_75G/cyy/uncertainty/Uncertainty-main/resnet50ckpt/resnet50-19c8e357.pth \
--save-dir ./Clothing1M_out/resnet50_out \
Clothing1M

python3 test.py \
--batch-size 64 \
--gpu 0 \
--nb-run 1 \
--model-name resnet50 \
--optim-name fmfp \
--crl-weight 0.2 \
--mixup-weight 1 \
--use-cosine True \
--save-dir ./Clothing1M_out/resnet50_out \
Clothing1M