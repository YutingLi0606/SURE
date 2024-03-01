########## run VGG-16
# Confirm the best CRL and mixup weight from ablations
# Priority: ours -> fmfp
# If ours results better than fmfp：baseline -> swa -> sam

## Ours
## Remember change the crl-weight and mixup weight from your ablations!!
## Different datasets should have different best weights!

## Ours
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 1 \
--model-name vgg19bn \
--optim-name fmfp \
--crl-weight 0.2 \
--mixup-weight 1 \
--mixup-beta 10 \
--use-cosine \
--save-dir ./Animal10N_out/vgg19bn_out \
Animal10N

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model-name vgg19bn \
--optim-name baseline \
--crl-weight 0.2 \
--mixup-weight 1 \
--use-cosine \
--save-dir ./Animal10N_out/vgg19bn_out \
Animal10N

## Baseline
python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 1 \
--model-name vgg19bn \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--mixup-beta 10 \
--save-dir ./Animal10N_out/vgg19bn_out \
Animal10N

python3 test.py \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model-name vgg19bn \
--optim-name baseline \
--crl-weight 0 \
--mixup-weight 0 \
--save-dir ./Animal10N_out/vgg19bn_out \
Animal10N