import torch
import torch.backends.cudnn
import torch.utils.tensorboard

import os
import numpy as np
import json

import train_finetune
import valid

import model.get_model
import optim
from torch.optim.swa_utils import AveragedModel
import data.dataset
import utils.utils
import utils.option

import resource


def freeze_bn_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

args = utils.option.get_args_parser()
torch.backends.cudnn.benchmark = True

load_path = args.save_dir + '/' + args.data_name + '_' + args.model_name + '_' + args.optim_name + '-mixup' + '_' + str(
    args.mixup_weight) + '-crl' + '_' + str(args.crl_weight)
save_pth_path = os.path.join(args.save_dir,
                             f"{args.data_name}_{args.model_name}_{args.optim_name}-mixup_{args.mixup_weight}-crl_{args.crl_weight}-finetune_{args.reweighting_type}")

if not os.path.exists(load_path):
    os.makedirs(load_path)
    os.makedirs(save_pth_path)
writer = torch.utils.tensorboard.SummaryWriter(save_pth_path)
logger = utils.utils.get_logger(save_pth_path)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

train_loader, valid_loader, _, nb_cls = data.dataset.get_loader(args.data_name, args.train_dir, args.val_dir,
                                                                args.test_dir,
                                                                args.batch_size, args.imb_factor, args.model_name)

for r in range(args.nb_run):
    prefix = '{:d} / {:d} Running'.format(r + 1, args.nb_run)
    logger.info(100 * '#' + '\n' + prefix)

    ## define model, optimizer
    net = model.get_model.get_model(args.model_name, nb_cls, logger, args)
    if args.optim_name == 'fmfp' or args.optim_name == 'swa':
        net = AveragedModel(net)
    net.load_state_dict(torch.load(os.path.join(load_path, f'best_acc_net_{r + 1}.pth')))
    freeze_bn_layers(net)
    optimizer, cos_scheduler, swa_model, swa_scheduler = optim.get_optimizer_scheduler(args.model_name,
                                                                                       args.optim_name,
                                                                                       net,
                                                                                       args.fine_tune_lr,
                                                                                       args.momentum,
                                                                                       args.weight_decay,
                                                                                       max_epoch_cos=args.epochs,
                                                                                       swa_lr=args.swa_lr)

    # make logger
    correct_log, best_acc, best_auroc, best_aurc = train_finetune.Correctness_Log(len(train_loader.dataset)), 0, 0, 1e6
    confidence_scores = None

    # start Train
    for epoch in range(1, args.fine_tune_epochs + 2):
        if epoch > 1:
            # 在第1个epoch之后，加载confidence_scores
            confidence_scores_path = os.path.join(save_pth_path, 'confidence_scores.npy')
            if os.path.exists(confidence_scores_path):
                confidence_scores = np.load(confidence_scores_path)
                logger.info('re-weighting...')
        train_finetune.train(train_loader, net, optimizer, epoch, correct_log, logger, writer, args, confidence_scores)

        if args.optim_name in ['swa', 'fmfp']:
            if epoch > args.swa_epoch_start:
                swa_model.update_parameters(net)
                swa_scheduler.step()
            else:
                cos_scheduler.step()
        else:
            cos_scheduler.step()

        # validation
        if epoch > args.swa_epoch_start and args.optim_name in ['swa', 'fmfp']:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device='cuda')
            net_val = swa_model.cuda()
        else:
            net_val = net
        res = valid.validation(valid_loader, net_val)
        log = [key + ': {:.3f}'.format(res[key]) for key in res]
        msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
        logger.info(msg)

        for key in res:
            if r < 1:
                writer.add_scalar('./Val/' + key, res[key], epoch)

        if res['Acc.'] > best_acc:
            acc = res['Acc.']
            msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
            logger.info(msg)
            best_acc = acc
            torch.save(net_val.state_dict(), os.path.join(save_pth_path, f'best_acc_finetune_net_{r + 1}.pth'))




