import torch.backends.cudnn
import torch.utils.tensorboard

import os
import json 

import train
import valid
from torch.optim.swa_utils import AveragedModel
import model.get_model
import optim

import data.dataset
import utils.utils
import utils.option



import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


args = utils.option.get_args_parser()
torch.backends.cudnn.benchmark = True

save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.model_name}_{args.optim_name}-mixup_{args.mixup_weight}-crl_{args.crl_weight}")
if not os.path.exists(save_path):
    os.makedirs(save_path)

writer = torch.utils.tensorboard.SummaryWriter(save_path)
logger = utils.utils.get_logger(save_path)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

train_loader, valid_loader, _, nb_cls = data.dataset.get_loader(args.data_name, args.train_dir, args.val_dir,args.test_dir,
                                                                      args.batch_size, args.imb_factor)

for r in range(args.nb_run):
    prefix = '{:d} / {:d} Running'.format(r + 1, args.nb_run)
    logger.info(100*'#' + '\n' + prefix)
    
    ## define model, optimizer 
    net = model.get_model.get_model(args.model_name, nb_cls, logger, args)
    print(net)
    if args.resume:
        if args.optim_name == 'fmfp' or args.optim_name == 'swa':
            net = AveragedModel(net)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}.pth')))
        logger.info(f"Loading checkpoints from {save_path}")
    optimizer, cos_scheduler, swa_model, swa_scheduler = optim.get_optimizer_scheduler(args.model_name,
                                                                                        args.optim_name,
                                                                                        net,
                                                                                        args.lr,
                                                                                        args.momentum,
                                                                                        args.weight_decay,
                                                                                        max_epoch_cos = args.epochs,
                                                                                        swa_lr = args.swa_lr)


    # make logger
    correct_log, best_acc, best_auroc, best_aurc = train.Correctness_Log(len(train_loader.dataset)), 0, 0, 1e6

    # start Train
    for epoch in range(1, args.epochs + 2):
        train.train(train_loader, net, optimizer, epoch, correct_log, logger, writer, args)

        if args.optim_name in ['swa', 'fmfp'] :
            if epoch > args.swa_epoch_start:
                print('111111111111')
                swa_model.update_parameters(net)
                swa_scheduler.step()
            else:
                cos_scheduler.step()
        else:
            cos_scheduler.step()

        # validation
        if epoch > args.swa_epoch_start and args.optim_name in ['swa', 'fmfp'] :
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device='cuda')
            net_val = swa_model.cuda()
        else : 
            net_val = net
        res = valid.validation(valid_loader, net_val)
        log = [key + ': {:.3f}'.format(res[key]) for key in res]
        msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
        logger.info(msg)

        for key in res :
            if r < 1:
                writer.add_scalar('./Val/' + key, res[key], epoch)

        if res['Acc.'] > best_acc :
            acc = res['Acc.']
            msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
            logger.info(msg)
            best_acc = acc
            torch.save(net_val.state_dict(),os.path.join(save_path, f'best_acc_net_{r+1}.pth'))
        
        # if res['AUROC'] > best_auroc :
        #     auroc = res['AUROC']
        #     msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
        #     logger.info(msg)
        #     best_auroc = auroc
        #     torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{r+1}.pth'))
        #
        # if res['AURC'] < best_aurc :
        #     aurc = res['AURC']
        #     msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
        #     logger.info(msg)
        #     best_aurc = aurc
        #     torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{r+1}.pth'))
    



