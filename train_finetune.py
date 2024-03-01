import utils.utils
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os


def compute_confidence_scores(net, train_loader):
    net.eval()  # Set the model to evaluation mode
    confidence_scores = np.zeros(len(train_loader.dataset))
    with torch.no_grad():
        for i, (image, target, image_idx) in enumerate(train_loader):
            image = image.cuda()
            output = net(image)
            softmax_scores = F.softmax(output, dim=1)
            max_scores, _ = softmax_scores.max(dim=1)
            confidence_scores[image_idx] = max_scores.cpu().numpy()

    return np.array(confidence_scores)


class Mixup_Criterion(nn.Module):
    def __init__(self, beta, cls_criterion):
        super().__init__()
        self.beta = beta
        self.cls_criterion = cls_criterion

    def get_mixup_data(self, image, target):
        beta = np.random.beta(self.beta, self.beta)
        index = torch.randperm(image.size()[0]).to(image.device)
        shuffled_image, shuffled_target = image[index], target[index]
        mixed_image = beta * image + (1 - beta) * shuffled_image
        return mixed_image, shuffled_target, beta

    def forward(self, image, target, net):
        mixed_image, shuffled_target, beta = self.get_mixup_data(image, target)
        pred_mixed = net(mixed_image)
        loss_mixup = beta * self.cls_criterion(pred_mixed, target) + (1 - beta) * self.cls_criterion(pred_mixed,
                                                                                                     shuffled_target)
        return loss_mixup


class Correctness_Log(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.max_correctness = 1

    # correctness update
    def update(self, data_idx, correctness):
        self.correctness[data_idx] += correctness.cpu().numpy()

    def resize(self, new_size):
        self.correctness = np.resize(self.correctness, new_size)

    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def _normalize(self, data):
        data_min = self.correctness.min()
        data_max = float(self.max_correctness)

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, idx1, idx2):
        idx1 = idx1.cpu().numpy()
        idx2 = idx2.cpu().numpy()
        correctness_norm = self._normalize(self.correctness)

        target1, target2 = correctness_norm[idx1], correctness_norm[idx2]

        # 1 for idx1 > idx2, 0 for idx1 = idx2, -1 for idx1 < idx2
        target = np.array(target1 > target2, dtype='float') + np.array(target1 < target2, dtype='float') * (-1)
        target = torch.from_numpy(target).float().cuda()

        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().cuda()

        return target, margin


class CRL_Criterion(nn.Module):
    '''
    Confidence-Aware Learning for Deep Neural Networks
    ICML 2020
    http://proceedings.mlr.press/v119/moon20a/moon20a.pdf
    code borrows from: https://github.com/daintlab/confidence-aware-learning/blob/master/crl_utils.py
    '''

    def __init__(self):
        super().__init__()
        self.rank_criterion = torch.nn.MarginRankingLoss(margin=0)

    def forward(self, output, image_idx, correct_log):
        conf, _ = F.softmax(output, dim=1).max(dim=1)
        conf_roll, image_idx_roll = torch.roll(conf, -1), torch.roll(image_idx, -1)

        # ranking target:
        # 1 for image_idx > image_idx_roll
        # 0 for image_idx = image_idx_roll
        # -1 for image_idx < image_idx_roll
        rank_target, rank_margin = correct_log.get_target_margin(image_idx, image_idx_roll)
        conf_roll = conf_roll + rank_margin / (rank_target + 1e-7)
        ranking_loss = self.rank_criterion(conf, conf_roll, rank_target)
        return ranking_loss


def compute_loss(args, net, image, target, image_idx, correct_log, cls_criterion, mixup_criterion, rank_criterion,
                 confidence_scores=None):
    output = net(image)
    loss_ce = cls_criterion(output, target)
    if confidence_scores is not None:
        batch_confidence_scores = torch.tensor(confidence_scores[image_idx], device=image.device)

        if args.reweighting_type == 'exp':
            if args.data_name == "Clothing1M":
                weights = torch.exp(args.t * batch_confidence_scores)
            else:
                weights = torch.exp(-args.t * batch_confidence_scores)
        elif args.reweighting_type == 'threshold':
            weights = (1.0 - batch_confidence_scores) * (batch_confidence_scores < args.alpha).float()
        elif args.reweighting_type == 'power':
            weights = (1.0 - batch_confidence_scores) ** args.p
        elif args.reweighting_type == 'linear':
            weights = 1.0 - batch_confidence_scores
        else:
            raise ValueError("Invalid reweighting_type. Choose 'exp', 'threshold', 'power' or 'linear'")

        weights_sum = weights.sum()
        if weights_sum > 0:
            weights = weights / weights_sum * len(weights)
            # weights = weights / weights_sum

        weights = torch.clamp(weights, min=0.0001, max=1.0)
        loss_ce = (loss_ce * weights).mean()

    loss_mixup = mixup_criterion(image, target, net)
    loss_crl = rank_criterion(output, image_idx, correct_log)
    loss = loss_ce + 0 * loss_mixup + args.crl_weight * loss_crl
    return loss, loss_ce, loss_mixup, loss_crl, output


def train(train_loader, net, optimizer, epoch, correct_log, logger, writer, args, confidence_scores=None):

    ## define criterion
    cls_criterion = torch.nn.CrossEntropyLoss()
    cls_criterion_confidence = torch.nn.CrossEntropyLoss(reduction='none')
    mixup_criterion = Mixup_Criterion(beta=args.mixup_beta, cls_criterion=cls_criterion)
    rank_criterion = CRL_Criterion()

    train_log = {
        'Top1 Acc.': utils.utils.AverageMeter(),
        'CLS Loss': utils.utils.AverageMeter(),
        'Mixup Loss': utils.utils.AverageMeter(),
        'CRL Loss': utils.utils.AverageMeter(),
        'Tot. Loss': utils.utils.AverageMeter(),
        'LR': utils.utils.AverageMeter(),

    }
    if epoch == 1:
        confidence_scores = compute_confidence_scores(net, train_loader)

        save_confidence_path = os.path.join(args.save_dir, f"{args.data_name}_{args.model_name}_{args.optim_name}-mixup_{args.mixup_weight}-crl_{args.crl_weight}-finetune_{args.reweighting_type}")
        np.save(os.path.join(save_confidence_path, 'confidence_scores.npy'), confidence_scores)

    net.train()
    msg = '####### --- Training Epoch {:d} --- #######'.format(epoch)
    logger.info(msg)
    for i, (image, target, image_idx) in enumerate(train_loader):
        image, target = image.cuda(), target.long().cuda()
        loss, loss_ce, loss_mixup, loss_crl, output = compute_loss(args,
                                                                   net,
                                                                   image,
                                                                   target,
                                                                   image_idx,
                                                                   correct_log,
                                                                   cls_criterion_confidence,
                                                                   mixup_criterion,
                                                                   rank_criterion,
                                                                   confidence_scores)
        optimizer.zero_grad()
        loss.backward()
        if args.optim_name in ['sam', 'fmfp']:

            optimizer.first_step(zero_grad=True)
            compute_loss(args, net, image, target, image_idx, correct_log, cls_criterion, mixup_criterion,
                         rank_criterion)[0].backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
        prec, correct = utils.utils.accuracy(output, target)
        correct_log.update(image_idx, correct)
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break

        train_log['Tot. Loss'].update(loss.item(), image.size(0))
        train_log['CLS Loss'].update(loss_ce.item(), image.size(0))
        train_log['Mixup Loss'].update(loss_mixup.item(), image.size(0))
        train_log['CRL Loss'].update(loss_crl.item(), image.size(0))
        train_log['Top1 Acc.'].update(prec.item(), image.size(0))
        train_log['LR'].update(lr, image.size(0))

        if i % 100 == 99:
            log = ['LR : {:.5f}'.format(train_log['LR'].avg)] + [key + ': {:.2f}'.format(train_log[key].avg) for key in
                                                                 train_log if key != 'LR']
            msg = 'Epoch {:d} \t Batch {:d}\t'.format(epoch, i) + '\t'.join(log)
            logger.info(msg)
            for key in train_log:
                train_log[key] = utils.utils.AverageMeter()
    correct_log.max_correctness_update(epoch)
    for key in train_log:
        writer.add_scalar('./Train/' + key, train_log[key].avg, epoch)



