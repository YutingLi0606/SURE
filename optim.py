import torch 
import torch.nn 
import utils.sam
import torch.optim
import torch.optim.lr_scheduler
import torch.optim.swa_utils



def get_optimizer_scheduler(model_name,
                            optim_name,
                            net,
                            lr,
                            momentum,
                            weight_decay,
                            max_epoch_cos = 200,
                            swa_lr = 0.05) :

    ## sgd + sam
    sgd_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sam_sgd = utils.sam.SAM(net.parameters(), torch.optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)

    ## adamw + sam
    adamw_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sam_adamw = utils.sam.SAM(net.parameters(), torch.optim.AdamW, lr=lr, weight_decay=weight_decay)

    ## convmixer uses adamw optimzer while cnn backbones uses sgd
    if model_name in ["convmixer", "vit_cifar"] :
        if optim_name in ['sam', 'fmfp'] : 
            optimizer = sam_adamw
        else :
            optimizer = adamw_optimizer
            
    else: 
        if optim_name in ['sam', 'fmfp'] : 
            optimizer = sam_sgd
        else :
            optimizer = sgd_optimizer

    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch_cos)
    
    ## swa model
    swa_model = torch.optim.swa_utils.AveragedModel(net)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)

    return optimizer, cos_scheduler, swa_model, swa_scheduler
