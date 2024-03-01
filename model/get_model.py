import model.resnet
import model.resnet18
import model.densenet_BC
import model.vgg
import model.mobilenet
import model.efficientnet
import model.wrn
import model.convmixer
import model.vit_cifar
import model.resnet50
import model.resnet_LT
import model.vit_seg
import model.classifier
import timm
import torch

def get_model(model_name, nb_cls, logger, args):
    if model_name == 'resnet18':
        net = model.resnet18.ResNet18(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == 'resnet110':
        net = model.resnet.resnet110(num_classes=nb_cls).cuda()
    elif model_name == 'resnet50':
        net = model.resnet50.resnet50(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp, pretrained=False).cuda()
        if args.resume_path:
            param = torch.load(args.resume_path)
            net.load_state_dict(param, strict=False)
    elif model_name == 'densenet':
        net = model.densenet_BC.DenseNet3(depth=100,
                                          num_classes=nb_cls,
                                          use_cos=args.use_cosine,
                                          cos_temp=args.cos_temp,
                                          growth_rate=12,
                                          reduction=0.5,
                                          bottleneck=True,
                                          dropRate=0.0).cuda()
    elif model_name == 'vgg':
        net = model.vgg.vgg16(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == 'vgg19bn':
        net = model.vgg.vgg19bn(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == 'wrn':
        net = model.wrn.WideResNet(28, nb_cls, args.use_cosine, args.cos_temp, 10).cuda()
    elif model_name == 'efficientnet':
        net = model.efficientnet.efficientnet(num_classes=nb_cls).cuda()
    elif model_name == 'mobilenet':
        net = model.mobilenet.mobilenet(num_classes=nb_cls).cuda()
    elif model_name == "cmixer":
        net = model.convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=nb_cls,
                                        ).cuda()
    elif model_name == "vit_cifar":
        net = model.vit_cifar.vit_cifar(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == "resnext":
        net = model.resnet_LT.Model(name='resnext50', num_classes=nb_cls,use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == "deit":
        net = timm.create_model('deit_base_patch16_224', checkpoint_path='/home/liyuting/lyt/deit_base_patch16_224-b5f2ef4d.pth').cuda()
        num_ftrs = net.head.in_features
        if args.use_cosine:
            net.head = model.classifier.Classifier(num_ftrs, nb_cls, args.cos_temp).cuda()
        else:
            net.head = torch.nn.Linear(num_ftrs, nb_cls).cuda()
    msg = 'Using {} ...'.format(model_name)
    logger.info(msg)
    return net