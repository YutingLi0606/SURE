import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import model.classifier
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

__all__ = [
    'VGG', 'vgg16', 'vgg19',]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10, use_cos=False, cos_temp=8):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
        )
        self.use_cos = use_cos
        self.fc = nn.Linear(512, num_classes)
        self.cos_classifier = model.classifier.Classifier(512, num_classes, cos_temp)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, feature_output=False):
        x = self.features(x)
        dim = x.size()[-1]
        # The resolution of CIFAR is 32x32
        # and the resolution of TinyImageNet is 64x64
        if dim > 1:
            x = F.avg_pool2d(x, dim)
        last_feature = x.view(x.size(0), -1)
        x = self.classifier(last_feature)
        if self.use_cos:
            x = self.cos_classifier(x)
        else:
            x = self.fc(x)
        if feature_output:
            return x, last_feature
        else:
            return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}




def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True),**kwargs)


def vgg19(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True),**kwargs)

def vgg19bn(num_classes=10, use_cos=False, cos_temp=8):
    net = models.vgg19_bn(pretrained=False)
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.2),
        nn.Linear(4096, num_classes)
    )
    if use_cos:
        net.classifier[-1] = model.classifier.Classifier(4096, num_classes, cos_temp)

    return net