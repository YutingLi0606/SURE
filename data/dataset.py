import numpy as np
import torchvision.transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
import data.dataset

from PIL import ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def generate_imbalanced_data(indices, labels, imb_factor, num_classes):
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    img_max = len(indices) / num_classes
    img_num_per_cls = [int(img_max * (imb_factor ** (i / (num_classes - 1.0)))) for i in range(num_classes)]

    imbalanced_indices = []
    for cls_idx, img_num in zip(range(num_classes), img_num_per_cls):
        cls_indices = class_indices[cls_idx]
        np.random.shuffle(cls_indices)
        selec_indices = cls_indices[:img_num]
        imbalanced_indices.extend(selec_indices)

    return imbalanced_indices

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, is_train=False, dataset_type='normal', imb_factor=None):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.is_train = is_train
        if self.is_train and dataset_type in ['cifar10_LT', 'cifar100_LT'] and imb_factor is not None:
            self.gen_imbalanced_data(imb_factor, dataset_type)

    def gen_imbalanced_data(self, imb_factor, dataset_type):
        targets_np = np.array(self.targets, dtype=np.int64)

        if dataset_type in ['cifar10','cifar10_LT']:
            num_classes = 10
        elif dataset_type in ['cifar100','cifar100_LT']:
            num_classes = 100
        assert dataset_type in ['cifar10','cifar10_LT','cifar100','cifar100_LT'], "error: new dataset should set num_classes"
        print(num_classes)
        imbalanced_indices = generate_imbalanced_data(np.arange(len(self.samples)), targets_np, imb_factor, num_classes)
        self.samples = [self.samples[i] for i in imbalanced_indices]
        self.targets = [self.targets[i] for i in imbalanced_indices]

    def __getitem__(self, index):
        img, label = super(CustomImageFolder, self).__getitem__(index)
        return img, label, index


def TrainDataLoader(img_dir, transform_train, batch_size, is_train=True, dataset_type='normal', imb_factor=None):
    train_set = CustomImageFolder(img_dir, transform_train, is_train=is_train, dataset_type=dataset_type, imb_factor=imb_factor)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    return train_loader

# test data loader
def TestDataLoader(img_dir, transform_test, batch_size):
    test_set = CustomImageFolder(img_dir, transform_test)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    return test_loader

def get_loader(dataset, train_dir, val_dir, test_dir, batch_size, imb_factor, model_name):


    if dataset in ['cifar10','cifar10_LT']:
        if model_name == 'deit':
            norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        nb_cls = 10
    elif dataset in ['cifar100', 'cifar100_LT']:
        if model_name == 'deit':
            norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        nb_cls = 100
    elif dataset == 'Animal10N':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nb_cls = 10
    elif dataset == 'Clothing1M':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nb_cls = 14
    elif dataset == 'Food101N':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nb_cls = 101
    elif dataset == 'TinyImgNet':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nb_cls = 200

    if dataset in ['cifar10', 'cifar10_LT', 'cifar100', 'cifar100_LT']:
        if model_name == 'deit':
            transform_train = torchvision.transforms.Compose([
                                                            torchvision.transforms.Resize(256),
                                                            torchvision.transforms.CenterCrop(224),
                                                            torchvision.transforms.RandomHorizontalFlip(),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(norm_mean, norm_std)])
        # transformation of the test set
            transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(norm_mean, norm_std)])
        else:
            transform_train = torchvision.transforms.Compose([
                                                            torchvision.transforms.RandomCrop(32, padding=4),
                                                            torchvision.transforms.RandomHorizontalFlip(),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(norm_mean, norm_std)])
        # transformation of the test set
            transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(norm_mean, norm_std)])
    elif dataset in ['Animal10N', 'TinyImgNet']:
        transform_train = torchvision.transforms.Compose([
                                                        torchvision.transforms.RandomCrop(64, padding=4),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(norm_mean, norm_std)])
        # transformation of the test set
        transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(norm_mean, norm_std)])
    elif dataset in ['Clothing1M', 'Food101N']:
        transform_train = torchvision.transforms.Compose([
                                                        torchvision.transforms.RandomResizedCrop(224),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(norm_mean, norm_std)])
        transform_test = torchvision.transforms.Compose([
                                                        torchvision.transforms.Resize(256),
                                                        torchvision.transforms.CenterCrop(224),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(norm_mean, norm_std)])

    train_loader = TrainDataLoader(train_dir, transform_train, batch_size, is_train=True, dataset_type=dataset, imb_factor=imb_factor)
    val_loader = TestDataLoader(val_dir, transform_test, batch_size)
    test_loader = TestDataLoader(test_dir, transform_test, batch_size)

    return train_loader, val_loader, test_loader, nb_cls

