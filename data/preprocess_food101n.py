import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
import PIL
import shutil
import argparse


def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def gen_train_list(out_dir):
    root_data_path = '/user/leuven/334/vsc33476/data_75G/cyy/kernelNested/data/Food-101N_release/meta/imagelist.tsv'
    class_list_path = '/user/leuven/334/vsc33476/data_75G/cyy/kernelNested/data/Food-101N_release/meta/classes.txt'

    file_path_prefix = '/user/leuven/334/vsc33476/data_75G/cyy/kernelNested/data/Food-101N_release/images'

    map_name2cat = dict()
    with open(class_list_path) as fp:
        for i, line in enumerate(fp):
            row = line.strip()
            if row == 'class_name':
                continue
            map_name2cat[row] = i - 1
    num_class = len(map_name2cat)
    print('Num Classes: ', num_class)

    train_dir = os.path.join(out_dir, 'train')
    check_folder(train_dir)

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        fp.readline()  # skip first line

        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            target = map_name2cat[class_name]
            img_path = os.path.join(file_path_prefix, line.strip())

            dir_target = os.path.join(train_dir, str(target))
            check_folder(dir_target)

            # copy images to the new train file
            shutil.copy(img_path, dir_target)

            targets.append(target)
            img_list.append(img_path)

    targets = np.array(targets)
    img_list = np.array(img_list)
    print('Num Train Images: ', len(img_list))

    save_dir = check_folder('./image_list')
    np.save(os.path.join(save_dir, 'train_images'), img_list)
    np.save(os.path.join(save_dir, 'train_targets'), targets)

    return map_name2cat


def gen_test_list(arg_map_name2cat, out_dir):
    map_name2cat = arg_map_name2cat
    root_data_path = '/user/leuven/334/vsc33476/data_75G/cyy/kernelNested/data/food-101/meta/test.txt'

    file_path_prefix = '/user/leuven/334/vsc33476/data_75G/cyy/kernelNested/data/food-101/images'

    test_dir = os.path.join(out_dir, 'test')
    check_folder(test_dir)

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            target = map_name2cat[class_name]
            img_path = os.path.join(file_path_prefix, line.strip() + '.jpg')

            dir_target = os.path.join(test_dir, str(target))
            check_folder(dir_target)

            # copy images to the new test file
            shutil.copy(img_path, dir_target)

            targets.append(target)
            img_list.append(img_path)

    targets = np.array(targets)
    img_list = np.array(img_list)

    save_dir = check_folder('./image_list')
    np.save(os.path.join(save_dir, 'test_images'), img_list)
    np.save(os.path.join(save_dir, 'test_targets'), targets)

    print('Num Test Images: ', len(img_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./Food101N/', help='data directory')
    args = parser.parse_args()

    map_name2cat = gen_train_list(args.out_dir)
    gen_test_list(map_name2cat, args.out_dir)