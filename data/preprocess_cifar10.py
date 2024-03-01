import os
import PIL.Image as Image
import numpy as np
from shutil import copyfile


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


ratio_val = 0.1  # 10% training data is used as validation

## ------------ cifar10 ---------- ##
img_dir = 'cifar-10-batches-py'
out_dir = 'CIFAR10'

train = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test = 'test_batch'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

train_dir = os.path.join(out_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

count = {}
for batch in train:
    dict = unpickle(os.path.join(img_dir, batch))
    labels = dict[b'labels']
    nb_data = len(labels)

    data = dict[b'data'].reshape((nb_data, 3, 32, 32)).transpose(0, 2, 3, 1)

    for i in range(nb_data):
        label_i = labels[i]
        if label_i not in count:
            count[label_i] = 1
        else:
            count[label_i] += 1

        cls_dir = train_dir

        dir_i = os.path.join(cls_dir, str(label_i))

        if not os.path.exists(dir_i):
            os.mkdir(dir_i)

        I_i = Image.fromarray(data[i]).save(os.path.join(dir_i, '{:d}.png'.format(count[label_i])))

count_train = {}
count_val = {}

## split to have a validation
val_dir = os.path.join(out_dir, 'val')
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

for cls in count:
    val_index = np.linspace(1, count[cls], int(count[cls] * ratio_val)).astype(int)
    count_train[cls] = count[cls] - len(val_index)
    count_val[cls] = len(val_index)

    val_i = os.path.join(val_dir, str(cls))
    if not os.path.exists(val_i):
        os.mkdir(val_i)

    train_i = os.path.join(train_dir, str(cls))

    for i in val_index:
        src = os.path.join(train_i, '{:d}.png'.format(i))
        dst = os.path.join(val_i, '{:d}.png'.format(i))
        copyfile(src, dst)

        cmd = 'rm {}'.format(src)
        os.system(cmd)

print('Cifar 10 train...')
print(count_train)

print('\nCifar 10 val...')
print(count_val)

test_dir = os.path.join(out_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

count = {}
dict = unpickle(os.path.join(img_dir, test))
labels = dict[b'labels']
nb_data = len(labels)

data = dict[b'data'].reshape((nb_data, 3, 32, 32)).transpose(0, 2, 3, 1)

for i in range(nb_data):
    label_i = labels[i]
    if label_i not in count:
        count[label_i] = 1
    else:
        count[label_i] += 1

    dir_i = os.path.join(test_dir, str(label_i))

    if not os.path.exists(dir_i):
        os.mkdir(dir_i)

    I_i = Image.fromarray(data[i]).save(os.path.join(dir_i, '{:d}.png'.format(count[label_i])))

print('\nCifar 10 test...')
print(count)