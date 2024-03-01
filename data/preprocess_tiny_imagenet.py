import os
import shutil
import random

base_dir = './data/tinyImageNet'
random.seed(42)
# We remove original test set and use original validation set as test set
# then split 10% training set as our validation set.

# 1. Restructure train set
train_dir = os.path.join(base_dir, 'train')
for class_dir in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_dir)
    images_dir = os.path.join(class_path, 'images')

    # Move images from images sub-directory to class directory
    for img in os.listdir(images_dir):
        shutil.move(os.path.join(images_dir, img), class_path)

    # Remove images sub-directory and boxes.txt
    shutil.rmtree(images_dir)
    os.remove(os.path.join(class_path, f"{class_dir}_boxes.txt"))

# 2. Create validation set
val_dir = os.path.join(base_dir, 'val')
os.makedirs(val_dir, exist_ok=True)

for class_dir in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_dir)
    images = os.listdir(class_path)
    val_images_count = int(0.1 * len(images))
    val_images = random.sample(images, val_images_count)

    val_class_dir = os.path.join(val_dir, class_dir)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in val_images:
        shutil.move(os.path.join(class_path, img), val_class_dir)

# 3. Restructure test set based on val_annotations.txt
test_dir = os.path.join(base_dir, 'test')
annotations_file = os.path.join(test_dir, 'val_annotations.txt')

with open(annotations_file, 'r') as f:
    for line in f.readlines():
        img_name, class_name = line.strip().split('\t')[:2]
        class_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        shutil.move(os.path.join(test_dir, 'images', img_name), class_dir)

shutil.rmtree(os.path.join(test_dir, 'images'))

print("Dataset restructured!")
