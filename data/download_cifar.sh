# You can also experiment on both CIFAR-10 and CIFAR-100
# Download CIFAR-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O ./CIFAR-10.tar.gz
tar -xvzf ./CIFAR-10.tar.gz
rm CIFAR-10.tar.gz

# Download CIFAR-100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -O ./CIFAR-100.tar.gz
tar -xvzf ./CIFAR-100.tar.gz
rm CIFAR-100.tar.gz

# Split into train / val(0.1 train) / test
python preprocess_cifar10.py

python preprocess_cifar100.py

## remove
rm -r cifar-10-batches-py
rm -r cifar-100-python