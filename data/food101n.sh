wget https://iudata.blob.core.windows.net/food101/Food-101N_release.zip

wget https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

## You can also download Food-101N from here(https://kuanghuei.github.io/Food-101N/)
## You can also download Food-101 from here(https://www.kaggle.com/datasets/dansbecker/food-101)
unzip Food-101N_release.zip
tar -xvf food-101.tar.gz

# change the data_path before preprocessing
python3 preprocess_food101n.py --out-dir ./Food101N