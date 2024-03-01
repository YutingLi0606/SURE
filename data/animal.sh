## You can experiment on Animal10N
# Download Animal10N
wget https://www.dropbox.com/s/vpctw4jhnh5apkg/raw_image_ver.zip?dl=0wget https://www.dropbox.com/s/vpctw4jhnh5apkg/raw_image_ver.zip?dl=0
mv raw_image_ver.zip?dl=0 raw_image_ver.zip
unzip raw_image_ver.zip
unzip raw_image.zip

# preprocess Animal10N training set
mv training train
cd train
python ../preprocess_animal.py

# preprocess Animal10N test set
cd ..
mv testing test
cd test
python ../preprocess_animal.py

# rm the other zip files
cd ..
mkdir Animal10N
mv train Animal10N/
mv test Animal10N/

rm -rf raw_image.zip
rm -rf raw_image_ver.zip
rm -rf readme.txt