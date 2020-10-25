rm -rf ./dataset/
wget https://github.com/derekfernandez/tp2-ia-rna/raw/main/dataset.tar.gz -P ./
tar -xvf ./dataset.tar.gz --strip-components=0 -C ./
rm ./dataset.tar.gz
