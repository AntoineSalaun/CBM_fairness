cd ..
gdown https://drive.google.com/u/0/uc?id=1fOUVzBpEt75s5IzY1HqcXiu6x9a830u7
unzip saved_models.zip
rm saved_models.zip

# Download the all-mpnet-base-v2 model for CBM evaluation
wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-mpnet-base-v2.zip
mkdir $PWD/saved_models/all-mpnet-base-v2
unzip all-mpnet-base-v2.zip -d $PWD/saved_models/all-mpnet-base-v2
rm all-mpnet-base-v2.zip