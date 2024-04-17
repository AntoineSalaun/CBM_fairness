# CBM Fairness Project

This repository contains the codebase for the CBM Fairness project done in the Decentralized Information Group at CSAIL MIT.

The data is preprocessed from the Doctor-Nurse, imSitu and MS-COCO datasets. The transformation from traditional model to CBM is made by the [Label-Free CBM implementation](https://github.com/Trustworthy-ML-Lab/Label-free-CBM). The data is then postprocessed. 

The main contributors of the codebase are [Vincent Yuan](https://www.csail.mit.edu/person/vincent-yuan) and [Hengzhi Li](https://www.csail.mit.edu/person/hengzhi-li). The project is done under the supervision of PhD student [Schrasing Tong](https://www.csail.mit.edu/person/schrasing-tong) and Prof. [Lalana Kagal](https://www.csail.mit.edu/person/lalana-kagal).

The README from the original imSitu and LF-CBM papers are available in this repository.

## Repository guide

This subsection will provide an high-level view of the repository.

### Documentation 

### Bash

This folder contains bash script used to run the different pipelines

### Data

This folder contains the different information related to data

- Classes contains the list of class for every dataset. These classes are then used during the CBM training and postprocessing process, to associate the neurons of the final layer to their classes
- Concept sets have every different concept set generated. The mention 'filtered' means they have passed the filtering part of LF-CBM. When adding 'Gender' at the end of the name, the concepts 'a male' and 'a female' are in the concept set.
- Datasets contains the different datasets. 
    - Dataset from original LF-CBM papers (CIFAR-10, CIFAR-100, CUB) can be downloaded by running the script `bash/download_models.sh` and `bash/download_models.sh`. Note: These datasets were not used throughout the project, so these scripts might be outdated. 
    - Doctor-Nurse dataset: You can download the dataset from [this repository](https://github.com/ghayat2/Agnostic_Fairness/tree/main/Datasets/doctor_nurse). The original dataset was the put in the folder `data/datasets/doctor_nurse_2/original_dataset`. This folder has two subfolders, `dr` and `nurse` which are present in the aforementionned repository.
    - imSitu dataset: The dataset was downloaded from the [imSitu website](http://imsitu.org/download/) and put in the folder `data/datasets/imSitu/original_data`. We took the 256*256 images. The metadata files were taken from the original [imSitu repository](https://github.com/my89/imSitu) and put in the folder `data/datasets/imSitu/original_metadata`. The metadata files taken are: `dev.json`, `test.json`, `train.json`
    - MS-COCO:

### Documentation

This section contains the README from imSitu and LF-CBM, and the MSc. Thesis of Vincent Yuan

### Logs

This section keeps the logs of the different runs. They are configured in the shell files of `bash/`

### Results

This section keeps the different results of the models.

### Saved activations

This section will be created when the LF-CBM pipeline will be ran. This section saves the activation of a given models, i.e. given a backbone model and a set of images as inputs, it computes and save the output (the embeddings) of the images put in the model.

### Saved models

This section stores all of the models, may they be the pre-trained models or the ones created when training LF-CBM.

### Fairness_cv_project

#### Datasets

This part contains the data preprocessing for every datasets. 

More specifically, doctor nurse contains training and testing models for resnet and alexnet - the model used was alexnet.

For imSitu, the data processing is made in `dataset_manipulation`, downloading the backbone resnet model is done in `download_models.py`, and the training for alexnet and resnet was made in `model_training` - resnet was chosen.

For MS-COCO,

#### Methods

This part contains only the code for Label_Free_CBM entry. 

- Notebooks: Contains the different notebooks used to preprocess the concept or postprocess the results
    - ConceptNet_conceptset generates the concepts from conceptNet, and is not used during our study.
    - GPT_initial_concepts: Allow to generate the concepts from GPT-3
    - GPT_conceptset_processor: Allow to filter the concepts created by GPT
    - evaluate_cbm: Allow to make sankey diagrams representation of a given CBM
    - experiments/ : This folder allows to make multiple experiments on existing models, like changing the weights form a model. 
- Src: This part is the core of the project
    - concept_engineering.py: this file generates, then filter the concepts for a given dataset present in the DATA_UTILS variables of `fairness_cv_project.methods.label_free_cbm.src.utils.data_utils`
    - Explainability: This file postprocess a given model, creating an excel spreadsheet containing, for every class, every concepts associated and their weight.
    - save_activation: This file saves the activation values of a given dataset on a given model
    - Similarity: helper function to do the cosine similarity
    - train_cbm: main function, training a LF-CBM: it trains the projection from the backbone to the concept layer, then the final sparse classifier.
    - Train_final_layer: This function trains the final sparse classifier from an already existing projection
    - Train_standard: train a standard model
    - The folder `downloaders/` allow to download the models and the datasets.

### methods_mscoco

### Postprocessing: 

This folder contains different notebooks used to process the different trained models

#### Tests

This folder contains the different test.

## CBM Creation Pipeline

This section explains how the CBM pipeline work, and how to add a new model/dataset on the LF-CBM pipeline. The reader is encouraged to read LF-CBM paper to understand deeply the model.

To add train on a new dataset/model:

1. In `utils/data_utils.py`, add the dataset path in `DATASET_ROOTS`, and the way to create a dataloader from the folder to the root in the `get_data` function. Add also the classes list in `LABEL_FILES`.
2. Add the model in `saved_models`, and add how to access it in the `get_target_model` function in `utils/data_utils.py`

## Onboarding

This codebase was ran on MIT supercloud server, and some features would work differently in different servers. The user can always contact the authors to ask questions.

Generally, the work should be done as such:

1. Verify you have the different dependencies from `requirements.txt` in your local environment
2. Create the different folders if necessary, such as `saved_activations` or `saved_models`
3. Download the datasets, as described above
4. Run the LF-CBM pipeline:

## Important notes

1. If you want to generate concepts again from GPT-3, you will need to add your OpenAI key in a file `~/openai_api_key` in your home folder. 
