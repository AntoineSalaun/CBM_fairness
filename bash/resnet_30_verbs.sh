#!/bin/bash
#SBATCH -o slurm_logs/resnet_30_verbs.sh.log-%j
#SBATCH --gres=gpu:volta:1

: <<COMMENT
# Train baseline on imSitu 30 verbs
python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py
python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 30_verbs/train_val_split/train_imbalanced_1 --save_folder 30_verbs/baseline/imbalanced_1/
python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 30_verbs/train_val_split/train_imbalanced_2 --save_folder 30_verbs/baseline/imbalanced_2/

# Training CBM on imSitu 30 verbs

# Genderless

# Sparse:
python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_balanced --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs/CBM/no_gender/sparse_balanced --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_imbalanced_1 --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs/CBM/no_gender/sparse_imbalanced_1 --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_imbalanced_2 --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs/CBM/no_gender/sparse_imbalanced_2 --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print

# Dense:
python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_balanced --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs/CBM/no_gender/dense_balanced --interpretability_cutoff 0.3 --n_iters 80 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_imbalanced_1 --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs/CBM/no_gender/dense_imbalanced_1 --interpretability_cutoff 0.3 --n_iters 80 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_imbalanced_2 --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs/CBM/no_gender/dense_imbalanced_2 --interpretability_cutoff 0.3 --n_iters 80 --print

# With gender

# Sparse:

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_balanced --backbone resnet50 --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print --protected_concepts "a male" "a female"  --concept_set data/concept_sets/imSitu_30_gender.txt --save_dir saved_models/imSitu/30_verbs/CBM/gender/sparse_balanced

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_imbalanced_1 --backbone resnet50 --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print --protected_concepts "a male" "a female"  --concept_set data/concept_sets/imSitu_30_gender.txt --save_dir saved_models/imSitu/30_verbs/CBM/gender/sparse_imbalanced_1

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_imbalanced_2 --backbone resnet50 --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print --protected_concepts "a male" "a female"  --concept_set data/concept_sets/imSitu_30_gender.txt --save_dir saved_models/imSitu/30_verbs/CBM/gender/sparse_imbalanced_2

# Dense

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_balanced --backbone resnet50  --interpretability_cutoff 0.3 --n_iters 80 --print --protected_concepts "a male" "a female"  --concept_set data/concept_sets/imSitu_30_gender.txt --save_dir saved_models/imSitu/30_verbs/CBM/gender/dense_balanced 

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_imbalanced_1 --backbone resnet50 --interpretability_cutoff 0.3 --n_iters 80 --print --protected_concepts "a male" "a female"  --concept_set data/concept_sets/imSitu_30_gender.txt --save_dir saved_models/imSitu/30_verbs/CBM/gender/dense_imbalanced_1

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_imbalanced_2 --backbone resnet50 --interpretability_cutoff 0.3 --n_iters 80 --print --protected_concepts "a male" "a female"  --concept_set data/concept_sets/imSitu_30_gender.txt --save_dir saved_models/imSitu/30_verbs/CBM/gender/dense_imbalanced_2

# Testing CBM on imSitu 30 verbs
# Baseline
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py 
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model 'saved_models/imSitu/30_verbs/baseline/imbalanced_1/model.pt' --path_result 'results/imSitu/30_verbs/baseline/imbalanced_1'
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model 'saved_models/imSitu/30_verbs/baseline/imbalanced_2/model.pt' --path_result 'results/imSitu/30_verbs/baseline/imbalanced_2'
COMMENT

# CBM no gender sparse
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/no_gender/sparse_balanced/imSitu_30_balanced_imSitu_30_filtered --path_result results/imSitu/30_verbs/CBM/no_gender/balanced_sparse 

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/no_gender/sparse_balanced/imSitu_30_balanced_imSitu_30_filtered --path_result results/imSitu/30_verbs/CBM/no_gender/balanced_sparse 

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/no_gender/sparse_imbalanced_1/imSitu_30_imbalanced_1_imSitu_30_filtered  --path_result results/imSitu/30_verbs/CBM/no_gender/imbalanced_1_sparse
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/no_gender/sparse_imbalanced_2/imSitu_30_imbalanced_2_imSitu_30_filtered --path_result results/imSitu/30_verbs/CBM/no_gender/imbalanced_2_sparse

# CBM no gender dense
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/no_gender/dense_balanced/imSitu_30_balanced_imSitu_30_filtered --path_result results/imSitu/30_verbs/CBM/no_gender/balanced_dense
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/no_gender/dense_imbalanced_1/imSitu_30_imbalanced_1_imSitu_30_filtered --path_result results/imSitu/30_verbs/CBM/no_gender/imbalanced_1_dense
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/no_gender/dense_imbalanced_2/imSitu_30_imbalanced_2_imSitu_30_filtered --path_result results/imSitu/30_verbs/CBM/no_gender/imbalanced_2_dense

# CBM gender sparse

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/gender/sparse_balanced/imSitu_30_balanced_imSitu_30_gender --path_result results/imSitu/30_verbs/CBM/gender/balanced_sparse

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/gender/sparse_imbalanced_1/imSitu_30_imbalanced_1_imSitu_30_gender --path_result results/imSitu/30_verbs/CBM/gender/imbalanced_1_sparse

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/gender/sparse_imbalanced_2/imSitu_30_imbalanced_2_imSitu_30_gender --path_result results/imSitu/30_verbs/CBM/gender/imbalanced_2_sparse

# CBM gender dense

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/gender/dense_balanced/imSitu_30_balanced_imSitu_30_gender --path_result results/imSitu/30_verbs/CBM/gender/balanced_dense

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/gender/dense_imbalanced_1/imSitu_30_imbalanced_1_imSitu_30_gender --path_result results/imSitu/30_verbs/CBM/gender/imbalanced_1_dense

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/gender/dense_imbalanced_2/imSitu_30_imbalanced_2_imSitu_30_gender --path_result results/imSitu/30_verbs/CBM/gender/imbalanced_2_dense


python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_balanced --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs/CBM/no_gender/sparse_balanced --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs/CBM/no_gender/sparse_balanced/imSitu_30_balanced_imSitu_30_filtered --path_result results/imSitu/30_verbs/CBM/no_gender/balanced_sparse 
