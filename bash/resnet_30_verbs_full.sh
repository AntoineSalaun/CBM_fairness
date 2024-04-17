#!/bin/bash
#SBATCH -o slurm_logs/resnet_30_verbs_full.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Train baseline
python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 30_verbs_full/train_val --save_folder 30_verbs_full/baseline/

# Train CBM
python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_full --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs_full/CBM/sparse_no_gender --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_full --backbone resnet50 --concept_set data/concept_sets/imSitu_30_filtered.txt --save_dir saved_models/imSitu/30_verbs_full/CBM/dense_no_gender --interpretability_cutoff 0.3 --n_iters 80 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_full --backbone resnet50 --concept_set data/concept_sets/imSitu_30_gender.txt --save_dir saved_models/imSitu/30_verbs_full/CBM/sparse_gender --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print --protected_concepts "a male" "a female"

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_30_full --backbone resnet50 --concept_set data/concept_sets/imSitu_30_gender.txt --save_dir saved_models/imSitu/30_verbs_full/CBM/dense_gender --interpretability_cutoff 0.3 --n_iters 80 --print --protected_concepts "a male" "a female"

# Test Baseline

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/30_verbs_full/baseline/model.pt --path_result results/imSitu/30_verbs_full/baseline --path_test_dataset data/datasets/imSitu/data/30_verbs_full/test_with_gender

# Test CBM

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs_full/CBM/sparse_no_gender/imSitu_30_full_imSitu_30_filtered  --path_test_dataset data/datasets/imSitu/data/30_verbs_full/test_with_gender --path_result results/imSitu/30_verbs_full/CBM/sparse_no_gender 

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs_full/CBM/dense_no_gender/imSitu_30_full_imSitu_30_filtered --path_result results/imSitu/30_verbs_full/CBM/dense_no_gender --path_test_dataset data/datasets/imSitu/data/30_verbs_full/test_with_gender

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs_full/CBM/sparse_gender/imSitu_30_full_imSitu_30_gender --path_result results/imSitu/30_verbs_full/CBM/sparse_gender --path_test_dataset data/datasets/imSitu/data/30_verbs_full/test_with_gender

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/30_verbs_full/CBM/dense_gender/imSitu_30_full_imSitu_30_gender --path_result results/imSitu/30_verbs_full/CBM/dense_gender --path_test_dataset data/datasets/imSitu/data/30_verbs_full/test_with_gender
