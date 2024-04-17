#!/bin/bash
#SBATCH -o slurm_logs/sparsity_imSitu.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Train 

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 0 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.00001 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 0.00001 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.0001 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 0.0001 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.0007 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 0.0007 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.001 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 0.001 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.01 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 0.01 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.1 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 0.1 --print

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_1 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 1 --print

# Test 
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/sparsity/lam_0/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/sparsity/lam_0 --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/sparsity/lam_0.00001/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/sparsity/lam_0.00001 --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/sparsity/lam_0.0001/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/sparsity/lam_0.0001 --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/sparsity/lam_0.001/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/sparsity/lam_0.001 --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/sparsity/lam_0.01/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/sparsity/lam_0.01 --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/sparsity/lam_0.1/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/sparsity/lam_0.1 --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/sparsity/lam_1/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/sparsity/lam_1 --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

# Explainability

python fairness_cv_project/methods/label_free_cbm/src/explainability.py --load_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0/imSitu_200_full_imSitu_200_filtered --file_path results/imSitu/200_verbs_full/sparsity/lam_0.xlsx

python fairness_cv_project/methods/label_free_cbm/src/explainability.py --load_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.00001/imSitu_200_full_imSitu_200_filtered --file_path results/imSitu/200_verbs_full/sparsity/lam_0.00001.xlsx

python fairness_cv_project/methods/label_free_cbm/src/explainability.py --load_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.0001/imSitu_200_full_imSitu_200_filtered --file_path results/imSitu/200_verbs_full/sparsity/lam_0.0001.xlsx

python fairness_cv_project/methods/label_free_cbm/src/explainability.py --load_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.001/imSitu_200_full_imSitu_200_filtered --file_path results/imSitu/200_verbs_full/sparsity/lam_0.001.xlsx

python fairness_cv_project/methods/label_free_cbm/src/explainability.py --load_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.01/imSitu_200_full_imSitu_200_filtered --file_path results/imSitu/200_verbs_full/sparsity/lam_0.01.xlsx

python fairness_cv_project/methods/label_free_cbm/src/explainability.py --load_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.1/imSitu_200_full_imSitu_200_filtered --file_path results/imSitu/200_verbs_full/sparsity/lam_0.1.xlsx

python fairness_cv_project/methods/label_free_cbm/src/explainability.py --load_dir saved_models/imSitu/200_verbs_full/sparsity/lam_1/imSitu_200_full_imSitu_200_filtered --file_path results/imSitu/200_verbs_full/sparsity/lam_1.xlsx

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.0007 --interpretability_cutoff 0 --clip_cutoff 0 --n_iters 80 --lam 0.0007 --print ; python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/sparsity/lam_0.0007/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/sparsity/lam_0.0007 --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt; python fairness_cv_project/methods/label_free_cbm/src/explainability.py --load_dir saved_models/imSitu/200_verbs_full/sparsity/lam_0.0007/imSitu_200_full_imSitu_200_filtered --file_path results/imSitu/200_verbs_full/sparsity/lam_0.0007.xlsx

 
