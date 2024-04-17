#!/bin/bash
#SBATCH -o slurm_logs/mscoco_single.sh.log-%j
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2023a

# Baseline:
python fairness_cv_project/datasets/mscoco/single_label/train.py --dataset "balanced" --save_name "balanced.pt"
python fairness_cv_project/datasets/mscoco/single_label/train.py --dataset "imbalanced1" --save_name "imbalanced1.pt"
python fairness_cv_project/datasets/mscoco/single_label/train.py --dataset "imbalanced2" --save_name "imbalanced2.pt"
# CBM_Sparse
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_balanced --backbone resnet50 --concept_set data/concept_sets/mscoco_single_filtered.txt --save_dir saved_models/mscoco/single_label/CBM/balanced_sparse --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_imbalanced1 --backbone resnet50 --concept_set data/concept_sets/mscoco_single_filtered.txt --save_dir saved_models/mscoco/single_label/CBM/imbalanced1_sparse --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_imbalanced2 --backbone resnet50 --concept_set data/concept_sets/mscoco_single_filtered.txt --save_dir saved_models/mscoco/single_label/CBM/imbalanced2_sparse --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print
# CBM_Dense
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_balanced --backbone resnet50 --concept_set data/concept_sets/mscoco_single_filtered.txt --save_dir saved_models/mscoco/single_label/CBM/balanced_dense --interpretability_cutoff 0.3 --n_iters 80 --print
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_imbalanced1 --backbone resnet50 --concept_set data/concept_sets/mscoco_single_filtered.txt --save_dir saved_models/mscoco/single_label/CBM/imbalanced1_dense --interpretability_cutoff 0.3 --n_iters 80 --print
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_imbalanced2 --backbone resnet50 --concept_set data/concept_sets/mscoco_single_filtered.txt --save_dir saved_models/mscoco/single_label/CBM/imbalanced2_dense --interpretability_cutoff 0.3 --n_iters 80 --print
#CBM_Sparse_Gender
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_balanced --backbone resnet50 --protected_concepts "a male" "a female" --concept_set data/concept_sets/mscoco_single_gender.txt --save_dir saved_models/mscoco/single_label/CBM/balanced_sparse_gender --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print  
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_imbalanced1 --backbone resnet50 --protected_concepts "a male" "a female" --concept_set data/concept_sets/mscoco_single_gender.txt --save_dir saved_models/mscoco/single_label/CBM/imbalanced1_sparse_gender --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print 
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_imbalanced2 --backbone resnet50 --protected_concepts "a male" "a female" --concept_set data/concept_sets/mscoco_single_gender.txt --save_dir saved_models/mscoco/single_label/CBM/imbalanced2_sparse_gender --interpretability_cutoff 0.3 --n_iters 80 --lam 0.01 --print
#CBM_Dense_Gender
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_balanced --backbone resnet50 --protected_concepts "a male" "a female" --concept_set data/concept_sets/mscoco_single_gender.txt --save_dir saved_models/mscoco/single_label/CBM/balanced_dense_gender --interpretability_cutoff 0.3 --n_iters 80 --print
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_imbalanced1 --backbone resnet50 --protected_concepts "a male" "a female" --concept_set data/concept_sets/mscoco_single_gender.txt --save_dir saved_models/mscoco/single_label/CBM/imbalanced1_dense_gender --interpretability_cutoff 0.3 --n_iters 80 --print
python fairness_cv_project/methods_mscoco/label_free_cbm/src/train_cbm.py --dataset mscoco_single_imbalanced2 --backbone resnet50 --protected_concepts "a male" "a female" --concept_set data/concept_sets/mscoco_single_gender.txt --save_dir saved_models/mscoco/single_label/CBM/imbalanced2_dense_gender --interpretability_cutoff 0.3 --n_iters 80 --print


# Test 
# Baseline:
python fairness_cv_project/datasets/mscoco/test.py --path_model 'saved_models/mscoco/single_label/baseline/balanced.pt' --path_result 'results/mscoco/single_label/baseline/balanced'
python fairness_cv_project/datasets/mscoco/test.py --path_model 'saved_models/mscoco/single_label/baseline/imbalanced1.pt' --path_result 'results/mscoco/single_label/baseline/imbalanced1'
python fairness_cv_project/datasets/mscoco/test.py --path_model 'saved_models/mscoco/single_label/baseline/imbalanced2.pt' --path_result 'results/mscoco/single_label/baseline/imbalanced2'
# CBM_Sparse:
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/balanced_sparse/mscoco_single_balanced_mscoco_single_filtered' --path_result 'results/mscoco/single_label/CBM/balanced_sparse'
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/imbalanced1_sparse/mscoco_single_imbalanced1_mscoco_single_filtered' --path_result 'results/mscoco/single_label/CBM/imbalanced1_sparse'
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/imbalanced2_sparse/mscoco_single_imbalanced2_mscoco_single_filtered' --path_result 'results/mscoco/single_label/CBM/imbalanced2_sparse'
# CBM_Dense:
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/balanced_dense/mscoco_single_balanced_mscoco_single_filtered' --path_result 'results/mscoco/single_label/CBM/balanced_dense'
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/imbalanced1_dense/mscoco_single_imbalanced1_mscoco_single_filtered' --path_result 'results/mscoco/single_label/CBM/imbalanced1_dense'
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/imbalanced2_dense/mscoco_single_imbalanced2_mscoco_single_filtered' --path_result 'results/mscoco/single_label/CBM/imbalanced2_dense'
#CBM_Sparse_Gender:
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/balanced_sparse_gender/mscoco_single_balanced_mscoco_single_gender' --path_result 'results/mscoco/single_label/CBM/balanced_sparse_gender'
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/imbalanced1_sparse_gender/mscoco_single_imbalanced1_mscoco_single_gender' --path_result 'results/mscoco/single_label/CBM/imbalanced1_sparse_gender'
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/imbalanced2_sparse_gender/mscoco_single_imbalanced2_mscoco_single_gender' --path_result 'results/mscoco/single_label/CBM/imbalanced2_sparse_gender'
#CBM_Dense_Gender:
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/balanced_dense_gender/mscoco_single_balanced_mscoco_single_gender' --path_result 'results/mscoco/single_label/CBM/balanced_dense_gender'
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/imbalanced1_dense_gender/mscoco_single_imbalanced1_mscoco_single_gender' --path_result 'results/mscoco/single_label/CBM/imbalanced1_dense_gender'
python fairness_cv_project/datasets/mscoco/test.py --is_cbm --path_model 'saved_models/mscoco/single_label/CBM/imbalanced2_dense_gender/mscoco_single_imbalanced2_mscoco_single_gender' --path_result 'results/mscoco/single_label/CBM/imbalanced2_dense_gender'


 