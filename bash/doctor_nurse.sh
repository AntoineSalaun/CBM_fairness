#!/bin/bash
#SBATCH -o slurm_logs/doctor_nurse.sh.log-%j
#SBATCH --gres=gpu:volta:1


# Train Baseline
python fairness_cv_project/datasets/doctor_nurse/train_alexnet.py --path_dataset data/datasets/doctor_nurse_2/train_test_split/train_balanced --path_save saved_models/doctor_nurse/alexnet/train_balanced

python fairness_cv_project/datasets/doctor_nurse/train_alexnet.py --path_dataset data/datasets/doctor_nurse_2/train_test_split/train_imbalanced_1 --path_save saved_models/doctor_nurse/alexnet/train_imbalanced_1

python fairness_cv_project/datasets/doctor_nurse/train_alexnet.py --path_dataset data/datasets/doctor_nurse_2/train_test_split/train_imbalanced_2 --path_save saved_models/doctor_nurse/alexnet/train_imbalanced_2

# Train doctor nurse

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset doctor_nurse_balanced --backbone alexnet_doctor_nurse --concept_set data/concept_sets/doctor_nurse_filtered_new.txt --feature_layer avgpool --save_dir saved_models/doctor_nurse/CBM/no_gender/balanced --n_iters 1000 --print --interpretability_cutoff 0 --clip_cutoff 0

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset doctor_nurse_imbalanced_1 --backbone alexnet_doctor_nurse --concept_set data/concept_sets/doctor_nurse_filtered_new.txt --feature_layer avgpool --save_dir saved_models/doctor_nurse/CBM/no_gender/imbalanced_1 --n_iters 1000 --print --interpretability_cutoff 0 --clip_cutoff 0

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset doctor_nurse_imbalanced_2 --backbone alexnet_doctor_nurse --concept_set data/concept_sets/doctor_nurse_filtered_new.txt --feature_layer avgpool --save_dir saved_models/doctor_nurse/CBM/no_gender/imbalanced_2 --n_iters 1000 --print --interpretability_cutoff 0 --clip_cutoff 0

# Train doctor nurse gender 
python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset doctor_nurse_balanced --backbone alexnet_doctor_nurse --concept_set data/concept_sets/doctor_nurse_with_gender.txt --feature_layer avgpool --save_dir saved_models/doctor_nurse/CBM/gender/balanced --n_iters 1000 --print --interpretability_cutoff 0 --clip_cutoff 0

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset doctor_nurse_imbalanced_1 --backbone alexnet_doctor_nurse --concept_set data/concept_sets/doctor_nurse_with_gender.txt --feature_layer avgpool --save_dir saved_models/doctor_nurse/CBM/gender/imbalanced_1 --n_iters 1000 --print --interpretability_cutoff 0 --clip_cutoff 0

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset doctor_nurse_imbalanced_2 --backbone alexnet_doctor_nurse --concept_set data/concept_sets/doctor_nurse_with_gender.txt --feature_layer avgpool --save_dir saved_models/doctor_nurse/CBM/gender/imbalanced_2 --n_iters 1000 --print --interpretability_cutoff 0 --clip_cutoff 0


# Test Baseline
python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --path_alexnet_model saved_models/doctor_nurse/alexnet/train_balanced/model.pt --path_result results/doctor_nurse/alexnet/train_balanced

python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --path_alexnet_model saved_models/doctor_nurse/alexnet/train_imbalanced_1/model.pt --path_result results/doctor_nurse/alexnet/train_imbalanced_1

python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --path_alexnet_model saved_models/doctor_nurse/alexnet/train_imbalanced_2/model.pt --path_result results/doctor_nurse/alexnet/train_imbalanced_2

# Test CBM 
python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --load_dir saved_models/doctor_nurse/CBM/no_gender/balanced/doctor_nurse_balanced_doctor_nurse_filtered_new --path_result results/doctor_nurse/CBM/no_gender/balanced

python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --load_dir saved_models/doctor_nurse/CBM/no_gender/imbalanced_1/doctor_nurse_imbalanced_1_doctor_nurse_filtered_new --path_result results/doctor_nurse/CBM/no_gender/imbalanced_1

python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --load_dir saved_models/doctor_nurse/CBM/no_gender/imbalanced_2/doctor_nurse_imbalanced_2_doctor_nurse_filtered_new --path_result results/doctor_nurse/CBM/no_gender/imbalanced_2

# Test CBM gender 
python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --load_dir saved_models/doctor_nurse/CBM/gender/balanced/doctor_nurse_balanced_doctor_nurse_with_gender --path_result results/doctor_nurse/CBM/gender/balanced

python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --load_dir saved_models/doctor_nurse/CBM/gender/imbalanced_1/doctor_nurse_imbalanced_1_doctor_nurse_with_gender --path_result results/doctor_nurse/CBM/gender/imbalanced_1

python fairness_cv_project/datasets/doctor_nurse/result_alexnet.py --load_dir saved_models/doctor_nurse/CBM/gender/imbalanced_2/doctor_nurse_imbalanced_2_doctor_nurse_with_gender --path_result results/doctor_nurse/CBM/gender/imbalanced_2
