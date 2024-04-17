#!/bin/bash
#SBATCH -o slurm_logs/baseline_200_verbs.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Train baseline
python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 200_verbs_full/train_val --save_folder baseline/four_zeros/full --lr 0.0001 --step_size 5 --gamma 0.1

python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 200_verbs/train_val_split/train_balanced --save_folder baseline/four_zeros/balanced --lr 0.0001 --step_size 5 --gamma 0.1

python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 200_verbs/train_val_split/train_imbalanced_1 --save_folder baseline/four_zeros/imbalanced_1 --lr 0.0001 --step_size 5 --gamma 0.1

python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 200_verbs/train_val_split/train_imbalanced_2 --save_folder baseline/four_zeros/imbalanced_2 --lr 0.0001 --step_size 5 --gamma 0.1

# Test Baseline

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/baseline/four_zeros/full/model.pt --path_result results/imSitu/baseline/four_zeros/full --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/baseline/four_zeros/balanced/model.pt --path_result results/imSitu/baseline/four_zeros/balanced --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/baseline/four_zeros/imbalanced_1/model.pt --path_result results/imSitu/baseline/four_zeros/imbalanced_1 --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/baseline/four_zeros/imbalanced_2/model.pt --path_result results/imSitu/baseline/four_zeros/imbalanced_2 --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/200_verbs_full/baseline/model.pt --path_result results/imSitu/200_verbs_full/baseline --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt