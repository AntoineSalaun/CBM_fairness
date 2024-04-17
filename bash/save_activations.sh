#!/bin/bash
#SBATCH -o slurm_logs/save_activations.sh.log-%j
#SBATCH --gres=gpu:volta:1

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_full

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_balanced

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_imbalanced_1

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_imbalanced_2

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_male

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_female

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_30_full

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_30_imbalanced_1

python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_30_imbalanced_2

# python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_full ; python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_balanced ; python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_imbalanced_1 ; python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_imbalanced_2 ; python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_male ; python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_200_female ; python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_30_full ; python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_30_imbalanced_1 ; python fairness_cv_project/methods/label_free_cbm/src/save_activations.py --dataset imSitu_30_imbalanced_2
