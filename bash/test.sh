#!/bin/bash
#SBATCH -o slurm_logs/test.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Train baseline
python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 200_verbs_full/train_val --save_folder 200_verbs_full/baseline/