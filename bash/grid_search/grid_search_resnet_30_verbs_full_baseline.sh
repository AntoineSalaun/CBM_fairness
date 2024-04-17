#!/bin/bash
#SBATCH -o slurm_logs/grid_search_resnet_30_verbs_full_baseline.sh.log-%j
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=40


# 30_verbs_full

train_baseline_path=fairness_cv_project/datasets/imSitu/model_training/train_resnet.py
train_path=fairness_cv_project/methods/label_free_cbm/src/train_cbm.py
test_path=fairness_cv_project/datasets/imSitu/model_training/test_resnet.py
explainability_path=fairness_cv_project/methods/label_free_cbm/src/explainability.py



# Train baseline
lr=(0.001 0.01 0.1)
MOMENTUM=(0.8 0.9 0.95)
STEP_SIZE=(5 7 10)
GAMMA=(0.1 0.5 0.9)

# lr=(0.001)
# MOMENTUM=(0.8)
# STEP_SIZE=(5)
# GAMMA=(0.1 0.5)

for learning_rate_value in "${lr[@]}"; do
    for momentum_value in "${MOMENTUM[@]}"; do
        for step_size_value in "${STEP_SIZE[@]}"; do
            for gamma_value in "${GAMMA[@]}"; do


                echo "______________________________________________________________________"
                echo "Learning rate: $learning_rate_value"
                echo "Momentum: $momentum_value"
                echo "Step size: $step_size_value"
                echo "Gamma: $gamma_value"

                save_folder="30_verbs_full/grid_search/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}"

                path_model="saved_models/imSitu/30_verbs_full/grid_search/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/model.pt"

                result_folder="results/imSitu/30_verbs_full/grid_search/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}"

                python ${train_baseline_path} --dataset 30_verbs_full/train_val --save_folder $save_folder --lr $learning_rate_value --momentum $momentum_value --step_size $step_size_value --gamma $gamma_value # --num_epochs 1
                
                python ${test_path} --path_model $path_model --path_result $result_folder --path_test_dataset data/datasets/imSitu/data/30_verbs_full/test_with_gender --num_classes 30 --path_list_verbs data/classes/imSitu_30_classes.txt

            done
        done
    done
done