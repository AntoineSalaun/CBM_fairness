#!/bin/bash
#SBATCH -o slurm_logs/grid_search_2_resnet_200_verbs_full_baseline.sh.log-%j
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=40


train_baseline_path=fairness_cv_project/datasets/imSitu/model_training/train_resnet.py
train_path=fairness_cv_project/methods/label_free_cbm/src/train_cbm.py
test_path=fairness_cv_project/datasets/imSitu/model_training/test_resnet.py
explainability_path=fairness_cv_project/methods/label_free_cbm/src/explainability.py



# Train baseline
lr=(0.0001 0.0005 0.001)
MOMENTUM=(0.5 0.8)
STEP_SIZE=(5 10)
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
                save_folder="200_verbs_full/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}"
                
                path_model="saved_models/imSitu/200_verbs_full/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/model.pt"
                
                result_folder="results/imSitu/200_verbs_full/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}"

                echo "Train"

                python ${train_baseline_path} --dataset 200_verbs_full/train_val --save_folder $save_folder --lr $learning_rate_value --momentum $momentum_value --step_size $step_size_value --gamma $gamma_value # --num_epochs 1

                echo "Test"

                python ${test_path} --path_model $path_model --path_result $result_folder --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

            done
        done
    done
done
