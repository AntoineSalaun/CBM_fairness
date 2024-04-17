#!/bin/bash
#SBATCH -o slurm_logs/grid_search_2_resnet_30_verbs_baseline.sh.log-%j
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=40


# 30_verbs

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

                save_folder_balanced="30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/balanced"
                result_folder_balanced="results/imSitu/30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/balanced"
                path_model_balanced="saved_models/imSitu/30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/balanced/model.pt"


                save_folder_imbalanced_1="30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/imbalanced_1"
                result_folder_imbalanced_1="results/imSitu/30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/imbalanced_1"
                path_model_imbalanced_1="saved_models/imSitu/30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/imbalanced_1/model.pt"

                save_folder_imbalanced_2="30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/imbalanced_2"
                result_folder_imbalanced_2="results/imSitu/30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/imbalanced_2"
                path_model_imbalanced_2="saved_models/imSitu/30_verbs/grid_search_2/baseline/lr_${learning_rate_value}_momentum_${momentum_value}_step_${step_size_value}_gamma_${gamma_value}/imbalanced_2/model.pt"
                
                echo "Baseline"

                python ${train_baseline_path} --dataset 30_verbs/train_val_split/train_balanced --save_folder $save_folder_balanced --lr $learning_rate_value --momentum $momentum_value --step_size $step_size_value --gamma $gamma_value #--num_epochs 1
                python ${test_path} --path_model $path_model_balanced --path_result $result_folder_balanced --path_test_dataset data/datasets/imSitu/data/30_verbs/test_with_gender --num_classes 30 --path_list_verbs data/classes/imSitu_30_classes.txt

                echo "Imbalanced 1"

                python ${train_baseline_path} --dataset 30_verbs/train_val_split/train_imbalanced_1 --save_folder $save_folder_imbalanced_1 --lr $learning_rate_value --momentum $momentum_value --step_size $step_size_value --gamma $gamma_value # --num_epochs 1
                python ${test_path} --path_model $path_model_imbalanced_1 --path_result $result_folder_imbalanced_1 --path_test_dataset data/datasets/imSitu/data/30_verbs/test_with_gender --num_classes 30 --path_list_verbs data/classes/imSitu_30_classes.txt
                
                echo "Imbalanced 2"

                python ${train_baseline_path} --dataset 30_verbs/train_val_split/train_imbalanced_2 --save_folder $save_folder_imbalanced_2 --lr $learning_rate_value --momentum $momentum_value --step_size $step_size_value --gamma $gamma_value # --num_epochs 1
                python ${test_path} --path_model $path_model_imbalanced_2 --path_result $result_folder_imbalanced_2 --path_test_dataset data/datasets/imSitu/data/30_verbs/test_with_gender --num_classes 30 --path_list_verbs data/classes/imSitu_30_classes.txt
            done
        done
    done
done