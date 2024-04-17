#!/bin/bash
#SBATCH -o slurm_logs/grid_search_resnet_200_verbs_cbm_gender.sh.log-%j
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=40


train_baseline_path=fairness_cv_project/datasets/imSitu/model_training/train_resnet.py
train_path=fairness_cv_project/methods/label_free_cbm/src/train_cbm.py
test_path=fairness_cv_project/datasets/imSitu/model_training/test_resnet.py
explainability_path=fairness_cv_project/methods/label_free_cbm/src/explainability.py

#interpretability_cutoffs=(0 0.1 0.3 0.5)
#lams=(0.00001 0.0007 0.007 0.07)
# clip_cutoffs=(0 0.1 0.2 0.3)
interpretability_cutoffs=(0.1)
lams=(0.07)
clip_cutoffs=(0.2)

# Loop over hyperparameters
for interpretability_cutoff in "${interpretability_cutoffs[@]}"; do
    for lam in "${lams[@]}"; do
        for clip_cutoff in "${clip_cutoffs[@]}"; do
            echo "______________________________________________________________________"
            echo "Interpretability cutoff: $interpretability_cutoff"
            echo "Lambda: $lam"
            echo "Clip cutoff: $clip_cutoff"

            # No Gender
            save_dir_gender_balanced="saved_models/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/balanced"
            save_dir_gender_imbalanced_1="saved_models/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/imbalanced_1"
            save_dir_gender_imbalanced_2="saved_models/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/imbalanced_2"

            echo "Training"
            python ${train_path} --dataset imSitu_200_balanced --backbone resnet50 --concept_set data/concept_sets/imSitu_200_gender.txt --save_dir $save_dir_gender_balanced --interpretability_cutoff $interpretability_cutoff --n_iters 80 --lam $lam --clip_cutoff $clip_cutoff --protected_concepts "a male" "a female"

            python ${train_path} --dataset imSitu_200_imbalanced_1 --backbone resnet50 --concept_set data/concept_sets/imSitu_200_gender.txt --save_dir $save_dir_gender_imbalanced_1 --interpretability_cutoff $interpretability_cutoff --n_iters 80 --lam $lam --clip_cutoff $clip_cutoff --protected_concepts "a male" "a female"

            python ${train_path} --dataset imSitu_200_imbalanced_2 --backbone resnet50 --concept_set data/concept_sets/imSitu_200_gender.txt --save_dir $save_dir_gender_imbalanced_2 --interpretability_cutoff $interpretability_cutoff --n_iters 80 --lam $lam --clip_cutoff $clip_cutoff --protected_concepts "a male" "a female"

            echo "Deleting activations"

            rm -rf saved_activations/imSitu_200_balanced_train_imSitu_200_gender
            rm -rf saved_activations/imSitu_200_balanced_val_imSitu_200_gender
            rm -rf saved_activations/imSitu_200_imbalanced_1_train_imSitu_200_gender
            rm -rf saved_activations/imSitu_200_imbalanced_1_val_imSitu_200_gender
            rm -rf saved_activations/imSitu_200_imbalanced_2_train_imSitu_200_gender
            rm -rf saved_activations/imSitu_200_imbalanced_2_val_imSitu_200_gender


            echo "Testing"
            saved_model_gender_balanced="saved_models/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/balanced/imSitu_200_balanced_imSitu_200_gender"
            path_result_gender_balanced="results/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/balanced" 

            saved_model_gender_imbalanced_1="saved_models/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/imbalanced_1/imSitu_200_imbalanced_1_imSitu_200_gender"
            path_result_gender_imbalanced_1="results/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/imbalanced_1" 

            saved_model_gender_imbalanced_2="saved_models/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/imbalanced_2/imSitu_200_imbalanced_2_imSitu_200_gender"
            path_result_gender_imbalanced_2="results/imSitu/200_verbs/grid_search/CBM/gender/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}/imbalanced_2" 


            python ${test_path} --is_cbm --path_model $saved_model_gender_balanced --path_result $path_result_gender_balanced --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

            python ${test_path} --is_cbm --path_model $saved_model_gender_imbalanced_1 --path_result $path_result_gender_imbalanced_1 --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

            python ${test_path} --is_cbm --path_model $saved_model_gender_imbalanced_2 --path_result $path_result_gender_imbalanced_2 --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

        done
    done
done

