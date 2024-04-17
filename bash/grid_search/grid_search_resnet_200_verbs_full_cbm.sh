#!/bin/bash
#SBATCH -o slurm_logs/grid_search_resnet_200_verbs_full_cbm.sh.log-%j
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=40


train_path=fairness_cv_project/methods/label_free_cbm/src/train_cbm.py
test_path=fairness_cv_project/datasets/imSitu/model_training/test_resnet.py
explainability_path=fairness_cv_project/methods/label_free_cbm/src/explainability.py

# Train and Test resnet 200 verbs full

interpretability_cutoffs=(0 0.1 0.3 0.5)
lams=(0.00001 0.0007 0.007 0.07)
clip_cutoffs=(0 0.1 0.2 0.3)

# interpretability_cutoffs=(0.1)
# lams=(0.00001)
# lip_cutoffs=(0.2 0.25)
# Loop over hyperparameters
for interpretability_cutoff in "${interpretability_cutoffs[@]}"; do
    for lam in "${lams[@]}"; do
        for clip_cutoff in "${clip_cutoffs[@]}"; do
            echo "______________________________________________________________________"
            echo "Interpretability cutoff: $interpretability_cutoff"
            echo "Lambda: $lam"
            echo "Clip cutoff: $clip_cutoff"
            # Define name suffix for saved model
            save_dir_gender="saved_models/imSitu/200_verbs_full/grid_search/CBM/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}_gender"
            save_dir_no_gender="saved_models/imSitu/200_verbs_full/grid_search/CBM/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}_no_gender"


            echo "Training"

            python ${train_path} --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir $save_dir_no_gender --interpretability_cutoff $interpretability_cutoff --n_iters 80 --lam $lam --clip_cutoff $clip_cutoff

            python ${train_path} --dataset imSitu_200_full --backbone resnet50 --concept_set data/concept_sets/imSitu_200_gender.txt --save_dir $save_dir_gender --interpretability_cutoff $interpretability_cutoff --n_iters 80 --lam $lam --clip_cutoff $clip_cutoff --protected_concepts "a male" "a female"

            # Delete activation maps
            
            echo "Deleting activation maps"
            rm -rf saved_activations/imSitu_200_full_train_imSitu_200_filtered
            rm -rf saved_activations/imSitu_200_full_val_imSitu_200_gender
            rm -rf saved_activations/imSitu_200_full_train_imSitu_200_filtered
            rm -rf saved_activations/imSitu_200_full_val_imSitu_200_gender

            # Test
            saved_model_no_gender="saved_models/imSitu/200_verbs_full/grid_search/CBM/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}_no_gender/imSitu_200_full_imSitu_200_filtered"
            path_result_no_gender="results/imSitu/200_verbs_full/grid_search/CBM/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}_no_gender" 

            saved_model_gender="saved_models/imSitu/200_verbs_full/grid_search/CBM/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}_gender/imSitu_200_full_imSitu_200_gender"
            path_result_gender="results/imSitu/200_verbs_full/grid_search/CBM/ic${interpretability_cutoff}_lam${lam}_cc${clip_cutoff}_no_gender" 

            echo "Testing"
            python ${test_path} --is_cbm --path_model $saved_model_no_gender --path_result $path_result_no_gender --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

            python ${test_path} --is_cbm --path_model $saved_model_gender --path_result $path_result_gender --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

        done
    done
done