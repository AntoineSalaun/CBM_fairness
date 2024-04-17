#!/bin/bash
#SBATCH -o slurm_logs/old_model_new_data.sh.log-%j
#SBATCH --gres=gpu:volta:1

# # Train and test baseline imbalanced
# python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 200_verbs_full/train_val --save_folder 200_verbs_full/baseline/ --lr 0.001 --momentum 0.8 --step_size 5 --gamma 0.1
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/200_verbs_full/baseline/model.pt --path_result results/imSitu/200_verbs_full/baseline --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

# # Train and test baseline balanced
python fairness_cv_project/datasets/imSitu/model_training/train_resnet.py --dataset 200_verbs/train_val_split/train_balanced --save_folder 200_verbs/balanced/baseline --lr 0.001 --momentum 0.8 --step_size 5 --gamma 0.1
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/200_verbs/balanced/baseline/model.pt --path_result results/imSitu/200_verbs/balanced/baseline --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt


# # Train and test CBM imbalanced
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_full  --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs_full/CBM --interpretability_cutoff 0.3 --n_iters 80 --lam 0.0007 --print
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_full/CBM/imSitu_200_full_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_full/CBM --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

# Train and test CBM balanced
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_balanced --backbone resnet50 --concept_set data/concept_sets/imSitu_200_filtered.txt --save_dir saved_models/imSitu/200_verbs/balanced/CBM --interpretability_cutoff 0.3 --n_iters 80 --lam 0.0007 --print
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs/balanced/CBM/imSitu_200_balanced_balanced_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --path_result results/imSitu/200_verbs/balanced/CBM --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt



# # Test baseline imbalanced
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/200_verbs_full/baseline/model.pt --path_result results/imSitu/200_verbs_daniel/old_model_new_data/200_verbs_imbalanced/baseline --path_test_dataset data/datasets/imSitu/data/200_verbs_daniel/200_verbs_imbalanced/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

# # Test baseline balanced
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/baseline/model.pt --path_result results/imSitu/200_verbs_daniel/200_verbs_balanced/baseline --path_test_dataset data/datasets/imSitu/data/200_verbs_daniel/200_verbs_balanced/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt



# # Test CBM imbalanced
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/CBM/imSitu_200_daniel_imbalanced_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_daniel/200_verbs_imbalanced/test_with_gender --path_result results/imSitu/200_verbs_daniel/200_verbs_imbalanced/CBM --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

# # Test CBM balanced
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/CBM/imSitu_200_daniel_balanced_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_daniel/200_verbs_balanced/test_with_gender --path_result results/imSitu/200_verbs_daniel/200_verbs_balanced/CBM --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
