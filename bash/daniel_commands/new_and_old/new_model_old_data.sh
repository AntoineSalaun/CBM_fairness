#!/bin/bash
#SBATCH -o slurm_logs/new_model_old_data.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Test baseline imbalanced
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/baseline/model.pt --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/baseline --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

# Test baseline balanced
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/baseline/model.pt --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/baseline --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt



# Test CBM imbalanced
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/CBM/imSitu_200_daniel_imbalanced_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/CBM --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

# Test CBM balanced
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/CBM/imSitu_200_daniel_balanced_imSitu_200_filtered  --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/CBM --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt



# Train and test visual scores CBM
#1
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_1/imSitu_200_daniel_imbalanced_visual_1_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_1/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_1/imSitu_200_daniel_balanced_visual_1_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_1/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#2
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_2/imSitu_200_daniel_imbalanced_visual_2_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_2/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_2/imSitu_200_daniel_balanced_visual_2_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_2/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#3
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_3/imSitu_200_daniel_imbalanced_visual_3_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_3/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_3/imSitu_200_daniel_balanced_visual_3_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_3/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#4
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_4/imSitu_200_daniel_imbalanced_visual_4_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_4/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_4/imSitu_200_daniel_balanced_visual_4_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_4/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#5
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_5/imSitu_200_daniel_imbalanced_visual_5_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_5/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_5/imSitu_200_daniel_balanced_visual_5_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_5/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#6
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_6/imSitu_200_daniel_imbalanced_visual_6_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_6/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_6/imSitu_200_daniel_balanced_visual_6_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_6/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#7
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_7/imSitu_200_daniel_imbalanced_visual_7_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_7/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_7/imSitu_200_daniel_balanced_visual_7_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_7/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#8
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_8/imSitu_200_daniel_imbalanced_visual_8_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_8/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_8/imSitu_200_daniel_balanced_visual_8_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_8/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#9
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_9/imSitu_200_daniel_imbalanced_visual_9_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_9/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_9/imSitu_200_daniel_balanced_visual_9_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_9/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt

#10
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_10/imSitu_200_daniel_imbalanced_visual_10_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_10/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_10/imSitu_200_daniel_balanced_visual_10_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_10/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt