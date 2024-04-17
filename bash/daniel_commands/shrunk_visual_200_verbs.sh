#!/bin/bash
#SBATCH -o slurm_logs/shrunk_visual_200_verbs.sh.log-%j
#SBATCH --gres=gpu:volta:1

#1
python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_1_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_1/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_1/200_verbs_shrunk/imSitu_200_shrunk_visual_1_imSitu_200_filtered --path_result results/imSitu/visual/visual_1/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #2
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_2_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_2/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_2/200_verbs_shrunk/imSitu_200_shrunk_visual_2_imSitu_200_filtered --path_result results/imSitu/visual/visual_2/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #3
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_3_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_3/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_3/200_verbs_shrunk/imSitu_200_shrunk_visual_3_imSitu_200_filtered --path_result results/imSitu/visual/visual_3/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #4
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_4_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_4/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_4/200_verbs_shrunk/imSitu_200_shrunk_visual_4_imSitu_200_filtered --path_result results/imSitu/visual/visual_4/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #5
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_5_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_5/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_5/200_verbs_shrunk/imSitu_200_shrunk_visual_5_imSitu_200_filtered --path_result results/imSitu/visual/visual_5/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #6
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_6_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_6/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_6/200_verbs_shrunk/imSitu_200_shrunk_visual_6_imSitu_200_filtered --path_result results/imSitu/visual/visual_6/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #7
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_7_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_7/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_7/200_verbs_shrunk/imSitu_200_shrunk_visual_7_imSitu_200_filtered --path_result results/imSitu/visual/visual_7/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #8
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_8_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_8/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_8/200_verbs_shrunk/imSitu_200_shrunk_visual_8_imSitu_200_filtered --path_result results/imSitu/visual/visual_8/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #9
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_9_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_9/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_9/200_verbs_shrunk/imSitu_200_shrunk_visual_9_imSitu_200_filtered --path_result results/imSitu/visual/visual_9/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 

# #10
# python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset imSitu_200_shrunk --backbone resnet50 --concept_set data/concept_sets/visual_concept_sets/visual_10_imSitu_200_filtered.txt --save_dir saved_models/imSitu/visual/visual_10/200_verbs_shrunk --interpretability_cutoff 0.0 --n_iters 80 --lam 0.0007 --clip_cutoff 0 --print 
# python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/visual/visual_10/200_verbs_shrunk/imSitu_200_shrunk_visual_10_imSitu_200_filtered --path_result results/imSitu/visual/visual_10/200_verbs_shrunk/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt 
