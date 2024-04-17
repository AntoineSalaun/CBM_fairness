result_str = ""
for i in range(1,11):
    result_str += "#" + str(i) + "\n"
    result_str += "python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_imbalanced/visual/visual_{0}/imSitu_200_daniel_imbalanced_visual_{0}_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_imbalanced/visual/visual_{0}/ --path_test_dataset data/datasets/imSitu/data/200_verbs_full/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt".format(str(i)) + "\n"
    result_str += "python fairness_cv_project/datasets/imSitu/model_training/test_resnet.py --is_cbm --path_model saved_models/imSitu/200_verbs_daniel/200_verbs_balanced/visual/visual_{0}/imSitu_200_daniel_balanced_visual_{0}_imSitu_200_filtered --path_result results/imSitu/200_verbs_daniel/new_model_old_data/200_verbs_balanced/visual/visual_{0}/ --path_test_dataset data/datasets/imSitu/data/200_verbs/test_with_gender --num_classes 200 --path_list_verbs data/classes/imSitu_200_classes.txt".format(str(i)) + "\n"
    result_str += "\n"

f = open("./dododo.txt", "a")
f.write(result_str)
f.close()
