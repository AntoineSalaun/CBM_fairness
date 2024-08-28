import sys, torch, argparse
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))
from methods.cbm import CBM
import methods.data_loader as ImSituLoader
import torch.nn as nn
from methods.leakage import Leakage, dataset_leakage, compute_MLP_leakage_of_layer, CBM_leakage
import torch.optim as optim
from gender_classifier import GenderClassifier, GenderClassifierNet
import pandas as pd
import matplotlib.pyplot as plt
from methods.clip_DNN import Clip_DNN
from methods.clip_zero_shot import Clip_zero_shot
from methods.resnet import ResNet


parser = argparse.ArgumentParser(description='Settings for creating Clip DNN')
parser.add_argument(f"--experiment", type=str, default=None) 
parser.add_argument(f"--loading", type=str, default=False)  
parser.add_argument(f"--zeta", type=float, default=False)       
args = parser.parse_args()

if __name__=='__main__':

    if args.experiment == None:
        
        exp_balance = 'balanced'
        print('=========================================',exp_balance,'=====================================================')
        #print('=========================================ResNet LEKAGE =====================================================')
        #resnet = ResNet(parsing = False, balance = exp_balance, num_epochs=25)
        #resnet.train(True)
        #CBM_leakage(resnet, retrain = False, concept = False)
        

        
        #print('=========================================Clip_DNN LEKAGE, lam =0  =====================================================')
        #clip_dnn = Clip_DNN(parsing = False, balance = exp_balance, batch_size = 800, lam = 0.0, n_iters = 2000, alpha = 0.99, lr = 1, experiment = None, target = 'verb')
        #clip_dnn.train(True)
        #CBM_leakage(clip_dnn, retrain = False, concept = False )

        #print('=========================================CBM not sparse lam = 0 =====================================================')
        #cbm = CBM(parsing = False, interpretability_cutoff = 0.28, lam=0, n_iters=2000, balance=exp_balance, adversarial=False, start_adv=2000, zeta = args.zeta)
        #cbm.train(False)
        #CBM_leakage(cbm, retrain = False)

        print('=========================================CBM sparse lam = 1e-3 =====================================================')
        cbm = CBM(parsing = False, interpretability_cutoff = 0.28, lam=1e-3, n_iters=2000, balance=exp_balance, adversarial=False, start_adv=2000, zeta = args.zeta)
        cbm.train(False)
        CBM_leakage(cbm, retrain = False)

        #print('=========================================Clip_zeroshot LEKAGE =====================================================')
        #clip_zs = Clip_zero_shot(parsing = False, dataset ='imSitu', which_clip = 'ViT-B-16', num_verb = 200, balance = exp_balance, batch_size = 800, experiment = None, target = 'verb')
        #CBM_leakage(clip_zs, retrain = False, concept = False)


    if args.experiment == 'leak_vs_zeta':
        
        results = pd.DataFrame(columns = ['zeta', 'Test_Accuracy','probability_leakage','model_leakage'])
        zetas = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

        for z in zetas:
            cbm = CBM(parsing = False, lam = 1e-3, adversarial=True, zeta=z, balance = 'original', n_iters=3000, start_adv=2000)
            
            cbm.train(use_existing_embeddings=False)
            test_loss, test_acc, test_f1, nnz_mean = cbm.test(use_existing_embeddings=False)
            
            train_prob_loader, val_prob_loader, test_prob_loader = cbm.leakage_loaders(layer='probabilities')
            prob_leakage, _ = compute_MLP_leakage_of_layer(train_prob_loader, val_prob_loader, test_prob_loader, leakage_type='probility_leakage', balance=cbm.balance)
            
            train_pred_loader, val_pred_loader, test_pred_loader = cbm.leakage_loaders(layer='predictions')
            model_leakage, _ = compute_MLP_leakage_of_layer(train_pred_loader, val_pred_loader, test_pred_loader, leakage_type='model_leakage', balance=cbm.balance)

            results = results.append({'zeta': z, 'Test_Accuracy': test_acc, 'probability_leakage': prob_leakage, 'model_leakage':model_leakage}, ignore_index=True)
        
        print(results)
        results.to_csv(cbm.path_save / 'leak_vs_zeta.csv', index=False)      

        plt.plot(results['zeta'], results['Test_Accuracy'])
        plt.xlabel('zeta')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs adversarial parameter')
        plt.savefig(cbm.path_save / 'leak_vs_zeta.png')


    if args.experiment == 'does_cbm_learn_gender':        
        
        ZS = Clip_zero_shot(parsing = False, balance = 'balanced', target = 'gender')
        test_loss, ZS_g, test_f1, _ = ZS.test(True)

        Clip_CBM_non_sparse = CBM(parsing = False, lam = 0, adversarial=False, balance = 'balanced', target = 'gender', n_iters=500)
        Clip_CBM_non_sparse.train(True)
        test_loss, CBM0_g, test_f1, _ = Clip_CBM_non_sparse.test(True)

        Clip_CBM_sparse = CBM(parsing = False, lam = 1e-3, adversarial=False, balance = 'balanced', target = 'gender', n_iters=500)
        Clip_CBM_sparse.train(True)
        test_loss, CBM3_g, test_f1,_= Clip_CBM_sparse.test(True)
        
        Clip_DNN_non_sparse = Clip_DNN(parsing = False, lam = 0, balance = 'balanced', target = 'gender', n_iters=500)
        Clip_DNN_non_sparse.train(True)
        test_loss, DNN0_g, test_f1, _ = Clip_DNN_non_sparse.test(True)

        Clip_DNN_sparse = Clip_DNN(parsing = False, lam = 1e-3, balance = 'balanced', target = 'gender', n_iters=500)
        Clip_DNN_sparse.train(True)
        test_loss, DNN3_g, test_f1, _ = Clip_DNN_sparse.test(True)

        
        
        ZS = Clip_zero_shot(parsing = False, balance = 'balanced')
        test_loss, ZS_v, test_f1, _ = ZS.test(True)

        Clip_CBM_non_sparse = CBM(parsing = False, lam = 0, adversarial=False, balance = 'balanced', n_iters=2000)
        Clip_CBM_non_sparse.train(True)
        test_loss, CBM0_v, test_f1, _ = Clip_CBM_non_sparse.test(True)

        Clip_CBM_sparse = CBM(parsing = False, lam = 1e-3, adversarial=False, balance = 'balanced', n_iters=2000)
        Clip_CBM_sparse.train(True)
        test_loss, CBM3_v, test_f1,_= Clip_CBM_sparse.test(True)
        
        Clip_DNN_non_sparse = Clip_DNN(parsing = False, lam = 0, balance = 'balanced', n_iters=2000)
        Clip_DNN_non_sparse.train(True)
        test_loss, DNN0_v, test_f1, _ = Clip_DNN_non_sparse.test(True)

        Clip_DNN_sparse = Clip_DNN(parsing = False, lam = 1e-3, balance = 'balanced', n_iters=2000)
        Clip_DNN_sparse.train(True)
        test_loss, DNN3_v, test_f1, _ = Clip_DNN_sparse.test(True)

        print('=======================RESULTS============================')
        print('Zero-shot - gender :', ZS_g, '- verb :', ZS_v)
        print('Clip_CBM_non_sparse - gender :', CBM0_g, '- verb :', CBM0_v)
        print('Clip_CBM_sparse - gender : ', CBM3_g, '- verb :', CBM3_v)
        print('Clip_DNN_non_sparse - gender : ', DNN0_g, '- verb :', DNN0_v)
        print('Clip_DNN_sparse - gender : ', DNN3_g, '- verb :', DNN3_v)

    if args.experiment == 'find_fair_cbm':
        '''
        print ('=========================================Finding a fair cbm (original) =====================================================')
        print('=========================================CBM + sparse concepts 50 =====================================================')
        cbm_5 = CBM(parsing = False, lam = 1e-3, adversarial=False, n_iters=2000, balance = 'original', sparse_concepts= 50)
        cbm_5.train(False)
        CBM_leakage(cbm_5, retrain = False, data_MLP = False, concept = False, model_MLP=False)


        print('=========================================CBM + adversarial 1e-2 =====================================================')
        cbm_1 = CBM(parsing = False, lam = 1e-3, adversarial=True, n_iters=3000, start_adv=2000, balance = 'original', zeta = 1e-2)
        cbm_1.train(False)
        #CBM_leakage(cbm_1, retrain = False, data_MLP = False, concept = False, model_MLP=False)
        

        print('=========================================CBM + adversarial 1e-2 + block 200 =====================================================')
        cbm_gender = CBM(parsing=False, target='gender',lam=1e-3, n_iters=500, balance = 'original')
        cbm_gender.train(use_existing_embeddings=True)
        cbm_gender.test(use_existing_embeddings=True)
        df_female, df_male = cbm_gender.extract_most_biased_concepts()
        block_how_many = 100
        block_concepts_idx = list(df_female['concept_idx'][:block_how_many]) + list(df_male['concept_idx'][:block_how_many])
        print('block_concepts_idx:', block_concepts_idx)
        
        #block_concepts_idx = [762, 491, 610, 259, 567, 577, 47, 711, 1064, 883, 194, 131, 474, 1572, 905, 380, 1010, 578, 1105, 1553, 141, 1611, 331, 1746, 538, 1425, 1610, 1514, 820, 1742, 422, 1186, 1764, 85, 507, 1773, 843, 546, 632, 797, 441, 729, 206, 618, 77, 530, 1588, 1241, 1757, 690, 998, 1165, 1003, 1135, 669, 1247, 514, 1664, 107, 707, 1533, 1243, 173, 487, 1287, 1733, 1329, 93, 1381, 1149, 1321, 1637, 1316, 289, 874, 719, 1313, 69, 675, 1639, 1770, 1207, 51, 976, 1740, 670, 420, 1273, 54, 456, 832, 192, 221, 662, 335, 911, 816, 1724, 977, 1339, 150, 727, 819, 442, 440, 299, 338, 475, 256, 906, 1252, 598, 201, 852, 1315, 1712, 1469, 1672, 124, 34, 1256, 1230, 1372, 743, 1371, 1158, 1086, 1570, 1376, 1391, 633, 39, 246, 211, 634, 829, 1041, 128, 991, 477, 1589, 485, 1161, 170, 994, 1698, 238, 1459, 463, 909, 1073, 752, 493, 330, 394, 639, 1355, 212, 619, 524, 1442, 1648, 1565, 304, 1771, 1017, 636, 1024, 1521, 139, 512, 1116, 513, 1343, 286, 1044, 466, 1254, 68, 1206, 348, 454, 869, 1328, 323, 1054, 1454, 1266, 486, 968, 497, 714, 1222, 496, 1456, 261, 228, 1409, 55, 1512]
        
        cbm_2 = CBM(parsing = False, lam = 1e-3, adversarial=True, n_iters=3000, start_adv=2000, balance = 'original', zeta = 1e-2, block = block_concepts_idx)       
        cbm_2.train(True)
        CBM_leakage(cbm_2, retrain = False, data_MLP = False, concept = False, model_MLP=False)
        '''
        
        print('=========================================learning biased concepts =====================================================')
        #cbm_gender = CBM(parsing=False, target='gender',lam=1e-3, n_iters=500, balance = 'original')
        #cbm_gender.train(use_existing_embeddings=True)
        #cbm_gender.test(use_existing_embeddings=True)
        #df_female, df_male = cbm_gender.extract_most_biased_concepts()
        #block_how_many = 100
        #block_concepts_idx = list(df_female['concept_idx'][:block_how_many]) + list(df_male['concept_idx'][:block_how_many])
        #print('block_concepts_idx:', block_concepts_idx)
        #print('d_female', df_female)
        #print('d_male',df_male)
        
        block_concepts_idx = [488, 391, 362, 425, 343, 874, 369, 663, 119, 41, 596, 312, 478, 741, 737, 84, 342, 531, 515, 378, 569, 4, 829, 73, 963, 908, 304, 900, 765, 704, 381, 540, 520, 287, 414, 850, 846, 327, 117, 151, 923, 689, 475, 313, 74, 100, 61, 183, 859, 95, 993, 716, 888, 935, 113, 396, 646, 350, 81, 591, 560, 220, 386, 504, 505, 49, 423, 745, 685, 647, 895, 742, 945, 305, 525, 452, 967, 521, 292, 71, 541, 286, 537, 50, 961, 777, 909, 168, 307, 193, 196, 706, 612, 7, 36, 198, 795, 735, 208, 709, 472, 649, 641, 642, 493, 322, 543, 104, 575, 814, 756, 82, 146, 820, 299, 490, 516, 944, 703, 784, 395, 696, 62, 781, 33, 47, 333, 904, 518, 776, 725, 969, 654, 253, 610, 1002, 602, 563, 970, 190, 489, 140, 174, 456, 595, 236, 675, 115, 579, 411, 665, 144, 564, 959, 317, 71, 686, 614, 906, 976, 971, 547, 6, 834, 633, 767, 399, 39, 293, 715, 58, 44, 72, 979, 828, 951, 978, 237, 482, 407, 630, 131, 947, 436, 60, 275, 56, 153, 851, 796, 126, 1003, 858, 134, 306, 212, 254, 452, 875, 327]
        

        print('=========================================CBM + sparse concepts 100 =====================================================')
        cbm_6 = CBM(parsing = False, lam = 1e-3, adversarial=False, n_iters=2000, balance = 'original', sparse_concepts= 100)
        cbm_6.train(True)
        CBM_leakage(cbm_6, retrain = False, data_MLP = False, concept = False, model_MLP=False)


        print('=========================================CBM + sparse concepts 100 + block 200 =====================================================')  
        cbm_7 = CBM(parsing = False, lam = 1e-3, adversarial=False, n_iters=2000, balance = 'original', sparse_concepts= 100, block = block_concepts_idx)
        #cbm_7.train(False)
        CBM_leakage(cbm_7, retrain = False, data_MLP = False, concept = False, model_MLP=False)

        
        print('=========================================CBM + adversarial 1e-2 + sparse concepts 100 =====================================================')
        cbm_3 = CBM(parsing = False, lam = 1e-3, adversarial=True, n_iters=3000, start_adv=2000, balance = 'original', zeta = 1e-2, sparse_concepts= 100)
        cbm_3.train(True)
        CBM_leakage(cbm_3, retrain = False, data_MLP = False, concept = False, model_MLP=False)
        
        print('=========================================CBM + adversarial 1e-2 + sparse concepts 100 + block 200 =====================================================')
        cbm_3 = CBM(parsing = False, lam = 1e-3, adversarial=True, n_iters=3000, start_adv=2000, balance = 'original', zeta = 1e-2, sparse_concepts= 100, block = block_concepts_idx)
        CBM_leakage(cbm_6, retrain = False, data_MLP = False, concept = False, model_MLP=False)






