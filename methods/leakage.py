import torch, sys, copy
from pathlib import Path

sys.path.insert(1, str(Path.cwd()))
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
#from torch.utils.data import DataLoader
import pandas as pd
#from sklearn.metrics import f1_score
import numpy as np
import argparse
from sklearn.metrics import f1_score
import os



from tqdm import tqdm as tqdm

from data_loader import ImSituLoader
from resnet import ResNet
from gender_classifier import GenderClassifier, GenderClassifierNet
from methods.cbm import CBM
from methods.clip_DNN import Clip_DNN
from methods.clip_zero_shot import Clip_zero_shot
from scipy import stats


class Leakage():
    def __init__(self, parsing=True, leakage_type='dataset_leakage', experiment=None, predictor='deterministic', \
                 attacked_model='CBM', perturbation_rate=0.0, balance='original', dataset='imSitu', \
                    dataset_dir='data/datasets/imSitu/', num_verb=200, device='cuda', num_epochs=10, batch_size=512, \
                        learning_rate=3e-3, hid_size=150, print_every=1, weight_decay=1e-2, block_how_many=50):

        self.params = [
            ("leakage_type", str, None, leakage_type),
            ("experiment", str, None, experiment),
            ("predictor", str, None, predictor),
            ("attacked_model", str, None, attacked_model),
            ("perturbation_rate", float, None, perturbation_rate),
            ("balance", str, None, balance),
            ("dataset", str, None, dataset),
            ("dataset_dir", str, None, dataset_dir),
            ("num_verb", int, None, num_verb),
            ("device", str, None, device),
            ("num_epochs", int, None, num_epochs),
            ("batch_size", int, None, batch_size),
            ("learning_rate", float, None, learning_rate),
            ("hid_size", int, None, hid_size),
            ("print_every", int, None, print_every),
            ("weight_decay", float, None, weight_decay),
            ("block_how_many", int, None, block_how_many)
        ] 
        if parsing == True :       
            parser = argparse.ArgumentParser(description='Settings for creating CBM')
            for param in self.params:
                parser.add_argument(f"--{param[0]}", type=param[1], default=param[2])
            args = parser.parse_args()

        self.arg_str = ""
        for param in self.params: # for each argument
            if parsing == True and getattr(args, param[0]) is not None: # if I can parse an argument
                setattr(self, param[0], getattr(args, param[0])) # Then I use it in self
                self.arg_str += f"{param[0]}: {str(getattr(args, param[0]))}\n"
            else:
                setattr(self, param[0], param[3]) # Other wise I use the default value from the __init__ function
                self.arg_str += f"{param[0]}: {param[3]}\n"
        
        # Write self.arg_str to a text file
        with open('arg_str.txt', 'w') as file: file.write(self.arg_str)

        self.num_verbs_str = str(self.num_verb) + '_verbs'

        if self.perturbation_rate > 0 and self.leakage_type == 'model_leakage':
            raise ValueError('You cannot use perturbation with model_leakage')



        # Store the datasets inside the class
        
    def compute_leakage(self):
        if self.leakage_type == 'dataset_leakage':
            self.train_data = ImSituLoader(perturbation_rate=self.perturbation_rate, balance=self.balance,dataset_dir=self.dataset_dir,  split = 'train')
            self.val_data = ImSituLoader(perturbation_rate=self.perturbation_rate, balance=self.balance, dataset_dir=self.dataset_dir, split = 'val')
            self.test_data = ImSituLoader(perturbation_rate=self.perturbation_rate, balance=self.balance, dataset_dir=self.dataset_dir,  split = 'test')

        in_features = len(self.train_data[1][1])
        print('Gender classifier MLP takes as in_Features :', len(self.train_data[1][1]))

        self.AttackerNet = GenderClassifierNet(in_features, self.hid_size).to(self.device)
        self.AttackerOptimizer = optim.Adam(self.AttackerNet.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        self.AttackerModel = GenderClassifier(predictor = self.predictor, net = self.AttackerNet, optimizer = self.AttackerOptimizer, leakage_type=self.leakage_type, balance = self.balance, num_epochs=self.num_epochs)
        
        # I got CUDA error Initialization error, so I changed num_workers from 6 to 0 and that worked
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size, shuffle = False, num_workers = 4, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size, shuffle = False, num_workers = 4, pin_memory = True)
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size = self.batch_size, shuffle = False, num_workers = 4, pin_memory = True)

        # trains the model and saves the best model
        train_metrics, train_acc, train_f1, val_acc, val_f1 = self.AttackerModel.train(train_loader, val_loader)

        # Test
        test_acc, test_f1 = self.AttackerModel.test(test_loader)

        self.AttackerModel.save_results(train_metrics, train_acc, train_f1, val_acc, val_f1, test_acc, test_f1)

        return train_acc, train_f1, test_acc, test_f1, val_acc, val_f1
    
    def predict_verbs_resnet(self):

        self.AttackedModel = ResNet(self.AttackerOptimizer)
        path_resnet_dict = Path.cwd() / 'saved_models' / self.dataset / self.num_verbs_str  / 'ResNet' / self.balance /'model_best.pt'
        self.AttackedModel.load_weights(path_resnet_dict)

        for data in [self.train_data, self.test_data, self.val_data]:
            loader = torch.utils.data.DataLoader(data, batch_size = self.batch_size, shuffle = False, num_workers = 0, pin_memory = True)

            _, acc, _, predicted_verbs = self.AttackedModel.epoch_pass(-1, loader, training = False)

            print('We used ', path_resnet_dict, ' to infer ',  data ,'- Accuracy : ', acc)
            
            data.change_verbs(predicted_verbs)

def dataset_leakage(rounds = 5, perturbation = None, predictor = 'deterministic', balance = 'imbalanced'):
    
    leak = Leakage(parsing = False, balance=balance, leakage_type='dataset_leakage', perturbation_rate=perturbation, predictor=predictor)

    test_accuracies = [] 
    if perturbation is None: perturbation = leak.perturbation_rate
    if predictor is not None: leak.predictor = predictor

    for round in range(rounds): 
        _, _, test_acc, _, _, _ = leak.compute_leakage() 
        print('round : ', round, 'dataset_leakage : ', test_acc)
        test_accuracies.append(test_acc)
    
    confidence = 0.95
    test_accuracies = np.array(test_accuracies)
    mean = np.mean(test_accuracies)
    std = np.std(test_accuracies)
    n = len(test_accuracies)
    margin_of_error = stats.t.ppf((1 + confidence) / 2, n - 1) * (std / np.sqrt(n))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    print('------------RESULTS------------')
    print("Mean:", mean)
    print("Standard Deviation:", std)
    print("Confidence Interval:", confidence_interval)

    return mean, std, confidence_interval

def compute_MLP_leakage_of_layer(train_layer, val_layer, test_layer, leakage_type, balance, wd = 1e-1): # ImSituLoader + layer as the target

    wd = 1e-2 if leakage_type in ['dataset_leakage','model_leakage'] else 1e-1

    leakage = Leakage(parsing = False, leakage_type=leakage_type, predictor='MLP', hid_size = 300, num_epochs=10, balance=balance, weight_decay=wd)
    print('leakage balance is ', leakage.balance)
    leakage.train_data, leakage.val_data, leakage.test_data = train_layer, val_layer, test_layer
    _, _, test_acc, _, _, _ = leakage.compute_leakage()

    return test_acc, leakage.AttackerModel.net

def CBM_leakage(cbm , retrain = True, data_MLP = True, data_deter = True, concept = True, prob = True, model_MLP = True, model_deter = True):
    
    # ATTENTION MODIFIe
    data_leakage, data_leakage_at_F1, concept_leakage, prob_leakage, model_leakage = 0,0,0, 0, 0
    probabilities_to_gender, preds_to_gender = None, None

        
    if retrain: cbm.train(True)
    _, test_acc, _, nnz = cbm.test(False)

    if data_MLP: data_leakage_MLP_at_F1, _, data_leakage_MLP_at_F1_CI = dataset_leakage(balance=cbm.balance, predictor='MLP', perturbation=(1-test_acc))
    if data_deter: data_leakage_deter_at_F1, _, data_leakage_deter_at_F1_CI = dataset_leakage(balance=cbm.balance, predictor='deterministic', perturbation=(1-test_acc))

    if concept: 
        train_concept_loader, val_concept_loader, test_concept_loader = cbm.leakage_loaders(layer='concept')
        concept_leakage, _ = compute_MLP_leakage_of_layer(train_concept_loader, val_concept_loader, test_concept_loader,balance=cbm.balance,  leakage_type='concept_leakage')

    if prob:
        train_prob_loader, val_prob_loader, test_prob_loader = cbm.leakage_loaders(layer='probabilities')
        prob_leakage, probabilities_to_gender = compute_MLP_leakage_of_layer(train_prob_loader, val_prob_loader, test_prob_loader, balance=cbm.balance, leakage_type='probability_leakage', wd=1e-1)

    if model_MLP:
        train_pred_loader, val_pred_loader, test_pred_loader = cbm.leakage_loaders(layer='predictions')
        model_leakage, preds_to_gender = compute_MLP_leakage_of_layer(train_pred_loader, val_pred_loader, test_pred_loader, balance=cbm.balance, leakage_type='model_leakage', wd=1e-2)

    if model_deter:
        leakage = Leakage(parsing = False, leakage_type='model_leakage', predictor='deterministic', num_epochs=10, balance=cbm.balance)
        leakage.train_data, leakage.val_data, leakage.test_data = cbm.leakage_loaders(layer='predictions')
        _, _, model_leakage_deter, _, _, _ = leakage.compute_leakage()
        

    print('---------Completed leakage study---------')
    print('CBM accuracy : ', test_acc, 'nnz weights :', nnz)
    if data_MLP: print('Data leakage (MLP):', data_leakage_MLP_at_F1, 'CI:', data_leakage_MLP_at_F1_CI)
    if data_deter: print('Data leakage (deterministic):', data_leakage_deter_at_F1, 'CI:', data_leakage_deter_at_F1_CI)
    if concept: print('Concept leakage:', concept_leakage)
    if prob: print('Probability leakage', prob_leakage)
    if model_MLP: print('Model leakage:', model_leakage)
    if model_deter: print('Model leakage (deterministic):', model_leakage_deter)

    return data_leakage, data_leakage_at_F1, concept_leakage, prob_leakage, model_leakage, probabilities_to_gender, preds_to_gender

if __name__ == '__main__':

    leak = Leakage(parsing= True)

    if leak.leakage_type == 'model_leakage':
        print('model leakage')
        if leak.attacked_model == 'ResNet': model = ResNet(parsing=False)
        elif leak.attacked_model == 'CBM': model = CBM(parsing=False)
        elif leak.attacked_model == 'clip_DNN': model = Clip_DNN(parsing=False)
        elif leak.attacked_model == 'clip_zero_shot': model = Clip_zero_shot(parsing=False)

        leak.train_data, leak.val_data, leak.test_data = model.leakage_loaders()
        leak.compute_leakage()
    
    elif leak.leakage_type == 'dataset_leakage':

        if leak.experiment == 'perturbation' : 
            perturbation_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
            results_df = pd.DataFrame(columns=['perturbation', 'det_mean', 'det_std', 'det_intv', 'mlp_mean', 'mlp_std', 'mlp_intv'])

            dir_results = Path(f'saved_models/{leak.dataset}/{leak.num_verb}_verbs/{leak.leakage_type}/{leak.balance}/{leak.experiment}')
            if not os.path.exists(dir_results): os.makedirs(dir_results)

            for perturbation in perturbation_values:
                det_mean,det_std , det_intv = dataset_leakage(5,perturbation,'deterministic')
                mlp_mean, mlp_std, mlp_intv = 0,0,0 
                results_df = results_df.append({'perturbation': perturbation, 'det_mean': det_mean, 'det_std': det_std, \
                                'det_intv': det_intv, 'mlp_mean': mlp_mean, 'mlp_std': mlp_std, 'mlp_intv': mlp_intv}, ignore_index=True)

            with open(dir_results / 'args.txt', 'w') as f:f.write(str(leak))
            results_df.to_csv(dir_results/'results.csv', index=False)  

        elif leak.experiment == 'hyperparameters' : 
            results_df = pd.DataFrame(columns=['weight_decay', 'learning_rate','3_round_leakage'])
            dir_results = Path(f'saved_models/{leak.dataset}/{leak.num_verb}_verbs/{leak.leakage_type}/{leak.balance}/{leak.experiment}')
            if not os.path.exists(dir_results):
                os.makedirs(dir_results)

            for wd in [1e-2, 7e-3, 5e-3, 3e-3, 1e-3]: 
                for lr in [1e-2, 7e-3, 5e-3, 3e-3, 1e-3, 7e-4, 5e-4]:           
                    leak.weight_decay = wd
                    leak.learning_rate = lr
                    print('wd : ', wd, 'lr : ', lr )
                    test_acc_mean = dataset_leakage(3)
                    results_df = results_df.append({'weight_decay': wd, 'learning_rate': lr, '3_round_leakage': test_acc_mean}, ignore_index=True)
            
            results_df.to_csv(dir_results/'results.csv', index=False)  

        else: dataset_leakage(5, perturbation=0.5590, balance='original', predictor='deterministic')

    elif leak.leakage_type == 'concept_leakage':
        if leak.attacked_model == 'CBM':
            #python methods/leakage.py --attacked_model CBM --predictor deterministic --leakage_type model_leakage
            cbm = CBM(parsing=False, target='verb',lam=5e-3, balance = leak.balance)
            #cbm.train()
            #_, test_acc, test_f1 = cbm.test()
            #data_leakage, _, _ = dataset_leakage(5, 1-test_acc)
            
            
            matrix_train = cbm.matrix_train.cpu().unbind(dim=0)
            matrix_val = cbm.matrix_val.cpu().unbind(dim=0)
            matrix_test = cbm.matrix_test.cpu().unbind(dim=0)

            leak.train_data = cbm.train_data
            leak.val_data = cbm.val_data
            leak.test_data = cbm.test_data


            for index in range(len(leak.train_data.image_metadata)):
                leak.train_data.image_metadata[index]['verb'] = matrix_train[index]

            for index in range(len(leak.val_data.image_metadata)):
                leak.val_data.image_metadata[index]['verb'] = matrix_val[index]

            for index in range(len(leak.test_data.image_metadata)):
                leak.test_data.image_metadata[index]['verb'] = matrix_test[index]

            _, _, model_leakage, _, _, _ = leak.compute_leakage()
    
    elif leak.leakage_type == 'probability_leakage':
        if leak.experiment == 'weight_decay':

            for wd in [1e-2, 5e-1, 1e-1, 5e-1, 1, 5e-3, 1e-3]: 
                l = Leakage(parsing = False, leakage_type='probability_leakage', predictor='MLP', hid_size = 300, num_epochs=10, balance=leak.balance, weight_decay=wd)
                print('-------------------------wd : ', wd)
                cbm = CBM(parsing=False, target='verb',lam=1e-3, balance = leak.balance)
                l.train_data, l.val_data, l.test_data = cbm.leakage_loaders(layer = 'probabilities')
                _, _, model_leakage, _, _, _ = l.compute_leakage()
        
        if leak.attacked_model == 'CBM':
            cbm = CBM(parsing=False, target='verb',lam=1e-3, balance = leak.balance)
            leak.train_data, leak.val_data, leak.test_data = cbm.leakage_loaders(layer = 'probabilities')
            _, _, model_leakage, _, _, _ = leak.compute_leakage()

    elif leak.leakage_type == 'amplification':
        if leak.attacked_model == 'ResNet': print('todo')
            
        elif leak.attacked_model == 'CBM':
            cbm = CBM(parsing=False, target='verb',lam=5e-3, balance = leak.balance)
            CBM_leakage(cbm, retrain = False)

        elif leak.attacked_model == 'clip_DNN':
            sparse_clip_dnn = Clip_DNN(parsing=False, lam=1e-4, balance=leak.balance)
            sparse_clip_dnn.train(True)
            _, test_acc, test_f1 = sparse_clip_dnn.test()
            data_leakage, _, _ = dataset_leakage(5, 1-test_acc)
            leak.train_data, leak.val_data, leak.test_data = sparse_clip_dnn.leakage_loaders()
            _, _, model_leakage, _, _, _ = leak.compute_leakage()

        elif leak.attacked_model == 'clip_zero_shot':
            czs = Clip_zero_shot(parsing=False, balance=leak.balance)
            f1 , test_acc = czs.test(True)
            data_leakage, _, _ = dataset_leakage(5, 1-test_acc)
            leak.train_data, leak.val_data, leak.test_data = czs.leakage_loaders()
            _, _, model_leakage, _, _, _ = leak.compute_leakage()
            
        print('===================results===================')
        print('accuracy : ', test_acc)
        print('data_leakage(test_acc) : ', data_leakage)
        print('Model_leakage : ', model_leakage)
        print('Amplification : ', model_leakage - data_leakage)
        print('Relative amplification : ', (model_leakage-data_leakage)/(1-data_leakage))



    elif leak.leakage_type == 'mitigation_leakage':
        
        if leak.experiment == 'block_most_biased_concepts':
            print('------creating a CBM that classifies gender------------')
            cbm_gender = CBM(parsing=False, target='gender',lam=5e-3, n_iters=500, balance = leak.balance)
            cbm_gender.test(use_existing_embeddings=True)
            print('------extracting the most biased concepts------------')
            df_female, df_male = cbm_gender.extract_most_biased_concepts()
            block_concepts_idx = list(df_female['concept_idx'][:leak.block_how_many]) + list(df_male['concept_idx'][:leak.block_how_many])
            
            print('------creating a CBM that classifies verbs------------')
            cbm = CBM(parsing=False, target='verb',lam=5e-3, n_iters=2000, balance = leak.balance)        
            #cbm.train(use_existing_embeddings=True)
            print('--------------testing with all concepts--------------')
            _, vanilla_CBM_acc, _, _ = cbm.test(use_existing_embeddings=True, block_concepts_idx=None)
            #cbm.give_me_an_example()
            print('--------------testing with blocked concepts--------------')
            _, mitigated_CBM_acc, _, _ = cbm.test(use_existing_embeddings=True, block_concepts_idx=block_concepts_idx)

            print('--------------getting all predicted verbs--------------')
            leak.train_data, leak.val_data, leak.test_data = cbm.leakage_loaders(block_concepts_idx=None)
            _, _, vanilla_CBM_leakage, _, _, _ = leak.compute_leakage()
            
            leak.train_data, leak.val_data, leak.test_data  = cbm.leakage_loaders(block_concepts_idx=block_concepts_idx)
            _, _, mitigated_CBM_leakage, _, _, _ = leak.compute_leakage()

            print('--------------dataset_leakage--------------')
            vanilla_CBM_data_leakage, vanilla_CBM_data_leakage_std, _ = dataset_leakage(10, 1-vanilla_CBM_acc)
            mitigated_CBM_data_leakage, mitigated_CBM_data_leakage_std, _ = dataset_leakage(10, 1-mitigated_CBM_acc)

            print('===================results===================')
            print('vanilla_CBM_acc : ', vanilla_CBM_acc)
            print('vanilla_CBM_data_leakage : ', vanilla_CBM_data_leakage)
            print('vanilla_CBM_leakage : ', vanilla_CBM_leakage)
            print('vanilla Amplification : ', vanilla_CBM_leakage - vanilla_CBM_data_leakage)

            print('mitigated_CBM_acc : ', mitigated_CBM_acc)
            print('mitigated_CBM_data_leakage : ', mitigated_CBM_data_leakage)
            print('mitigated_CBM_leakage : ', mitigated_CBM_leakage)
            print('mitigated Amplification : ', mitigated_CBM_leakage - mitigated_CBM_data_leakage)
    


