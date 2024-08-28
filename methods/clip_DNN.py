import argparse, random, torch, os, math, json, sys, sklearn, copy
import numpy as np
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))
import torch
from data_loader import ImSituLoader
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm as tqdm
import torch.nn as nn
import torch.nn.functional as F
# importing sys
import methods.clip.clip as clip
from methods.data_loader import IndexedTensorDataset
from methods.cbm import CBM
from sklearn.metrics import f1_score
from methods.final_layer import Final_layer
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt


class Clip_DNN(nn.Module):
    def __init__(self, parsing = True, dataset = 'imSitu', which_clip = 'ViT-B-16', num_verb = 200,\
                  balance = 'imbalanced', batch_size = 800, interpretability_cutoff = 0.25, lam = 0.0,\
                      n_iters = 2000, alpha = 0.99, lr = 1, experiment = None, target = 'verb'):
        
        super(Clip_DNN, self).__init__()
        
        self.params = [
        ("dataset", str, None, dataset),
        ("which_clip", str, None, which_clip),
        ("num_verb", int, None, num_verb),
        ("balance", str, None, balance),
        ("batch_size", int, None, batch_size),
        ("interpretability_cutoff", float, None, interpretability_cutoff),
        ("lam", float, None,lam),
        ("n_iters", int, None,n_iters),
        ("alpha", float, None,alpha),
        ("lr", float, None,lr),
        ("experiment", str, None,experiment),
        ("target", str, None,target)
        ]        
        
        if parsing == True:
            parser = argparse.ArgumentParser(description='Settings for creating Clip DNN')

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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        path_verbs = Path(f'data/datasets/{self.dataset}/data_processed/{self.num_verb}_verbs/{self.num_verb}_verbs.txt')
        path_saved_models = Path(f'saved_models/clip_models')
        if self.target == 'verb': self.path_save = Path(f'saved_models/{self.dataset}/{self.num_verb}_verbs/Clip-DNN/{self.balance}')
        elif self.target == 'gender': self.path_save = Path(f'saved_models/{self.dataset}/2_genders/Clip-DNN/{self.balance}')
        
        os.makedirs(self.path_save, exist_ok=True)
        with open(self.path_save / 'args.txt', 'w') as f: f.write(self.arg_str) # save the arguments in a txt file
        
        if self.target == 'verb': 
            with open(path_verbs, "r") as f: self.targets = f.read().split("\n")
        elif self.target == 'gender': self.targets = ['female', 'male']

        self.pretrained_clip, self.preprocess = clip.load(path_saved_models,"ViT-B-16.pt", device=self.device)
        
        if self.dataset == 'imSitu':
            self.train_data = ImSituLoader(balance=self.balance, split = 'train', transform_name='clip', transform = self.preprocess)
            self.val_data = ImSituLoader(balance=self.balance, split = 'val', transform_name='clip', transform = self.preprocess)
            self.test_data = ImSituLoader(balance=self.balance, split = 'test', transform_name='clip', transform = self.preprocess)
        else: raise('Dataset not supported yet')

        self.train_targets = self.train_data.get_verbs() if self.target == 'verb' else self.train_data.get_genders()
        self.val_targets = self.val_data.get_verbs() if self.target == 'verb' else self.val_data.get_genders()
        self.test_targets = self.test_data.get_verbs() if self.target == 'verb' else self.test_data.get_genders()

        if Path(f'{self.path_save}/embeddings/train_image_embeddings.pt').exists() and Path(f'{self.path_save}/embeddings/val_image_embeddings.pt').exists():
            self.train_image_embeddings = torch.load(Path(f'{self.path_save}/embeddings/train_image_embeddings.pt'), map_location='cpu').float().to(self.device)
            self.val_image_embeddings = torch.load(Path(f'{self.path_save}/embeddings/val_image_embeddings.pt'), map_location='cpu').float().to(self.device)
            self.test_image_embeddings = torch.load(Path(f'{self.path_save}/embeddings/test_image_embeddings.pt'), map_location='cpu').float().to(self.device)

        if os.path.exists(self.path_save / "W_l.pt") and os.path.exists(self.path_save / "b_l.pt"):
            self.W_fl = torch.load(self.path_save / "W_l.pt", map_location='cpu').float().to(self.device)
            self.b_fl = torch.load(self.path_save / "b_l.pt", map_location='cpu').float().to(self.device)
        else:
            self.W_fl = None
            self.b_fl = None


        self.fl = Final_layer(512, self.train_targets.size(1),  self.path_save, lr = self.lr , lam = self.lam, alpha = self.alpha, n_iters =self.n_iters , device = self.device, \
                             W_fl = self.W_fl, b_fl =  self.b_fl, adv_debiasing = False, start_adv = 10000000, zeta = 0, balance = self.balance)
        

    def train(self, use_existing_embeddings = False):
        
        if use_existing_embeddings == False:
            os.makedirs(self.path_save / "embeddings", exist_ok=True)
            self.train_image_embeddings = CBM.encode_image(self.train_data, self.path_save, pretrained_clip=self.pretrained_clip).float() #([N, 512])
            torch.save(self.train_image_embeddings, Path(f'{self.path_save}/embeddings/train_image_embeddings.pt'))
            self.val_image_embeddings = CBM.encode_image(self.val_data, self.path_save, pretrained_clip=self.pretrained_clip).float() #([N, 512])
            torch.save(self.val_image_embeddings, Path(f'{self.path_save}/embeddings/val_image_embeddings.pt'))
            self.test_image_embeddings = CBM.encode_image(self.test_data, self.path_save, pretrained_clip=self.pretrained_clip).float()
            torch.save(self.test_image_embeddings, Path(f'{self.path_save}/embeddings/test_image_embeddings.pt'))

        train_loader = DataLoader(IndexedTensorDataset(self.train_image_embeddings, self.train_targets, self.train_data.get_genders()), batch_size=self.batch_size, shuffle=False) #removed IndexedTensorDataset
        val_loader = DataLoader(IndexedTensorDataset(self.val_image_embeddings, self.val_targets, self.val_data.get_genders()), batch_size=self.batch_size, shuffle=False)
        
        _, train_acc, train_f1, val_acc, val_f1, nnz_avg  = self.fl.train(train_loader, val_loader)

        return train_acc, train_f1, val_acc, val_f1, nnz_avg
    
    def test(self, use_existing_embeddings = False):

        test_loader = DataLoader(IndexedTensorDataset(self.test_image_embeddings, self.test_targets, self.test_data.get_genders()), batch_size=self.batch_size, shuffle=False) #removed IndexedTensorDataset
        
        test_loss, test_acc, test_f1, nnz = self.fl.test(test_loader, plot=False)

        with open(self.path_save / 'test_results.txt', 'w') as f:      
            f.write('test_loss: {0:.4f}, test_acc: {1:.4f}, test_f1: {2:.4f}'.format( test_loss, test_acc, test_f1))
            print('test_loss: {0:.4f}, test_acc: {1:.4f}, test_f1: {2:.4f}'.format( test_loss, test_acc, test_f1))

        return test_loss, test_acc, test_f1, nnz

    def get_all_predicted_verbs(self, probabilities = False):
            
        train_loader = DataLoader(IndexedTensorDataset(self.train_image_embeddings, self.train_data.get_verbs()), batch_size=self.batch_size, shuffle=False) #removed IndexedTensorDataset
        val_loader = DataLoader(IndexedTensorDataset(self.val_image_embeddings, self.val_data.get_verbs()), batch_size=self.batch_size, shuffle=False) #removed IndexedTensorDataset
        test_loader = DataLoader(IndexedTensorDataset(self.test_image_embeddings, self.test_data.get_verbs()), batch_size=self.batch_size, shuffle=False) #removed IndexedTensorDataset
        
        _, _, _, train_preds, _, train_prob, _, _ = self.fl.epoch_pass(-1, train_loader, training=False)
        _, _, _, val_preds, _, val_prob, _, _ = self.fl.epoch_pass(-1, val_loader, training=False)
        _, _, _, test_preds, _, test_prob, _, _ = self.fl.epoch_pass(-1, test_loader, training=False)

        if probabilities == True:
            return train_prob, val_prob, test_prob
        else:
            return train_preds, val_preds, test_preds 

    def leakage_loaders(self, block_concepts_idx = None, layer = 'predicitons'):
        
        train_data_copy = copy.deepcopy(self.train_data)
        val_data_copy = copy.deepcopy(self.val_data)
        test_data_copy = copy.deepcopy(self.test_data)

        probabilities = True if layer == 'probabilities' else False
        train_pv, val_pv, test_pv = self.get_all_predicted_verbs(probabilities = probabilities)

        train_data_copy.change_verbs(train_pv)
        val_data_copy.change_verbs(val_pv)
        test_data_copy.change_verbs(test_pv)
        
        return train_data_copy, val_data_copy, test_data_copy

if __name__=='__main__':
    clip_dnn= Clip_DNN()
    clip_dnn.train(True)
    clip_dnn.test(True)




        


