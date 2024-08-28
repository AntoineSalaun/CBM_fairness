import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
import math, os, random, sys, argparse
import numpy as np
import argparse
from PIL import Image


from tqdm import tqdm as tqdm

from data_loader import ImSituLoader



class GenderClassifier(nn.Module):
    def __init__(self, predictor = 'deterministic', net = None, optimizer = None , leakage_type = 'model_leakage', device='cuda', balance='imbalanced', \
                num_verb=200, image_dir='data/datasets/imSitu/classes/imSitu_200_classes.txt', \
                hid_size=300, num_epochs=15, learning_rate=0.00001, print_every=1, batch_size=512, \
                    dataset='imSitu', dataset_dir='data/datasets/imSitu/', perturbation=0.67):
        super(GenderClassifier, self).__init__()

        if net is None: 
           self.net = GenderClassifierNet(hid_size=hid_size) 
        else: self.net = net


        self.leakage_type = leakage_type
        self.device = device
        self.balance = balance
        self.num_verb = num_verb
        self.image_dir = image_dir
        self.hid_size = hid_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.print_every = print_every
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.perturbation = perturbation

        self.predictor = predictor

        if self.predictor == 'MLP':
            self.net = net.to(self.device)
            if optimizer is None:
                self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
            else: self.optimizer = optimizer

        print('leakage_type :', self.leakage_type, 'predictor : ', self.predictor)

        self.save_path = Path(f'saved_models/{self.dataset}/{self.num_verb}_verbs/{self.leakage_type}/{self.balance}')
        os.makedirs(self.save_path, exist_ok=True)


    
    def train(self, train_loader, val_loader):
        
        train_metrics = {'epoch':[], 'train_accuracy':[], 'train_loss':[], 'train_f1':[], 'val_accuracy':[], 'val_loss':[], 'val_f1':[]}
        if self.predictor == 'MLP':         ## train the model with a MLP
            best_val_acc = 0.0
            for epoch in tqdm(range(1, self.num_epochs + 1)):

                # train
                train_loss, train_acc, train_f1 = self.epoch_pass(epoch , train_loader, training=True)

                # val
                val_loss, val_acc, val_f1 = self.epoch_pass(epoch , val_loader, training = False)
                
                train_metrics['epoch'].append(epoch)
                train_metrics['train_accuracy'].append(train_acc)
                train_metrics['train_loss'].append(train_loss)
                train_metrics['train_f1'].append(train_f1)
                train_metrics['val_accuracy'].append(val_acc)
                train_metrics['val_loss'].append(val_loss)
                train_metrics['val_f1'].append(val_f1)


                if epoch % self.print_every == 0:
                    print('train, {0}, train loss: {1:.4f}, train acc: {2:.4f}'.format(epoch, train_loss, train_acc))
                    print('val, {0}, val loss: {1:.4f}, val acc: {2:.4f}'.format(epoch, val_loss, val_acc))
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({'epoch': epoch, 'state_dict': self.net.state_dict()}, self.save_path / 'model_best.pth.tar')

                if epoch >= self.num_epochs - 2 and val_acc < best_val_acc:
                    last_3_epochs_acc = train_metrics['val_accuracy'][-3:]
                    if all(acc < best_val_acc for acc in last_3_epochs_acc): # ça ne fonctionne pas
                        print(' last three epochs had val accuracies of ', last_3_epochs_acc, 'which is below the best val accuracy of ', best_val_acc, 'so stopping training')
                        break


            # Load the weights of the best model
            self.net.load_state_dict(torch.load(self.save_path / 'model_best.pth.tar', map_location='cpu')['state_dict'])
            _, train_acc, train_f1 = self.epoch_pass(-1 , train_loader, training=False)
            _, val_acc, val_f1 = self.epoch_pass(-1, val_loader, training = False)

        elif self.predictor == 'deterministic':
            if self.leakage_type not in ['dataset_leakage','data_leakage','model_leakage']: raise(self.leakage_type, 'is not compatible with the deterministic predictor. Use the MLP instead')
            
            # counting the number of male and female in each class
            male_counts, female_count, self.predict_this = [0] * self.num_verb, [0] * self.num_verb, [0] * self.num_verb
            for _, verbs, _, _, genders in train_loader:
                genders_num = np.argmax(genders, axis=1)
                verbs_num = np.argmax(verbs, axis=1)

                male_counts += np.bincount(verbs_num[genders_num == 1], minlength=self.num_verb)
                female_count += np.bincount(verbs_num[genders_num == 0], minlength=self.num_verb)

            # we predict the gender in majority in each class
            self.predict_this = [1 if male_counts[i] > female_count[i] else 0 for i in range(len(male_counts))]
            male_ratio = [male_counts[i]/(male_counts[i]+female_count[i]) if male_counts[i]+female_count[i] != 0 else 0.5 for i in range(len(male_counts))]
            majority_ratio = [1 - ratio if ratio < 0.5 else ratio for ratio in male_ratio]

            # computing the average majority_ratio weighted by the class population
            weighted_majority_ratio = sum([majority_ratio[i] * (female_count[i] + male_counts[i]) for i in range(len(majority_ratio))]) / sum(female_count + male_counts)
            train_acc = weighted_majority_ratio
            train_f1 = -1
            
            train_acc = sum(majority_ratio) / len(majority_ratio)

            # computing validation accuracy
            preds, truth = list(), list()
            for _, verbs, _, _, genders in val_loader:
                verbs_num = np.argmax(verbs, axis=1)

                predictions = [self.predict_this[int(verbs_num[i])] for i in range(len(verbs_num))]

                preds += predictions
                truth += np.argmax(genders, axis=1)

            val_acc = np.mean(np.array(truth) == np.array(preds))
            val_f1 = f1_score(np.array(truth), np.array(preds), average='macro')
        else:
            raise ValueError('predictor not found')

        return train_metrics, train_acc, train_f1, val_acc, val_f1



    def epoch_pass(self, epoch, data_loader, training):

        t_loss = 0.0
        n_processed = 0
        preds = list()
        truth = list()

        if training:
            self.net.train()
        else:
            self.net.eval()

        # [image , verb , path , image_name , gender]
        for idx, (_, targets, _, image_ids , genders) in enumerate(data_loader): # images are not provided
            
            targets = targets.cuda()
            genders = genders.cuda()

            predictions = self.net(targets)

            loss = F.cross_entropy(predictions, genders, reduction='mean')
            predictions = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)

            preds += predictions.tolist()
            truth += genders.max(1, keepdim=False)[1].cpu().numpy().tolist() # does this work ???????
            # I fell like ths would make more sense : genders_num = np.argmax(genders, axis=1)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            t_loss += loss.item()
            n_processed += len(genders)

        acc = np.mean(np.array(truth) == np.array(preds))
        f1 = f1_score(np.array(truth), np.array(preds), average='macro')
        
        return  loss, acc, f1

    def save_results(self, train_metrics, train_acc, train_f1, val_acc, val_f1, test_acc, test_f1):    

        print('train_acc: ',train_acc, 'train_f1 : ', train_f1, 'val_acc : ', val_acc, 'val_f1 : ', val_f1, 'test_acc : ', test_acc, 'test_f1 : ', test_f1 )
        
        with open(self.save_path / 'results.txt', 'w') as f:
            f.write('train_acc: {0:.4f}, train_f1: {1:.4f}, val_acc: {2:.4f}, val_f1: {3:.4f}, test_acc: {4:.4f}, test_f1: {5:.4f}'.format(train_acc, train_f1, val_acc, val_f1, test_acc, test_f1))
        
        with open(self.save_path / 'args.txt', 'w') as f:
            f.write(str(self))
        
        if self.predictor == 'MLP':
            train_metrics_df = pd.DataFrame({
                'train_accuracy': torch.tensor(train_metrics['train_accuracy']).detach().cpu().numpy(),
                'train_loss': torch.tensor(train_metrics['train_loss']).detach().cpu().numpy(),
                'train_f1': torch.tensor(train_metrics['train_f1']).detach().cpu().numpy(),
                'val_accuracy': torch.tensor(train_metrics['val_accuracy']).detach().cpu().numpy(),
                'val_loss': torch.tensor(train_metrics['val_loss']).detach().cpu().numpy(),
                'val_f1': torch.tensor(train_metrics['val_f1']).detach().cpu().numpy()
            })
            train_metrics_df.to_csv(self.save_path / 'train_metrics.csv', index=False)
    
    def test(self, test_loader):

        if self.predictor == 'MLP':        
            self.net.load_state_dict(torch.load(self.save_path / 'model_best.pth.tar',map_location='cpu')['state_dict'])

            # test the model by giving the last epoch saved results
            _, test_acc, test_f1 = self.epoch_pass(-1, test_loader, training = False)
        
        elif self.predictor == 'deterministic':
            preds, truth = list(), list()
            for _, verbs, _, _, genders in test_loader:
                verbs_num = np.argmax(verbs, axis=1)

                predictions = [self.predict_this[int(verbs_num[i])] for i in range(len(verbs_num))]

                preds += predictions
                truth += np.argmax(genders, axis=1)

            test_acc = np.mean(np.array(truth) == np.array(preds))
            test_f1 = f1_score(np.array(truth), np.array(preds), average='macro')
        else:
            raise ValueError('predictor not found')
        return test_acc, test_f1


class GenderClassifierNet(nn.Module):
    def __init__(self, in_features = 200, hid_size = 300):
        super(GenderClassifierNet, self).__init__()

        mlp = []
        mlp.append(nn.BatchNorm1d(in_features))
        mlp.append(nn.Linear(in_features, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, 2, bias=True))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, input_rep):

        return self.mlp(input_rep)
