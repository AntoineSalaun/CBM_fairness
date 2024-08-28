    
import argparse, random, torch, os, math, json, sys, sklearn, copy
import numpy as np
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))
import torch
from data_loader import ImSituLoader
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm as tqdm
import torch.nn as nn
import torch.nn.functional as F
# importing sys
import methods.clip.clip as clip
from sklearn.metrics import f1_score, accuracy_score
from gender_classifier import GenderClassifier, GenderClassifierNet

#import methods.leakage as Leakage

import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt


class Final_layer(nn.Module):
    def __init__(self, in_features, out_features,  path_save, concepts = ['not a concept'] * 512, lr = 1e-3 , \
                 lam = 0.0007, alpha = 0.1, zeta = 0, n_iters =1000 , device = 'cuda', W_fl = None, \
                    b_fl = None, adv_debiasing = False, start_adv = 500, train_data = None, val_data = None, test_data = None, balance = None):
        super(Final_layer, self).__init__()
        
        self.device = device 
        self.in_features = in_features
        self.out_features = out_features       
        self.layer = nn.Linear(in_features, out_features).to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.layer.parameters(), self.lr)
        
        if W_fl is None: nn.init.kaiming_uniform_(self.layer.weight)
        else : self.layer.weight.data = W_fl

        if b_fl is None: self.layer.bias.data.zero_()
        else: self.layer.bias.data = b_fl
        
        self.path_save = path_save
        self.n_iters = n_iters
        self.lam = lam
        self.alpha = alpha
        self.zeta = zeta
        self.adv_debiasing = adv_debiasing
        self.start_adv = start_adv
        self.balance = balance
        self.gc = None
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        path_verbs = Path(f'data/datasets/imSitu/data_processed/200_verbs/200_verbs.txt')
        path_concepts = Path(f'data/datasets/imSitu/data_processed/200_verbs/imSitu_200_filtered.txt') #ATTENTION ! IT MAY CHANGE
        
        if self.out_features >2: 
            with open(path_verbs, "r") as f: self.verbs = f.read().split("\n")     # read the verbs
        else: 
            self.verbs = ['female','male']
        self.concepts = concepts
        

    def train(self, train_loader, val_loader):
        print('lambda:', self.lam, 'alpha:', self.alpha, 'zeta:', self.zeta) 
        # Reseting weight and bias value
        nn.init.kaiming_uniform_(self.layer.weight)
        self.layer.bias.data.zero_()
        
        train_metrics = []
        best_acc = 0.0
        nnz_avg = 2000
        best_weight = None
        best_bias = None
        best_acc_epoch = 0

        #for epoch in tqdm(range(self.n_iters)):
        for epoch in range(self.n_iters):  
            
            # In case, we are not in the adversarial phase, do a regular epoch pass
            if self.adv_debiasing == False or epoch < self.start_adv: 
                train_loss, train_acc, train_f1, train_preds, train_truth, train_prob, train_adv_loss, train_p_leak = self.epoch_pass(epoch , train_loader, training=True)
                val_loss, val_acc, val_f1, val_preds, val_truth, val_prob, val_adv_loss,val_p_leak = self.epoch_pass(epoch , val_loader, training = False)

            # In case, we are in the adversarial phase, do this
            if self.adv_debiasing and epoch >= self.start_adv:
                if epoch == self.start_adv:
                    print('epoch : ', epoch, '-> training the first gender classifier')

                    train_data_with_prob = copy.deepcopy(self.train_data) 
                    val_data_with_prob = copy.deepcopy(self.val_data)
                    
                    train_data_with_prob.change_verbs(train_prob)
                    val_data_with_prob.change_verbs(val_prob)

                    self.AttackerNet = GenderClassifierNet(200, 300).to(self.device)
                    self.AttackerOptimizer = optim.Adam(self.AttackerNet.parameters(), lr=1e-3, weight_decay = 2e-1)
                    self.AttackerModel = GenderClassifier(predictor = 'MLP', net = self.AttackerNet, optimizer = self.AttackerOptimizer, leakage_type='probability_lekage', balance = self.balance, num_epochs=12)

                    train_loader_with_prob = torch.utils.data.DataLoader(train_data_with_prob, batch_size = 512, shuffle = False, num_workers = 6, pin_memory = True)
                    val_loader_with_prob = torch.utils.data.DataLoader(val_data_with_prob, batch_size = 512, shuffle = False, num_workers = 6, pin_memory = True)

                    for layer in self.layer.parameters(): layer.requires_grad = False #freezing the main layer
                    # trains the model and saves the best model
                    self.AttackerModel.train(train_loader_with_prob, val_loader_with_prob)

                    print('after training, freezing the adversarial network and unfreezing the main network')
                    for layer in self.layer.parameters(): layer.requires_grad = True
                    for layer in self.AttackerModel.net.parameters(): layer.requires_grad = False     
                    self.AttackerModel.net.eval() 


                train_loss, train_acc, train_f1, train_preds, train_truth, train_prob, train_adv_loss, train_p_leak = self.epoch_pass(epoch , train_loader, training=True, gender_classifier=self.AttackerModel.net)
                val_loss, val_acc, val_f1, val_preds, val_truth, val_prob, val_adv_loss,val_p_leak = self.epoch_pass(epoch , val_loader, training = False, gender_classifier=self.AttackerModel.net)

                if epoch > self.start_adv and epoch % 100 == 0:
                    train_data_with_prob = copy.deepcopy(self.train_data)
                    train_data_with_prob.change_verbs(train_prob)
                    val_data_with_prob = copy.deepcopy(self.val_data)
                    val_data_with_prob.change_verbs(val_prob)

                    train_loader_with_prob = torch.utils.data.DataLoader(train_data_with_prob, batch_size = 512, shuffle = False, num_workers = 0, pin_memory = True)
                    val_loader_with_prob = torch.utils.data.DataLoader(val_data_with_prob, batch_size = 512, shuffle = False, num_workers = 0, pin_memory = True)

                    print('epoch : ', epoch, 'unfreezing the gender classifier')
                    for layer in self.AttackerModel.net.parameters(): layer.requires_grad = True     
                    
                    print('training it 1 epoch')
                    _, gc_train_acc, _ = self.AttackerModel.epoch_pass(epoch , train_loader_with_prob, training=True)
                    _, gc_val_acc, _ = self.AttackerModel.epoch_pass(epoch , val_loader_with_prob, training = False)
                    print('After retraining the gender classifier, it has train_acc: ', gc_train_acc, 'val_acc: ', gc_val_acc, ' aftert that, we freeze it again')
                    
                    for layer in self.AttackerModel.net.parameters(): layer.requires_grad = False     
                    self.GenderClassifierNet = self.AttackerModel.net.eval()

            train_metrics.append(f'Epoch {epoch} -  train_loss: {train_loss:.4f} train_adv_loss: {train_adv_loss:.6f} train_acc: {train_acc:.4f} train_p_leak: {train_p_leak:.4f} val_loss: {val_loss:.4f}  val_adv_loss: {val_adv_loss:.6f} val_acc: {val_acc:.4f} val_p_leak: {val_p_leak:.4f} nnz_avg: {nnz_avg:.0f}')
            if (epoch+1) % 50 == 0: print(train_metrics[-1])


            if self.adv_debiasing == False :
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_acc_epoch = epoch
                    with torch.no_grad():
                        best_weight = self.layer.weight.data
                        best_bias = self.layer.bias.data
                        torch.save(best_weight, self.path_save / "W_l.pt")
                        torch.save(best_bias, self.path_save / "b_l.pt")
                    nnz_avg,_,_ = self.sparsity_measurement(W_fl = best_weight)
                #print('I saved the new weights')
            
            if self.adv_debiasing and epoch >= self.start_adv:
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_acc_epoch = epoch
                    with torch.no_grad():
                        best_weight = self.layer.weight.data
                        best_bias = self.layer.bias.data
                        torch.save(best_weight, self.path_save / "W_l.pt")
                        torch.save(best_bias, self.path_save / "b_l.pt")
                    nnz_avg,_,_ = self.sparsity_measurement(W_fl = best_weight)
                #print('I saved the new weights')
        
        if self.adv_debiasing: self.gc = self.AttackerModel.net

        if self.adv_debiasing == False:
            print('best weights were optained at epoch : ', best_acc_epoch) # Do one last run with the best weights to get the best results
            
            self.layer.weight.data = torch.load(self.path_save / "W_l.pt", map_location='cpu').float().to(self.device)
            self.layer.bias.data = torch.load(self.path_save / "b_l.pt", map_location='cpu').float().to(self.device)
            _, train_acc, train_f1, _, _, _, train_adv_loss, train_leak = self.epoch_pass(best_acc_epoch , train_loader, training=False)
            _, val_acc, val_f1,_ ,_, _, val_adv_loss,val_leak = self.epoch_pass(best_acc_epoch, val_loader, training = False)

        self.save_train_results(train_metrics, train_acc, train_f1, val_acc, val_f1, best_weight, best_bias, nnz_avg)

        return train_metrics, train_acc, train_f1, val_acc, val_f1, nnz_avg

    def epoch_pass(self, epoch , data_loader, training=True, gender_classifier = None):
        if training: self.layer.train()
        else: self.layer.eval()
        if gender_classifier is not None: 
            for layer in gender_classifier.parameters(): layer.requires_grad = False #just making sure it's well frozen
              
        epoch_loss = 0
        preds = list()
        truth = list()
        pred_probabilities = list()

        g_preds = list()
        g_truth = list()
        epoch_adv_loss = 0

        for batch in data_loader:
            #batch = batch.to(self.device)
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            gender = batch[2].to(self.device)

            layer_predictions = self.layer(inputs)
            predictions = np.argmax(F.softmax(layer_predictions, dim=1).cpu().detach().numpy(), axis=1) 

            # if we are not doing adversarial debiasing, we just compute the loss
            if self.adv_debiasing == False or epoch < self.start_adv:
                batch_loss, batch_adv_loss = self.elastic_loss(self.layer, labels, layer_predictions, self.lam, self.alpha), 0
                g_preds += [0] * inputs.size(0)
                g_truth += [1] * inputs.size(0)
                
            else:
                gender_predictions = gender_classifier(layer_predictions)
                g_preds += np.argmax(F.softmax(gender_predictions, dim=1).cpu().detach().numpy(), axis=1).tolist()
                g_truth += np.argmax(gender.cpu().detach(), axis=1).tolist()
                batch_loss, batch_adv_loss = self.adversarial_loss(self.layer, labels, layer_predictions, gender_predictions, gender, self.lam, self.alpha, self.zeta)

            pred_probabilities += F.softmax(layer_predictions, dim=1).cpu().detach().numpy().tolist()
            preds += predictions.tolist()
            truth += np.argmax(labels.cpu().detach(), axis=1).tolist()
            epoch_loss += batch_loss.item()
            epoch_adv_loss += batch_adv_loss

            if training:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

        epoch_acc = accuracy_score(truth, preds)
        epoch_f1 = f1_score(truth, preds, average='macro')
        epoch_leakage = accuracy_score(g_truth, g_preds)

        return epoch_loss, epoch_acc, epoch_f1, preds, truth, pred_probabilities, epoch_adv_loss, epoch_leakage


    def elastic_loss(self, linear, labels, predictions, lam, alpha): 
        weight, bias = list(linear.parameters())
        l1 = lam * alpha * weight.norm(p=1)  #lambda is a general normalization factor
        l2 = 0.5 * lam * (1 - alpha) * (weight**2).sum() # a high alpha favorizes l1 over l2 
        #print('labels:', labels)
        if labels.dim() > 1: labels = torch.argmax(labels, dim=1)  
        l = F.cross_entropy(predictions,labels, reduction='mean') + l1 + l2
        return l
        
    def adversarial_loss(self, linear, labels, prediction_probs, gender_prediction, true_genders, lam, alpha, zeta):
        weight, bias = list(linear.parameters())
        l1 = lam * alpha * weight.norm(p=1)
        l2 = 0.5 * lam * (1 - alpha) * (weight**2).sum()
        #with torch no grad ????
        l3 = - zeta * F.cross_entropy(gender_prediction, true_genders, reduction='mean')
        l = F.cross_entropy(prediction_probs, labels, reduction='mean') 
        return l + l1 + l2 + l3, l3
    


    def test(self, test_loader, plot=True):
        
        self.layer.eval() 
        if self.gc is not None : self.gc = self.gc.eval() 

        test_loss, test_acc, test_f1, _, _, _, test_adv_loss, test_leak = self.epoch_pass(-1, test_loader, training=False, gender_classifier=self.gc)      
        
        self.plot_all_class_weights()      

        return test_loss, test_acc, test_f1, test_leak



    def save_train_results(self, train_metrics, train_acc, train_f1, val_acc, val_f1, best_weight, best_bias, nnz_avg):    
        
        print('iters : ', self.n_iters , 'lam : ',self.lam , 'alpha : ',self.alpha )
        print('train_acc: ',train_acc, 'train_f1 : ', train_f1, 'val_acc : ', val_acc, 'val_f1 : ', val_f1, 'test_acc : ')

        with open(self.path_save / 'train_results.txt', 'w') as f:
            f.write('train_acc: {0:.4f}, train_f1: {1:.4f}, val_acc: {2:.4f}, val_f1: {3:.4f} nnz_avg : {3:.0f}\n'.format(train_acc, train_f1, val_acc, val_f1, nnz_avg))
        
        with open(self.path_save / 'args.txt', 'w') as f:
            f.write(f"{self.optimizer} iters: {self.n_iters}\nlam: {self.lam}\nalpha: {self.alpha}")
        
        with open(self.path_save / 'train_metrics.txt', 'w') as f:
            for item in train_metrics:
                f.write("%s\n" % item)
    
    def verb_to_concepts(self, verb_idx, threshold = 1e-3, print_concepts = False):
        
        weights = self.layer.weight.data
        if weights.shape[1] != len(self.concepts):raise('weights shape and concepts length does not match. Fix it by saving the concepts when computing the matrix')
        weights = weights[verb_idx]
        nnz_concepts =[]
        
        selected_idx = torch.nonzero(weights > threshold).squeeze()
        selected_idx = selected_idx.cpu().tolist()

        if isinstance(selected_idx, list) and len(selected_idx) > 0:
            nnz_values = weights[selected_idx].cpu().tolist()
            nnz_concepts = [self.concepts[i] for i in selected_idx]

            if print_concepts:
                print('In order to predict the class:', self.verbs[verb_idx], 'we need the following', len(nnz_concepts), 'non-zero concepts:')
                for idx in range(len(nnz_concepts)):
                    print(nnz_concepts[idx], ':', nnz_values[idx])
        else: 
            nnz_values = [0]
            print('No non-zero concepts for the verb:', self.verbs[verb_idx])
            nnz_concepts = ['No non-zero concepts for the verb:', self.verbs[verb_idx]]
            selected_idx = []

        return nnz_concepts, nnz_values, selected_idx
    
    def sparsity_measurement(self, threshold = 1e-3, W_fl = None):
        if W_fl == None : W_fl = torch.load(self.path_save / "W_l.pt", map_location='cpu').float().to(self.device)

        number_of_concepts_to_predict = []
        for verb_idx in range(len(W_fl)):
            nnz_concepts, nnz_value, _ = self.verb_to_concepts(verb_idx, threshold =threshold)
            number_of_concepts_to_predict.append(len(nnz_concepts))
            
        median = torch.median(torch.tensor(number_of_concepts_to_predict))
        minimum = torch.min(torch.tensor(number_of_concepts_to_predict))
        maximum = torch.max(torch.tensor(number_of_concepts_to_predict))

        return median, minimum, maximum
    
    def plot_class_weights(self, verb_idx = None):

        os.makedirs(self.path_save / "plots", exist_ok=True)

        if verb_idx == None: verb_idx = random.randint(0, self.out_features-1)

        concepts, weights = ['not a concept'], [0]
        concepts, weights,_ = self.verb_to_concepts(verb_idx, print_concepts = False)

        if len(concepts) == len(weights):

            # Create a DataFrame from concepts and contribution
            df = pd.DataFrame({'Concepts': concepts, 'weights': weights})
            df_sorted = df.sort_values(by='weights', key=lambda x: abs(x), ascending=False)
            total_weights = df_sorted['weights'].sum()
            threshold = total_weights * 0.005
            filtered_df = df_sorted.head(25)

            # Calculate the sum of the remaining concepts' contribution
            sum_weights = df_sorted.iloc[25:]['weights'].sum()

            filtered_df = pd.concat([filtered_df, pd.DataFrame.from_records([{'Concepts': f'Sum of {len(df_sorted)-25} other concepts', 'weights': sum_weights}], index=[25])])

            # Save the dataframes
            df_sorted.to_csv(self.path_save / "plots"/ f"{self.verbs[verb_idx]}_concepts_weights.csv", index=False)
            filtered_df.to_csv(self.path_save / "plots" / f"{self.verbs[verb_idx]}_filtered_concepts_wegihts.csv", index=False)

            f, ax = plt.subplots(figsize=(6, 8))
            sns.set_color_codes("pastel")
            sns.barplot(x='weights', y='Concepts', data=filtered_df, label="Total", color="b")
            ax.axhline(y=0, color='black', linewidth=0.05)  # Add a thin horizontal line
            ax.set(xlim=(0, 0.6), ylabel="",
            xlabel="class-related weights")
            sns.despine(left=True, bottom=True)

            # Set the title and labels
            ax.set_title("Top-25 concepts for {}".format(self.verbs[verb_idx]))
            plt.tight_layout()
            # Save the plot
            plt.savefig(self.path_save / "plots" / f"{self.verbs[verb_idx]}_concept_weight.png")
            plt.close()

    def plot_all_class_weights(self):
        for verb_idx in range(self.out_features):
            self.plot_class_weights(verb_idx)

        
    