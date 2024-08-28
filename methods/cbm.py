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
from sklearn.metrics import f1_score
from methods.final_layer import Final_layer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#python methods/cbm.py --lam 5e-3 --alpha 0.99 --n_iters 1200 --interpretability_cutoff 0.25
class CBM(nn.Module):
    def __init__(self, parsing = True, dataset = 'imSitu', which_clip = 'ViT-B-16', num_verb = 200, concept_set = 'imSitu_200_filtered.txt', \
                 balance = 'imbalanced', batch_size = 800, interpretability_cutoff = 0.28, lam = 1e-3, n_iters = 2000, alpha = 0.99, step_size = 0.1, \
                    lr = 5e-4, experiment = None, target = 'verb', adversarial = False, start_adv = 2000, zeta = 0, sparse_concepts = None, block = None): 
        super(CBM, self).__init__()
        
        
        self.params = [
        ("dataset", str, None, dataset),
        ("which_clip", str, None, which_clip),
        ("num_verb", int, None, num_verb),
        ("balance", str, None, balance),
        ("concept_set", str, None, concept_set),
        ("batch_size", int, None, batch_size),
        ("interpretability_cutoff", float, None, interpretability_cutoff),
        ("lam", float, None,lam),
        ("n_iters", int, None,n_iters),
        ("alpha", float, None,alpha),
        ("step_size", float, None,step_size),
        ("lr", float, None,lr),
        ("experiment", str, None,experiment),
        ("target", str, None,target),
        ("adversarial", str, None,adversarial),
        ("start_adv", int, None,start_adv),
        ("zeta", int, None,zeta),
        ("sparse_concepts", int, None, sparse_concepts),
        ("block", list, None, block)
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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        path_verbs = Path(f'data/datasets/{self.dataset}/data_processed/{self.num_verb}_verbs/{self.num_verb}_verbs.txt')
        self.path_concepts = Path(f'data/datasets/{self.dataset}/data_processed/{self.num_verb}_verbs/{self.concept_set}')
        path_saved_models = Path(f'saved_models/clip_models')
        if self.target == 'verb': self.path_save = Path(f'saved_models/{self.dataset}/{self.num_verb}_verbs/CBM/{self.balance}')
        elif self.target == 'gender': self.path_save = Path(f'saved_models/{self.dataset}/2_genders/CBM/{self.balance}')
        if adversarial == 'True': self.path_save = Path(f'saved_models/{self.dataset}/{self.num_verb}_verbs/adversarial_CBM/{self.balance}')
        
        os.makedirs(self.path_save, exist_ok=True) # create the folder if it does not exist
        with open(self.path_save / 'args.txt', 'w') as f: f.write(self.arg_str) # save the arguments in a txt file

        with open(path_verbs, "r") as f: self.verbs = f.read().split("\n")     # read the verbs
        

        self.pretrained_clip, self.preprocess = clip.load(path_saved_models,"ViT-B-16.pt", device=self.device) # load the clip model and revieve the preprocessinig associated with it
        

        self.train_data = ImSituLoader(balance=self.balance, split = 'train', transform_name='clip', transform = self.preprocess)
        self.val_data = ImSituLoader(balance=self.balance, split = 'val', transform_name='clip', transform = self.preprocess)
        self.test_data = ImSituLoader(balance=self.balance, split = 'test', transform_name='clip', transform = self.preprocess)

        self.train_targets = self.train_data.get_verbs() if self.target == 'verb' else self.train_data.get_genders()
        self.val_targets = self.val_data.get_verbs() if self.target == 'verb' else self.val_data.get_genders()
        self.test_targets = self.test_data.get_verbs() if self.target == 'verb' else self.test_data.get_genders()

        if os.path.exists(self.path_save / "embeddings/train_concept_matrix.pt") and os.path.exists(self.path_save / "embeddings/val_concept_matrix.pt") and os.path.exists(self.path_save / "embeddings/test_concept_matrix.pt"):
            self.matrix_train = torch.load(Path(f'{self.path_save}/embeddings/train_concept_matrix.pt'), map_location='cpu').float().to(self.device)
            self.matrix_val = torch.load(Path(f'{self.path_save}/embeddings/val_concept_matrix.pt'), map_location='cpu').float().to(self.device)
            self.matrix_test = torch.load(Path(f'{self.path_save}/embeddings/test_concept_matrix.pt'), map_location='cpu').float().to(self.device)
            self.concepts = torch.load(Path(f'{self.path_save}/embeddings/filtered_concepts.pt'), map_location='cpu')
        else:
            self.concepts = [concept for concept in open(self.path_concepts, 'r').read().split("\n")]

        if os.path.exists(self.path_save / "W_l.pt") and os.path.exists(self.path_save / "b_l.pt"):
            self.W_fl = torch.load(self.path_save / "W_l.pt", map_location='cpu').float().to(self.device)
            self.b_fl = torch.load(self.path_save / "b_l.pt", map_location='cpu').float().to(self.device)
        else:
            self.W_fl = None
            self.b_fl = None

        self.fl = Final_layer(len(self.concepts), self.train_targets.size(1), self.path_save, self.concepts, lr = self.lr , lam = self.lam, alpha = self.alpha, n_iters =self.n_iters , device = self.device, \
                                W_fl = self.W_fl, b_fl =  self.b_fl, adv_debiasing = self.adversarial, start_adv = self.start_adv, zeta = self.zeta, train_data=self.train_data, val_data=self.val_data, test_data=self.test_data, balance = self.balance)
        if self.adversarial and self.start_adv >= self.n_iters: print('Non-blocking error : start_adv should be smaller than n_iters')

    def encode_text(pretrained_clip, concepts, save_path, batch_size=4000):
        
        text_features = []
        with torch.no_grad():
            for i in range(math.ceil(len(concepts)/batch_size)):
                text_features.append(pretrained_clip.encode_text(concepts[batch_size*i:batch_size*(i+1)]))
        text_features = torch.cat(text_features, dim=0)

        # Normalizing the embeddings, train_cbm did it after saving th activations
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features  # Return as a tensor [N,512] on device
    
    def encode_image(data, save_name,  pretrained_clip, batch_size=3000, device = 'cuda'):
        print('encoding images')
        loader = torch.utils.data.DataLoader(data, shuffle = False, num_workers = 4, pin_memory = True)
        all_embedd = []

        with torch.no_grad():
            for idx, (images , targets, _, image_ids , genders) in enumerate(loader): # images are not provided
                image_emmeddings = pretrained_clip.encode_image(images.to(device))
                all_embedd.append(image_emmeddings.cpu())
        
        embeddings = torch.cat(all_embedd, dim=0).to(device)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalizing the embeddings, train_cbm did it after saving th activations

        return embeddings  # Return as a tensor on device
    
    def top_activating_concepts(matrix, concepts, cutoff):
    
        highest = torch.mean(torch.topk(matrix, dim=0, k=5)[0], dim=0)    
        concepts_idx = [i for i in range(len(concepts)) if highest[i] > cutoff]
        new_concepts = [concepts[i] for i in concepts_idx]

        return new_concepts, concepts_idx


    def compute_concept_matrix(self , data, phase = 'train',  device = 'cuda', block_concept_filtering = False, sparse_concepts = None):
        
        if phase == 'train': 
            with open(self.path_concepts) as f: concepts = f.read().split("\n") # read all the raw concepts from the file
        else : concepts = self.concepts # use the existing (filtered) concepts

        tokenized_text = clip.tokenize(["{}".format(concept) for concept in concepts]).to(device)

        text_embeddings = CBM.encode_text(self.pretrained_clip, tokenized_text, self.path_save).float() # ([M, 512])
        image_embeddings = CBM.encode_image(data, self.path_save, pretrained_clip=self.pretrained_clip).float() #([N, 512])

        concept_matrix = image_embeddings @ text_embeddings.T  #([N, M]) = ([N, 512]) @ ([512, M])
    
        if phase == 'train' and block_concept_filtering == False : # if we are in training mode, we should apply an activation filtering on the concepts
            self.concepts, concepts_idx = CBM.top_activating_concepts(concept_matrix, concepts, cutoff = self.interpretability_cutoff)
            print('Before filtrering, we had ', len(concepts), 'concepts After filtrering, we ended with ', len(self.concepts), 'concepts')        
            concept_matrix = concept_matrix[:, concepts_idx]
            text_embeddings = text_embeddings[concepts_idx]
            # because the concepts have changed, I need to update the final layer
            self.fl = Final_layer(len(self.concepts), self.train_targets.size(1), self.path_save, self.concepts, lr = self.lr , lam = self.lam, alpha = self.alpha, n_iters =self.n_iters , device = self.device, W_fl = self.W_fl, b_fl = self.b_fl, adv_debiasing = self.adversarial, start_adv = self.start_adv, zeta = self.zeta)

        # Center and normalize the matrix along columns (for each concept)
        concept_matrix = concept_matrix - torch.mean(concept_matrix, dim=0)
        concept_matrix = concept_matrix / torch.std(concept_matrix, dim=0)

        os.makedirs(self.path_save / "embeddings", exist_ok=True)
        torch.save(text_embeddings, Path(f'{self.path_save}/embeddings/{phase}_text_embeddings.pt'))
        torch.save(image_embeddings, Path(f'{self.path_save}/embeddings/{phase}_image_embeddings.pt'))
        torch.save(concept_matrix, Path(f'{self.path_save}/embeddings/{phase}_concept_matrix.pt'))
        torch.save(self.concepts, Path(f'{self.path_save}/embeddings/filtered_concepts.pt'))


        if self.sparse_concepts is not None:
            concept_matrix_topk = torch.topk(concept_matrix, k=self.sparse_concepts, dim=1)[0]
            concept_matrix[concept_matrix < concept_matrix_topk[:, -1].unsqueeze(1)] = 0
            #print('for example image 3500 has the following activations for all the concepts : ', concept_matrix[3500])

        if len(text_embeddings) != len(self.concepts): 
            print('text_embeddings and concepts should have the same length')
            print('len(text_embeddings) : ', len(text_embeddings), 'len(self.concepts) : ', len(self.concepts))
            raise('text_embeddings and concepts should have the same length')

        return concept_matrix, self.concepts, image_embeddings, text_embeddings
    

    def train(self, use_existing_embeddings = False):
        print('-----------Training CBM----------- (use_existing_embeddings :', use_existing_embeddings,')' )
        self.W_fl, self.b_fl = None, None
        if use_existing_embeddings == False:
            self.matrix_train, self.concepts, _, _ = self.compute_concept_matrix(self.train_data, phase = 'train')
            self.matrix_val,self.concepts, _, _ = self.compute_concept_matrix(self.val_data, phase ='val')
        
        self.fl = Final_layer(len(self.concepts), self.train_targets.size(1), self.path_save, self.concepts, lr = self.lr , lam = self.lam, alpha = self.alpha, n_iters =self.n_iters , device = self.device, \
                                W_fl = self.W_fl, b_fl =  self.b_fl, adv_debiasing = self.adversarial, start_adv = self.start_adv, zeta = self.zeta, train_data=self.train_data, val_data=self.val_data, test_data=self.test_data, balance = self.balance)

        train_loader = DataLoader(IndexedTensorDataset(self.matrix_train.cpu(), self.train_targets.cpu(), self.train_data.get_genders().cpu()), batch_size=self.batch_size, shuffle=False, num_workers=0) #removed IndexedTensorDataset
        val_loader = DataLoader(IndexedTensorDataset(self.matrix_val.cpu(), self.val_targets.cpu(), self.val_data.get_genders().cpu()), batch_size=self.batch_size, shuffle=False, num_workers=0) #removed IndexedTensorDataset
        
        self.fl.train_data = self.train_data
        self.fl.val_data = self.val_data
        self.fl.test_data = self.test_data

        _, train_acc, train_f1, val_acc, val_f1, nnz_avg  = self.fl.train(train_loader, val_loader)

        self.W_fl = self.fl.layer.weight.data
        self.b_fl = self.fl.layer.bias.data

        print('at the end of the training, the number of concepts is ', len(self.concepts), 'and the CBM weights have shape : ', self.W_fl.shape, 'while the final layer weights have shape are ', self.fl.layer.weight.shape)
        return train_acc, train_f1, val_acc, val_f1, nnz_avg


    def test(self, use_existing_embeddings = False, block_concepts_idx = None):
        print('-----------Testing CBM----------- (use_existing_embeddings :', use_existing_embeddings,')' )
        print('at the begining of testing, the number of concepts is ', len(self.concepts), 'and the CBM weights have shape : ', self.W_fl.shape, 'while the final layer weights have shape are ', self.fl.layer.weight.shape)

        if use_existing_embeddings == False:
            self.matrix_test,self.concepts, _, _ = self.compute_concept_matrix(self.test_data, phase ='test')

        test_loader = DataLoader(IndexedTensorDataset(self.matrix_test, self.test_targets, self.test_data.get_genders()), batch_size=self.batch_size, shuffle=False, num_workers=0) #removed IndexedTensorDataset
        
        if self.block is not None: self.block_concepts(self.block) # compute new weights

        test_loss, test_acc, test_f1, test_leak = self.fl.test(test_loader)
        nnz_mean, _, _ = self.fl.sparsity_measurement()

        with open(self.path_save / 'test_results.txt', 'w') as f:      
            f.write('test_loss: {0:.4f}, test_acc: {1:.4f}, test_f1: {2:.4f}, nnz_avg: {3:.0f}, test_leak{4:.4f}'.format(test_loss, test_acc, test_f1, nnz_mean, test_leak))
            print('test_loss: {0:.4f}, test_acc: {1:.4f}, test_f1: {2:.4f}, nnz_avg: {3:.0f}, test_leak{4:.4f}'.format(test_loss, test_acc, test_f1, nnz_mean, test_leak))

        return test_loss, test_acc, test_f1, float(nnz_mean)

    def block_concepts(self, block_concepts_idx):
        self.W_fl = self.W_fl.cpu()
        for i in block_concepts_idx:
            for j in range(len(self.b_fl)):   
                with torch.no_grad():
                    self.W_fl[j][i] = 0
        print('blocking', len(block_concepts_idx), 'concepts')

        self.W_fl = self.W_fl.to(self.device)
        self.b_fl = self.b_fl.to(self.device)
        self.fl = Final_layer(len(self.concepts), self.train_targets.size(1), self.path_save, self.concepts, lr = self.lr , lam = self.lam, alpha = self.alpha, n_iters =self.n_iters , device = self.device, \
                                 W_fl = self.W_fl, b_fl = self.b_fl, adv_debiasing = self.adversarial, start_adv = self.start_adv, zeta = self.zeta)
        return self.W_fl, self.b_fl
        
    def extract_most_biased_concepts(self): #must be executed after training a model for gender prediction
        
        concepts,weights,concept_idx = self.fl.verb_to_concepts(0) # weights to predict female
        
        df_female = pd.DataFrame({'Concepts': concepts, 'weights': weights, 'concept_idx': concept_idx})
        df_female_sorted = df_female.sort_values(by='weights', key=lambda x: abs(x), ascending=False)
        
        concepts,weights,concept_idx = self.fl.verb_to_concepts(1) # weights to predict male
        df_male = pd.DataFrame({'Concepts': concepts, 'weights': weights, 'concept_idx': concept_idx})
        df_male_sorted = df_male.sort_values(by='weights', key=lambda x: abs(x), ascending=False)
        return df_female_sorted, df_male_sorted

    def gender_class_study(self):
        test_loader = DataLoader(IndexedTensorDataset(self.matrix_test, self.test_targets), batch_size=self.batch_size, shuffle=False)   
        
        class_results = {'male_success':[0]*len(self.verbs), 'male_count':[0]*len(self.verbs), 'male_accuracy':[0]*len(self.verbs), 'female_success':[0]*len(self.verbs), 'female_count':[0]*len(self.verbs), 'female_accuracy':[0]*len(self.verbs)}

        genders = self.test_data.get_genders()
        true_class = torch.argmax(self.test_targets, dim=1)            
        _, agg_acc, _, preds, truth, _, adv_loss, _ = self.fl.epoch_pass(-1, test_loader, training=False)
        
        for i in range(len(preds)):
            verb_idx = preds[i]
            true_verb_idx = truth[i]
            gender = 'male' if torch.argmax(genders[i])==1 else 'female'
            class_results[gender + '_count'][true_verb_idx] += 1
            if verb_idx == true_verb_idx:
                class_results[gender + '_success'][true_verb_idx] += 1
            class_results[gender + '_accuracy'][true_verb_idx] = class_results[gender + '_success'][true_verb_idx] / class_results[gender + '_count'][true_verb_idx]
            
        return class_results

    def get_all_predicted_verbs(self, block_concepts_idx = None,  probabilities = False):
        
        if self.block is not None: self.block_concepts(self.block)
        
        train_loader = DataLoader(IndexedTensorDataset(self.matrix_train, self.train_data.get_verbs(), self.train_data.get_genders()), batch_size=self.batch_size, shuffle=False)
        _, _, _, train_preds, _,train_prob, adv_loss, train_leak = self.fl.epoch_pass(-1, train_loader, training=False)
 
        val_loader = DataLoader(IndexedTensorDataset(self.matrix_val, self.val_data.get_verbs(), self.val_data.get_genders()), batch_size=self.batch_size, shuffle=False)
        _, _, _, val_preds, _ ,val_prob, adv_loss, val_leak = self.fl.epoch_pass(-1, val_loader, training=False)

        test_loader = DataLoader(IndexedTensorDataset(self.matrix_test, self.test_data.get_verbs(), self.test_data.get_genders()), batch_size=self.batch_size, shuffle=False)
        _, _, _, test_preds, _,test_prob, adv_loss, test_leak  = self.fl.epoch_pass(-1, test_loader, training=False)

        if probabilities == True:
            return train_prob, val_prob, test_prob
        else:
            return train_preds, val_preds, test_preds 

    def leakage_loaders(self, block_concepts_idx = None, layer = 'predicitons'):
        
        train_data_copy = copy.deepcopy(self.train_data)
        val_data_copy = copy.deepcopy(self.val_data)
        test_data_copy = copy.deepcopy(self.test_data)
    
        if layer == 'predictions' or layer == 'probabilities':
            probabilities = True if layer == 'probabilities' else False
            train_pv, val_pv, test_pv = self.get_all_predicted_verbs(block_concepts_idx = self.block, probabilities = probabilities)

            train_data_copy.change_verbs(train_pv)
            val_data_copy.change_verbs(val_pv)
            test_data_copy.change_verbs(test_pv)

        elif layer == 'concept':
            matrix_train = self.matrix_train.cpu().unbind(dim=0)
            matrix_val = self.matrix_val.cpu().unbind(dim=0)
            matrix_test = self.matrix_test.cpu().unbind(dim=0)

            for index in range(len(train_data_copy.image_metadata)):
                train_data_copy.image_metadata[index]['verb'] = matrix_train[index]

            for index in range(len(val_data_copy.image_metadata)):
                val_data_copy.image_metadata[index]['verb'] = matrix_val[index]

            for index in range(len(test_data_copy.image_metadata)):
                test_data_copy.image_metadata[index]['verb'] = matrix_test[index]

        return train_data_copy, val_data_copy, test_data_copy


    def verb_to_concepts(self, weights, verb_idx, threshold = 1e-3):

        weights = weights[verb_idx]
        selected_idx = torch.nonzero(weights > threshold).squeeze().tolist()
        nnz_values = weights[selected_idx].cpu().tolist()
        nnz_concepts = [self.concepts[i] for i in selected_idx]

        return nnz_concepts, nnz_values
       


if __name__=='__main__':
    cbm= CBM()

    if cbm.experiment == 'sparsity':
        lambdas = [5e-1, 1e-1,  5e-2, 1e-2, 5e-3]
        alphas = [ 0.99, 0.95, 0.9, 0.7, 0.5]
        results = []
        for i in range(len(lambdas)):
            for j in range(len(alphas)):
            
                print('lambda : ', lambdas[i], 'alpha : ', alphas[j])
                cbm = CBM(lam =lambdas[i], alpha = alphas[j])
                
                #CHECK THIS !!!
                train_acc, train_f1, val_acc, val_f1, nnz_avg = cbm.train(use_existing_embeddings=True)
                test_loss, test_acc, test_f1, nnz_avg = cbm.test(use_existing_embeddings=True)
                print('inside sparsity experminet : ', nnz_avg)
                result = {
                    'Lambda': cbm.lam,
                    'Alpha': cbm.alpha,
                    'Test Loss': test_loss,
                    'Test_Accuracy': test_acc,
                    'Test F1': test_f1,
                    'non_zero_weights_avg': nnz_avg
                }
                results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(cbm.path_save / 'sparsity_exp.csv', index=False)

        # Plot the results
        plt.plot(df['non_zero_weights_avg'], df['Test_Accuracy'])
        plt.xlabel('Proportion of non-zero weights in FL')
        plt.ylabel('Test Accuracy')
        plt.title('FL sparsity vs accuracy')
        plt.legend()

        # Save the plot
        plt.savefig(cbm.path_save / 'sparsity_vs_accuracy.png')
    
    elif cbm.experiment == 'example':
        cbm = CBM(parsing= False,lam = 1e-3, alpha = 0.99, n_iters = 2000, balance = 'original', adversarial = False, start_adv = 2000, zeta = 0,sparse_concepts = 25)
        cbm.train(False)
        cbm.test(False)

        verb_idx = random.randint(0, len(cbm.verbs))
        image_idx = random.choice([i for i in range(len(cbm.train_data)) if torch.argmax(cbm.train_data[i][1]) == verb_idx])

        saving_spot = cbm.path_save / 'example_sparse'
        os.makedirs(saving_spot, exist_ok=True)

        verb_idx = 50
        image_idx = next((i for i, (_, verb, _, _, _) in enumerate(cbm.train_data) if torch.argmax(verb) == verb_idx), None)

        if image_idx is not None:
            concept_scores = cbm.matrix_train[image_idx, :]
        else:
            print(f"No image found with verb index {verb_idx}")

        image, verb, path, _, gender = cbm.train_data[image_idx]
        gender_str = "Female" if torch.argmax(gender) == 0 else "Male"
        verb_str = cbm.verbs[torch.argmax(verb)]

        # [2nd] Concept activation of the image
        top_concepts = [cbm.concepts[i] for i in torch.argsort(concept_scores, descending=True)[:20]]
        top_concepts_score = concept_scores[torch.argsort(concept_scores, descending=True)[:20]]

        concept_str = "Top associated concepts:\n"
        for idx in range(len(top_concepts)): concept_str += f"{top_concepts[idx]} : {100*top_concepts_score[idx].item():.2f}%\n"

        # loading the model and making predictions
        test_loader = DataLoader(IndexedTensorDataset(cbm.matrix_train[image_idx].unsqueeze(0) , cbm.train_targets[image_idx].unsqueeze(0)), batch_size=1, shuffle=False)
        _, _, _, test_preds, _, pred_probabilities, adv_loss, test_leak = cbm.fl.epoch_pass(-1, test_loader, training=False)        
        
        # [5th] Build a data frame with the prediction probabilities
        pred_prob_df = pd.DataFrame({'Prediction Probability': pred_probabilities[0]})
        pred_prob_df['Verb'] = cbm.verbs
        pred_prob_df = pred_prob_df.sort_values(by='Prediction Probability', ascending=False)
        pred_prob_df.to_csv(saving_spot / 'pred_prob_df.csv', index=False)
        
        # [1st] I want to show the image without the preprocessing
        cbm.train_data.data_transforms = None
        image_not_processed= cbm.train_data[image_idx][0]
        plt.savefig(saving_spot / 'image_not_processed.png')

        # [3rd] Finding the weights from all the concepts to the predicted class
        _, weights_to_predicted_class, nnz_idx = cbm.fl.verb_to_concepts(test_preds[0])
        concepts_selected = [cbm.concepts[i] for i in nnz_idx]
        top_weights, top_indices = torch.topk(torch.abs(torch.Tensor(weights_to_predicted_class)), k=25)
        top_concepts_w = [concepts_selected[i] if top_weights [i] > 0 else f"NOT {concepts_selected[i]}" for i in range(len(top_weights))]
        df_weights = pd.DataFrame({'Concept': top_concepts_w, 'Weight': top_weights})
        other_weights_sum = torch.sum(torch.abs(torch.Tensor(weights_to_predicted_class))) - torch.sum(top_weights)
        df_weights = df_weights.append({'Concept': f"{len(weights_to_predicted_class) - len(top_weights)} other weights", 'Weight': other_weights_sum}, ignore_index=True)
        df_weights.to_csv(saving_spot / 'df_weights.csv', index=False)

        # [4th] Finding the top 10 concepts that contribute to the prediction
        contributions_to_prediction = np.multiply(weights_to_predicted_class, concept_scores[nnz_idx].detach().cpu().tolist())
        #print('contributions_to_prediction', contributions_to_prediction)
        df_contribution = pd.DataFrame({'Concepts': concepts_selected, 'Contributions': contributions_to_prediction})
        df_contribution = df_contribution.reindex(df_contribution['Contributions'].abs().sort_values(ascending=False).index).head(25)
        df_contribution = df_contribution.append({'Concepts': f"{len(contributions_to_prediction) - len(df_contribution)} other concepts", 'Contributions': sum(contributions_to_prediction[25:])}, ignore_index=True)        
        df_contribution.to_csv(saving_spot / 'df_contribution.csv', index=False)

        plt.figure(figsize=(30,10))
        plt.subplot(1, 5, 1)
        plt.imshow(image_not_processed)
        plt.title(f"Gender: {gender_str}\nVerb: {verb_str}")
        
        plt.subplot(1, 5, 2)
        plt.text(0, 0.5, concept_str, fontsize=12, ha='left', va='center')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(1, 5, 3)
        plt.barh(range(len(df_weights)), df_weights['Weight'])
        plt.yticks(range(len(df_weights)), df_weights['Concept'])
        plt.title(f'Weights predicting {cbm.verbs[test_preds[0]]}')
        plt.xlabel('Absolute value of weights')

        plt.subplot(1, 5, 4)
        plt.barh(range(len(df_contribution)), df_contribution['Contributions'])
        plt.xlabel('Contribtution : fc(x)*W ')
        plt.title(f'Contribution to the prediction of {cbm.verbs[test_preds[0]]}')
        plt.yticks(range(len(df_contribution)), df_contribution['Concepts'])
        plt.tight_layout()
        sns.despine(left=True, bottom=True)

        plt.subplot(1, 5, 5)
        plt.barh(range(5), pred_prob_df['Prediction Probability'][:5])
        plt.xlabel('Prediction Probability')
        plt.title(f'Prediction {cbm.verbs[test_preds[0]]}')
        plt.yticks(range(5), pred_prob_df['Verb'][:5])
        plt.tight_layout()
        sns.despine(left=True, bottom=True)

        plt.show()
        plt.savefig(saving_spot / 'example.png')
        
        print('--------------------second example, not sparse-------------------')
        cbm = CBM(parsing= False,lam = 1e-3, alpha = 0.99, n_iters = 2000, balance = 'original', adversarial = False, start_adv = 2000, zeta = 0)
        cbm.train(False)
        cbm.test(False)

        verb_idx = random.randint(0, len(cbm.verbs))
        image_idx = random.choice([i for i in range(len(cbm.train_data)) if torch.argmax(cbm.train_data[i][1]) == verb_idx])
        
        saving_spot = cbm.path_save / 'example_not_sparse'
        os.makedirs(saving_spot, exist_ok=True)
        verb_idx = 50
        image_idx = next((i for i, (_, verb, _, _, _) in enumerate(cbm.train_data) if torch.argmax(verb) == verb_idx), None)

        if image_idx is not None:
            concept_scores = cbm.matrix_train[image_idx, :]
        else:
            print(f"No image found with verb index {verb_idx}")

        image, verb, path, _, gender = cbm.train_data[image_idx]
        gender_str = "Female" if torch.argmax(gender) == 0 else "Male"
        verb_str = cbm.verbs[torch.argmax(verb)]

        # [2nd] Concept activation of the image
        top_concepts = [cbm.concepts[i] for i in torch.argsort(concept_scores, descending=True)[:20]]
        top_concepts_score = concept_scores[torch.argsort(concept_scores, descending=True)[:20]]

        concept_str = "Top associated concepts:\n"
        for idx in range(len(top_concepts)): concept_str += f"{top_concepts[idx]} : {100*top_concepts_score[idx].item():.2f}%\n"

        # loading the model and making predictions
        test_loader = DataLoader(IndexedTensorDataset(cbm.matrix_train[image_idx].unsqueeze(0) , cbm.train_targets[image_idx].unsqueeze(0)), batch_size=1, shuffle=False)
        _, _, _, test_preds, _, pred_probabilities, adv_loss, test_leak = cbm.fl.epoch_pass(-1, test_loader, training=False)        
        
        # [5th] Build a data frame with the prediction probabilities
        pred_prob_df = pd.DataFrame({'Prediction Probability': pred_probabilities[0]})
        pred_prob_df['Verb'] = cbm.verbs
        pred_prob_df = pred_prob_df.sort_values(by='Prediction Probability', ascending=False)
        pred_prob_df.to_csv(saving_spot / 'pred_prob_df.csv', index=False)
        
        # [1st] I want to show the image without the preprocessing
        cbm.train_data.data_transforms = None
        image_not_processed= cbm.train_data[image_idx][0]
        plt.savefig(saving_spot / 'image_not_processed.png')

        # [3rd] Finding the weights from all the concepts to the predicted class
        _, weights_to_predicted_class, nnz_idx = cbm.fl.verb_to_concepts(test_preds[0])
        concepts_selected = [cbm.concepts[i] for i in nnz_idx]
        top_weights, top_indices = torch.topk(torch.abs(torch.Tensor(weights_to_predicted_class)), k=25)
        top_concepts_w = [concepts_selected[i] if top_weights [i] > 0 else f"NOT {concepts_selected[i]}" for i in range(len(top_weights))]
        df_weights = pd.DataFrame({'Concept': top_concepts_w, 'Weight': top_weights})
        other_weights_sum = torch.sum(torch.abs(torch.Tensor(weights_to_predicted_class))) - torch.sum(top_weights)
        df_weights = df_weights.append({'Concept': f"{len(weights_to_predicted_class) - len(top_weights)} other weights", 'Weight': other_weights_sum}, ignore_index=True)
        df_weights.to_csv(saving_spot / 'df_weights.csv', index=False)

        # [4th] Finding the top 10 concepts that contribute to the prediction
        contributions_to_prediction = np.multiply(weights_to_predicted_class, concept_scores[nnz_idx].detach().cpu().tolist())
        #print('contributions_to_prediction', contributions_to_prediction)
        df_contribution = pd.DataFrame({'Concepts': concepts_selected, 'Contributions': contributions_to_prediction})
        df_contribution = df_contribution.reindex(df_contribution['Contributions'].abs().sort_values(ascending=False).index).head(25)
        df_contribution = df_contribution.append({'Concepts': f"{len(contributions_to_prediction) - len(df_contribution)} other concepts", 'Contributions': sum(contributions_to_prediction[25:])}, ignore_index=True)        
        df_contribution.to_csv(saving_spot / 'df_contribution.csv', index=False)

        plt.figure(figsize=(30,10))
        plt.subplot(1, 5, 1)
        plt.imshow(image_not_processed)
        plt.title(f"Gender: {gender_str}\nVerb: {verb_str}")
        
        plt.subplot(1, 5, 2)
        plt.text(0, 0.5, concept_str, fontsize=12, ha='left', va='center')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(1, 5, 3)
        plt.barh(range(len(df_weights)), df_weights['Weight'])
        plt.yticks(range(len(df_weights)), df_weights['Concept'])
        plt.title(f'Weights predicting {cbm.verbs[test_preds[0]]}')
        plt.xlabel('Absolute value of weights')

        plt.subplot(1, 5, 4)
        plt.barh(range(len(df_contribution)), df_contribution['Contributions'])
        plt.xlabel('Contribtution : fc(x)*W ')
        plt.title(f'Contribution to the prediction of {cbm.verbs[test_preds[0]]}')
        plt.yticks(range(len(df_contribution)), df_contribution['Concepts'])
        plt.tight_layout()
        sns.despine(left=True, bottom=True)

        plt.subplot(1, 5, 5)
        plt.barh(range(5), pred_prob_df['Prediction Probability'][:5])
        plt.xlabel('Prediction Probability')
        plt.title(f'Prediction {cbm.verbs[test_preds[0]]}')
        plt.yticks(range(5), pred_prob_df['Verb'][:5])
        plt.tight_layout()
        sns.despine(left=True, bottom=True)

        plt.show()
        plt.savefig(saving_spot / 'example.png')



    elif cbm.experiment == 'accuracy_vs_cutoff':

        results = pd.DataFrame(columns = ['Cutoff', 'Test_Accuracy', 'Test_F1', 'Non_zero_weights_avg', 'len(self.concepts)'])
        cutoffs = [0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295, 0.30, 0.305, 0.31]

        for cutoff in cutoffs:
            cbm = CBM(parsing = True, interpretability_cutoff=cutoff, lam = 0)
            cbm.train(use_existing_embeddings=False)
            test_loss, test_acc, test_f1, nnz_mean = cbm.test(use_existing_embeddings=False)
            results = results.append({'Cutoff': cutoff, 'Test_Accuracy': test_acc, 'Test_F1': test_f1, 'Non_zero_weights_avg': nnz_mean, 'len(self.concepts)': len(cbm.concepts)}, ignore_index=True)
            if len(cbm.concepts) < 10: 
                print('stopping at cutoff ', cutoff, 'because were only have ', len(cbm.concepts), 'concepts')
                break

        print(results)
        results.to_csv(cbm.path_save / 'accuracy_vs_cutoff.csv', index=False)
        plt.figure(figsize=(12, 6))

        # Left subplot: Accuracy against cutoff
        plt.subplot(1, 2, 1)
        plt.plot(results['Cutoff'], results['Test_Accuracy'])
        plt.xlabel('Intrerpretability cutoff')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy vs Cutoff')

        # Right subplot: Number of concepts against cutoff
        plt.subplot(1, 2, 2)
        plt.plot(results['Cutoff'], results['len(self.concepts)'])
        plt.xlabel('Intrerpretability cutoff')
        plt.ylabel('Number of Concepts after filtering')
        plt.title('Number of Concepts vs Cutoff')

        plt.tight_layout()
        plt.savefig(cbm.path_save / 'accuracy_vs_cutoff.png')
        plt.show()

    elif cbm.experiment == 'accuracy_vs_lam':
        results = pd.DataFrame(columns = ['Lambda', 'Test_Accuracy', 'Test_F1', 'Non_zero_weights_avg', 'len(self.concepts)', 'interpretability_cutoff'])
        #lambdas = [1e-1, 3e-1, 1e-2, 1e-3, 1e-4]
        lambdas = np.logspace(-1, -5, 12)
        #lambdas = [1e-1, 1e-2]
        intp = 0.28

        for lam in lambdas:
            cbm = CBM(parsing = False, lam=lam,interpretability_cutoff = intp)
            use_existing_embeddings = False if lam == lambdas[0] else True
            cbm.train(use_existing_embeddings=use_existing_embeddings)
            test_loss, test_acc, test_f1, nnz_mean = cbm.test(use_existing_embeddings=use_existing_embeddings)
            results = results.append({'Lambda': lam, 'Test_Accuracy': test_acc, 'Test_F1': test_f1, 'Non_zero_weights_avg': nnz_mean, 'len(self.concepts)': len(cbm.concepts),'interpretability_cutoff':intp}, ignore_index=True)

        print(results)
        results.to_csv(cbm.path_save / 'accuracy_vs_lam.csv', index=False)

        plt.figure(figsize=(12, 6))

        # Left subplot: Accuracy against lambda
        plt.subplot(1, 2, 1)
        plt.semilogx(results['Lambda'], results['Test_Accuracy'])
        plt.xlabel('Lambda')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy vs Lambda')

        # Right subplot: Number of concepts against lambda
        plt.subplot(1, 2, 2)
        plt.semilogx(results['Lambda'], results['Non_zero_weights_avg'])
        plt.xlabel('Lambda')
        plt.ylabel('Non-zero weights per class')
        plt.title('Sparsity of concepts vs Lambda')

        plt.tight_layout()
        plt.savefig(cbm.path_save / 'accuracy_vs_lambda.png')
        plt.show()

    elif cbm.experiment == 'accuracy_vs_cutoff_vs_lamda':
        results = pd.DataFrame(columns = ['Lambda', 'cutoff','Test_Accuracy','Non_zero_weights_avg'])
        lambdas = np.logspace(-1, -5, 10)
        lambdas = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
        cutoffs = [0.25, 0.26, 0.27, 0.28, 0.29]

        for lam in lambdas:
            for cutoff in cutoffs:
                cbm = CBM(parsing = False, lam=lam,interpretability_cutoff = cutoff, balance = 'imbalanced')
                cbm.train(use_existing_embeddings=False)
                test_loss, test_acc, test_f1, nnz_mean = cbm.test(use_existing_embeddings=False)
                print('Resuls - lambda : ', lam, 'cutoff : ', cutoff, 'test_acc : ', test_acc, 'nnz_mean : ', nnz_mean,'----------------------:::::::::')
                results = results.append({'Lambda': lam, 'cutoff':cutoff, 'Test_Accuracy': test_acc, 'Non_zero_weights_avg': nnz_mean}, ignore_index=True)
        
        print(results)
        results.to_csv(cbm.path_save / 'accuracy_vs_cutoff_vs_lamda_heatmap.csv', index=False)      

        sns.set_theme()

        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(results, annot=True, fmt="d", linewidths=.5, ax=ax)
        sns.heatmap(results, annot=results['Non_zero_weights_avg'], fmt="d", linewidths=.5, ax=ax)
        

    elif cbm.experiment == 'how_sparsity_affects_gender_classification':
        df = pd.DataFrame(columns = ['Lambda',  'non_zero_weights', 'Test_Accuracy', 'Test_F1'])  
        for lambda_value in [3e-1, 2e-1, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2,4e-2, 3e-2]:
            print('lambda : ', lambda_value)
            cbm = CBM(parsing = False, lam=lambda_value, target='gender',n_iters=500,alpha=0.99)
            os.makedirs(cbm.path_save/'sparsity_vs_acc', exist_ok=True)
            cbm.train(use_existing_embeddings=True)
            test_loss, test_acc, test_f1, nnz_mean = cbm.test(use_existing_embeddings=True)
            df = df.append({'Lambda': lambda_value, 'non_zero_weights': nnz_mean, 'Test_Accuracy': test_acc, 'Test_F1': test_f1}, ignore_index=True)

        df.to_csv(cbm.path_save /'sparsity_vs_acc'/ 'results.csv', index=False)
        plt.plot(df['non_zero_weights'], df['Test_Accuracy'])
        plt.xlabel('Non Zero Weights')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs Non Zero Weights')
        plt.savefig(cbm.path_save / 'sparsity_vs_acc' / 'test_accuracy_vs_non_zero_weights.png')

    elif cbm.experiment == 'how_sparsity_affects_verb_classification':
        df = pd.DataFrame(columns = ['Lambda',  'non_zero_weights', 'Test_Accuracy', 'Test_F1'])  
        for lambda_value in [3e-1, 2e-1, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2,4e-2, 3e-2]:
            print('lambda : ', lambda_value)
            cbm = CBM(parsing = False, lam=lambda_value, target='verb',n_iters=500,alpha=0.99)
            os.makedirs(cbm.path_save/'sparsity_vs_acc', exist_ok=True)
            cbm.train(use_existing_embeddings=True)
            test_loss, test_acc, test_f1, nnz_mean = cbm.test(use_existing_embeddings=True)
            df = df.append({'Lambda': lambda_value, 'non_zero_weights': nnz_mean, 'Test_Accuracy': test_acc, 'Test_F1': test_f1}, ignore_index=True)

        df.to_csv(cbm.path_save /'sparsity_vs_acc'/ 'results.csv', index=False)
        plt.plot(df['non_zero_weights'], df['Test_Accuracy'])
        plt.xlabel('Non Zero Weights')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs Non Zero Weights')
        plt.savefig(cbm.path_save / 'sparsity_vs_acc' / 'test_accuracy_vs_non_zero_weights.png')


    elif cbm.experiment == 'wd_vs_gender_accuracy':
        os.makedirs(cbm.path_save / 'sparsity_bias', exist_ok=True)
        lambda_values = np.logspace(-1.5, -4, num=100)
        accuracies = pd.DataFrame(index=range(len(lambda_values)), columns=['Lambda'])
        aggregated_accuracies = pd.DataFrame(index= range(len(lambda_values)),columns=['Lambda', 'non_zero_weights', 'Aggregate_Accuracy', 'female_acc', 'male_acc'])

        for verb in cbm.verbs:
            accuracies[f'{verb}_male_accuracy'] = np.nan
            accuracies[f'{verb}_female_accuracy'] = np.nan
            accuracies[f'{verb}_accuracy'] = np.nan

        for i in range(len(lambda_values)):
            cbm = CBM(parsing=True, lam=lambda_values[i], n_iters=2000,balance='original')
            cbm.train(use_existing_embeddings=True)
            _, gender_aggregated_acc, _, nnz_mean = cbm.test(use_existing_embeddings=True)
            results = cbm.gender_class_study()

            accuracies.loc[i, 'Lambda'] = lambda_values[i]

            for idx, verb in enumerate(cbm.verbs):
                accuracies.loc[i, f'{verb}_male_accuracy'] = results['male_accuracy'][idx]
                accuracies.loc[i, f'{verb}_female_accuracy'] = results['female_accuracy'][idx]
                accuracies.loc[i, f'{verb}_accuracy'] = (results['female_success'][idx] + results['male_success'][idx]) / (results['female_count'][idx] + results['male_count'][idx])

            aggregated_accuracies = aggregated_accuracies.append(
                {'Lambda': lambda_values[i], 'non_zero_weights': int(nnz_mean),
                 'Aggregate_Accuracy': gender_aggregated_acc,
                 'female_acc': sum(results['female_success']) / sum(results['female_count']),
                 'male_acc': sum(results['male_success']) / sum(results['male_count'])}, ignore_index=True)

        accuracies.to_csv(cbm.path_save / 'sparsity_bias' / '0_accuracies.csv', index=False)
        aggregated_accuracies.to_csv(cbm.path_save / 'sparsity_bias' / '0_aggregated_accuracies.csv', index=False)
        plt.figure(figsize=(10, 6))
        plt.semilogx(aggregated_accuracies['Lambda'], aggregated_accuracies['male_acc'], marker='o', label='Male Accuracy')
        plt.semilogx(aggregated_accuracies['Lambda'], aggregated_accuracies['female_acc'], marker='o', label='Female Accuracy')
        plt.semilogx(aggregated_accuracies['Lambda'], aggregated_accuracies['Aggregate_Accuracy'], marker='o', label='Aggregate Accuracy')
        plt.xlabel('Lambda')
        plt.ylabel('Accuracy')
        plt.title('Male Accuracy vs Female Accuracy')
        plt.legend()
        plt.savefig(cbm.path_save / 'sparsity_bias' / '0_male_vs_female_accuracy.png')

        for verb_idx in range(len(cbm.verbs)):
            plt.figure(figsize=(10, 6))
            plt.semilogx(accuracies['Lambda'], accuracies[f'{cbm.verbs[verb_idx]}_male_accuracy'], marker='o', label='Male Accuracy')
            plt.semilogx(accuracies['Lambda'], accuracies[f'{cbm.verbs[verb_idx]}_female_accuracy'], marker='o', label='Female Accuracy')
            plt.semilogx(aggregated_accuracies['Lambda'], aggregated_accuracies['Aggregate_Accuracy'], marker='o', label='overall Accuracy')
            plt.xlabel('Lambda')
            plt.ylabel('Accuracy')
            plt.title('Accuracy for verb : ' + cbm.verbs[verb_idx])
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(cbm.path_save / 'sparsity_bias' / f'{cbm.verbs[verb_idx]}_accuracy.png')
            plt.close()
   
    elif cbm.experiment == "sparse_concepts":

        results = pd.DataFrame(columns = ['sparse_concepts', 'Test_Accuracy','Non_zero_weights_avg'])
        sparse_concepts = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

        for sp in sparse_concepts:
            cbm = CBM(parsing = True, lam = 0, sparse_concepts = int(sp))
            cbm.train(use_existing_embeddings=False)
            test_loss, test_acc, test_f1, nnz_mean = cbm.test(use_existing_embeddings=False)
            results = results.append({'sparse_concepts': sp, 'Test_Accuracy': test_acc, 'Non_zero_weights_avg': nnz_mean}, ignore_index=True)
        
        print(results)
        results.to_csv(cbm.path_save / 'acc_vs_sparse_concepts.csv', index=False)      

        plt.plot(results['sparse_concepts'], results['Test_Accuracy'])
        plt.xlabel('#concepts / image')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs concept sparsity')
        plt.savefig(cbm.path_save / 'acc_vs_sparse_concepts.png')


    elif cbm.experiment is None:
        cbm.train(use_existing_embeddings=False)
        cbm.test(use_existing_embeddings=False)        
        




