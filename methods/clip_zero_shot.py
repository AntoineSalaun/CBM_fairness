import argparse, random, torch, os, math, json, sys, sklearn, copy
import numpy as np
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))
import torch
from data_loader import ImSituLoader
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import methods.clip.clip as clip
from sklearn.metrics import f1_score
from methods.cbm import CBM



class Clip_zero_shot(nn.Module):
    def __init__(self, parsing = True, dataset ='imSitu', which_clip = 'ViT-B-16', num_verb = 200, \
                 balance = 'imbalanced', batch_size = 800, experiment = None, target = 'verb'):
        super(Clip_zero_shot, self).__init__()

        self.params = [
        ("dataset", str, None, dataset),
        ("which_clip", str, None, which_clip),
        ("num_verb", int, None, num_verb),
        ("balance", str, None, balance),
        ("batch_size", int, None, batch_size),
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
        if self.target == 'verb': self.path_save = Path(f'saved_models/{self.dataset}/{self.num_verb}_verbs/clip_zero_shot/{self.balance}')
        elif self.target == 'gender': self.path_save = Path(f'saved_models/{self.dataset}/2_genders/clip_zero_shot/{self.balance}')
        else : raise('Target not supported yet')

        
        os.makedirs(self.path_save, exist_ok=True)
        with open(self.path_save / 'args.txt', 'w') as f: f.write(self.arg_str) # save the arguments in a txt file
        
        if self.target == 'verb':
            with open(path_verbs, "r") as f: self.targets = f.read().split("\n")
            with open(path_verbs, "r") as f: self.verbs = f.read().split("\n")
        elif self.target == 'gender':
            self.targets =  ['female', 'male']
     
        self.pretrained_clip, self.preprocess = clip.load(path_saved_models,"ViT-B-16.pt", device=self.device)
        
        if self.dataset == 'imSitu':
           self.train_data = ImSituLoader(balance=self.balance, split = 'train', transform_name='clip', transform = self.preprocess)
           self.val_data = ImSituLoader(balance=self.balance, split = 'val', transform_name='clip', transform = self.preprocess)
           self.test_data = ImSituLoader(balance=self.balance, split = 'test', transform_name='clip', transform = self.preprocess)
        else:
            raise('Dataset not supported yet')
        
    def test(self, use_exising_embeddings = True):
        
        if use_exising_embeddings == True:

            concept_matrix = torch.load(Path(f'{self.path_save}/embeddings/test_concept_matrix.pt'), map_location='cpu').float().to(self.device)
        else:
            print('self.targets', self.targets)

            tokenized_text = clip.tokenize(["{}".format(concept) for concept in self.targets]).to(self.device)

            text_embeddings = CBM.encode_text(self.pretrained_clip, tokenized_text, self.path_save).float() # ([M, 512])
            image_embeddings = CBM.encode_image(self.test_data, self.path_save, pretrained_clip=self.pretrained_clip).float() #([N, 512])

            concept_matrix = image_embeddings @ text_embeddings.T  #([N, M]) = ([N, 512]) @ ([512, M])
        
            # Center and normalize the matrix along columns (for each concept)
            concept_matrix = concept_matrix - torch.mean(concept_matrix, dim=0)
            concept_matrix = concept_matrix / torch.std(concept_matrix, dim=0)

            os.makedirs(self.path_save / "embeddings", exist_ok=True)
            torch.save(concept_matrix, Path(f'{self.path_save}/embeddings/test_concept_matrix.pt'))
        
        predictions = np.argmax(F.softmax(concept_matrix, dim=1).cpu().numpy(), axis=1) 
        
        if self.target == 'verb':
            labels = np.argmax(self.test_data.get_verbs().numpy(),axis=1)
        elif self.target == 'gender':
            labels = np.argmax(self.test_data.get_genders().numpy(),axis=1)
        
        accuracy = np.mean(labels == predictions)
        f1 = f1_score(labels, predictions, average='macro')

        print('F1 score:', f1, 'Accuracy:', accuracy)        
        with open(self.path_save / 'test_results.txt', 'w') as f:    f.write('test_acc: {:.4f}, test_f1: {:.4f}'.format(accuracy, f1))

        return f1_score, accuracy, 0, 0

    def get_all_predicted_verbs(self, use_existing_embeddings= True, probabilities = False):
        
        matrix_train, matrix_val = self.train_and_val_matrix()
        train_prob = F.softmax(matrix_train, dim=1).cpu().numpy()
        train_preds = np.argmax(train_prob, axis=1)
        
        val_prob = F.softmax(matrix_val, dim=1).cpu().numpy()
        val_preds = np.argmax(val_prob, axis=1)

        matrix_test =  torch.load(Path(f'{self.path_save}/embeddings/test_concept_matrix.pt'), map_location='cpu').float().to(self.device)
        test_prob = F.softmax(matrix_test, dim=1).cpu().numpy()
        test_preds = np.argmax(test_prob, axis=1)

        if probabilities == True:
            return train_prob, val_prob, test_prob
        else:
            return train_preds, val_preds, test_preds 

    def leakage_loaders(self, block_concepts_idx = None, layer = 'predicitons'):

        train_data_copy = copy.deepcopy(self.train_data)
        val_data_copy = copy.deepcopy(self.val_data)
        test_data_copy = copy.deepcopy(self.test_data)

        probabilities = True if layer == 'probabilities' else False
        train_pv, val_pv, test_pv = self.get_all_predicted_verbs(True)
        
        train_data_copy.change_verbs(train_pv)
        val_data_copy.change_verbs(val_pv)
        test_data_copy.change_verbs(test_pv)

        return train_data_copy, val_data_copy, test_data_copy

    def train_and_val_matrix(self):
        tokenized_text = clip.tokenize(["{}".format(concept) for concept in self.targets]).to(self.device)

        text_embeddings = CBM.encode_text(self.pretrained_clip, tokenized_text, self.path_save).float() # ([M, 512])
        train_image_embeddings = CBM.encode_image(self.train_data, self.path_save, pretrained_clip=self.pretrained_clip).float() #([N, 512])
        val_image_embeddings = CBM.encode_image(self.val_data, self.path_save, pretrained_clip=self.pretrained_clip).float()

        train_concept_matrix = train_image_embeddings @ text_embeddings.T  #([N, M]) = ([N, 512]) @ ([512, M])
        val_concept_matrix = val_image_embeddings @ text_embeddings.T

        # Center and normalize the matrix along columns (for each concept)
        train_concept_matrix = train_concept_matrix - torch.mean(train_concept_matrix, dim=0)
        train_concept_matrix = train_concept_matrix / torch.std(train_concept_matrix, dim=0)

        val_concept_matrix = val_concept_matrix - torch.mean(val_concept_matrix, dim=0)
        val_concept_matrix = val_concept_matrix / torch.std(val_concept_matrix, dim=0)

        os.makedirs(self.path_save / "embeddings", exist_ok=True)
        torch.save(train_concept_matrix, Path(f'{self.path_save}/embeddings/train_concept_matrix.pt'))
        torch.save(val_concept_matrix, Path(f'{self.path_save}/embeddings/val_concept_matrix.pt'))
       
        return train_concept_matrix, val_concept_matrix

if __name__=='__main__':
    CZS= Clip_zero_shot()
    CZS.train_and_val_matrix()
    CZS.test(False)
    CZS.leakage_loaders()
