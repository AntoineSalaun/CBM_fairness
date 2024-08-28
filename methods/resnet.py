import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from pathlib import Path
import math, os, random, sys, argparse, copy
from tqdm import tqdm as tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torchvision.models import ResNet50_Weights
from data_loader import ImSituLoader
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, parsing = True, dataset = 'imSitu', num_verb = 200, balance = 'imbalanced', lr = 0.001, weight_decay = 0.8,gamma = 0.1, num_epochs = 25, seed = 0, target = 'verb'):

        super(ResNet, self).__init__()
        self.params = [
        ("dataset", str, None, dataset),
        ("num_verb", int, None, num_verb),
        ("balance", str, None, balance),
        ("lr", float, None, lr),
        ("weight_decay", float, None, weight_decay),
        ("gamma", float, None, gamma),
        ("num_epochs", int, None, num_epochs),
        ("seed", int, None, seed),
        ("target", str, None, target)
        ] 
        
        if parsing == True :       
            parser = argparse.ArgumentParser(description='Settings for creating a ResNet')

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

        self.net = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.net.parameters(): 
            param.requires_grad = False
        self.net.fc = nn.Linear(self.net.fc.in_features, self.num_verb)  # replace the last FC layer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = self.net.to(self.device)

        self.optimizer = optim.Adam(self.net.fc.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        if self.target == 'verb': self.path_save = Path(f'saved_models/{self.dataset}/{self.num_verb}_verbs/ResNet/{self.balance}')
        elif self.target == 'gender': self.path_save = Path(f'saved_models/{self.dataset}/2_genders/ResNet/{self.balance}')

        os.makedirs(self.path_save, exist_ok=True)
        self.momentum = self.weight_decay
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)         # Decay LR by a factor of 0.1 every 5 epochs


        self.train_data = ImSituLoader(balance=self.balance, split = 'train', transform_name='ResNet')
        self.val_data = ImSituLoader(balance=self.balance, split = 'val', transform_name='ResNet')
        self.test_data = ImSituLoader(balance=self.balance, split = 'test', transform_name='ResNet')

        self.train_targets = self.train_data.get_verbs() if self.target == 'verb' else self.train_data.get_genders()
        self.val_targets = self.val_data.get_verbs() if self.target == 'verb' else self.val_data.get_genders()
        self.test_targets = self.test_data.get_verbs() if self.target == 'verb' else self.test_data.get_genders()

        if os.path.exists(self.path_save / 'model_best.pt'):
            self.net.load_state_dict(torch.load(self.path_save / 'model_best.pt',map_location='cpu'))
        
        path_verbs = Path(f'data/datasets/{self.dataset}/data_processed/{self.num_verb}_verbs/{self.num_verb}_verbs.txt')
        with open(path_verbs, "r") as f: self.verbs = f.read().split("\n")     # read the verbs


    def train(self, use_existing_embeddings = True):
        train_metrics = []
        # Reset the weights with Xavier uniform initialization
        nn.init.xavier_uniform_(self.net.fc.weight)
        nn.init.constant_(self.net.fc.bias, 0.0)
        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_acc = 0.0

        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)
        
        
        for epoch in tqdm(range(self.num_epochs)):
            
            train_loss, train_acc, train_f1, train_pred, train_prob = self.epoch_pass(epoch , train_loader, training=True)
            val_loss, val_acc, val_f1, train_pred, train_prob = self.epoch_pass(epoch , val_loader, training=False)

            train_metrics.append(f'Epoch {epoch} -  train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_f1: {train_f1:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_f1: {val_f1:.4f}')
            print(train_metrics[-1])

            self.scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.net.state_dict())

        self.net.load_state_dict(best_model_wts)
        _, train_acc, train_f1, train_pred, train_prob = self.epoch_pass(-1 , train_loader, training=False)
        _, val_acc, val_f1, train_pred, train_prob = self.epoch_pass(-1, val_loader, training = False)

        torch.save(self.net.state_dict(), self.path_save / 'model_best.pt')

        return train_metrics, train_acc, train_f1, val_acc, val_f1

    def epoch_pass(self, epoch, data_loader, training):
        preds = list()
        truth = list()
        pred_probabilities = list()
        epoch_loss = 0

        if training:
            self.net.train()
        else:
            self.net.eval()

        for inputs, verbs, _, image_name, _ in data_loader:
            inputs = inputs.to(self.device)
            verbs = verbs.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                layer_predictions = self.net(inputs)
                probabilities = F.softmax(layer_predictions, dim=1)
                predictions = np.argmax(probabilities.cpu().detach().numpy(), axis=1) 

                batch_loss = self.criterion(layer_predictions, verbs)
                #print('image ', image_name[0], 'was predicted as ', self.verbs[predictions[0]])
                pred_probabilities += probabilities.cpu().tolist()
                preds += predictions.tolist()
                truth += np.argmax(verbs.cpu().detach().numpy(), axis=1).tolist()
                epoch_loss += batch_loss.item()

                if training:
                    batch_loss.backward()
                    self.optimizer.step()

        epoch_acc = accuracy_score(truth, preds)
        epoch_f1 = f1_score(truth, preds, average='weighted')

        return epoch_loss, epoch_acc, epoch_f1, preds, pred_probabilities
       
    def save_results(self, train_metrics, train_acc, train_f1, val_acc, val_f1, test_acc, test_f1):    
        
        print(str(self.arg_str))
        print('train_acc: ',train_acc, 'train_f1 : ', train_f1, 'val_acc : ', val_acc, 'val_f1 : ', val_f1, 'test_acc : ', test_acc, 'test_f1 : ', test_f1 )

        with open(self.path_save / 'results.txt', 'w') as f:
            f.write('train_acc: {0:.4f}, train_f1: {1:.4f}, val_acc: {2:.4f}, val_f1: {3:.4f}, test_acc: {4:.4f}, test_f1: {5:.4f}'.format(train_acc, train_f1, val_acc, val_f1, test_acc, test_f1))
        
        with open(self.path_save / 'args.txt', 'w') as f:
            f.write(str(self.args))
        
        with open(self.path_save / 'train_metrics.txt', 'w') as f:
            for item in train_metrics:
                f.write("%s\n" % item)

    def test(self, use_existing_embeddings = True):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)
        self.net.load_state_dict(torch.load(self.path_save / 'model_best.pt',map_location='cpu'))
        
        test_loss, test_acc, test_f1, pred, prob = self.epoch_pass(-1, test_loader, training = False)

        return test_loss, test_acc, test_f1, 0
    
        
    def leakage_loaders(self, layer = 'predicitions'):
        
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size = 16, shuffle = False, num_workers = 0, pin_memory = True)
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size = 16, shuffle = False, num_workers = 0, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size = 16, shuffle = False, num_workers = 0, pin_memory = True)
        
        #train_loader = DataLoader(IndexedTensorDataset(self.matrix_train, self.train_data.get_verbs(), self.train_data.get_genders()), batch_size=self.batch_size, shuffle=False)
        #_, _, _, train_preds, _,train_prob, adv_loss, train_leak = self.fl.epoch_pass(-1, train_loader, training=False)


        train_loss, train_acc, train_f1, train_pred, train_prob = self.epoch_pass(-1, train_loader, training=False)
        val_loss, val_acc, val_f1, val_pred, val_prob = self.epoch_pass(-1, val_loader, training=False)
        test_loss, test_acc, test_f1, test_pred, test_prob = self.epoch_pass(-1, test_loader, training=False)

        train_data_copy = copy.deepcopy(self.train_data)
        val_data_copy = copy.deepcopy(self.val_data)
        test_data_copy = copy.deepcopy(self.test_data)

        if layer == 'probabilities':
            train_pv = train_prob
            val_pv = val_prob
            test_pv = test_prob
        else:
            train_pv = train_pred
            val_pv = val_pred
            test_pv = test_pred 
                
        #print('train_pv[0]', train_pv[0], 'val_pv[0]', val_pv[0], 'test_pv[0]', test_pv[0])

        if len(train_data_copy.image_metadata) != len(train_pv):
            raise ValueError(f"Length of train_pv ({len(train_pv)}) does not match train_data_copy.image_metadata ({len(train_data_copy.image_metadata)})")
        if len(val_data_copy.image_metadata) != len(val_pv):
            raise ValueError(f"Length of val_pv ({len(val_pv)}) does not match val_data_copy.image_metadata ({len(val_data_copy.image_metadata)})")
        if len(test_data_copy.image_metadata) != len(test_pv):
            raise ValueError(f"Length of test_pv ({len(test_pv)}) does not match test_data_copy.image_metadata ({len(test_data_copy.image_metadata)})")

        train_data_copy.change_verbs(train_pv)
        val_data_copy.change_verbs(val_pv)
        test_data_copy.change_verbs(test_pv)
        
        return train_data_copy, val_data_copy, test_data_copy



if __name__ == '__main__':
    # Assuming you have a ResNet class definition available as provided
    resnet = ResNet(parsing=False, balance="imbalanced", num_epochs=25)  # Adjust parameters as necessary

    # Call the leakage loaders
    train_data, val_data, test_data = resnet.leakage_loaders('predicitions')
    train_data, val_data, test_data = resnet.leakage_loaders('probabilities')



