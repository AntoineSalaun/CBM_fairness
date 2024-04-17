import os
import sys
from pathlib import Path

sys.path.insert(1, str(Path.cwd()))

import json
import os
import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import fairness_cv_project.methods.label_free_cbm.src.similarity as similarity
from fairness_cv_project.methods.label_free_cbm.src.glm_saga.elasticnet import (
    IndexedTensorDataset, glm_saga)
from fairness_cv_project.methods.label_free_cbm.src.utils import (data_utils,
                                                                  utils)


def train_final_layer(saved_projection_path, dataset, save_name, len_classes, lam, n_iters=80, saga_batch_size=256, device='cuda'):
    
    # Loading the projection layer
    W_c = torch.load(os.path.join(saved_projection_path ,"W_c.pt"), map_location='cpu').to(device)
    proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
    proj_layer.load_state_dict({"weight":W_c})
    
    # Loading the activation
    path_saved_activation = Path('saved_activations/saved')
    target_save_name = path_saved_activation / f'{dataset}_train' / 'resnet50_layer4_avg.pt'
    val_target_save_name = path_saved_activation / f'{dataset}_val' / 'resnet50_layer4_avg.pt'

    # Loading the targets
    d_train = f'{dataset}_train'
    d_val = f'{dataset}_val'
    
    train_targets = data_utils.get_targets_only(d_train)
    val_targets = data_utils.get_targets_only(d_val)
    
    # Projecting the activation to CBL, then normalizing CBL and putting values into Tensor
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float().to(device)
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float().to(device)

        train_c = proj_layer(target_features)
        val_c = proj_layer(val_target_features)
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std
        
        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std
        
        val_y = torch.LongTensor(val_targets)

        val_ds = TensorDataset(val_c,val_y)

    # Creating train loader
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1], len_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, n_iters, ALPHA, epsilon=1, k=1,
                      val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len_classes)
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    save_dir = f'saved_models/imSitu/custom/{save_name}'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    torch.save(train_mean, os.path.join(save_dir, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_dir, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_dir ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(save_dir, "b_g.pt"))
    
    
    shutil.copy(Path(saved_projection_path) / 'args.txt', os.path.join(save_dir, 'args.txt'))
    
    with open(os.path.join(save_dir, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_projection_path', type=str, default='saved_models/imSitu/200_verbs_full/CBM/dense_no_gender/imSitu_200_full_imSitu_200_filtered')
    parser.add_argument('--dataset', type=str, default='imSitu_200_imbalanced_1')
    parser.add_argument('--save_name', type=str, default='CBM_imSitu_200_full/activation_imSitu_200_imbalanced_1')
    parser.add_argument('--len_classes', type=int, default=200)
    parser.add_argument('--lam', type=float, default=0.0007)
    args = parser.parse_args()

    saved_projection_path = args.saved_projection_path
    dataset = args.dataset
    save_name = args.save_name
    len_classes = args.len_classes
    lam = args.lam

    train_final_layer(saved_projection_path, dataset, save_name, len_classes, lam)