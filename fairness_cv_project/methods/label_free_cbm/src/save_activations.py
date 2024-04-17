import os
import sys
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))

import os
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader


from fairness_cv_project.methods.label_free_cbm.src.utils import data_utils

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook



def save_activation(target_name, save_name, d_probe, batch_size=512, target_layer = 'layer4', pool_mode='avg', device='cuda'):
    
    target_model, target_preprocess = data_utils.get_target_model(target_name, device, d_probe)
    dataset = data_utils.get_data(d_probe, target_preprocess)
    
    all_features = []
    hooks = {}
        
    command = "target_model.{}.register_forward_hook(get_activation(all_features, pool_mode))".format(target_layer)
    hooks[target_layer] = eval(command)
        
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    if not os.path.exists(save_name.parent):
        os.makedirs(save_name.parent)
    
    
    torch.save(torch.cat(all_features), save_name)
    hooks[target_layer].remove()
    
    # Free memory
    del all_features
    torch.cuda.empty_cache()
    return
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imSitu_200_full')
    parser.add_argument('--target_name', type=str, default='resnet50')
    
    args = parser.parse_args()
    dataset = args.dataset
    target_name = args.target_name

    d_train = f'{dataset}_train'
    d_val = f'{dataset}_val'
    save_folder = Path('saved_activations/saved/')

    for d_probe in [d_train, d_val]:
        save_name = save_folder / d_probe / 'resnet50_layer4_avg.pt'
        save_activation(target_name, save_name, d_probe)