import os
import sys
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from fairness_cv_project.methods.label_free_cbm.src.utils import data_utils, utils
import fairness_cv_project.methods.label_free_cbm.src.similarity as similarity
from fairness_cv_project.methods.label_free_cbm.src.glm_saga.elasticnet import IndexedTensorDataset, glm_saga

def train_cbm_and_save(args):

    logging.info('Loading concept and classes')    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.concept_set==None:
        # args.concept_set = "data/concept_sets/{}_filtered_new.txt".format(args.dataset)
        args.concept_set = "data/concept_sets/{}_gpt3_ensemble2.txt".format(args.dataset)
        
    similarity_fn = similarity.cos_similarity_cubed_single
    
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    
    # Get concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

     
    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    # Define protected concepts
    protected_concepts = args.protected_concepts
    
    #save activations and get save_paths
    logging.info('Saving activations from backbone and CLIP')
    for d_probe in [d_train, d_val]:
        utils.save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
    logging.info('Loading activations from backbone and CLIP')
    logging.debug('Fetching save names for activations')
    target_save_name, clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer, d_train, args.concept_set, "avg", args.activation_dir, args.name_suffix)
    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir, args.name_suffix)
    
    logging.debug('Loading activations from backbone and CLIP, and normalizing')

    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
    
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        logging.debug('Calculating CLIP similarity matrix')        
        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features
    
    #filter concepts not activating highly
    logging.info('Filtering concepts not activating highly')
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)
    
    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i]<=args.clip_cutoff and concept not in protected_concepts:
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
    concepts_idx = [i for i in range(len(concepts)) if highest[i]>args.clip_cutoff or concepts[i] in protected_concepts]
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.clip_cutoff or concepts[i] in protected_concepts]

    #save memory by recalculating
    del clip_features

    logging.info('Recomputing similarity matrix, with filtered concepts')
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[concepts_idx]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
    
        clip_features = image_features @ text_features.T
        del image_features, text_features    
    val_clip_features = val_clip_features[:, concepts_idx]

    logging.info('Training projection layer from backbone to concepts')    
    #learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()

    logging.info('Training done, saving best projection layer') 
    proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
    #delete concepts that are not interpretable
    logging.info('Deleting concepts that are not interpretable in validation data')
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff
        
    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i]<=args.interpretability_cutoff and concept not in protected_concepts:
                print("Deleting {}, Interpretability:{:.3f}".format(concept, sim[i]))

    concepts_idx = [i for i in range(len(concepts)) if interpretable[i] or concepts[i] in protected_concepts]
    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i] or concepts[i] in protected_concepts]

    logging.debug('Remaining concepts: {}'.format(concepts)) 
    del clip_features, val_clip_features
    
    logging.info('Creating new projection layer with interpretable concepts')
    W_c = proj_layer.weight[concepts_idx]
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    
    logging.info('Creating dataset with interpretable concepts predicted by the projection layer and final targets')
    train_targets = data_utils.get_targets_only(d_train)
    val_targets = data_utils.get_targets_only(d_val)
    
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
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


    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    logging.info('Training the final linear model from interpretable concepts to final targets')
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                      val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes))
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    logging.info('Saving the final model')
    concept_set_name = args.concept_set.split("/")[-1].split(".")[0]  
    save_name = f"{args.save_dir}/{args.dataset}_{concept_set_name}_cbm_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    save_name = f"{args.save_dir}/{args.dataset}_{concept_set_name}"
    os.mkdir(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)
 

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Settings for creating CBM')

    logging.debug('Parsing dataset, concept set, backbone and clip name')
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--concept_set", type=str, default=None, 
                        help="path to concept set name")
    parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

    logging.debug('Parsing torch device')
    parser.add_argument("--device", type=str, default="cuda", help="Which device to use")

    logging.debug('Parsing hyperparameters')
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
    parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
    parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

    parser.add_argument("--feature_layer", type=str, default='layer4', 
                        help="Which layer to collect activations from. Should be the name of second to last layer in the model")
    parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
    parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
    parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
    parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
    parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
    parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
    parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
    parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix to add to saved model name")
    parser.add_argument('--protected_concepts', nargs='+', default=[], help='protected concepts')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args

# Defining log files

log_file = 'run_{}.log'.format(datetime.datetime.now().strftime("%Y%m%d%H%M"))
log_path = Path.cwd() / 'logs' / log_file

logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


if __name__=='__main__':
    random.seed(0)
    torch.manual_seed(0)
    args = parse_args()
    train_cbm_and_save(args)