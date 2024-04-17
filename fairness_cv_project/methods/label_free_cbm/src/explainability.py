import torch
import argparse
import os
import sys
import random
import json
from pathlib import Path

import pandas as pd
sys.path.insert(1, str(Path.cwd()))
from fairness_cv_project.methods.label_free_cbm.src.plots import plots
from fairness_cv_project.methods.label_free_cbm.src.utils import data_utils, utils
from fairness_cv_project.methods.label_free_cbm.src.models import cbm

def explainability(load_dir, file_name):
    # change this to the correct model dir, everything else should be taken care of

    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True)

    root = Path.cwd()
    path_model_dir = root / load_dir
    device = "cuda" if torch.cuda.is_available() else 'cpu'


    with open(root / os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
        
    dataset = args["dataset"]
    model = cbm.load_cbm(path_model_dir, device)

    cls_file = data_utils.LABEL_FILES[dataset]

    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().split("\n")

    # Save weights to excel

    with pd.ExcelWriter(file_name) as writer:  
        all_weights = []
        for verb in classes:
            i = classes.index(verb)
            data = []
            for j in range(len(concepts)):
                if torch.abs(model.final.weight[i,j])>0.01:
                    data.append({"Concept": concepts[j], "Weight": model.final.weight[i,j].item(), "Class": classes[i]})
            data.append({"Concept": 'DEFAULT', "Weight": 0, "Class": classes[i]})
            df = pd.DataFrame(data)
            df.sort_values(['Class', 'Weight'], ascending=[True, False], inplace=True)
            df.to_excel(writer, sheet_name=verb, index=False)
            all_weights.append(df)
        
        # For the first sheet, concatenate all dataframes and write to the sheet
        pd.concat(all_weights).to_excel(writer, sheet_name='All Weights', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default='saved_models/imSitu/30_verbs/CBM/no_gender/dense/balanced/imSitu_30_balanced_imSitu_30_filtered')
    parser.add_argument('--file_path', type=str, default='results/imSitu/30_verbs/dense_balanced_no_gender.xlsx')

    args = parser.parse_args()
    load_dir = Path(args.load_dir)
    file_path = Path(args.file_path)

    explainability(load_dir, file_path)