#!/bin/bash
#SBATCH -o slurm_logs/run_cbm.log-%j

python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py --dataset phoning_eating_balanced --backbone alexnet_phoning_eating --concept_set data/concept_sets/phoning_eating_filtered_new.txt --feature_layer avgpool --n_iters 1000 --print
