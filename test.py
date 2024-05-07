from rdkit import RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
import pickle
import pandas as pd
from tqdm import tqdm
#import yaml
import os.path as osp
import torch
from argparse import ArgumentParser

from dataset.utils import *
from rdkit.Chem.rdMolAlign import GetBestRMS

parser = ArgumentParser()
parser.add_argument('--save', type=str, default='TorSeq', help='input filename')
args = parser.parse_args()


save_name = './sampled_pickle/' + args.save +'.pkl'
csv_dir = './data/DRUGS/test_smiles.csv'
pickle_dir = './data/DRUGS/test_mols.pkl'
real_mol_set = open_pickle(pickle_dir)
test_mol_set = open_pickle(save_name)
test_data = pd.read_csv(csv_dir).values


def add_to_dict(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]
    return d

recall_rate_list = []
recall_rate_75_list = []
prec_rate_list = []
prec_rate_75_list = []

for smi_idx, (raw_smi, n_confs, smi) in enumerate(test_data):
    real_mol = real_mol_set[raw_smi]
    if not real_mol:
        print('no good ground real')
    
    generate_mol = None
    try:
        generate_mol = test_mol_set[raw_smi][0]
    except:
        print(f'mol {smi_idx} has no good_ground truth')
        print(generate_mol)
        continue
    try:
        real_mol = clean_confs_test(smi, real_mol)
    except:
        print('bad real')
        continue

    if generate_mol and real_mol:
        if len(generate_mol) < 2 * len(real_mol):
            print(f'mol {smi_idx} fail in generate')
            continue
        if len(generate_mol) > 2 * len(real_mol):
            generate_mol = generate_mol[:len(real_mol)*2]
    else:
        print('gen_mol',generate_mol)
        print('real_mol', real_mol)
        print(f'mol {smi_idx} has 2222')
        continue

    generate_set = []
    real_set = []
    for mol in real_mol:
        real_set.append(Chem.RemoveHs(mol))
    
    for mol in generate_mol:
        generate_set.append(Chem.RemoveHs(mol))

    score_matrix = []
    try:
        for i in tqdm(real_set):
            score_list = []
            for j in generate_set:
                score_list.append(GetBestRMS(i,j))
            score_matrix.append(score_list)
    except:
        print(f'mol {smi_idx}  in fail to compare RMS')
        continue
    

    score_matrix = torch.tensor(score_matrix)
    best_prec, prec_min_indice = torch.min(score_matrix, dim=0)
    best_recall , recall_min_indice = torch.min(score_matrix, dim=1)
    recall_rate_75 = (torch.sum((best_recall < 0.75).float())/len(best_recall)).item()
    recall_rate = (torch.mean(best_recall)).item()
    prec_rate_75 = (torch.sum((best_prec < 0.75).float())/len(best_prec)).item()
    prec_rate = (torch.mean(best_prec)).item()
    recall_rate_list.append(recall_rate)
    recall_rate_75_list.append(recall_rate_75*100)
    prec_rate_list.append(prec_rate)
    prec_rate_75_list.append(prec_rate_75*100)
    if smi_idx %5 ==0:
        print(smi_idx , "{:.1f}".format(torch.mean(torch.tensor(recall_rate_75_list)).item()),"{:.1f}".format(torch.median(torch.tensor(recall_rate_75_list)).item()),
    "{:.3f}".format(torch.mean(torch.tensor(recall_rate_list)).item()),"{:.3f}".format(torch.median(torch.tensor(recall_rate_list)).item()),
    "{:.1f}".format(torch.mean(torch.tensor(prec_rate_75_list)).item()),"{:.1f}".format(torch.median(torch.tensor(prec_rate_75_list)).item()),
    "{:.3f}".format(torch.mean(torch.tensor(prec_rate_list)).item()),"{:.3f}".format(torch.median(torch.tensor(prec_rate_list)).item()),
        )   

print(f'tested_mol {len(recall_rate_75_list)}')
print(f'the last mol recall list shape{len(best_recall)}, prec recall list shape {len(best_prec)} ')
print(smi_idx , "{:.1f}".format(torch.mean(torch.tensor(recall_rate_75_list)).item()),"{:.1f}".format(torch.median(torch.tensor(recall_rate_75_list)).item()),
"{:.3f}".format(torch.mean(torch.tensor(recall_rate_list)).item()),"{:.3f}".format(torch.median(torch.tensor(recall_rate_list)).item()),
"{:.1f}".format(torch.mean(torch.tensor(prec_rate_75_list)).item()),"{:.1f}".format(torch.median(torch.tensor(prec_rate_75_list)).item()),
"{:.3f}".format(torch.mean(torch.tensor(prec_rate_list)).item()),"{:.3f}".format(torch.median(torch.tensor(prec_rate_list)).item()),
    )   
    