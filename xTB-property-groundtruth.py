import pickle, tqdm
from argparse import ArgumentParser
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from utils.xtb import *
import numpy as np
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--in_confs', type=str, required=True, help='Pickle with input conformers')
parser.add_argument('--skip', type=int, default=1, help='Frequency for running procedure')
parser.add_argument('--level', type=str, default="normal", help='xTB optimization level')
parser.add_argument('--xtb_path', type=str, default=None, help='Specifies local path to xTB installation')
args = parser.parse_args()

## this is 'corrected_smiles'
random_100 = pd.read_csv('data/DRUGS/rand-100.csv')
random_100_list = list(random_100['Random-100'])
test_data = pd.read_csv('data/DRUGS/test_smiles.csv').values
test_data = test_data[::args.skip]
new_mols = pickle.load(open(args.in_confs, 'rb'))

# find raw file change "\"" to "\\"" and "/"" to "_" 
fixed_smi = [
"CCCCCC_C(=N\\NS(C)(=O)=O)c1ccccc1",
"CN1C(=S)N(c2ccccc2)C(=O)_C1=C_c1ccco1",
"COc1cc(_C=N_Nc2cccc(C)c2)cc(Br)c1O",
"COc1ccc(C(=O)O_N=C(\\N)c2ccc(OC)c(OC)c2)cc1",
"Cc1ccc(NCC(=O)N_N=C\c2ccc([N+](=O)[O-])o2)cc1",
"O=C(N_N=C1\\NC(=O)C(CO)(CO)S1)c1ccccc1",
"S=c1[nH]nc(COc2ccccc2)n1_N=C_C=C_c1ccco1",
]

def clean_confs_with_weight(smi, confs, limit=None):
    good_ids = []
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    except Exception as e:
        print('Error', smi, e)
        return []
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c['rd_mol'], sanitize=False),
                                    isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
        if len(good_ids) == limit:
            break
    
    return [confs[i]['rd_mol'] for i in good_ids], np.array([confs[i]['boltzmannweight'] for i in good_ids])

ground_truth = {}
data_path = './data/DRUGS/drugs/'
fixed_idx = 0
for smi, _, _ in test_data:
    if smi in new_mols.keys():
        try:
        #raw_data = pickle.load(open(data_path + named_smi +'.pickle', 'rb')) 
            raw_data = pickle.load(open(data_path + smi +'.pickle', 'rb'))
        except:
            raw_data = pickle.load(open(data_path + fixed_smi[fixed_idx] +'.pickle', 'rb'))
            fixed_idx +=1
        #ground_truth[smi] = [1]
        #continue
        smi = raw_data['smiles']
        confs = raw_data['conformers']
        mol_ = Chem.MolFromSmiles(smi)
        canonical_smi = Chem.MolToSmiles(mol_)
        confs, weight = clean_confs_with_weight(canonical_smi, confs)
        normalize_weight = weight / np.sum(weight)

        for conf in tqdm(confs):
            if args.xtb_path:
                #if xtb_energy:
                #    success = xtb_optimize(conf, args.level, path_xtb=args.xtb_path)
                #    if not success: continue
                res = xtb_energy(conf, dipole=True, path_xtb=args.xtb_path)
                if not res: continue
                conf.xtb_energy, conf.xtb_dipole, conf.xtb_gap, conf.xtb_runtime = res['energy'], res['dipole'], res['gap'], res['runtime']

        total_xtb_energy = []
        total_xtb_dipole = []
        total_xtb_gap = []
        for idx, conf in enumerate(confs):
            total_xtb_energy.append(conf.xtb_energy)
            total_xtb_dipole.append(conf.xtb_dipole)
            total_xtb_gap.append(conf.xtb_gap)
        ground_truth[smi] = [total_xtb_energy , total_xtb_dipole , total_xtb_gap, normalize_weight ]
        #print(smi, ground_truth[smi],total_weight)

open('./xTB_pickle/xTB_groundtruth_weighted.pkl', 'wb').write(pickle.dumps(ground_truth))