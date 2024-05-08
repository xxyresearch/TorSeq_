#import block

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pickle
import pandas as pd
from tqdm import tqdm
#import yaml
import os.path as osp
import torch
from utils.s_sampling import *
from utils.parsing import parse_train_args
from torch.utils.data import  DataLoader
from model.score_model import TensorProductScoreModel


args = parse_train_args()

model_name = 'TorSeq+TorDiff'
path = './weighted_model/'
save_name = './sampled_pickle/' + model_name  +'.pkl'
csv_dir = './data/DRUGS/test_smiles.csv'
pickle_dir = './data/DRUGS/test_mols.pkl'

weighted_model = 'weighted_model/TorSeq+TorDiff.pth'

inference_steps = 20

model = TensorProductScoreModel(in_node_features=args.in_node_features, in_edge_features=args.in_edge_features,
                        ns=args.ns, nv=args.nv, sigma_embed_dim=args.sigma_embed_dim,
                        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                        num_conv_layers=args.num_conv_layers,
                        max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim,
                        scale_by_sigma=args.scale_by_sigma,
                        use_second_order_repr=args.use_second_order_repr,
                        residual=not args.no_residual, batch_norm= not args.no_batch_norm, rnn= not args.no_use_lstm,
                        rnn_layers= args.lstm_layer_num)

model.load_state_dict(torch.load(weighted_model)) 
model.eval()
model.to(args.device)

print(model)
print(f'sample pkl name {save_name}')
print(f'denoising steps {inference_steps}')
print(f'model name {weighted_model}')

def sample(conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,
           ode=False, likelihood=None, pdb=None, device = 'cpu'):

    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(dataset = conf_dataset, batch_size = batch_size, shuffle = False, collate_fn=collate_batch)

    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps

    for batch_idx, data in enumerate(tqdm(loader)):
        dlogp = torch.zeros(len(data.name))
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule):

            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)

            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()
            perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
            data_gpu.pos = data.pos.to(device)

            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()

    return conformers

def sample_confs(raw_smi, n_confs, smi,  batch_size = args.batch_size, device = args.device):
    
    mol, data = get_seed(smi, dataset=args.data_type)
    if not mol:
        print('Failed to get seed', smi)
        return None
    n_rotable_bonds = int(data.edge_mask.sum())
    conformers = embed_seeds(mol, data, n_confs, embed_func=embed_func)
    conformers = perturb_seeds(conformers)
    if n_rotable_bonds > 0.5:
        conformers = sample(conformers, model, args.sigma_max, args.sigma_min, inference_steps,
                             batch_size = args.batch_size, device = args.device)

    mols = [pyg_to_mol(mol, conf, rmsd = not False ) for conf in conformers]

    
    return mols

real_mol_set = open_pickle(pickle_dir)
test_data = pd.read_csv(csv_dir).values
new_conf_dict = {}
import pickle

for smi_idx, (raw_smi, n_confs, smi) in enumerate(test_data):
    real_mols = real_mol_set[raw_smi]
    real_mols = clean_confs(smi, real_mols)
    if real_mols == []:
        print(f'mol {smi_idx}  has bad groundtruth')
        continue
    generate_mols= sample_confs(raw_smi, 200, smi, batch_size = args.batch_size, device = args.device)  #, seq_length 
    mol_name = raw_smi
    new_conf_dict[mol_name] = generate_mols

with open(save_name , 'wb') as file:
    pickle.dump(new_conf_dict , file)
print('successfully saved')
