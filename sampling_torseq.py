from tqdm import tqdm
import torch
import torch.nn as nn
from copy import deepcopy
from utils.utils import *
import model.torus as torus
from math import pi
from argparse import ArgumentParser
from rdkit.Chem.rdMolTransforms import SetDihedralRad
import time

parser = ArgumentParser()
parser.add_argument('--split_path', type=str, default='./data/DRUGS/split.npy', help='Path of file defining the split')
parser.add_argument('--data_type', type=str, default='drugs', help='Path of file defining the split')
parser.add_argument('--max_seq_length', type=int, default=100, help='Max Sequence Length')
parser.add_argument('--conf_num', type=int, default=30, help='the number of generate conf')
parser.add_argument('--std_dir', type=str, default='/data/DRUGS/standardized_pickles', help='Folder in which the pickle are put after standardisation/matching')
parser.add_argument('--device', type=str, default='cpu', help='the device id')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--mpnn_conv', action='store_false', default= True, help='Whether to use mpnn conv')
parser.add_argument('--gnn_layer_num', type=int, default=3, help='Number of interaction layers')
parser.add_argument('--latent_dim', type=int, default=10, help='the noise dim of mpnn input')
parser.add_argument('--hidden_dim', type=int, default=64, help='the hidden dim of mpnn layer')
parser.add_argument('--model_name', type=str, default='TorSeq', help='the name of saved model')

parser.add_argument('--no_use_lstm', action='store_true', default = False, help='whether not use lstm for torsional angle')
parser.add_argument('--no_res_lstm', action='store_true', default = False, help='whether not use residual structure for lstm')
parser.add_argument('--use_motif_gcn', action='store_true', default = False, help='whether use motif-level gcn')
parser.add_argument('--no_random_start', action='store_true', default = False, help='whether not start frin random degree')
parser.add_argument('--no_local_feature', action='store_true', default = False, help='whether not use local feature')

parser.add_argument('--lstm_layer_num', type=int, default=2, help='the number of lstm layer')
parser.add_argument('--lstm_hidden_dim', type=int, default=512, help='the hidden dim of lstm layer')
parser.add_argument('--max_length', type=int, default=100, help='max length of torsional sequence')
parser.add_argument('--bidirection', action='store_false', default= True, help='Whether to use BiDirectional LSTM')
parser.add_argument('--in_node_features', type=int, default=74, help='Dimension of node features: 74 for drugs and xl, 44 for qm9')
parser.add_argument('--in_edge_features', type=int, default=4, help='Dimension of edge feature (do not change)')
parser.add_argument('--pickle_name', type=str, default='default', help='the name of saved mol')
args = parser.parse_args()


conf_mode_type = 'Original'

weighted_model = "./weighted_model/" + args.model_name + ".pth"

model = get_model(args)
test_loader = get_test_dataloader(conf_mode_type, args)
model.to(args.device)
model.load_state_dict(torch.load(weighted_model, map_location=torch.device(args.device)))
model.eval()
generation_time = 0
conf_num_total = 0
new_conf_dict = {}

with torch.no_grad():
    for data_idx, batch_data in enumerate(tqdm(test_loader )):
        start_time = time.time()
        data = batch_data.to(args.device) 

        if torch.sum(data.dihedral_indx) == 0:
            mol_name = data.smi
            new_conf_dict[mol_name] = [data.gen_mol_list, data.dihedral_indx.shape[0]]    
            continue       
        gen_num = len(data.gen_mol_list)
        gen_mol_list = data.gen_mol_list
        generate_mols = []

        dihedral_rad_pred_matrix = torch.zeros((gen_num, data.dihedral_indx.shape[0]), dtype = torch.float).to(args.device)
        x = data.x
        x = x.unsqueeze(1).repeat(1, gen_num ,1)
        x = x.permute(1,0,2)
        latent = torch.rand([gen_num, x.size(1), args.latent_dim]).to(args.device)
        x_with_noise = torch.cat((x, latent), dim=-1)

        for conf_idx in range(len(gen_mol_list)):
            data = batch_data.to(args.device)
            dihedral_lstm_mask = (data.dihedral_lstm_index - data.dihedral_lstm_index[:,0].view(-1,1)) != 0
            dihedral_lstm_mask[:,0] = True
            x_feature = x_with_noise[conf_idx]
            if not args.no_random_start:
                random_start = torch.rand(len(dihedral_rad_pred_matrix[conf_idx]),1).to(args.device) * 2 * pi - pi
            else:
                random_start = torch.zeros(len(dihedral_rad_pred_matrix[conf_idx]),1).to(args.device) 
            dihedral_rad = model(data, args, random_start, x_feature, conf_idx, test_mode = True)
            dihedral_rad_pred_matrix[conf_idx] = torch.squeeze(dihedral_rad)

        for mol_idx, dihedral_group in enumerate(dihedral_rad_pred_matrix):
            gen_mol = deepcopy(data.gen_mol_list[mol_idx])
            conf = gen_mol.GetConformer()
            for dihedral_id, bond in enumerate(data.dihedral_indx):
                phi_hat = dihedral_group
                SetDihedralRad(conf, int(bond[0]), int(bond[1]), int( bond[2]), int( bond[3]), float(phi_hat[dihedral_id]) )
            generate_mols.append(gen_mol)

        mol_name = data.smi
        new_conf_dict[mol_name] = [generate_mols, data.dihedral_indx.shape[0]]
        end_time = time.time()
        generation_time += end_time - start_time 
        conf_num_total += gen_num

import pickle
save_name = './sampled_pickle/' + args.pickle_name  +'.pkl'
with open(save_name , 'wb') as file:
    pickle.dump(new_conf_dict , file)
print('successfully saved')

print(generation_time / conf_num_total )
