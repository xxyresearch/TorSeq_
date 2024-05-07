import os.path as osp
import numpy as np
import pandas as pd
import glob
import torch
from torch_geometric.data import  Data, InMemoryDataset
from typing import Callable, Optional
from tqdm import tqdm
import glob
import time
from dataset.utils import *

class OriginDataset(InMemoryDataset):
    def __init__(self, cmt, args,  mode, split_num,
                    root = '.',
                    transform: Optional[Callable] = None,
                    pre_transform: Optional[Callable] = None,
                    pre_filter: Optional[Callable] = None,       
                    ):
        self.cmt = cmt
        self.mode = mode
        self.root = root
        self.split_num = split_num
        self.std_dir = args.std_dir
        self.split_path = args.split_path
        self.data_type = args.data_type
        self.max_seq_length = args.max_seq_length
        self.max_conf_num = args.conf_num

        super(OriginDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return self.cmt + self.data_type + self.mode + str(self.split_num) + '.pt'

    def process(self):
        data_list = []
        if self.data_type == 'qm9':
            atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            data_dir = self.root + '/data/QM9/qm9/'
        else:
            atom_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
                'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
                'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
                'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
            data_dir = self.root + '/data/DRUGS/drugs/'
        all_files = sorted(glob.glob(osp.join(data_dir, '*.pickle')))
        split_idx = 0 if self.mode == 'Train' else 1 if self.mode == 'Val' else 2
        #mol_num = 20000 if self.mode == 'train' else 2000
        split = np.load(self.split_path, allow_pickle=True)[split_idx]
        pickle_files = [f for i, f in enumerate(all_files) if i in split]
        rd_std_root = self.root + self.std_dir 
        rd_std_files = sorted(glob.glob(osp.join(rd_std_root , '*.pickle')))
        pickle_file_name = [s.replace('.pickle', "") for s in [t.replace(data_dir, "") for t in pickle_files]]       

        if self.data_type == 'drugs' and self.mode == 'Train':
            upper_bound = min(self.split_num * 60, len(rd_std_files))
            lower_bound = (self.split_num - 1)* 60
            data_range = range(lower_bound , upper_bound )
        elif self.data_type == 'qm9' and self.mode == 'Train':
            
            upper_bound = min(self.split_num * 30, len(rd_std_files))
            lower_bound = (self.split_num - 1)* 30
            #print(upper_bound, lower_bound,len(rd_std_files))
            data_range = range(lower_bound , upper_bound )

        else: data_range = range(len(rd_std_files))

        for std_pickle_idx in data_range:
            print(f'pickle ID {std_pickle_idx}, start.')
            rd_std_mols = open_pickle(rd_std_files[std_pickle_idx] )
            rd_std_mols_keys = list(rd_std_mols.keys())

            for pickle_idx, key in enumerate(tqdm(rd_std_mols_keys)):
                if key in pickle_file_name:
                    pickle_name = data_dir + key +'.pickle'
                    gen_data = rd_std_mols[key]
                    raw_data = open_pickle(pickle_name)

                    gen_mol_list = [i['rd_mol'] for i in gen_data['conformers']]
                    smi = raw_data['smiles']
                    confs = raw_data['conformers']
                    mol_ = Chem.MolFromSmiles(smi)
                    # skip mol cannot intrinsically handle
                    if mol_:
                        canonical_smi = Chem.MolToSmiles(mol_)
                    else:
                        print(f'pickle ID {std_pickle_idx}, data ID.{pickle_idx} rdkit cannot intrinsically handle')
                        continue
                        
                    confs = sort_confs(confs)
                    real_mol_list = clean_confs(canonical_smi, confs, limit= self.max_conf_num )

                    if len(real_mol_list) != len(gen_mol_list):
                        print(f'pickle ID {std_pickle_idx}, data ID.{pickle_idx} gen_mol cannot match real_mol')
                        continue
                        
                    # skip mol with fragments
                    if '.' in smi:
                        print(f'pickle ID {std_pickle_idx}, data ID.{pickle_idx} conformers with fragments')
                        continue

                    ##################### CLEAN DATA ABOVE ###############
                    ##################### PROCESS DATA BELOW #############
                    mol = real_mol_list[0]
                    x, edge_index, edge_attr, bond_index, chiral_info = one_hop_process(mol, types = atom_types) 
                    h_mask = get_h_mask(mol)
                    rotatable_bonds = get_rotatable_bonds(mol, edge_index) 
                    fragments = split_mol(mol, edge_index)
                    atom_to_fragments = get_af(fragments, x)
                    frag_edges =  torch.sort(atom_to_fragments[rotatable_bonds])[0]
                    motif_edge_index = atom_to_fragments[torch.cat((rotatable_bonds ,torch.flip(rotatable_bonds, [1])),0).t()]
                    sorted_rotatable_bonds = get_frag_graph_info(frag_edges, rotatable_bonds, atom_to_fragments )

                    #hexagon_frag = get_hexagon_ring(atom_to_fragments, h_mask, x)
                    symmetrical_frags_end, symmetrical_frags_chain = get_symmetrical_frag(atom_to_fragments, h_mask, x, sorted_rotatable_bonds, edge_index)
                    sym_dihedral_batch = get_sym_dihedral_batch(atom_to_fragments, sorted_rotatable_bonds, symmetrical_frags_chain)
                    atom_neighbor = get_atom_neighbor(mol, x, bond_index, atom_to_fragments)                     
                    dihedral_indx = get_dihedral_info(sorted_rotatable_bonds, edge_index, atom_neighbor, atom_to_fragments)
                    frag, frag_mask = get_frag_info(fragments, self.max_seq_length)
                    chain_core_points = get_chain_geo_points(dihedral_indx, atom_to_fragments, sorted_rotatable_bonds, symmetrical_frags_chain)  
                    seq_length = dihedral_indx.shape[0]    
                    rnn_length = seq_length
                    conf_padded_num = self.max_conf_num - len(gen_mol_list)
                    sym_mask = get_sym_mask(atom_to_fragments, sorted_rotatable_bonds, symmetrical_frags_end, symmetrical_frags_chain)
                    seq_mask = get_seq_mask(rnn_length, self.max_seq_length) 
                    
                    phi_fake, phi_chain_fake, local_fake, pos_fake = get_feature(gen_mol_list, dihedral_indx, chain_core_points, sym_mask )
                    phi_fake, phi_chain_fake = sym_h(gen_mol_list, dihedral_indx, phi_fake, phi_chain_fake, h_mask, atom_neighbor)

                    if  len(gen_mol_list) == self.max_conf_num:
                        conf_mask = torch.ones(self.max_conf_num).bool().unsqueeze(0)
                    else:
                        conf_mask = F.pad(torch.ones(len(gen_mol_list)),(0,conf_padded_num) , "constant", 0).bool().unsqueeze(0)

                    padded_phi = F.pad(phi_fake, (0,0,0, 0 ,0,conf_padded_num ), "constant", 0) .permute(1,0,2)       
                    padded_phi_chain = F.pad(phi_chain_fake, (0,0,0, 0 ,0,conf_padded_num ), "constant", 0).permute(1,0,2) 
                    padded_local = F.pad(local_fake , (0,0,0, 0 ,0,conf_padded_num ), "constant", 0).permute(1,0,2)    
                    ########### PROCESS DATA END###########################
        
                    #except:
                    #    print(f'pickle ID {std_pickle_idx}, data ID.{pickle_idx}  atom meet get feature problem')
                    #   continue 
                    
                    new_data = Data(
                                #NEW!
                                motif_edge_index = motif_edge_index,
                                sym_dihedral_batch = sym_dihedral_batch,
                                #
                                phi = padded_phi,
                                phi_chain = padded_phi_chain,
                                local = padded_local,
                                #pos = padded_pos,
                                seq_mask = seq_mask, 
                                atom_to_fragments = atom_to_fragments ,
                                x = x, 
                                edge_index = edge_index, 
                                edge_attr = edge_attr, 
                                chiral_info = chiral_info,
                                dihedral_indx = dihedral_indx,
                                #frag = frag,
                                #frag_mask = frag_mask,
                                conf_mask = conf_mask,
                    )
                    #assert(1==2)
                    data_list.append(new_data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class OriginDataset_test(InMemoryDataset):
    def __init__(self, cmt, args,
                    root = '.',
                    transform: Optional[Callable] = None,
                    pre_transform: Optional[Callable] = None,
                    pre_filter: Optional[Callable] = None,       
                    ):
        self.cmt = cmt
        self.root = root
        self.std_dir = args.std_dir
        self.split_path = args.split_path
        self.data_type = args.data_type
        self.max_seq_length = args.max_seq_length

        super(OriginDataset_test, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return self.cmt + self.data_type + 'Test' + '.pt'


    def process(self):
        data_list = []
        if self.data_type == 'qm9':
            atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            atom_type_column = 5
            pickle_dir = self.root + '/data/QM9/test_mols.pkl'
            csv_dir = self.root + '/data/QM9/test_smiles.csv'
            
        else:
            atom_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
                'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
                'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
                'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
            atom_type_column = 35
            pickle_dir = self.root + '/data/DRUGS/test_mols.pkl'
            csv_dir = self.root + '/data/DRUGS/test_smiles.csv'


        real_mol_set = open_pickle(pickle_dir)
        test_data = pd.read_csv(csv_dir).values
        rdkit_time = 0
        procession_time = 0 
        total_conf = 0

        for smi_idx, (raw_smi, n_confs, smi) in enumerate(tqdm(test_data)):
            #Tor Diff filter No.1 mol in test_dataset for potential problem during its pre-procession,
            #All experimental data are borrowed from TorDiff, therefore we manually filter No.1.
            if smi_idx == 1:  
                continue
            real_mols = real_mol_set[raw_smi]
            real_mol_list = clean_confs_test(raw_smi, real_mols)
            num_confs = len(real_mol_list) * 2
            
            if real_mol_list  == []:
                print(f'mol {smi_idx}  has bad groundtruth')
                continue

            total_conf += num_confs
            mol = real_mol_list[0]
            start_time = time.time()

            ############# rdkit generate mol

            gen_mol_list = get_rd_gen_mol(mol, num_confs)
            rdkit_time += time.time() - start_time
            try:
                x, edge_index, edge_attr, bond_index, chiral_info = one_hop_process(mol, types = atom_types)
            except:
                print(f'mol {smi_idx}  Bad Conformer Id')
                continue
               
            h_mask = get_h_mask(mol)
            rotatable_bonds = get_rotatable_bonds(mol, edge_index) 
            if len(rotatable_bonds) <1:
                print(f'mol {smi_idx}  has no rotatable bond')
                new_data = Data(
                        raw_smi = raw_smi,
                        h_mask = h_mask,
                        motif_edge_index = motif_edge_index,
                        atom_to_fragments = atom_to_fragments ,
                        x = x, 
                        edge_index = edge_index, 
                        edge_attr = edge_attr, 
                        gen_mol_list = gen_mol_list,
                        perturb_gen_mol_list = gen_mol_list,
                        chiral_info = chiral_info,

                        dihedral_indx = torch.zeros(1,9),
                        frag = torch.zeros(1),
                        frag_mask = torch.zeros(1),
                        local_fake = torch.zeros(1,100,10),
                        padded_local_perturb = torch.zeros(1,100,10),
                        )
                data_list.append(new_data)
                continue
            fragments = split_mol(mol, edge_index)
            atom_to_fragments = get_af(fragments, x)
            frag_edges =  torch.sort(atom_to_fragments[rotatable_bonds])[0]
            motif_edge_index = atom_to_fragments[torch.cat((rotatable_bonds ,torch.flip(rotatable_bonds, [1])),0).t()]
            sorted_rotatable_bonds = get_frag_graph_info(frag_edges, rotatable_bonds, atom_to_fragments )
            symmetrical_frags_end, symmetrical_frags_chain = get_symmetrical_frag(atom_to_fragments, h_mask, x, sorted_rotatable_bonds, edge_index)
            #sym_dihedral_batch = get_sym_dihedral_batch(atom_to_fragments, sorted_rotatable_bonds, symmetrical_frags_chain)
            atom_neighbor = get_atom_neighbor(mol, x, bond_index, atom_to_fragments)                     
            dihedral_indx = get_dihedral_info(sorted_rotatable_bonds, edge_index, atom_neighbor, atom_to_fragments)
            frag, frag_mask = get_frag_info(fragments, self.max_seq_length)
            chain_core_points = get_chain_geo_points(dihedral_indx, atom_to_fragments, sorted_rotatable_bonds, symmetrical_frags_chain)  
            dihedral_padded_num = self.max_seq_length - len(dihedral_indx)
            #print(dihedral_indx .shape)
            #seq_length = dihedral_indx.shape[0]    
            #rnn_length = seq_length
            #sym_mask = get_sym_mask(atom_to_fragments, sorted_rotatable_bonds, symmetrical_frags_end, symmetrical_frags_chain)
            #seq_mask = get_seq_mask(rnn_length, self.max_seq_length) 
            try:
                local_fake = get_feature_local(gen_mol_list, dihedral_indx, chain_core_points)
                padded_local = F.pad(local_fake , (0,0,0, dihedral_padded_num, 0 ,0, ), "constant", 0)
                procession_time += time.time() - start_time

                perturb_gen_mol_list = perturb_conf(gen_mol_list, dihedral_indx)
                local_fake_perturb = get_feature_local(perturb_gen_mol_list, dihedral_indx, chain_core_points)
                padded_local_perturb = F.pad(local_fake_perturb  , (0,0,0, dihedral_padded_num, 0 ,0, ), "constant", 0)
            except:
                print(f'mol {smi_idx}  Bad Local')
                continue
               

            new_data = Data(
                        raw_smi = raw_smi,
                        h_mask = h_mask,
                        motif_edge_index = motif_edge_index,
                        atom_to_fragments = atom_to_fragments ,
                        x = x, 
                        edge_index = edge_index, 
                        edge_attr = edge_attr, 
                        chiral_info = chiral_info,
                        dihedral_indx = dihedral_indx,
                        frag = frag,
                        frag_mask = frag_mask,
                        local_fake = padded_local,
                        padded_local_perturb = padded_local_perturb,
                        gen_mol_list = gen_mol_list,
                        perturb_gen_mol_list = perturb_gen_mol_list,
            )
            print(new_data)
            data_list.append(new_data)

        print(f'rdkit generate local{rdkit_time / total_conf}')
        print(f'total precession time include rdkit generate local, {procession_time /total_conf}')
        torch.save(self.collate(data_list), self.processed_paths[0])
