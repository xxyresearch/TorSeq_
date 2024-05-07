from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


def concat_new_data(data, new_data, dim, add_num):
    if add_num != 0:
        new_data = torch.add(new_data, add_num)
    data = torch.cat((data, new_data), dim=dim)
    return data


def collate_perturb_batch(batch_data):
    max_torsional_edge_num = 100
    atom_num = torch.tensor(0) 
    torsion_num = torch.tensor(0)
    ptr = [0]
    for idx, data in enumerate(batch_data):
        if idx==0:  
            x = data.x 
            edge_index = data.edge_index 
            edge_attr = data.edge_attr
            z = data.z
            pos = data.pos

            mask_edges = torch.tensor(data.mask_edges)
            rotatable_bonds = data.rotatable_bonds
            edge_mask = data.edge_mask 
            node_sigma = data.node_sigma
            edge_rotate = data.edge_rotate
            torsion_rnn_index = F.pad(torch.arange(len(data.edge_rotate)), (0,max_torsional_edge_num - len(data.edge_rotate)), "constant", 0).unsqueeze(0)
            seq_mask = torch.zeros_like(torsion_rnn_index).bool()
            seq_mask[:,:len(data.edge_rotate)] = 1
            rnn_length = [len(data.edge_rotate)]
            #batch
            batch = torch.ones(x.shape[0]) * idx
            batch_torsion = torch.ones(len(data.edge_rotate)) * idx
        else:
            x = concat_new_data(x, data.x , dim=0, add_num=0)
            edge_index = concat_new_data(edge_index, data.edge_index, dim=1, add_num=atom_num)
            edge_attr = concat_new_data(edge_attr, data.edge_attr, dim=0, add_num=0)
            z = concat_new_data(z, data.z, dim=0, add_num=0)
            pos = concat_new_data(pos, data.pos, dim=0, add_num=0)

            mask_edges = concat_new_data(mask_edges, torch.tensor(data.mask_edges), dim=0, add_num=0)
            rotatable_bonds = concat_new_data(rotatable_bonds, data.rotatable_bonds, dim=0, add_num=atom_num)
            edge_mask = concat_new_data(edge_mask, data.edge_mask, dim=0, add_num=0) 
            node_sigma = concat_new_data(node_sigma, data.node_sigma, dim=0, add_num=0)
            edge_rotate = concat_new_data(edge_rotate, data.edge_rotate, dim=0, add_num=0)
            new_torsion_rnn_index = F.pad(torch.arange(len(data.edge_rotate)), (0,max_torsional_edge_num - len(data.edge_rotate)), "constant", 0).unsqueeze(0)
            new_seq_mask = torch.zeros_like(new_torsion_rnn_index).bool()
            new_seq_mask[:,:len(data.edge_rotate)] = 1

            torsion_rnn_index = concat_new_data(torsion_rnn_index, new_torsion_rnn_index, dim=0, add_num=torsion_num)
            seq_mask = concat_new_data(seq_mask, new_seq_mask, dim=0, add_num=0)

            rnn_length.append(len(data.edge_rotate))
            new_batch = torch.ones(data.x.shape[0]) * idx
            new_batch_torsion  = torch.ones(len(data.edge_rotate)) * idx
            batch = concat_new_data(batch, new_batch, dim=0, add_num=0)
            batch_torsion = concat_new_data(batch_torsion, new_batch_torsion, dim=0, add_num=0)
            
        
        atom_num += data.x.shape[0]
        ptr.append(deepcopy(atom_num))
        torsion_num += len(data.edge_rotate)
        
    
    ptr = torch.tensor(ptr)
    rnn_length = torch.tensor(rnn_length)
    data = Data(
            x = x, 
            edge_index = edge_index, 
            edge_attr = edge_attr,
            z = z,
            pos = pos,

            mask_edges = mask_edges,
            #mask_rotate = mask_rotate,
            rotatable_bonds = rotatable_bonds,
            #sorted_rotatable_bonds = sorted_rotatable_bonds,
            #sorted_rotatable_indice = sorted_rotatable_indice,
            edge_mask = edge_mask, 
            node_sigma = node_sigma,
            edge_rotate = edge_rotate,

            ### RNN part
            #sort_indice = sort_indice,
            #recover_indice = recover_indice,
            torsion_rnn_index = torsion_rnn_index, 
            seq_mask = seq_mask,
            rnn_length = rnn_length,
            #batch
            batch = batch.long(),
            batch_torsion = batch_torsion.long(),
            ptr = ptr,
            )
    return data


def collate_original_batch(batch_data):
    max_dihedral_num = 100 # this is hard code
    atom_num = torch.tensor(0) 
    edge_num = torch.tensor(0)
    dihedral_num = torch.tensor(0)
    sym_dihedral_num = torch.tensor(0)
    frag_num = torch.tensor(0)
    for idx, data in enumerate(batch_data):
        if idx==0:  
            #tgt, tgt_chain, local_feature = get_train_materials( data, max_conf_num)
            conf_mask = data.conf_mask  
            tgt = data.phi.permute(1, 0 , 2)
            tgt_chain = data.phi_chain.permute(1, 0 , 2)
            local_feature = data.local.permute(1, 0 , 2)
            edge_attr = data.edge_attr 
            edge_index = data.edge_index
            motif_edge_index = data.motif_edge_index
            dihedral_indx_p1 = data.dihedral_indx[:,:4]
            dihedral_indx_p2 = data.dihedral_indx[:,4:7]
            dihedral_indx_p3 = data.dihedral_indx[:,7:]
            dihedral_lstm_index = F.pad(torch.arange(len(data.dihedral_indx)), (0,max_dihedral_num-len(data.dihedral_indx)), "constant", 0).unsqueeze(0)
            seq_mask = data.seq_mask.unsqueeze(0) 
            x = data.x
            atom_to_frag = data.atom_to_fragments

            rnn_length = [data.dihedral_indx.shape[0]]
            pos_embed = torch.arange(len(data.dihedral_indx))
            #batch
            batch_atom = torch.ones(x.shape[0]) * idx
            batch_sym_dihedral = data.sym_dihedral_batch
            batch_dihedral = torch.ones(torch.max(data.sym_dihedral_batch)+1) * idx
            batch_frag = torch.ones(int(torch.max(data.atom_to_fragments )+1)) * idx

        else:
            #new_tgt, new_tgt_chain, new_local_feature = get_train_materials( data, max_conf_num)
            new_tgt = data.phi.permute(1, 0 , 2)
            new_tgt_chain = data.phi_chain.permute(1, 0 , 2)
            new_local_feature = data.local.permute(1, 0 , 2)
            tgt = concat_new_data(tgt, new_tgt, dim=1, add_num=0)
            tgt_chain = concat_new_data(tgt_chain, new_tgt_chain, dim=1, add_num=0)
            local_feature = concat_new_data(local_feature, new_local_feature , dim=1, add_num=0)

            conf_mask = concat_new_data(conf_mask, data.conf_mask, dim=0, add_num=0)
            edge_attr = concat_new_data(edge_attr, data.edge_attr, dim=0, add_num=0)
            edge_index = concat_new_data(edge_index, data.edge_index, dim=1, add_num=atom_num)
            motif_edge_index = concat_new_data(motif_edge_index, data.motif_edge_index, dim=1, add_num=frag_num)
            x = concat_new_data(x, data.x, dim=0, add_num=0)
            ## dihedral
            dihedral_indx_p1 = concat_new_data(dihedral_indx_p1, data.dihedral_indx[:,:4], dim=0, add_num=atom_num)
            dihedral_indx_p2 = concat_new_data(dihedral_indx_p2, data.dihedral_indx[:,4:7], dim=0, add_num=edge_num)
            dihedral_indx_p3 = concat_new_data(dihedral_indx_p3, data.dihedral_indx[:,7:], dim=0, add_num=frag_num)
            # frag
            #frag = concat_new_data(frag, data.frag, dim = 0, add_num = frag_num)
            #frag_mask = concat_new_data(frag_mask, data.frag_mask, dim=0, add_num =0)
            atom_to_frag = concat_new_data(atom_to_frag, data.atom_to_fragments, dim=0, add_num = frag_num)
            #

            rnn_length.append(data.dihedral_indx.shape[0]) 
            ### DIHEDRAL LSTM
            dihedral_lstm_index = concat_new_data(dihedral_lstm_index, F.pad(torch.arange(len(data.dihedral_indx)), (0,max_dihedral_num -len(data.dihedral_indx)), "constant", 0).unsqueeze(0), dim=0, add_num = dihedral_num)
            pos_embed = concat_new_data(pos_embed, torch.arange(len(data.dihedral_indx)), dim=0, add_num = 0)
            seq_mask = concat_new_data(seq_mask, data.seq_mask.unsqueeze(0) , dim=0, add_num =0) 
            batch_atom = concat_new_data(batch_atom, torch.ones(data.x.shape[0]) * idx, dim=0, add_num = 0 )
            #batch_dihedral = concat_new_data(batch_dihedral, torch.ones(data.dihedral_indx.shape[0]) * idx , dim=0, add_num = 0)
            batch_frag = concat_new_data(batch_frag , torch.ones(int(torch.max(data.atom_to_fragments )+1)) * idx , dim=0, add_num = 0)
            batch_sym_dihedral = concat_new_data(batch_sym_dihedral, data.sym_dihedral_batch, dim=0, add_num=sym_dihedral_num)
            batch_dihedral = concat_new_data(batch_dihedral, torch.ones(torch.max(data.sym_dihedral_batch)+1) * idx, dim=0, add_num=0)

        sym_dihedral_num += (torch.max(data.sym_dihedral_batch)+1).long()
        atom_num += data.x.shape[0]
        edge_num += data.edge_attr.shape[0]
        dihedral_num += data.dihedral_indx.shape[0]
        frag_num += (torch.max(data.atom_to_fragments )+1).long()
    pos_embed = F.one_hot(pos_embed , num_classes=max_dihedral_num).float()
    dihedral_indx = torch.cat((dihedral_indx_p1, dihedral_indx_p2, dihedral_indx_p3), 1)
    rnn_length = torch.tensor(rnn_length)

    data = Data(
            ## for all
            pos_embed = pos_embed,
            batch_sym_dihedral = batch_sym_dihedral.long(),
            conf_mask = conf_mask,
            tgt = tgt, 
            tgt_chain = tgt_chain,
            local_feature = local_feature, 
            edge_attr = edge_attr ,
            edge_index = edge_index.long(),
            motif_edge_index = motif_edge_index.long(),
            dihedral_indx = dihedral_indx.long(),
            rnn_length =  rnn_length,
            dihedral_lstm_index = dihedral_lstm_index.long(),
            x = x,
            #frag = frag,
            #frag_mask = frag_mask.bool(),
            atom_to_frag = atom_to_frag.long(),
            batch_atom = batch_atom.long(),
            batch_dihedral = batch_dihedral.long(),
            batch_frag = batch_frag.long(),
            seq_mask = seq_mask,
            #start_mask = start_mask,
            #end_mask = end_mask,
            #dihedral_weight = dihedral_weight,
            )
    return data


def collate_original_batch_test(batch_data):
    max_dihedral_num = 100 # this is hard code
    data = batch_data[0]
    #tgt, tgt_chain, local_feature = get_train_materials( data, max_conf_num)
    local_feature = data.padded_local_perturb[:,:data.dihedral_indx.shape[0],:]
    edge_attr = data.edge_attr 
    edge_index = data.edge_index
    motif_edge_index = data.motif_edge_index
    dihedral_indx = data.dihedral_indx
    x = data.x
    atom_to_frag = data.atom_to_fragments
    dihedral_lstm_index = F.pad(torch.arange(len(data.dihedral_indx)), (0,max_dihedral_num-len(data.dihedral_indx)), "constant", 0).unsqueeze(0)
    rnn_length = [data.dihedral_indx.shape[0]]
    batch_atom = torch.zeros(x.shape[0])
    batch_frag = torch.zeros(int(torch.max(data.frag)+1))
    batch_dihedral = torch.zeros(data.dihedral_indx.shape[0]) 
    gen_mol_list = data.perturb_gen_mol_list
    smi = data.raw_smi

    data = Data(
        smi = smi,
        gen_mol_list = gen_mol_list ,
        local_feature = local_feature, 
        edge_attr = edge_attr ,
        edge_index = edge_index.long(),
        motif_edge_index = motif_edge_index.long(),
        dihedral_indx = dihedral_indx.long(),
        rnn_length =  rnn_length,
        dihedral_lstm_index = dihedral_lstm_index.long(),
        x = x,
        atom_to_frag = atom_to_frag.long(),
        batch_atom = batch_atom.long(),
        batch_frag = batch_frag.long(),
        batch_dihedral = batch_dihedral.long(),
        )
    return data