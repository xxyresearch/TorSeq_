import torch
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as F
from torch_geometric.nn import Sequential, GCNConv, global_add_pool
import torch.nn as nn
from math import pi
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GINEConv,GINConv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import Sequential, GCNConv

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter, scatter_sum

#from model.utils import *


eps=1e-8 

class GLSTM(nn.Module):
    def __init__(self, node_dim = 75, edge_dim = 5, hidden_dim = 100, 
                 latent_dim = 10, lstm_layer_num = 2, gnn_layer_num = 3, 
                 use_lstm = True, use_motif_graph = True, use_local_feature = True):
        super(GLSTM, self).__init__()
        self.motif_graph = use_motif_graph
        self.hidden_dim =  hidden_dim
        self.act = F.relu
        self.node_extractor = MLP(in_dim = node_dim + latent_dim , out_dim= hidden_dim, num_layers=2)
        if use_local_feature:
            self.local_extractor = MLP(in_dim = 10 , out_dim= hidden_dim, num_layers=2)
        self.edge_extractor = MLP(in_dim = edge_dim, out_dim= hidden_dim, num_layers=1)
        self.edge_extractor_plus_1 = MLP(in_dim = hidden_dim*2, out_dim= hidden_dim, num_layers=1)
        self.frag_extractor_1 = MLP(in_dim = hidden_dim, out_dim= hidden_dim, num_layers=1)
        
        self.gnn = GNN(node_dim=hidden_dim, edge_dim=hidden_dim, hidden_dim=hidden_dim, depth= gnn_layer_num)
        if use_motif_graph:
            self.motif_gcn = Sequential('x, edge_index', [
                (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
                (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            ])
            self.frag_extractor_2 = MLP(in_dim = hidden_dim, out_dim= hidden_dim, num_layers=1)


        self.rotation_side_extractor_1 = MLP(in_dim = hidden_dim, out_dim= hidden_dim, num_layers=1)
        self.rotation_side_extractor_2 = MLP(in_dim = hidden_dim, out_dim= hidden_dim, num_layers=1)
        self.mol_extractor = MLP(in_dim = 2*hidden_dim, out_dim= hidden_dim, num_layers=1)
        if use_local_feature:
            self.dihedral_extractor = nn.Sequential(
                nn.Linear(14 * hidden_dim, 8 * hidden_dim),
                nn.BatchNorm1d(8 *hidden_dim),
                nn.ReLU(),
                nn.Linear(8 *hidden_dim, 4 *hidden_dim),
            )  
        else:
            self.dihedral_extractor = nn.Sequential(
                nn.Linear(13 * hidden_dim, 8 * hidden_dim),
                nn.BatchNorm1d(8 *hidden_dim),
                nn.ReLU(),
                nn.Linear(8 *hidden_dim, 4 *hidden_dim),
            )             
        #"""
        
        if use_lstm:
            self.dihedral_seq_model = nn.LSTM(input_size = 4 * hidden_dim, 
                                hidden_size = 4 * hidden_dim, 
                                num_layers = lstm_layer_num,
                                batch_first = True,
                                bidirectional = True)
            self.output_buffer = MLP(8 * hidden_dim, 4 * hidden_dim, num_layers=2)

        ####    
        self.rand_encoder_1 = MLP(in_dim = 1, out_dim = hidden_dim , num_layers=1)
        self.rand_decoder_1 = MLP(in_dim = hidden_dim, out_dim = 1, num_layers=1)
        self.rand_encoder_2 = MLP(in_dim = 1, out_dim = hidden_dim * 1, num_layers=1)
        self.rand_decoder_2 = MLP(in_dim = 1 * hidden_dim, out_dim = 1, num_layers=1)

        
        self.output = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),#),
            nn.BatchNorm1d(2 *hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 1 * hidden_dim),#, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),#, bias=False),
            nn.Tanh()
        ) 

    def forward (self, data, args, random_start, x_with_noise, idx, test_mode = False):
        x = self.node_extractor(x_with_noise)
        if not test_mode or not args.no_local_feature:
            local_feature  = data.local_feature[idx]
            chain_fix_detach = local_feature[:,8:].detach()   
        if not args.no_local_feature:
            h_local = self.local_extractor(local_feature.float())       
        edge_attr = self.edge_extractor(data.edge_attr)
        edge_in = x[data.edge_index[0]]
        edge_out = x[data.edge_index[1]]
        edge_attr = self.edge_extractor_plus_1(torch.cat((edge_attr, edge_in + edge_out),1))
        x, edge_attr = self.gnn(x, data.edge_index, edge_attr)

        # local 
        row, col = data.edge_index
        x_row = x[row]
        x_scatter = scatter_sum(x_row, col, dim=0, dim_size=x.size(0))
        x1_in = self.rotation_side_extractor_1(x_scatter[data.dihedral_indx[:,1]] - x[data.dihedral_indx[:,2]])
        x2_in = self.rotation_side_extractor_2(x_scatter[data.dihedral_indx[:,2]] - x[data.dihedral_indx[:,1]])

        # Motif Level
        h_frag = scatter(x, data.atom_to_frag, dim=0, reduce='sum')
        h_frag = self.frag_extractor_1(h_frag)
        if self.motif_graph:
            h_frag = self.motif_gcn(h_frag, data.motif_edge_index)
            h_frag = self.frag_extractor_2(h_frag)
            #assert(1==2)
      
        # mol feature
        h_mol_atom = scatter(x, data.batch_atom, dim=0, reduce='sum')
        h_mol_frag = scatter(h_frag, data.batch_frag, dim=0, reduce='sum')
        h_mol = self.mol_extractor(torch.cat((h_mol_atom, h_mol_frag),1))

        if test_mode:
            h_mol = h_mol[data.batch_dihedral]
        else:
            h_mol = h_mol[data.batch_dihedral[data.batch_sym_dihedral]]

        # basic_feature
        x0 = x[data.dihedral_indx[:,0]]
        x1 = x[data.dihedral_indx[:,1]]
        x2 = x[data.dihedral_indx[:,2]]
        x3 = x[data.dihedral_indx[:,3]]
        e1 = edge_attr[data.dihedral_indx[:,4]]
        e2 = edge_attr[data.dihedral_indx[:,5]]
        e3 = edge_attr[data.dihedral_indx[:,6]]
        f1 = h_frag[data.dihedral_indx[:,7]]
        f2 = h_frag[data.dihedral_indx[:,8]]
        
        r1 = self.rand_encoder_1(random_start)
        re_r1 = self.rand_decoder_1(r1)
        r2 = self.rand_encoder_2(random_start)
        re_r2 = self.rand_decoder_2(r2)
        if not args.no_local_feature:
            h_t = torch.cat((x1_in, x2_in, x0, x1, x2, x3, e1, e2, e3, f1, f2, h_mol, h_local, r1),1) 
        else:
            h_t = torch.cat((x1_in, x2_in, x0, x1, x2, x3, e1, e2, e3, f1, f2, h_mol, r1),1)             
        h_t = self.dihedral_extractor(h_t)

        if not args.no_use_lstm:
            if test_mode:
                h_t_seq = h_t.unsqueeze(0)
                h_t_seq, _ = self.dihedral_seq_model(h_t)
                h_t_seq = torch.squeeze(h_t_seq)
                if not args.no_res_lstm:
                    h_t = h_t_seq + torch.cat((h_t, h_t),1)
                else:
                    h_t = h_t_seq
            else:
                h_t_seq = h_t[data.dihedral_lstm_index]*data.seq_mask[:,:-2].float().unsqueeze(-1)
                packed_h_t_seq  = pack_padded_sequence(h_t_seq, data.rnn_length.detach().cpu(), batch_first=True, enforce_sorted=False) 
                packed_h_t_seq , (hn, cn) = self.dihedral_seq_model(packed_h_t_seq)
                h_t_seq , output_lengths = pad_packed_sequence(packed_h_t_seq , batch_first=True)       
                max_length = torch.max(output_lengths)
                mask = data.seq_mask[:,:max_length]
                h_t = h_t_seq[mask] + torch.cat((h_t, h_t),1)

            h_t = self.output_buffer(h_t)
            if test_mode:
                h_t = h_t.view(-1, args.hidden_dim * 4)

        h_t = torch.cat((h_t, r2),1) 
        dihedral_rad = self.output(h_t) * pi
        
        if test_mode:
            return dihedral_rad + random_start
        
        else:
            res = dihedral_rad + random_start
            res_chain = torch.sum(torch.cat((res, chain_fix_detach), 1), dim=1).reshape(-1,1)
            return res, res_chain, re_r1, re_r2



class MLP(nn.Module):
    """
    Creates a NN using nn.ModuleList to automatically adjust the number of layers.
    For each hidden layer, the number of inputs and outputs is constant.
    Inputs:
        in_dim (int):               number of features contained in the input layer.
        out_dim (int):              number of features input and output from each hidden layer, 
                                    including the output layer.
        num_layers (int):           number of layers in the network
        activation (torch function): activation function to be used during the hidden layers
    """
    def __init__(self, in_dim, out_dim, num_layers, activation=torch.nn.ReLU(), layer_norm=False, batch_norm=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        h_dim = in_dim if out_dim < 50 else out_dim
        # create the input layer
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(nn.Linear(in_dim, h_dim))
            else:
                self.layers.append(nn.Linear(h_dim, h_dim))
            if layer_norm: self.layers.append(nn.LayerNorm(h_dim))
            if batch_norm: self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(activation)
        self.layers.append(nn.Linear(h_dim, out_dim))
        
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class MetaLayer(torch.nn.Module):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.
    """
    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.edge_eps = nn.Parameter(torch.Tensor([0]))
        self.node_eps = nn.Parameter(torch.Tensor([0]))
        #self.reset_parameters()

    #def reset_parameters(self):
    #    for item in [self.node_model, self.edge_model]:
    #        if hasattr(item, 'reset_parameters'):
    #            item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """"""
        if self.edge_model is not None:
            edge_attr = (1 + self.edge_eps) * edge_attr + self.edge_model(x, edge_attr, edge_index)
        if self.node_model is not None:
            x = (1 + self.node_eps) * x + self.node_model(x, edge_index, edge_attr, batch)

        return x, edge_attr

class EdgeModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(EdgeModel, self).__init__()
        self.edge = Lin(hidden_dim, hidden_dim)
        self.node_in = Lin(hidden_dim, hidden_dim, bias=False)
        self.node_out = Lin(hidden_dim, hidden_dim, bias=False)
        self.mlp = MLP(hidden_dim, hidden_dim, n_layers)

    def forward(self, x, edge_attr, edge_index):
        # source, target: [2, E], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs (we don't have any of these yet)
        # batch: [E] with max entry B - 1.

        f_ij = self.edge(edge_attr)
        f_i = self.node_in(x)
        f_j = self.node_out(x)
        row, col = edge_index

        out = F.relu(f_ij + f_i[row] + f_j[col])
        return self.mlp(out)

class NodeModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = MLP(hidden_dim, hidden_dim, n_layers)
        self.node_mlp_2 = MLP(hidden_dim, hidden_dim, n_layers)

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [N, h], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u] (N/A)
        # batch: [N] with max entry B - 1.
        # source, target = edge_index
        _, col = edge_index
        out = self.node_mlp_1(edge_attr)
        out = scatter_sum(out, col, dim=0, dim_size=x.size(0))
        return self.node_mlp_2(out)

class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=300, depth=3, n_layers=1, need_init = False):
        super(GNN, self).__init__()
        self.need_init = need_init
        self.depth = depth
        self.node_init = MLP(node_dim, hidden_dim, n_layers)
        self.edge_init = MLP(edge_dim, hidden_dim, n_layers)
        self.update = nn.ModuleList()
        for layer in range(depth):
            self.update.append(MetaLayer(EdgeModel(hidden_dim, n_layers), NodeModel(hidden_dim, n_layers)))

    def forward(self, x, edge_index, edge_attr):
        if self.need_init:
            x = self.node_init(x)
            edge_attr = self.edge_init(edge_attr)
        for i in range(len(self.update)):
            x, edge_attr = self.update[i](x, edge_index, edge_attr)
        
        return x, edge_attr
    