
"""dataset utils import"""
import pickle
from math import pi
import numpy as np
from copy import deepcopy
import networkx as nx
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from torch_geometric.data import  Data, InMemoryDataset
from torch_geometric.utils import to_networkx

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
import torch
import torch.nn.functional as F
import os.path as osp

from torch_geometric.data import  Data, InMemoryDataset
from typing import Callable, Optional
import numpy as np
from tqdm import tqdm
import glob


class PerturbDataset(InMemoryDataset):
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

        super(PerturbDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return self.cmt + self.data_type + self.mode + str(self.split_num) + '.pt'

    def process(self):
        data_list = []
        #file_idx = 0
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
            upper_bound = min(self.split_num * 60, len(pickle_files))
            lower_bound = (self.split_num - 1)* 60
            data_range = range(lower_bound , upper_bound )
        elif self.data_type == 'qm9' and self.mode == 'Train':
            upper_bound = min(self.split_num * 25, len(pickle_files))
            lower_bound = (self.split_num - 1)* 25
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
                    real_confs = raw_data['conformers']

                    # filter mol
                    mol_ = Chem.MolFromSmiles(smi)
                    # skip mol cannot intrinsically handle
                    if mol_:
                        canonical_smi = Chem.MolToSmiles(mol_)
                    else:
                        print(f'pickle ID {std_pickle_idx}, data ID.{pickle_idx} rdkit cannot intrinsically handle')
                        continue
                        
                    real_confs = sort_confs(real_confs)
                    real_mol_list = clean_confs(canonical_smi, real_confs, limit = self.max_conf_num )

                    if len(real_mol_list) != len(gen_mol_list):
                        print(f'pickle ID {std_pickle_idx}, data ID.{pickle_idx} gen_mol cannot match real_mol')
                        continue
                        
                    # skip mol with fragments
                    if '.' in smi:
                        print(f'pickle ID {std_pickle_idx}, data ID.{pickle_idx} conformers with fragments')
                        continue

                    correct_mol = gen_mol_list[0]

                    pos = []
                    for mol in gen_mol_list:
                        pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
                    
                    data = featurize_mol(correct_mol, atom_types)
                    data.canonical_smi, data.mol, data.pos = [canonical_smi], [correct_mol], [pos]
                    data = get_transformation_mask(data)
                    rotatable_bonds = data.edge_index.t()[data.mask_edges]

                    if len(rotatable_bonds) == 0:
                        print(f'pickle ID {std_pickle_idx}, data ID.{pickle_idx} has no rotatable bond')
                        continue

                    data.rotatable_bonds = rotatable_bonds 
                    data_list.append(data)
        #print(data_list)
        torch.save(self.collate(data_list), self.processed_paths[0])

    
"""load data"""
def open_pickle(mol_path):
    with open(mol_path, "rb") as f:
        dic = pickle.load(f)
    return dic

""" Chemistry Basic Cell """
dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}
                                 
PATT = Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')
ATOMS = ['H', 'C', 'N', 'O', 'F']

ATOM_DEGREE = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

BODN_TYPES = {
    0: Chem.rdchem.BondType.SINGLE, 
    1: Chem.rdchem.BondType.DOUBLE,
    2: Chem.rdchem.BondType.TRIPLE,
    3: Chem.rdchem.BondType.AROMATIC}

BODN_TYPE_DEGREE = {
    Chem.rdchem.BondType.SINGLE: 1, 
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3}


"""Function for One Hop """
"""We use GEOMOL """
def get_cycle_values(cycle_list, start_at=None):
    start_at = 0 if start_at is None else cycle_list.index(start_at)
    while True:
        yield cycle_list[start_at]
        start_at = (start_at + 1) % len(cycle_list)

def get_cycle_indices(cycle, start_idx):
    cycle_it = get_cycle_values(cycle, start_idx)
    indices = []

    end = 9e99
    start = next(cycle_it)
    a = start
    while start != end:
        b = next(cycle_it)
        indices.append(torch.tensor([a, b]))
        a = b
        end = b

    return indices

def get_current_cycle_indices(cycles, cycle_check, idx):
    c_idx = [i for i, c in enumerate(cycle_check) if c][0]
    current_cycle = cycles.pop(c_idx)
    current_idx = current_cycle[(np.array(current_cycle) == idx.item()).nonzero()[0][0]]
    return get_cycle_indices(current_cycle, current_idx)

def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def featurize_mol(mol, types):
    """
    Part of the featurisation code taken from GeoMol https://github.com/PattanaikL/GeoMol
    Returns:
        x:  node features
        z: atomic numbers of the nodes (the symbol one hot is included in x)
        edge_index: [2, E] tensor of node indices forming edges
        edge_attr: edge features
    """
    #if type(types) is str:
    #    if types == 'qm9':
    #        types = qm9_types
    #    elif types == 'drugs':
    #        types = drugs_types
    
    N = mol.GetNumAtoms()
    atom_type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type_idx.append(types[atom.GetSymbol()])
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                              1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    x1 = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z)

def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    pyg_data.mask_edges = mask_edges
    pyg_data.mask_rotate = mask_rotate 
    return pyg_data

def is_rotatable_bond(mol, bond_idx):
    bond_idx = bond_idx.tolist()
    a1_idx, a2_idx = bond_idx
    bond = mol.GetBondBetweenAtoms(a1_idx, a2_idx)
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    if bond.IsInRing() or a1_idx > a2_idx:
        return False
    a1_neighbors = [atom for atom in a1.GetNeighbors()] #if atom.GetAtomicNum() != 1]
    a2_neighbors = [atom for atom in a2.GetNeighbors()] #if atom.GetAtomicNum() != 1]
    return len(a1_neighbors) >= 2 and len(a2_neighbors) >= 2

def get_rotatable_bonds(mol, edge_index):
    edge_index_t = edge_index.t()
    # Check if each bond is rotatable
    rotatable_bonds_mask = torch.tensor([is_rotatable_bond(mol, bond_idx) for bond_idx in edge_index_t ])
    rotatable_edges = (edge_index.t())[rotatable_bonds_mask]
    return rotatable_edges


def split_mol(mol, edge_index):
    # Build a graph from the molecule
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        first_node = bond.GetBeginAtomIdx()
        second_node = bond.GetEndAtomIdx()
        if first_node > second_node: G.add_edge(second_node , first_node)
        else:  G.add_edge(first_node , second_node)
    
    # Get the rotatable bonds
    rotatable_bonds = get_rotatable_bonds(mol, edge_index)
    # Split the molecule by removing the rotatable bonds
    for bond in rotatable_bonds:
        bond = bond.tolist()
        G.remove_edge(bond[0], bond[1])

    
    # Get the fragments as lists of atom indices
    fragments = list(nx.connected_components(G))
    return fragments

def get_af(fragments, x):
    af = torch.zeros_like(x[:,0])
    for i in range(len(fragments)):
        for node_num in fragments[i]:
            af[node_num] = i
    return af.long()

def get_frag_graph_info(frag_edges, rotatable_bonds, atom_to_fragments ):
    G, longest_path, edges_in_path, frag_weight = get_weighted_graph(frag_edges)
    start = longest_path[0]
    visited_edges = dfs_weighted_edges(G,start)    
    indices_ve, reverse_indices_ve = get_reverse_indices(visited_edges)
    indices_fe, reverse_indices_fe = get_reverse_indices(frag_edges)
    return rotatable_bonds[indices_fe][reverse_indices_ve], frag_weight[atom_to_fragments]


def get_weighted_graph(frag_edge):
    # This will retrun a graph of frag, its longest path and the edge in path.
    # As no ring in this graph, this graph is a Tree
    # Also, the graph with weighted node.
    # The weight of sub-branch (3) , branch (2) ,and main-chain (1).

    G = nx.Graph()
    G.add_edges_from(frag_edge.numpy())
    for node in G.nodes:
        G.nodes[node]['weight'] = 3
    longest_path, edges_in_path = get_longest_path(G)

    for node in longest_path:
        G.nodes[node]['weight'] = 1

    sgs = deepcopy(G)
    for edge in edges_in_path:
        sgs.remove_edge(*edge)

    sgs = [sgs.subgraph(c).copy() for c in nx.connected_components(sgs)]
    for sg in sgs:
        if len(sg) == 1:
            continue
        else:
            root_node = max(sg.nodes(data=True), key=lambda x: x[1]['weight'])[0]
            edges = list(bfs_edges(sg, root_node))
            farthest_node = edges[-1][1]  
            sg_path = nx.shortest_path(sg, root_node, farthest_node)
            for node in sg_path:
                G.nodes[node]['weight'] = 2
            G.nodes[root_node]['weight'] = 3
    frag_weight = []
    for node in G.nodes:
        frag_weight.append(G.nodes[node]['weight'])
    return G, longest_path, edges_in_path, torch.tensor(frag_weight)


def dfs_weighted_edges(graph, start):
    # This dfs will search the edge w largest weight
    # The weight of sub-branch (3) , branch (2) ,and main-chain (1).
    visited, stack = set(), [(None, start)]
    edges = []
    while stack:
        parent, node = stack.pop()
        if node not in visited:
            if parent is not None:
                edges.append((parent, node))
            visited.add(node)
            neighbors = sorted(graph.neighbors(node), key=lambda x: graph.nodes[x]['weight'])
            stack.extend((node, neighbor) for neighbor in neighbors)
    return torch.sort(torch.tensor(edges))[0]

def get_reverse_indices(bond_index):
    # x is a N*2 tensor
    # node id << 1000: so we transfer sort 2D to 1D;
    edge_number = bond_index[:,0] * 1000 + bond_index[:,1]

    # Sort the tensor and get the sorted indices
    sorted_tensor, indices = edge_number.sort(dim=0)

    # Compute the reverse indices
    reverse_indices = torch.argsort(indices, dim=0)

    return indices, reverse_indices

def get_longest_path(G):
    # use two BFS to find the longest chain in graph
    for i in G.nodes:
        start_node = i
        break

    edges = nx.bfs_tree(G, start_node).edges()
    last_node_in_bfs = list(edges)[-1][1]
    edges = nx.bfs_tree(G, last_node_in_bfs).edges()
    other_end_of_diameter = list(edges)[-1][1]

    # node in the longest path
    longest_path = nx.shortest_path(G,  other_end_of_diameter, last_node_in_bfs)

    # edge in the longest path
    edges_in_path = [(longest_path[i], longest_path[i+1]) for i in range(len(longest_path)-1)]
    return longest_path, edges_in_path

def get_sorted_indice(A,B):
    # get indice of B in A 
    indices = []

    # Iterate over each row in B
    for row in B:
        # Find the index in A where the row from B matches
        match = (A == row).all(dim=1).nonzero(as_tuple=True)[0]
        indices.append(match.item())

    return indices

def sort_confs(confs):
    return sorted(confs, key=lambda conf: -conf['boltzmannweight'])

def clean_confs(smi, confs, limit=None):
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
    
    return [confs[i]['rd_mol'] for i in good_ids]