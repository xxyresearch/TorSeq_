"""dataset utils import"""
import pickle
import random
from math import pi
import numpy as np
import random
from copy import deepcopy
import math
import ot

import networkx as nx
from networkx.algorithms.traversal.breadth_first_search import bfs_edges

from rdkit import Chem
from rdkit.Chem.rdMolTransforms import SetDihedralRad, GetDihedralRad
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleRad, GetDihedralRad
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdMolAlign import AlignMol, GetBestRMS

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse


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

"""One Hop Procession and Get Ground Truth"""
"""We use GEOMOL """
def one_hop_process(mol, types):
    """
    Transfer Molecular data to basic one hop graph data
    :mol - rdkit mol data
    :return 0 - x, 
    :return 1 - edge_index, 
    :return 2 - edge_attr, 
    :return 3 - bond_index, 
    :return 4 - chiral_tag, 
    """
    N = mol.GetNumAtoms()
    type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    neighbor_dict = {}
    conf = mol.GetConformer()
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx.append(types[atom.GetSymbol()])
        n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(n_ids) > 1:
            neighbor_dict[i] = torch.tensor(n_ids)
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                                1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))  #7 ---->   #
        
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [  #5   #20~24              #drug - 40 -sp2; 41- sp3
                                Chem.rdchem.HybridizationType.SP,
                                Chem.rdchem.HybridizationType.SP2,
                                Chem.rdchem.HybridizationType.SP3,
                                Chem.rdchem.HybridizationType.SP3D,
                                Chem.rdchem.HybridizationType.SP3D2]))
        #"""
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))  #7   #25~31
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))  #3 32~34
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                int(ring.IsAtomInRingOfSize(i, 4)),
                                int(ring.IsAtomInRingOfSize(i, 5)),
                                int(ring.IsAtomInRingOfSize(i, 6)),
                                int(ring.IsAtomInRingOfSize(i, 7)),
                                int(ring.IsAtomInRingOfSize(i, 8))])    #6   35~40
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))  #4  41~44
        #"""
    z = torch.tensor(atomic_number, dtype=torch.long)
    chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

    row, col, edge_type, bond_features = [], [], [], []
    bond_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_index.append([start,end])
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bt = tuple(sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
        bond_features += 2 * [int(bond.IsInRing()),
                                int(bond.GetIsConjugated()),
                                int(bond.GetIsAromatic())]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x3 = chiral_tag.view(-1,1)
    x = torch.cat([x1.to(torch.float), x2, x3.to(torch.float)], dim=-1)
    

    bond_index = torch.tensor(bond_index)
    return x, edge_index, edge_attr, bond_index, chiral_tag

def get_h_mask(mol):
    # Initialize an empty list to store boolean values
    atom_mask = []

    # Traverse each atom in the molecule
    for atom in mol.GetAtoms():
        # If atom is Hydrogen, append True otherwise False
        atom_mask.append(atom.GetSymbol() == 'H')

    # Convert the list into PyTorch tensor
    atom_mask_tensor = torch.tensor(atom_mask, dtype=torch.bool)
    return atom_mask_tensor

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

def mask_process_r_bond(to_do_list, known_node_mask, unknown_node_mask, rotatable_edges):
    # 
    for atom_num in to_do_list:
        target_bond = rotatable_edges[torch.any(rotatable_edges == atom_num, dim=1)]
        bond_pair_atom_set = target_bond[target_bond!= atom_num ]
        for bond_pair_atom in bond_pair_atom_set:
            known_node_mask[bond_pair_atom] = True
            unknown_node_mask[bond_pair_atom] = False
    return known_node_mask, unknown_node_mask

def transform_points(points, first, second):
    # Translate points so v0 is at origin
    translated_points = points - points[first]

    # Rotate points around z-axis to bring v1 to xz-plane
    theta = torch.atan2(translated_points[second, 1], translated_points[second, 0])
    R_z = torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                        [-torch.sin(theta), torch.cos(theta), 0],
                        [0, 0, 1]])
    rotated_points_1 = (R_z @ translated_points.T).T

    # Rotate points around y-axis to bring v1 to z-axis
    rho = torch.atan2(rotated_points_1[second, 0], rotated_points_1[second, 2])
    R_y = torch.tensor([[torch.cos(rho), 0, -torch.sin(rho)],
                        [0, 1, 0],
                        [torch.sin(rho), 0, torch.cos(rho)]])
    rotated_points_2 = (R_y @ rotated_points_1.T).T

    return rotated_points_2

def rotate_along_z_axis(points, theta_rad):
    # Define the rotation matrix for rotation about the Z-axis
    rotation_matrix =torch.tensor([
        [torch.cos(theta_rad), -torch.sin(theta_rad), 0],
        [torch.sin(theta_rad), torch.cos(theta_rad), 0],
        [0, 0, 1]
    ])

    # Transpose the points so they are in 3xN format, rotate them, and then transpose the result back to Nx3 format
    rotated_points = torch.mm(rotation_matrix, points.t()).t()

    return rotated_points


def get_best_RMSD(src, tgt):
    """For Calculate loss ONLY. Align the source to the target using the Kabsch algorithm"""
    # src and tgt are N x 3 tensors
    #src = src[~h_atom_mask]
    #tgt = tgt[~h_atom_mask]

    # kabsch_alignmen
    # Compute centroids
    src_centroid = torch.mean(src, dim=0)
    tgt_centroid = torch.mean(tgt, dim=0)

    # Subtract centroids
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    # Compute covariance matrix
    cov = src_centered.t().mm(tgt_centered)

    # Singular value decomposition
    u, s, v = torch.svd(cov)

    # Compute optimal rotation
    d = (v.mm(u.t())).det()
    rot_matrix = v.mm(torch.diag(torch.tensor([1, 1, torch.sign(d)]))).mm(u.t())

    # Apply rotation to the source
    src_aligned = (src_centered.mm(rot_matrix.t())) + tgt_centroid

    # calculate RMSD
    diff = src_aligned - tgt
    squared_diff = diff ** 2
    sum_squared_diff = torch.sum(squared_diff)
    mean_squared_diff = sum_squared_diff / src.shape[0]

    return torch.sqrt(mean_squared_diff)

def get_random_mol_list(confs, canonical_smi):
    mol_list = []
    for idx, conf in enumerate(confs):
        mol = conf['rd_mol']

        # skip mols with atoms with more than 4 neighbors for now
        n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if np.max(n_neighbors) > 4:
            continue

        # filter for conformers that may have reacted
        try:
            conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception as e:
            continue

        if conf_canonical_smi != canonical_smi:
            continue

        mol_list.append(mol)
    return mol_list

def get_atom_neighbor(mol, x, rotatable_edges, atom_to_fragments, use_H = True):
    neighbor_list = []
    h_atom = (x[:,35] == 1).nonzero(as_tuple=True)[0]
    one_neighbor_atom = find_one_neighbor_atom(x, mol)
    rotatable_edges_list = rotatable_edges.tolist()
    for i in range(len(x)):
        atom_neighbor_renew = []
        atom_idx = i
        atom = mol.GetAtomWithIdx(atom_idx)
        neighbor = [idx.GetIdx() for idx in atom.GetNeighbors()]
        a = [] # h atom
        b = [] # one neighbor atom
        c = [] # rotatable atom
        d = [] # ring atom
        for n_atom in neighbor:
            if n_atom in h_atom:
                a.append(n_atom)
            elif n_atom in one_neighbor_atom:
                b.append(n_atom)
            elif [i, n_atom] in rotatable_edges_list or [n_atom, i] in rotatable_edges_list:
                c.append(n_atom)
            else: d.append(n_atom)
        for i in c: atom_neighbor_renew.append(i)
        for i in d: atom_neighbor_renew.append(i)
        for i in b: atom_neighbor_renew.append(i)
        if use_H:
            for i in a: atom_neighbor_renew.append(i)
        neighbor_list.append(atom_neighbor_renew)
    return neighbor_list

def find_one_neighbor_atom(x, mol):
    """this function is used to find non-H atom with one non-H neighbour"""
    one_neighbor_atom = []
    for atom_idx, atom_attr in enumerate(x):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_neighbor = [idx.GetIdx() for idx in atom.GetNeighbors()]
        if len(atom_neighbor) == 1 and atom_attr[35] != 1:
            one_neighbor_atom.append(atom_idx)
        elif atom_has_one_non_H_neighbor(mol, atom_idx):
            one_neighbor_atom.append(atom_idx)


    return torch.tensor(one_neighbor_atom)

def atom_has_one_non_H_neighbor(mol, atom_idx):
    # Get the atom
    atom = mol.GetAtomWithIdx(atom_idx)

    # Initialize a count of non-Hydrogen neighbors
    non_H_neighbors = 0

    # Iterate over the neighbors of the atom
    for neighbor in atom.GetNeighbors():
        # If the neighbor is not Hydrogen, increment the counter
        if neighbor.GetAtomicNum() != 1:
            non_H_neighbors += 1

    # If the atom has exactly one non-Hydrogen neighbor, return True
    return non_H_neighbors == 1

def edge_idx_calculation(index, n0, n1):
    """
    Calculate the index of an edge in edge_index
    : input 0 - transpose of edge_index, shape should be [2 * n]
    : input 1 - first_node_index
    : input 2 - second_node_index
    :return - index number
    """
    idx_0 = index[0]
    idx_1 = index[1]
    part_one = (n0 == idx_0 ).nonzero(as_tuple=True)[0]
    part_two = (n1 == idx_1 ).nonzero(as_tuple=True)[0]  
    part_one_repeat = part_one.repeat(len(part_two),1).t()
    return (part_one[(part_one_repeat == part_two).nonzero(as_tuple=True)[0]])

def get_af(fragments, x):
    af = torch.zeros_like(x[:,0])
    for i in range(len(fragments)):
        for node_num in fragments[i]:
            af[node_num] = i
    return af.long()

def get_groundtruth(a0, a1, a2, a3, mol_list):
    
    dihedral = []
    for mol in mol_list:
        conf = mol.GetConformer()
        dihedral.append(GetDihedralRad (conf,a0, a1, a2, a3 )) 
    
    return torch.tensor(dihedral)

def get_three_hop_cal_info(a0, a1, a2, a3, mol_list):
    three_hop_distance = []
    hab = []

    for mol in mol_list:
        conf = mol.GetConformer()
        pos = torch.tensor(conf.GetPositions())
        pos_normalized = transform_points(pos, a1, a2)
        three_hop_distance.append(torch.sqrt(torch.sum((pos_normalized[a0] - pos_normalized[a3]) ** 2)))

        h = torch.abs(pos_normalized[a0][2] - pos_normalized[a3][2]) 
        a = torch.sqrt(torch.sum((pos_normalized[a0][:2] ** 2)))
        b = torch.sqrt(torch.sum((pos_normalized[a3][:2] ** 2)))
        hab.append([a,b,h])
    hab = torch.mean(torch.tensor(hab),dim=0)
    three_hop_distance = torch.tensor(three_hop_distance)

    return three_hop_distance, hab

def get_dihedral_info(sorted_rotatable_bond, edge_index, atom_neighbor, atom_to_fragments):
    # get dihedral info
    dihedral_index = []
    for bond_start, bond_end in sorted_rotatable_bond:
        neighbor_start = deepcopy(atom_neighbor[bond_start])
        neighbor_end = deepcopy(atom_neighbor[bond_end])
        #try:
        neighbor_start.remove(int(bond_end))
        #except:
        #    xxxx = 1
        #try:
        neighbor_end.remove(int(bond_start))
        #except:
        #    xxxx = 1
        a0 = neighbor_start[0]
        a1 = int(bond_start)
        a2 = int(bond_end)
        a3 = neighbor_end[0]
        e0 = edge_idx_calculation(edge_index, a0, a1)
        e1 = edge_idx_calculation(edge_index, a1, a2)
        e2 = edge_idx_calculation(edge_index, a2, a3)
        f0 = atom_to_fragments[a1]
        f1 = atom_to_fragments[a2]
        dihedral_index.append(torch.tensor([a0, a1, a2, a3, e0, e1, e2, f0, f1]))
    dihedral_index = torch.stack(dihedral_index)
    dihedral_index = dihedral_index.long()
    #h_mask_rotation = ~(h_mask[dihedral_index[:,0]] | h_mask[dihedral_index[:,3]])
    #sorted_rotatable_bond = sorted_rotatable_bond[h_mask_rotation]
    #dihedral_index = dihedral_index[h_mask_rotation]
    return dihedral_index

def get_frag_info(fragments, max_rotatable_bond_num):
      
    frag = torch.zeros(max_rotatable_bond_num + 10)
    for i in range(len(fragments)):
        frag[i] = i
    frag_mask = (frag != 0).float()
    frag_mask[0] = 1
    return frag, frag_mask

def get_h_mask(mol):
    # Initialize an empty list to store boolean values
    atom_mask = []

    # Traverse each atom in the molecule
    for atom in mol.GetAtoms():
        # If atom is Hydrogen, append True otherwise False
        atom_mask.append(atom.GetSymbol() == 'H')

    # Convert the list into PyTorch tensor
    atom_mask_tensor = torch.tensor(atom_mask, dtype=torch.bool)
    return atom_mask_tensor

def get_rotation_mask(edge_index, x,  dihedral_indx):
    rotation_mask = torch.zeros(dihedral_indx.shape[0], x.shape[0]).bool()
    edges = [tuple(edge) for edge in edge_index.t().numpy()]
    G = nx.Graph()
    G.add_edges_from(edges)
    for idx, i in enumerate(dihedral_indx):
        a0 = int(i[1])
        a1 = int(i[2])
        G_sub = deepcopy(G)
        G_sub.remove_edge(a0,a1)
        subgraphs = nx.connected_components(G_sub)
        for subgraph_idx, subgraph in enumerate(subgraphs):
            if subgraph_idx!=0:
                rotation_mask[idx][list(subgraph)] = True
    return rotation_mask

def get_real_pos(mol_list):
    real_pos = []
    for mol in mol_list:
        real_pos.append(torch.tensor(mol.GetConformer().GetPositions()))
    return torch.stack(real_pos)


def get_fake_info(mol, dihedral_indx, max_conf_num):
    fake_pos = []
    
    randomly_rotate_rad = []
    gen_mol = deepcopy(mol)
    AllChem.EmbedMolecule(gen_mol)
    gen_conf = gen_mol.GetConformer()
    verify_pos = torch.tensor(gen_conf.GetPositions())
    for i in range(max_conf_num):
        random_rad = []
        for j in dihedral_indx:
            a0 = int(j[0])
            a1 = int(j[1])
            a2 = int(j[2])
            a3 = int(j[3])
            random_radian = random.uniform(0, 2*math.pi)
            random_rad .append(random_radian )
            SetDihedralRad(gen_conf ,a0,a1,a2,a3,random_radian)
        randomly_rotate_rad.append(random_rad)

        fake_pos.append(torch.tensor(gen_conf.GetPositions()))
    fake_pos = torch.stack(fake_pos)
    randomly_rotate_rad = torch.tensor(randomly_rotate_rad)
    return fake_pos.permute(1,0,2),  verify_pos, randomly_rotate_rad.transpose(1,0)


def get_side_leaf_info(edge_index, dihedral_indx,h_mask):
    edges = [tuple(edge) for edge in edge_index.t().numpy()]
    G = nx.Graph()
    G.add_edges_from(edges)
    s_side_info = []
    s_leaf_info = []
    r_side_info = []
    r_leaf_info = []
    for idx, dihedral_indx_sample in enumerate(dihedral_indx):
        a0 = int(dihedral_indx_sample[0])
        a1 = int(dihedral_indx_sample[1])
        a2 = int(dihedral_indx_sample[2])
        a3 = int(dihedral_indx_sample[3])
        
        # Convert the hop counts to a list of tuples [(node, hop_count), ...],
        # sorted by node
        # Extract just the hop counts, in order
        # Convert the list of hop counts to a tensor
        A_hops = nx.shortest_path_length(G, a0)
        A_sorted_hops = sorted(A_hops.items(), key=lambda x: int(x[0]))
        A_hop_counts = [hop_count for node, hop_count in A_sorted_hops ]
        A_hops = torch.tensor(A_hop_counts)

        B_hops = nx.shortest_path_length(G, a1)
        B_sorted_hops = sorted(B_hops.items(), key=lambda x: int(x[0]))
        B_hop_counts = [hop_count for node, hop_count in B_sorted_hops ]
        B_hops = torch.tensor(B_hop_counts)

        C_hops = nx.shortest_path_length(G, a2)
        C_sorted_hops = sorted(C_hops.items(), key=lambda x: int(x[0]))
        C_hop_counts = [hop_count for node, hop_count in C_sorted_hops ]
        C_hops = torch.tensor(C_hop_counts)

        D_hops = nx.shortest_path_length(G, a3)
        D_sorted_hops = sorted(D_hops.items(), key=lambda x: int(x[0]))
        D_hop_counts = [hop_count for node, hop_count in D_sorted_hops ]
        D_hops = torch.tensor(D_hop_counts)

        s_side = (B_hops < C_hops) & (B_hops != 0) #& (~h_mask)
        r_side = (C_hops < B_hops) & (C_hops != 0) #& (~h_mask)
        s_leaf = (B_hops - 1 == A_hops) & s_side
        r_leaf = (C_hops - 1 == D_hops) & r_side

        s_side_indices = torch.nonzero(s_side).squeeze()
        s_side_hop = B_hops[s_side_indices] - 1
        s_side_index = torch.ones_like(s_side_hop) * idx
        s_side_info.append(torch.stack((s_side_indices, s_side_hop, s_side_index)).t().view(-1,3))

        s_leaf_indices = torch.nonzero(s_leaf).squeeze()
        s_leaf_hop = A_hops[s_leaf_indices]
        s_leaf_index = torch.ones_like(s_leaf_hop) * idx
        s_leaf_info.append(torch.stack((s_leaf_indices, s_leaf_hop, s_leaf_index)).t().view(-1,3))

        r_side_indices = torch.nonzero(r_side).squeeze()
        r_side_hop = C_hops[r_side_indices] - 1
        r_side_index = torch.ones_like(r_side_hop) * idx
        r_side_info.append(torch.stack((r_side_indices, r_side_hop, r_side_index)).t().view(-1,3))

        r_leaf_indices = torch.nonzero(r_leaf).squeeze()
        r_leaf_hop = D_hops[r_leaf_indices]
        r_leaf_index = torch.ones_like(r_leaf_hop) * idx
        r_leaf_info.append(torch.stack((r_leaf_indices, r_leaf_hop, r_leaf_index)).t().view(-1,3))
    try:
        s_side_info = torch.cat(s_side_info,0)
        s_leaf_info = torch.cat(s_leaf_info,0)
        r_side_info = torch.cat(r_side_info,0)
        r_leaf_info = torch.cat(r_leaf_info,0)
    except:
        print(s_side_info)
        print(s_leaf_info)
        print(r_side_info)
        print(r_leaf_info)
        assert(1==2)
    return s_side_info, s_leaf_info, r_side_info, r_leaf_info


def calculate_phi(v1, v2, v3):
    normal_1 = np.cross(v1, v2)
    normal_2 = np.cross(v2, v3)
    # Normalize the normals
    normal_1 = normal_1 / np.linalg.norm(normal_1, axis=1, keepdims=True)
    normal_2 = normal_2 / np.linalg.norm(normal_2, axis=1, keepdims=True)

    # Calculate the dot product of the normalized vectors
    dot_product = np.einsum('ij,ij->i', normal_1, normal_2)

    # To ensure the value is within the domain for arccos (between -1 and 1)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle
    angle = np.arccos(dot_product)

    # Calculate the sign of the dihedral angle
    reference = np.cross(normal_1, normal_2)
    sign = np.sign(np.einsum('ij,ij->i', reference, v2))

    # Multiply the angle by the sign
    angle *= sign
    return torch.tensor(angle)

def calculate_structure_weight(v1, v2):
    d1 = torch.norm(v1,dim=1)
    d2 = torch.norm(v2,dim=1)
    dot_product = torch.matmul(v1.unsqueeze(dim=1), v2.unsqueeze(dim=2)).view(-1)
    cos_angle = dot_product / (d1 * d2 + 1e-10)
    cos_angle = torch.where(cos_angle >1, 1, cos_angle)
    cos_angle = torch.where(cos_angle <-1, -1, cos_angle)
    weight = torch.sin(torch.acos(cos_angle ).view(-1) )
    return weight

def get_another_r_part(r_bond, num):
    rows, cols = (r_bond == num).nonzero(as_tuple=True)
    for row, col in zip(rows, cols):
        other_col = 1 - col
        output  = r_bond[row, other_col]
    return output


def get_chain_geo_points(dihedral_indx, atom_to_fragments, sorted_rotatable_bond, symmetrical_frags_chain):
    output = []
    for bond in dihedral_indx:
        atom_num = torch.arange(0 , len(atom_to_fragments))
        rotatable_bond_atoms = torch.unique(sorted_rotatable_bond)
        r_mask = torch.zeros_like(atom_to_fragments).bool()
        r_mask[rotatable_bond_atoms] = True

        a10 = bond[1]
        f1_mask = atom_to_fragments == atom_to_fragments[a10]
        f1_mask[a10] = False
        a00_mask = f1_mask & r_mask
        a00_new = atom_num[a00_mask]
        if len(a00_new) == 0 or atom_to_fragments[a10] in symmetrical_frags_chain:
            a00 = bond[0]
            a01 = bond[1]
        else: 
            a01 = a00_new[0]
            a00 = get_another_r_part(sorted_rotatable_bond, a01)
            
        a11 = bond[2]
        f2_mask = atom_to_fragments == atom_to_fragments[a11]
        f2_mask[a11] = False
        a20_mask = f2_mask & r_mask
        a20_new = atom_num[a20_mask]
        if len(a20_new) == 0 or atom_to_fragments[a11] in symmetrical_frags_chain:
            a20 = bond[2]
            a21 = bond[3]
        else: 
            a20 = a20_new[0]
            a21 = get_another_r_part(sorted_rotatable_bond, a20)

        output.append([a00, a01, a10, a11, a20, a21])

    return torch.tensor(output)

def get_feature(mol_list, dihedral_indx, chain_core_points, sym_mask, real_mol =True ):
    # get groundtruth phi and phi_chain
    # get local feature phi_delta_in, phi_delta_out, and distance of outer bound to axis and along axis
    phi_all = []
    phi_chain_all = []
    local_feature_all = []
    pos_all = []

    for mol in mol_list:
        if real_mol:
            pos = torch.tensor(mol.GetConformer().GetPositions())
            pos_all.append(pos)
        else:
            pos = torch.tensor(mol.GetPositions())
            pos_all.append(pos)
        v1 = pos[dihedral_indx[:,1]] - pos[dihedral_indx[:,0]] 
        v2 = pos[dihedral_indx[:,2]] - pos[dihedral_indx[:,1]] 
        v3 = pos[dihedral_indx[:,3]] - pos[dihedral_indx[:,2]] 
        v1_chain = pos[chain_core_points[:,1]] - pos[chain_core_points[:,0]] 
        v3_chain = pos[chain_core_points[:,5]] - pos[chain_core_points[:,4]] 

        phi = calculate_phi(v1, v2, v3)
        chain_phi = calculate_phi(v1_chain, v2, v3_chain)
        phi_in = calculate_phi(-v1_chain, v2, v1)
        phi_out = calculate_phi(v3, v2, -v3_chain)

        #for i, p in enumerate(phi):
        #    if torch.cos(GetDihedralRad((mol.GetConformer()), int(dihedral_indx[:,0][i]), int(dihedral_indx[:,1][i]), int(dihedral_indx[:,2][i]), int(dihedral_indx[:,3][i])) - p) < 0.999:
        #        assert(1==2)
                        
        # C--> AB,  C-->A along AB
        pos_in_inner = torch.stack((pos[dihedral_indx[:,1]],pos[dihedral_indx[:,2]], pos[chain_core_points[:,1]]),1)
        pos_in_outer = torch.stack((pos[dihedral_indx[:,1]],pos[dihedral_indx[:,2]], pos[chain_core_points[:,0]]),1)
        pos_out_inner = torch.stack((pos[dihedral_indx[:,2]],pos[dihedral_indx[:,1]], pos[chain_core_points[:,4]] ),1)
        pos_out_outer  = torch.stack((pos[dihedral_indx[:,2]],pos[dihedral_indx[:,1]], pos[chain_core_points[:,5]] ),1)
        distance_in_inner = calculate_distances(pos_in_inner.numpy())
        distance_in_outer = calculate_distances(pos_in_outer.numpy())
        distance_out_inner = calculate_distances(pos_out_inner.numpy())
        distance_out_outer = calculate_distances(pos_out_outer.numpy())
        local_distance_feature = torch.cat((distance_in_inner, distance_in_outer, distance_out_inner, distance_out_outer),1)
        local_delta_phi_feature = torch.stack((phi_in, phi_out)).t()
        local_geo_feature = torch.cat((local_distance_feature, local_delta_phi_feature),1)
        phi_all.append(phi)
        phi_chain_all.append(chain_phi)
        local_feature_all.append(local_geo_feature )

    phi = get_sym_tgt(torch.stack(phi_all), sym_mask)
    phi_chain = get_sym_tgt (torch.stack(phi_chain_all), sym_mask)
    local_feature = torch.stack(local_feature_all)
    pos = torch.stack(pos_all)
    
    return phi, phi_chain, local_feature, pos


def get_feature_local(mol_list, dihedral_indx, chain_core_points):
    # get groundtruth phi and phi_chain
    # get local feature phi_delta_in, phi_delta_out, and distance of outer bound to axis and along axis
    local_feature_all = []
    pos_all = []

    for mol in mol_list:
        pos = torch.tensor(mol.GetConformer().GetPositions())
        pos_all.append(pos)
        v1 = pos[dihedral_indx[:,1]] - pos[dihedral_indx[:,0]] 
        v2 = pos[dihedral_indx[:,2]] - pos[dihedral_indx[:,1]] 
        v3 = pos[dihedral_indx[:,3]] - pos[dihedral_indx[:,2]] 
        v1_chain = pos[chain_core_points[:,1]] - pos[chain_core_points[:,0]] 
        v3_chain = pos[chain_core_points[:,5]] - pos[chain_core_points[:,4]] 

        phi_in = calculate_phi(-v1_chain, v2, v1)
        phi_out = calculate_phi(v3, v2, -v3_chain)

        # C--> AB,  C-->A along AB
        pos_in_inner = torch.stack((pos[dihedral_indx[:,1]],pos[dihedral_indx[:,2]], pos[chain_core_points[:,1]]),1)
        pos_in_outer = torch.stack((pos[dihedral_indx[:,1]],pos[dihedral_indx[:,2]], pos[chain_core_points[:,0]]),1)
        pos_out_inner = torch.stack((pos[dihedral_indx[:,2]],pos[dihedral_indx[:,1]], pos[chain_core_points[:,4]] ),1)
        pos_out_outer  = torch.stack((pos[dihedral_indx[:,2]],pos[dihedral_indx[:,1]], pos[chain_core_points[:,5]] ),1)
        distance_in_inner = calculate_distances(pos_in_inner.numpy())
        distance_in_outer = calculate_distances(pos_in_outer.numpy())
        distance_out_inner = calculate_distances(pos_out_inner.numpy())
        distance_out_outer = calculate_distances(pos_out_outer.numpy())
        local_distance_feature = torch.cat((distance_in_inner, distance_in_outer, distance_out_inner, distance_out_outer),1)
        local_delta_phi_feature = torch.stack((phi_in, phi_out)).t()
        local_geo_feature = torch.cat((local_distance_feature, local_delta_phi_feature),1)
        #phi_all.append(phi)
        #phi_chain_all.append(chain_phi)
        local_feature_all.append(local_geo_feature )

    local_feature = torch.stack(local_feature_all)
    
    return local_feature

def get_hyper_mask(sym_mask):
    hyper_bond_mask = torch.zeros_like(sym_mask).bool()
    for idx, mask in enumerate(sym_mask):
        if idx+2 < len(sym_mask) and sym_mask[idx] ==1 and sym_mask[idx+2] ==1:
            hyper_bond_mask[idx+1] = True
            
    return hyper_bond_mask


def pertube_gen_mol(gen_mol_list, dihedral_indx):
    for mol in gen_mol_list:
        #print(dihedral_indx)
        conf = mol.GetConformer()
        pos_start = deepcopy(mol.GetConformer().GetPositions())
        for idx, d_index in enumerate(dihedral_indx):
            random_number = random.uniform(-pi, pi)
            SetDihedralRad(conf,int(d_index[0]) , int(d_index[1]), int(d_index[2]), int(d_index[3]), random_number)
        #print(pos_start - mol.GetConformer().GetPositions())
        #assert(1==2)
    return gen_mol_list

def get_similar_indice(delta_chain_phi_one,delta_chain_phi_two):
    indice_list = []
    local_diff = torch.acos(torch.cos(delta_chain_phi_one.unsqueeze(0) - delta_chain_phi_one.unsqueeze(1))) + torch.acos(torch.cos(delta_chain_phi_two.unsqueeze(0) - delta_chain_phi_two.unsqueeze(1)))
    local_diff = torch.mean(local_diff, dim=-1)
    k = min(len(local_diff), 5)
    for i in local_diff:
        values, indices = torch.topk(-i, k)
        indice_list.append(indices)
    return torch.stack(indice_list)

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

def get_frag_graph_info(frag_edges, rotatable_bonds, atom_to_fragments ):
    G, longest_path, edges_in_path, frag_weight = get_weighted_graph(frag_edges)
    start = longest_path[0]
    visited_edges = dfs_weighted_edges(G,start)    
    indices_ve, reverse_indices_ve = get_reverse_indices(visited_edges)
    indices_fe, reverse_indices_fe = get_reverse_indices(frag_edges)
    return rotatable_bonds[indices_fe][reverse_indices_ve]#, frag_weight[atom_to_fragments]

def calculate_distances(pos):
    # Extract points A, B, C
    # Calculate vectors AB, AC for the whole batch
    AB = pos[:, 1, :] - pos[:, 0, :]
    AC = pos[:, 2, :] - pos[:, 0, :]

    # Calculate the distance from C to the line defined by A and B for the whole batch
    numerator = np.linalg.norm(np.cross(AB, AC, axis=1), axis=1)
    denominator = np.linalg.norm(AB, axis=1)
    distance_C_to_AB = numerator / denominator

    # Calculate the distance from C to A along the line defined by A and B for the whole batch
    dot_product = np.einsum('ij,ij->i', AC, AB)
    norm_squared = np.linalg.norm(AB, axis=1) ** 2
    projection_AC_on_AB = (dot_product / norm_squared)[:, None] * AB
    distance_C_to_A_along_AB = np.linalg.norm(projection_AC_on_AB, axis=1)

    return torch.tensor(np.stack((distance_C_to_AB, distance_C_to_A_along_AB), axis=-1))

def clean_mol_list(mol, mol_list):
    cleaned_list = []
    print(mol)
    sample_mol = deepcopy(Chem.RemoveHs(mol))
    canonical_smi = Chem.MolToSmiles(sample_mol)
    for check_mol in mol_list:
        try:
            conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(deepcopy(check_mol)))
        except Exception as e:
            continue

        if conf_canonical_smi != canonical_smi:
            continue

        cleaned_list.append(check_mol)
    return cleaned_list


def get_rd_gen_mol(mol, gen_num):
    AllChem.EmbedMultipleConfs(mol, gen_num+1, numThreads=8)
    mols = []
    for idx, conf_id in enumerate(mol.GetConformers()):
        if idx == 0:
            continue
        new_mol = Chem.Mol(mol)  # Create a copy of the molecule
        new_mol.RemoveAllConformers()  # Remove all conformations from the copy
        conf = conf_id.GetOwningMol().GetConformer(conf_id.GetId())  # Get the conformation
        new_mol.AddConformer(conf, assignId=True)  # Add the conformation to the copy
        mols.append(new_mol)  # Add the copy to the list
    return mols

def cosine_similarity_matrix(mat1, mat2, eps=1e-8):
    norm_mat1 = mat1 / (np.linalg.norm(mat1, axis=1, keepdims=True) + eps)
    norm_mat2 = mat2 / (np.linalg.norm(mat2, axis=1, keepdims=True) + eps)
    cos_sim_mat = np.dot(norm_mat1, norm_mat2.T)
    return cos_sim_mat

def top_k_indices(mat, k):
    top_k = np.argsort(mat, axis=1)[:, ::-1][:, :k]
    return top_k

def get_gen_feature_indice(real_feature, fake_feature, gen_num):
    # use emd to match the best gen local for real mol:
    # input should be (N_real * H, N_fake * H, int) np array
    # return N_real tensor  #if N_real < N_gen, return N_gen tensor
    if len(real_feature) < gen_num:
        return torch.arange(gen_num)
    cos_sim_mat = cosine_similarity_matrix(real_feature, fake_feature)
    diff_matrix = 1 - cos_sim_mat
    H_2 = np.ones(diff_matrix.shape[0]) / diff_matrix.shape[0]
    H_1 = np.ones(diff_matrix.shape[1]) / diff_matrix.shape[1]
    ot_mat = ot.emd(a=H_2, b=H_1, M=np.max(np.abs(diff_matrix)) + diff_matrix, numItermax=10000)
    gen_conf = top_k_indices(ot_mat, 1)
    return torch.tensor(gen_conf.copy()).reshape(-1)


def get_hexagon_ring(atom_to_fragments, h_mask, x):
    frag_atom = atom_to_fragments[~h_mask]
    frag_atom_sp_2 = x[:,40][~h_mask]
    
    frag_num = []
    for i in range(max(frag_atom)+1):
        atom_num_in_frag = len(frag_atom[frag_atom == i])
        sp_2 = frag_atom_sp_2[frag_atom == i]
        if atom_num_in_frag != 6:
            continue
        if (sp_2 == 1).all():
            frag_num.append(i)
            continue
    return torch.tensor(frag_num)

def find_nodes_3_hops_node(graph, node):
    # find the three hop node
    nodes_3_hops_away = []
    
    for new_node, depth in nx.single_source_shortest_path_length(graph, node).items():
        if depth == 3:
            nodes_3_hops_away.append(new_node)
    
    return nodes_3_hops_away

def find_all_paths(graph, source, target):
    return list(nx.all_simple_paths(graph, source, target))

def get_symmetrical_frag(atom_to_fragments, h_mask, x, sorted_rotatable_bonds, edge_index):
    hexagon_frag = get_hexagon_ring(atom_to_fragments, h_mask, x)
    rotatable_atom = torch.unique(sorted_rotatable_bonds)
    symmetrical_frags_end = []
    symmetrical_frags_chain = []
    for i in hexagon_frag:
        atom_number = torch.arange(len(atom_to_fragments))[[~h_mask]]
        hexagon_atom_number = atom_number[atom_to_fragments[~h_mask]  == i]

        # check at most 2 points in rotatable bonds
        rotatable_hexagon_atom = hexagon_atom_number[torch.isin(hexagon_atom_number , rotatable_atom)]
        if len(rotatable_hexagon_atom) > 2:
            continue

        #generate_graph
        edge_mask = torch.isin(edge_index[0], hexagon_atom_number) & torch.isin(edge_index[1], hexagon_atom_number)
        filtered_edge_index = edge_index[:, edge_mask]
        edges = list(map(tuple, filtered_edge_index.t().tolist()))
        G = nx.from_edgelist(edges)

        ## Check if the degree of each node is 2
        degrees = [degree for node, degree in G.degree()]
        is_degree_two = all(degree == 2 for degree in degrees)
        if is_degree_two == False: continue

        # if one in rotatable bond, find the farthest one:
        if len(rotatable_hexagon_atom) == 1:
            atom_1 = int(rotatable_hexagon_atom[0])
            atom_2 = find_nodes_3_hops_node(G, atom_1)[0]
            path1, path2 = find_all_paths(G, int(atom_1), int(atom_2))
            atom_tpye_1 = x[:,35][path1]
            atom_tpye_2 = x[:,35][path2]
            if torch.equal(atom_tpye_1 , atom_tpye_2 ) : symmetrical_frags_end.append(i)

        else: 
            atom_1,  atom_2 = rotatable_hexagon_atom
            path1, path2 = find_all_paths(G, int(atom_1), int(atom_2))
            atom_tpye_1 = x[:,35][path1]
            atom_tpye_2 = x[:,35][path2]
            if torch.equal(atom_tpye_1 , atom_tpye_2 ) : symmetrical_frags_chain.append(i)


    return torch.tensor(symmetrical_frags_end), torch.tensor(symmetrical_frags_chain)

def add_hyper_bond(symmetrical_frags_chain, atom_to_fragments, sorted_rotatable_bonds, dihedral_weight):
    eps = 1e-10
    hyper_bonds = []
    #print('a new data')
    #print(symmetrical_frags_chain)
    for s_f_indice in symmetrical_frags_chain:
        #print('a new hyper')
        #print('before', sorted_rotatable_bonds)
        
        srb_in_f = atom_to_fragments[sorted_rotatable_bonds]
        indices_i = (srb_in_f  == s_f_indice).any(dim=1).nonzero(as_tuple=True)[0]
        hyper_start = sorted_rotatable_bonds[indices_i[0]][srb_in_f [indices_i[0]] != s_f_indice]
        hyper_end = sorted_rotatable_bonds[indices_i[1]][srb_in_f [indices_i[1]] != s_f_indice ]
        hyper_bond = torch.tensor([[hyper_start, hyper_end]])
        hyper_bonds.append([hyper_start, hyper_end])

        top = sorted_rotatable_bonds[:indices_i[1]]
        botton = sorted_rotatable_bonds[indices_i[1]:]
        #print(top, hyper_bond, botton)
        sorted_rotatable_bonds = torch.cat((top, hyper_bond, botton), 0 )

        top_weight =  dihedral_weight[:indices_i[1]]
        botton_weight = dihedral_weight[indices_i[1]:]
        #print('after', sorted_rotatable_bonds)
        #print(top_weight,  botton_weight)
        hyper_bond_weight = torch.tensor([(top_weight[-1] + botton_weight[0])/2]) 
        dihedral_weight = torch.cat((top_weight, hyper_bond_weight, botton_weight), 0 )
        dihedral_weight = dihedral_weight/(torch.sum(dihedral_weight)+eps)
        if torch.isnan(dihedral_weight ).any().item():
            print(dihedral_weight)
            assert(1==2)

    sorted_rotatable_bonds, indices = sorted_rotatable_bonds.sort(dim=1)
    return sorted_rotatable_bonds, torch.tensor(hyper_bonds), dihedral_weight

def update_edge_feature(hyper_bond, edge_index, edge_attr):
    new_edge = torch.cat((hyper_bond ,hyper_bond.flip(1)), 0 ).t()
    new_attr = F.one_hot(torch.ones(new_edge.shape[1]).long()*4, num_classes=5).float()
    new_edge_index = torch.cat((edge_index, new_edge), 1)
    new_edge_attr = torch.cat((edge_attr, torch.zeros(len(edge_attr),1)),1)
    new_edge_attr = torch.cat((new_edge_attr,new_attr),0)
    return new_edge_index, new_edge_attr

def get_sym_mask(atom_to_fragments, sorted_rotatable_bonds, symmetrical_frags_end, symmetrical_frags_chain):
    sym_mask_1 = torch.isin(atom_to_fragments[sorted_rotatable_bonds], symmetrical_frags_end).any(dim=1)
    sym_mask_2 = torch.isin(atom_to_fragments[sorted_rotatable_bonds], symmetrical_frags_chain).any(dim=1)
    sym_mask = sym_mask_1 | sym_mask_2
    return sym_mask.float()

def get_sym_tgt (tgt, sym_mask):
    tgt_sym = torch.remainder(tgt + pi * sym_mask + pi, 2*pi) - pi
    tgt = torch.stack((tgt, tgt, tgt, tgt_sym, tgt_sym, tgt_sym),dim=-1)
    return tgt

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

def clean_confs_test(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]

def get_dihedral_weight(edge_index, rotatable_bonds, h_mask):
    eps = 1e-8
    edge_index_np = edge_index.numpy()

    # Create a networkx graph from numpy array
    G = nx.from_edgelist(edge_index_np.transpose())

    weight = [] 
    for bond in rotatable_bonds:
        G_copy = deepcopy(G)
        bond = bond.tolist()
        G_copy .remove_edge(bond[0], bond[1])
        fragments = list(nx.connected_components(G_copy))
        atoms_in_frag_a = torch.sum(~h_mask[torch.tensor(list(fragments[0]))]) - 1
        atoms_in_frag_b = torch.sum(~h_mask[torch.tensor(list(fragments[1]))]) - 1
        weight_new = min(atoms_in_frag_a , atoms_in_frag_b)
        assert(weight_new >= 0)
        weight.append(weight_new)
    weight = torch.tensor(weight)
    #norm_weight = weight / torch.sum(weight) + eps
    norm_weight = weight / (torch.sum(weight) + eps)
    if torch.isnan(norm_weight ).any().item():
        print(edge_index)
        print(weight)
        assert(1==2)
    return norm_weight 

def get_seq_mask(rnn_length, max_rotatable_bond_num):
    padded_num = max_rotatable_bond_num + 2 - rnn_length
    seq_mask = torch.ones(rnn_length)
    seq_mask = F.pad(seq_mask,(0,padded_num) , "constant", 0).bool()
    return seq_mask

def get_sym_dihedral_batch(atom_to_fragments, sorted_rotatable_bonds, symmetrical_frags_chain):
    #print(edge_index.t())
    #print(hexagon_frag)
    #print(sorted_rotatable_bonds)
    #print(symmetrical_frags_chain )

    sym_dihedral_batch = []
    sym_batch_id = -1
    sym_batch_input = atom_to_fragments[sorted_rotatable_bonds]
    for idx, tgt in enumerate(sym_batch_input):
        if idx != 0:
            if tgt[0] in symmetrical_frags_chain or tgt[1] in symmetrical_frags_chain:
                if sym_batch_input[idx-1][0] in symmetrical_frags_chain or sym_batch_input[idx-1][1] in symmetrical_frags_chain:
                    sym_dihedral_batch.append(sym_batch_id) 
                    continue

        sym_batch_id += 1
        sym_dihedral_batch.append(sym_batch_id) 

    return torch.tensor(sym_dihedral_batch)
    #print(sym_batch_input)
    #print(torch.tensor(sym_dihedral_batch))
    #assert(1==2)

def sym_h(gen_mol_list, dihedral_indx, phi_fake, phi_chain_fake, h_mask, atom_neighbor):
    for mol_id, sample_mol in enumerate(gen_mol_list):
        conf = sample_mol.GetConformer()
        for id, d_idx in enumerate(dihedral_indx):
            # get dihedral info
            bond_start = d_idx[1]
            bond_end = d_idx[2]
            angle = []
            
            if h_mask[d_idx[0]]:
                neighbor_start = deepcopy(atom_neighbor[bond_start])
                neighbor_start.remove(int(bond_end))
                for i in neighbor_start:
                    angle.append(GetDihedralRad(conf, i, int(d_idx[1]), int(d_idx[2]), int(d_idx[3])))
                diff = torch.tensor(angle) - angle[0]
                diff = F.pad(diff,(0,3-len(diff)), "constant", 0)
                diff = torch.cat((diff,diff),0)
                phi_fake[mol_id][id] += diff
                phi_chain_fake[mol_id][id] += diff

            elif h_mask[d_idx[3]]:
                neighbor_end = deepcopy(atom_neighbor[bond_end]) 
                neighbor_end.remove(int(bond_start)) 
                for i in neighbor_end:
                    angle.append(GetDihedralRad(conf, int(d_idx[0]), int(d_idx[1]), int(d_idx[2]), i))
                diff = torch.tensor(angle) - angle[0]
                diff = F.pad(diff,(0,3-len(diff)), "constant", 0)
                diff = torch.cat((diff,diff), 0)
                phi_fake[mol_id][id] += diff
                phi_chain_fake[mol_id][id] += diff
    
    phi_fake = torch.remainder(phi_fake+pi, 2*pi) - pi
    phi_chain_fake = torch.remainder(phi_chain_fake +pi, 2*pi) - pi

    return phi_fake, phi_chain_fake


def perturb_conf(gen_mol_list, dihedral_indx):
    perturb_gen_mol_list = deepcopy(gen_mol_list)
    for mol in perturb_gen_mol_list:
        conf = mol.GetConformer()
        for tor_idx in dihedral_indx:
            random_number = random.uniform(-math.pi, math.pi)
            SetDihedralRad(conf, int(tor_idx[0]), int(tor_idx[1]), int(tor_idx[2]), int(tor_idx[3]), random_number)
    return perturb_gen_mol_list