from dataset.PerturbDataset import *
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Dataset
import copy
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from utils.xtb import *

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def featurize_mol_from_smiles(smiles, dataset='drugs'):
    if dataset == 'qm9':
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    elif dataset == 'drugs' or dataset == 'bace':
        types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
                'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
                'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
                'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}

    # filter fragments
    if '.' in smiles:
        return None, None

    # filter mols rdkit can't intrinsically handle
    mol = Chem.MolFromSmiles(smiles)
    
    if mol:
        mol = Chem.AddHs(mol)
    else:
        return None, None
    N = mol.GetNumAtoms()

    # filter out mols model can't make predictions for
    if not mol.HasSubstructMatch(dihedral_pattern):
        return None, None
    if N < 4:
        return None, None

    data = featurize_mol(mol, types)
    data = get_transformation_mask(data)
    rotatable_bonds = data.edge_index.t()[data.mask_edges]
    data.name = smiles
    if len(rotatable_bonds) < 1:
        return mol, data
    fragments = split_mol(mol, data.edge_index)
    atom_to_fragments = get_af(fragments, data.x)
    frag_edges =  torch.sort(atom_to_fragments[rotatable_bonds])[0]
    sorted_rotatable_bonds, node_chain_weight = get_frag_graph_info(frag_edges, rotatable_bonds, atom_to_fragments )
    sort_indice = get_sorted_indice(rotatable_bonds, sorted_rotatable_bonds)

    #print(sorted_rotatable_bonds == rotatable_bonds[sort_indice])
    data.rotatable_bonds = rotatable_bonds 
    data.sorted_rotatable_bonds = sorted_rotatable_bonds
    data.sorted_rotatable_indice = sort_indice
    #data.name = smiles
    return mol, data

def get_transformation_mask_sampling(pyg_data):
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

    return mask_edges, mask_rotate

def get_seed(smi, seed_confs=None, dataset='drugs'):
    mol, data = featurize_mol_from_smiles(smi, dataset=dataset)
    if not mol:
        return None, None
    data.edge_mask, data.mask_rotate = get_transformation_mask_sampling(data)
    data.edge_mask = torch.tensor(data.edge_mask)
    return mol, data

def embed_seeds(mol, data, n_confs, embed_func=None, ):
    embed_num_confs = n_confs
    try:
        mol = embed_func(mol, embed_num_confs)
    except Exception as e:
        print(e.output)
        pass
    if len(mol.GetConformers()) != embed_num_confs:
        print(len(mol.GetConformers()), '!=', embed_num_confs)
        return []
    
    conformers = []
    for i in range(n_confs):
        data_conf = copy.deepcopy(data)
        seed_mol = copy.deepcopy(mol)
        [seed_mol.RemoveConformer(j) for j in range(n_confs) if j != i]

        data_conf.pos = torch.from_numpy(seed_mol.GetConformers()[0].GetPositions()).float()
        data_conf.seed_mol = copy.deepcopy(seed_mol)

        conformers.append(data_conf)
    if mol.GetNumConformers() > 1:
        [mol.RemoveConformer(j) for j in range(n_confs) if j != 0]
    return conformers

def embed_func(mol, numConfs):
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=5)
    return mol

def to_np_array(item):
    if isinstance(item, np.ndarray):
        return item
    elif isinstance(item, list):
        return np.array(item)
    else:
        raise ValueError("The input should be either a list or a numpy array.")



def modify_conformer(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        #print(edge_index)
        #print(idx_edge, u, v)
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v] # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos

def perturb_seeds(data, pdb=None):
    for i, data_conf in enumerate(data):
        torsion_updates = np.random.uniform(low=-np.pi,high=np.pi, size=data_conf.edge_mask.sum())
        data_conf.pos = modify_conformer(data_conf.pos, data_conf.edge_index.T[data_conf.edge_mask],
                                         data_conf.mask_rotate, torsion_updates)
        data_conf.total_perturb = torsion_updates
    return data


class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        for i, d in enumerate(data_list):
            d.idx = i
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def apply_torsion_and_update_pos(self, data, torsion_updates):
        pos_new, torsion_updates = perturb_batch(data, torsion_updates, split=True, return_updates=True)
        for i, idx in enumerate(data.idx):
            try:
                self.data[idx].total_perturb += torsion_updates[i]
            except:
                pass
            self.data[idx].pos = pos_new[i]

def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    """
    if type(data) is Data:
        return modify_conformer(data.pos, 
            data.edge_index.T[data.edge_mask], 
            data.mask_rotate, torsion_updates)
    """
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new

def pyg_to_mol(mol, data, mmff=False, rmsd=True, copy=True):
    if not mol.GetNumConformers():
        conformer = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conformer)
    coords = data.pos
    if type(coords) is not np.ndarray:
        coords = coords.double().numpy()
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        except Exception as e:
            pass
    try:
        if rmsd:
            mol.rmsd = AllChem.GetBestRMS(
                Chem.RemoveHs(data.seed_mol),
                Chem.RemoveHs(mol)
            )
        mol.total_perturb = data.total_perturb
    except:
        pass
    mol.n_rotable_bonds = data.edge_mask.sum()
    if not copy: return mol
    return deepcopy(mol)

def populate_likelihood(mol, data, water=False, xtb=None):
    try:
        mol.dlogp = data.dlogp
    except:
        mol.dlogp = 0
    mol.inertia_tensor = inertia_tensor(data.pos)
    mol.log_det_jac = log_det_jac(data)
    mol.euclidean_dlogp = mol.dlogp - 0.5 * np.log(np.abs(np.linalg.det(mol.inertia_tensor))) - mol.log_det_jac
    mol.mmff_energy = mmff_energy(mol)
    if not xtb: return
    res = xtb_energy(mol, dipole=True, path_xtb=xtb)
    if res:
        mol.xtb_energy, mol.xtb_dipole, mol.xtb_gap, mol.xtb_runtime = res['energy'], res['dipole'], res['gap'], res['runtime']
    else:
        mol.xtb_energy = None
    if water:
        mol.xtb_energy_water = xtb_energy(mol, water=True, path_xtb=xtb)['energy']

def inertia_tensor(pos):  # n, 3
    if type(pos) != np.ndarray:
        pos = pos.numpy()
    pos = pos - pos.mean(0, keepdims=True)
    n = pos.shape[0]
    I = (pos ** 2).sum() * np.eye(3) - (pos.reshape(n, 1, 3) * pos.reshape(n, 3, 1)).sum(0)
    return I

def dx_dtau(pos, edge, mask):
    u, v = pos[edge]
    bond = u - v
    bond = bond / np.linalg.norm(bond)
    u_side, v_side = pos[~mask] - u, pos[mask] - u
    u_side, v_side = np.cross(u_side, bond), np.cross(v_side, bond)
    return u_side, v_side


def log_det_jac(data):
    pos = data.pos
    if type(data.pos) != np.ndarray:
        pos = pos.numpy()

    pos = pos - pos.mean(0, keepdims=True)
    I = inertia_tensor(pos)
    jac = []
    for edge, mask in zip(data.edge_index.T[data.edge_mask], data.mask_rotate):
        dx_u, dx_v = dx_dtau(pos, edge, mask)
        dx = np.zeros_like(pos)
        dx[~mask] = dx_u
        dx = dx - dx.mean(0, keepdims=True)
        L = np.cross(pos, dx).sum(0)
        omega = np.linalg.inv(I) @ L
        dx = dx - np.cross(omega, pos)
        jac.append(dx.flatten())
    jac = np.array(jac)
    _, D, _ = np.linalg.svd(jac)
    return np.sum(np.log(D))

def mmff_energy(mol):
    energy = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')).CalcEnergy()
    return energy

def collate_batch(batch_data):
    max_torsional_edge_num = 128
    atom_num = torch.tensor(0) 
    torsion_num = torch.tensor(0)
    ptr = [0]
    for data_idx, data in enumerate(batch_data): 

        if data_idx == 0:
            x = data.x
            edge_index = data.edge_index   # --> update by atom_num
            edge_attr = data.edge_attr
            z = data.z
            
            #np array list append
            mask_edges = [data.mask_edges]              
            mask_rotate = [data.mask_rotate]

            # other to list
            name = [data.name]
            seed_mol = [data.seed_mol]
            total_perturb = [data.total_perturb]
            idx = [data.idx] # --> change to tensor @ final

            # other tensor data
            edge_mask = data.edge_mask
            pos = data.pos

            ### RNN part
            # need add torsional number            
            torsion_rnn_index = F.pad(torch.arange(len(data.rotatable_bonds)), (0,max_torsional_edge_num - len(data.rotatable_bonds)), "constant", 0).unsqueeze(0)
            seq_mask = torch.zeros_like(torsion_rnn_index).bool()
            seq_mask[:,:len(data.rotatable_bonds)] = 1
            rnn_length = [len(data.rotatable_bonds)]
            sort_indice = torch.tensor(data.sorted_rotatable_indice)
            recover_indice = torch.argsort(sort_indice)

            #batch
            batch = torch.ones(data.x.shape[0]) * data_idx
        
        if data_idx >= 1:
            x = concat_new_data(x, data.x)
            edge_index = concat_new_data(edge_index, data.edge_index, dim=1, add_num = atom_num)  # --> update by atom_num
            edge_attr = concat_new_data(edge_attr, data.edge_attr)
            z = concat_new_data(z, data.z)
            
            #np array list append
            mask_edges.append(data.mask_edges)           
            mask_rotate.append(data.mask_rotate)

            # other to list
            name.append(data.name)
            seed_mol.append(data.seed_mol)
            total_perturb.append(data.total_perturb)
            idx.append(data.idx) # --> change to tensor @ final

            # other tensor data
            edge_mask = concat_new_data(edge_mask, data.edge_mask)
            pos = concat_new_data(pos, data.pos)

            ### RNN part
            new_sort_indice = torch.tensor(data.sorted_rotatable_indice)
            new_recover_indice = torch.argsort(new_sort_indice)

            sort_indice = concat_new_data(sort_indice, new_sort_indice , dim=0, add_num=torsion_num)
            recover_indice = concat_new_data(recover_indice, new_recover_indice, dim=0, add_num=torsion_num)

            #####
            new_torsion_rnn_index = F.pad(torch.arange(len(data.rotatable_bonds)), (0,max_torsional_edge_num - len(data.rotatable_bonds)), "constant", 0).unsqueeze(0)
            new_seq_mask = torch.zeros_like(new_torsion_rnn_index).bool()
            new_seq_mask[:,:len(data.rotatable_bonds)] = 1

            torsion_rnn_index = concat_new_data(torsion_rnn_index, new_torsion_rnn_index, dim=0, add_num=torsion_num)
            seq_mask = concat_new_data(seq_mask, new_seq_mask, dim=0, add_num=0)

            rnn_length.append(len(data.rotatable_bonds))

            #batch     
            new_batch = torch.ones(data.x.shape[0]) * data_idx
            #new_batch_torsion  = torch.ones(len(data.edge_rotate)) * idx
            batch = concat_new_data(batch, new_batch, dim=0, add_num=0)
            #batch_torsion = concat_new_data(batch_torsion, new_batch_torsion, dim=0, add_num=0)      
        atom_num += data.x.shape[0]
        ptr.append(deepcopy(atom_num))
        torsion_num += len(data.rotatable_bonds)

    
    ptr = torch.tensor(ptr)
    rnn_length = torch.tensor(rnn_length)
    data = Data(x= x,
                edge_index= edge_index,
                edge_attr= edge_attr,
                z= z,
                # list part
                mask_edges= mask_edges,
                mask_rotate= mask_rotate,
                pos= pos,
                seed_mol= seed_mol,
                name= name,
                total_perturb = total_perturb,
                # other tensor
                edge_mask= edge_mask,
                idx = torch.tensor(idx),
                ### RNN part
                torsion_rnn_index = torsion_rnn_index, 
                seq_mask = seq_mask,
                rnn_length = rnn_length,
                ptr = ptr,
                sort_indice  = sort_indice  ,
                recover_indice = recover_indice,
                batch = batch,
                )
    return data

def concat_new_data(data, new_data, dim=0, add_num=0):
    if add_num != 0:
        new_data = torch.add(new_data, add_num)
    data = torch.cat((data, new_data), dim=dim)
    return data

def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]