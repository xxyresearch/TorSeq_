from torch_geometric.transforms import BaseTransform
import random
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np


def modify_conformer(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v] # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos

class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi, boltzmann_weight=False):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.boltzmann_weight = boltzmann_weight

    def __call__(self, data):
        # select conformer
        if self.boltzmann_weight:
            data.pos = random.choices(data.pos[0], data.weights, k=1)[0]
        else:
            data.pos = random.choice(data.pos[0])

        try:
            edge_mask, mask_rotate = data.edge_mask, data.mask_rotate
        except:
            edge_mask, mask_rotate = data.mask_edges, data.mask_rotate
            data.edge_mask = torch.tensor(data.mask_edges)

        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        data.node_sigma = sigma * torch.ones(data.num_nodes)

        torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
        data.pos = modify_conformer(data.pos, data.edge_index.T[edge_mask], mask_rotate, torsion_updates)
        data.edge_rotate = torch.tensor(torsion_updates)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')
