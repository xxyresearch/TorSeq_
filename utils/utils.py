import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch_scatter import scatter
import ot
import numpy as np


from model.score_model import TensorProductScoreModel
from model.mpnn_model import GLSTM
from dataset.PerturbDataset import PerturbDataset
from dataset.OriginDataset import OriginDataset, OriginDataset_test

from utils.dataloader import collate_perturb_batch, collate_original_batch, collate_original_batch_test


def get_lr(optimizer):
    for p in optimizer.param_groups:
        return p["lr"]

def get_dataloader(conf_mode_type, args, transform=None):
    if conf_mode_type == 'Perturb':
        valdata = PerturbDataset(cmt = conf_mode_type, args=args,  mode = 'Val', split_num = 1, transform=transform)
        traindata_1 = PerturbDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 1, transform=transform)
        traindata_2 = PerturbDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 2, transform=transform)
        traindata_3 = PerturbDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 3, transform=transform)
        traindata_4 = PerturbDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 4, transform=transform)
        traindata_5 = PerturbDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 5, transform=transform)
        traindata = ConcatDataset([traindata_1, traindata_2, traindata_3, traindata_4, traindata_5])
        train_loader = DataLoader(dataset = traindata, batch_size= args.batch_size, shuffle = True, num_workers=2, collate_fn=collate_perturb_batch)
        val_loader = DataLoader(dataset = valdata, batch_size= args.batch_size, shuffle = True, num_workers=2, collate_fn=collate_perturb_batch)
    else:
        valdata = OriginDataset(cmt = conf_mode_type, args=args,  mode = 'Val', split_num = 1, transform=None)
        traindata_1 = OriginDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 1, transform=None)
        traindata_2 = OriginDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 2, transform=None)
        traindata_3 = OriginDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 3, transform=None)
        traindata_4 = OriginDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 4, transform=None)
        traindata_5 = OriginDataset(cmt = conf_mode_type, args=args, mode = 'Train', split_num = 5, transform=None)
        traindata = ConcatDataset([traindata_1, traindata_2, traindata_3, traindata_4, traindata_5])
        train_loader = DataLoader(dataset = traindata, batch_size= args.batch_size, shuffle = True, num_workers=4, collate_fn=collate_original_batch)
        val_loader = DataLoader(dataset = valdata, batch_size= args.batch_size, shuffle = True, num_workers=4, collate_fn=collate_original_batch)
    return train_loader, val_loader

def get_test_dataloader(conf_mode_type, args):
    testdata = OriginDataset_test(cmt = conf_mode_type, args=args)
    #print(len(testdata))
    #assert(1==2)
    test_loader = DataLoader(dataset = testdata, batch_size= 1, shuffle = False, num_workers=4, collate_fn=collate_original_batch_test)
    return test_loader

def get_model(args):
    # original method got about 0.707 loss in val
    if args.mpnn_conv == False:
        model = TensorProductScoreModel(in_node_features=args.in_node_features, in_edge_features=args.in_edge_features,
                                ns=args.ns, nv=args.nv, sigma_embed_dim=args.sigma_embed_dim,
                                sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                num_conv_layers=args.num_conv_layers,
                                max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim,
                                scale_by_sigma=args.scale_by_sigma,
                                use_second_order_repr=args.use_second_order_repr,
                                residual= not args.no_residual, batch_norm= not args.no_batch_norm, rnn= not args.no_use_lstm,
                                rnn_layers= args.lstm_layer_num)
    else:
        model = GLSTM(node_dim=args.in_node_features+1, edge_dim=args.in_edge_features, hidden_dim=args.hidden_dim,
                      lstm_layer_num = args.lstm_layer_num, gnn_layer_num = args.gnn_layer_num, latent_dim=args.latent_dim,
                      use_lstm= not args.no_use_lstm, use_motif_graph = args.use_motif_gcn,
                      use_local_feature= not args.no_local_feature)

    return  model

def get_optimizer_and_scheduler(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
        #                                                       patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def get_loss(pred, tgt, data):
    # pred: conf_num * torsion_num  
    # tgt: conf_num * torsion_num * 2
    pred = pred.unsqueeze(-1).repeat(1,1,6).unsqueeze(0)
    tgt = tgt.unsqueeze(1)
    diff = 1 - torch.cos(pred -tgt) 
    diff = diff.permute(2,0,1,3)
    diff = scatter(diff, data.batch_sym_dihedral, dim=0, reduce = 'sum')
    diff, _ = torch.min(diff, dim=-1)   
    diff = scatter(diff, data.batch_dihedral, dim=0, reduce = 'sum')
    # --> batch * conf_num * conf_num
    return diff

def ot_emd_loss(loss_matrix, mask, args, debug=False):
    true_conf_num = int(torch.sum(mask))
    H_1 = np.ones(true_conf_num) / true_conf_num 
    H_2 = np.ones(args.conf_num ) / args.conf_num 
    cost_mat_detach = loss_matrix.detach().cpu().numpy()
    cost_mat_i = cost_mat_detach[:true_conf_num]
    ot_mat = ot.emd(a=H_1, b=H_2, M=np.max(np.abs(cost_mat_i)) + cost_mat_i, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=args.device, requires_grad=False).float()
    loss = ot_mat_attached * loss_matrix[:true_conf_num]
    if debug==True:
        print(loss)
        print(ot_mat_attached)
        print(loss_matrix)
        assert(1==2)
    loss = torch.sum(loss)
    return loss, ot_mat

def get_loss_for_print(target_loss_matrix, conf_mask, mol_index, ot_emd): 
    true_conf_num = int(torch.sum(conf_mask[mol_index]))
    loss_sample = target_loss_matrix[:true_conf_num,:true_conf_num,mol_index].detach().cpu()
    loss_sample = torch.sum(loss_sample * ot_emd)
    return loss_sample