from tqdm import tqdm
import torch
import torch.nn as nn
from copy import deepcopy
from utils.parsing import parse_train_args
from utils.utils import *
import model.torus as torus
from math import pi

args = parse_train_args()
conf_mode_type = 'Original'
model = get_model(args)
model.to(args.device)
print(model)
optimizer, scheduler = get_optimizer_and_scheduler(args, model)
print(optimizer, scheduler)

model_name = args.model_name
best_epoch = 0
min_val_loss = 9e9 
best_model_name = "./weighted_model/" + model_name + "_BEST.pth"
final_model_name = "./weighted_model/" + model_name + "_FINAL.pth"

print(best_model_name )
print(args)

train_loader, val_loader = get_dataloader(conf_mode_type, args)


for epoch_idx in range(args.n_epochs):
    train_loss = []
    val_loss = []
    model.train()

    for batch_idx, batch_data in enumerate(tqdm(train_loader)): 
        atom_type = batch_data.x[:,35]
        optimizer.zero_grad() 
        data = batch_data.to(args.device)
        x = data.x
        x = x.unsqueeze(1).repeat(1, args.conf_num ,1)
        x = x.permute(1,0,2)
        
        latent = torch.rand([args.conf_num, x.size(1), args.latent_dim]).to(args.device)
        x_with_noise = torch.cat((x, latent), dim=-1)
        dihedral_rad_tar = data.tgt
        dihedral_rad_chain_tar = data.tgt_chain
        dihedral_rad_pred_matrix = torch.zeros((args.conf_num, dihedral_rad_tar.shape[1]), dtype = torch.float).to(args.device)
        dihedral_rad_chain_pred_matrix= torch.zeros((args.conf_num, dihedral_rad_chain_tar.shape[1]), dtype = torch.float).to(args.device)

        loss_rand = []
        for idx, x_feature in enumerate(x_with_noise):
            if not args.no_random_start:
                random_start = torch.rand(len(dihedral_rad_chain_pred_matrix[idx]),1).to(args.device) * 2 * pi - pi
            else:
                random_start = torch.zeros(len(dihedral_rad_chain_pred_matrix[idx]),1).to(args.device) 
            dihedral_rad, dihedral_chain_pred, r1, r2 = model(data, args, random_start, x_feature, idx)
            dihedral_rad_pred_matrix[idx] = torch.squeeze(dihedral_rad)
            dihedral_rad_chain_pred_matrix[idx] = torch.squeeze(dihedral_chain_pred)
            loss_rand.append(torch.mean(1-torch.cos(random_start-r1))/2 + torch.mean(1-torch.cos(random_start-r2))/2) 

        pred_loss = get_loss(dihedral_rad_pred_matrix, dihedral_rad_tar, data)
        chain_pred_loss = get_loss(dihedral_rad_chain_pred_matrix, dihedral_rad_chain_tar, data)
        diff = (pred_loss + chain_pred_loss)/2

        total_loss = []
        for diff_idx, diff_i in enumerate(diff):
            loss_item, ot_mat = ot_emd_loss(diff[diff_idx], data.conf_mask[diff_idx],  args=args) 
            total_loss.append(loss_item)

        loss_rand = torch.mean(torch.stack(loss_rand))
        loss  = torch.sum(torch.stack(total_loss))/(data.dihedral_indx.shape[0]) + loss_rand
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5, norm_type=2)
        optimizer.step()
        train_loss.append(loss.detach().cpu())

    with torch.no_grad():
        model.eval()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)): 
            data = batch_data.to(args.device)
            x = data.x
            x = x.unsqueeze(1).repeat(1, args.conf_num ,1)
            x = x.permute(1,0,2)
            
            latent = torch.rand([args.conf_num, x.size(1), args.latent_dim]).to(args.device)
            x_with_noise = torch.cat((x, latent), dim=-1)
            dihedral_rad_tar = data.tgt
            dihedral_rad_chain_tar = data.tgt_chain
            dihedral_rad_pred_matrix = torch.zeros((args.conf_num, dihedral_rad_tar.shape[1]), dtype = torch.float).to(args.device)
            dihedral_rad_chain_pred_matrix= torch.zeros((args.conf_num, dihedral_rad_chain_tar.shape[1]), dtype = torch.float).to(args.device)

            loss_rand = []
            for idx, x_feature in enumerate(x_with_noise):
                if not args.no_random_start:
                    random_start = torch.rand(len(dihedral_rad_chain_pred_matrix[idx]),1).to(args.device) * 2 * pi - pi
                else:
                    random_start = torch.zeros(len(dihedral_rad_chain_pred_matrix[idx]),1).to(args.device) 
                dihedral_rad, dihedral_chain_pred, r1, r2 = model(data, args, random_start, x_feature, idx)
                dihedral_rad_pred_matrix[idx] = torch.squeeze(dihedral_rad)
                dihedral_rad_chain_pred_matrix[idx] = torch.squeeze(dihedral_chain_pred)
                loss_rand.append(torch.mean(1-torch.cos(random_start-r1))/2 + torch.mean(1-torch.cos(random_start-r2))/2) 


            pred_loss = get_loss(dihedral_rad_pred_matrix, dihedral_rad_tar, data)
            chain_pred_loss = get_loss(dihedral_rad_chain_pred_matrix, dihedral_rad_chain_tar, data)
            diff = (pred_loss + chain_pred_loss)/2

            total_loss = []
            for diff_idx, diff_i in enumerate(diff):
                loss_item, ot_mat = ot_emd_loss(diff[diff_idx], data.conf_mask[diff_idx],  args=args) 
                total_loss.append(loss_item)

            loss_rand = torch.mean(torch.stack(loss_rand))
            loss  = torch.sum(torch.stack(total_loss))/(data.dihedral_indx.shape[0]) + loss_rand
            val_loss.append(loss.detach().cpu())

    
    scheduler.step(torch.mean(torch.tensor(val_loss)))
    lr = get_lr(optimizer)

    train_phi_loss = np.mean(train_loss)
    val_phi_loss = np.mean(val_loss)
    print('epoch,[%d/%d], lr_1= %.7f'%((epoch_idx+1), args.n_epochs, lr ))
    print('phi loss:        %.6f, val: %.6f' %(train_phi_loss, val_phi_loss))
    
    if epoch_idx >= 5:
        if val_phi_loss <  min_val_loss:
            min_val_loss = val_phi_loss
            best_epoch = deepcopy(epoch_idx)
            torch.save(model.state_dict(), best_model_name )
        torch.save(model.state_dict(), final_model_name)


print(f'The best val loss is in epoch {best_epoch}, the loss is {min_val_loss:.4f}')
print(f'final model name {final_model_name}')
print(f'best model name {best_model_name}')
torch.save(model.state_dict(), final_model_name)