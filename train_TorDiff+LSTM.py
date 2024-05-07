from tqdm import tqdm
import torch
from copy import deepcopy

from utils.parsing import parse_train_args
from utils.noise_transform import TorsionNoiseTransform
from utils.utils import *
import model.torus as torus

args = parse_train_args()
print(args)
model = get_model(args)

model_name = 'ResLSTM'


best_epoch = 0
min_val_loss = 9e9 
best_model_name = "./weighted_model/" + model_name + "_BEST.pth"
final_model_name = "./weighted_model/" + model_name + "_FINAL.pth"

transform = TorsionNoiseTransform(sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                    boltzmann_weight=args.boltzmann_weight)
conf_mode_type = 'Perturb'
train_loader, val_loader = get_dataloader(conf_mode_type, args, transform)


model.to(args.device)
#print(model)
optimizer, scheduler = get_optimizer_and_scheduler(args, model)
print(model)
print(optimizer)
print(scheduler)

for epoch_idx in range(args.n_epochs):
    model.train()
    loss_tot = []
    base_tot = []
    for batch_idx, batch_data in enumerate(tqdm(train_loader)):  
        optimizer.zero_grad() 
        
        data = batch_data.to(args.device)
        data = model(data)

        pred = data.edge_pred
        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score, device=pred.device)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm, device=pred.device)
        loss = ((score - pred) ** 2 / score_norm).mean()
        loss.backward()
        optimizer.step()
        loss_tot.append(loss.item()) 
        base_tot.append((score ** 2 / score_norm).mean().item()) 

    loss_avg = sum(loss_tot) / len(loss_tot)
    base_avg = sum(base_tot) / len(base_tot)
    
    with torch.no_grad():
        model.eval()
        loss_tot = []
        base_tot = []
        for batch_data in tqdm(val_loader):  
            data = batch_data.to(args.device)
            data = model(data)

            pred = data.edge_pred
            score = torus.score(
                data.edge_rotate.cpu().numpy(),
                data.edge_sigma.cpu().numpy())
            score = torch.tensor(score, device=pred.device)
            score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
            score_norm = torch.tensor(score_norm, device=pred.device)
            loss = ((score - pred) ** 2 / score_norm).mean()
            loss_tot.append(loss.item()) 
            base_tot.append((score ** 2 / score_norm).mean().item()) 

        val_loss_avg = sum(loss_tot) / len(loss_tot)
        val_base_avg = sum(base_tot) / len(base_tot)

        lr = get_lr(optimizer)
        scheduler.step(loss)
        print(f'epoch {epoch_idx}, train_loss {loss_avg:.4f}, base {base_avg:.4f}') 
        print(f'lr {lr:.6f}, val_loss {val_loss_avg:.4f}, base {val_base_avg:.4f}') 


    if epoch_idx >= 2:
        if val_loss_avg <  min_val_loss:
            min_val_loss = val_loss_avg
            best_epoch = deepcopy(epoch_idx)
            torch.save(model.state_dict(), best_model_name )
        torch.save(model.state_dict(), final_model_name)

print(f'The best val loss is in epoch {best_epoch}, the loss is {min_val_loss:.4f}')
print(f'final model name {final_model_name}')
print(f'best model name {best_model_name}')
torch.save(model.state_dict(), final_model_name)
 