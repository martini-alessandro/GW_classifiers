import numpy as np 
import pandas as pd 
import torch
from time import time 
from tqdm import tqdm
from device import moveTo 


def train_network(model, optimizer, loss_func, train_dataloader, test_dataloader = None, epochs = 50, score_funcs = None,\
                  learning_rate = .01, device = 'cpu', checkpoint_file = None, save_every = 10): 
    to_track = ['epoch', 'total_time', 'train loss'] 
    if test_dataloader is not None: 
        to_track.append('test loss')
    for eval_score in score_funcs: 
        to_track.append(f'train {eval_score}')
        if test_dataloader is not None: 
            to_track.append(f'test {eval_score}')         
    
    total_train_time = 0
    results = {} 
    model.to(device)
    for item in to_track: 
        results[item] = [] 

    
    for epoch in epochs: 
        model.train() 
        time = run_epoch(model, loss_func, optimizer, train_dataloader, score_funcs, device, results) 
        
        results['total time'].append(time)
        results['epoch'].append('epoch')

        if test_dataloader is not None: 
            model.eval() 
            with torch.no_grad(): 
                run_epoch(model, loss_func, optimizer, test_dataloader, score_funcs, device, results, prefix = 'test', desc = 'test') 

            if checkpoint_file is not None: 
                if epoch % save_every == 0: 
                    state_dict = model.state_dict() 
                    torch.save({'epoch': epoch, 
                                'state_dict': state_dict,
                                'optimizer_state_dict': optimizer.state_dict(), 
                                'results': results}, checkpoint_file)
        return pd.DataFrame(results), checkpoint_file 

    return pd.DataFrame 

def run_epoch(model, loss_func, optimizer, data_loader, score_funcs, device, results, prefix = 'train', desc = 'Training'): 

    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()

    for inputs, labels in tqdm(data_loader, desc = desc, leave = False):
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

    y_hat = model(inputs)
    loss = loss_func(y_hat, labels)
    if model.training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    running_loss.append(loss.item())

    if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()

        y_true.extend(labels.tolist())
        y_pred.extend(y_hat.tolist())

    end = time.time()
    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis = 1)

    results[prefix + ' loss'].append(np.mean(running_loss))
    for keys, score_func in score_funcs.items():
        try:
            results[prefix + ' ' + keys].append(score_func(y_true, y_pred))
        except:
            results[prefix + ' ' + keys].append(float("NaN"))

    return end-start
    
