import numpy as np 
import pandas as pd 
import torch
from time import time 
from tqdm import tqdm
from device import moveTo 
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import GridSearchCV
from importlib import import_module
from GW_classifiers.utils import load_module 


def train_model(name, X_train, y_train, config): 
    model = load_module(config["path"], name)
    if hasattr(model, "fit"): 
        if config["cv"] > 0: 
            model = GridSearchCV(model, param_grid = config["params"], cv = config["cv"], scoring = 'roc_auc')
        model.fit(X_train, y_train)
    else: 
        train_torch(model, X_train, y_train, config)

    return model 

    

def train_torch(model, X_train, y_train, config):
    #Convert to torch tensors  
    model_config = config[model]
    X = torch.Tensor(X_train, dtype = torch.float32)
    y = torch.Tensor(y_train, dtype = torch.long)
    n_train = mode_config["train_size"] * len(X) 
    n_test = len(X) - N_train 
    train_dataset, test_dataset = random_split(TensorDataset(X,y), [n_train, n_test])
        

    #Define Dataloader 
    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = config['shuffle'])
    test_dataloader = DataLoader(test_dataset)

    optimizer = importlib.import_module()
    loss_func = import_module.import_module(nn.Loss) 

    train_network(model, optimizer, loss_func, train_dataloader, test_dataloader, config['epochs'] )

    return model 



def train_network(model, optimizer, loss_func, train_dataloader, test_dataloader = None, epochs = 50, score_funcs = None,\
                  device = 'cpu', checkpoint_file = 'results/state_dict', save_every = 10): 
    """
    Trains a neural network model using the provided optimizer and loss function.
    
    Parameters:
    model: The neural network model to be trained.
    optimizer: The optimizer to use for training the model.
    loss_func: The loss function to compute the training loss.
    train_dataloader: DataLoader for the training dataset.
    test_dataloader: DataLoader for the test dataset (optional).
    epochs: Number of epochs to train the model.
    score_funcs: Dictionary of evaluation functions to compute scores during training.
    device: The device to run the training on (e.g., 'cpu' or 'cuda').
    checkpoint_file: File path to save the model checkpoints (optional).
    save_every: Frequency of saving checkpoints (in epochs).
    
    Returns:
    A DataFrame containing the training results, including loss and evaluation scores.
    """

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

    best_loss = np.inf 
    best_state_dict = None    
    for epoch in epochs: 
        model.train() 
        tota_train_time += run_epoch(model, loss_func, optimizer, train_dataloader, score_funcs, device, results) 
        
        results['total time'].append(total_train_time)
        results['epoch'].append('epoch')

        if test_dataloader is not None: 
            model.eval() 
            with torch.no_grad(): 
                run_epoch(model, loss_func, optimizer, test_dataloader, score_funcs, device, results,\
                          prefix = 'test', desc = 'test') 

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
    """
    Runs a single epoch of training or evaluation on the model using the provided data loader.   
    """
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
    
