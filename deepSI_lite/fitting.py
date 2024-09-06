
import numpy as np
import torch

def data_batcher(*arrays, batch_size=256, seed=0, device=None, indices=None):
    rng = np.random.default_rng(seed=seed)
    if indices is None:
        indices = np.arange(arrays[0].shape[0])
    dataset_size = len(indices)
    assert all(array.shape[0] == arrays[0].shape[0] for array in arrays)
    assert batch_size <= dataset_size
    while True:
        perm = rng.permutation(indices)
        start, end = 0, batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays) #arrays are already torch arrays
            start, end = start + batch_size, end + batch_size

def compute_clamp_NMSE(*A, NRMS_clamp_level=0.1, min_steps=None) -> torch.Tensor:
    from math import ceil
    model, *xarrays, yarray = A
    yout = model(*xarrays)
    errs = torch.mean((yout-yarray)**2/model.norm.ystd**2,dim=0) #[error step 0, error step 1, error step 2, ..]
    if min_steps==None:
        min_steps = ceil(model.nx/(model.ny if isinstance(model.ny,int) else 1))
    return (torch.sum(errs[:min_steps]) + torch.sum(torch.clamp(errs[min_steps:],min=None, max=NRMS_clamp_level**2)))/len(errs)

def compute_NMSE(*A) -> torch.Tensor:
    model, *xarrays, yarray = A
    yout = model(*xarrays)
    return torch.mean((yout-yarray)**2/model.norm.ystd**2)

import cloudpickle, os
from secrets import token_urlsafe
from copy import deepcopy
from tqdm.auto import tqdm
from torch import nn, optim
from nonlinear_benchmarks import Input_output_data
import time

def fit(model: nn.Module, train:Input_output_data, val:Input_output_data, n_its:int, T:int=50, \
        batch_size:int=256, stride:int=1, val_freq:int=250, optimizer:optim.Optimizer=None, loss_fun=compute_NMSE):
    """
    Trains a PyTorch model (e.g. a SUBNET model).

    Args:
        model (nn.Module): The neural network model to be trained. This network needs to have a method of `.create_arrays(train, T, stride)` such that the training arrays can be created
        train (Input_output_data): Training data.
        val (Input_output_data): Validation data.
        n_its (int): Number of training iterations (i.e. batch updates).
        T (int, optional): Sequence length considerd in the loss (unroll lenght). Default is 50.
        batch_size (int, optional): Size of training batches. Default is 256.
        stride (int, optional): Stride for data batching. Default is 1.
        val_freq (int, optional): Validation frequency (in iterations). Default is 250.
        optimizer (optim.Optimizer, optional): Optimizer for training. Default is Adam.
        loss_fun (callable, optional): Loss function. Default is compute_NMSE.

    Returns:
        dict: Contains the best model and optimizer states, and training/validation loss histories.
    """
    code = token_urlsafe(4).replace('_','0').replace('-','a')
    save_filename = os.path.join(get_checkpoint_dir(), f'{model.__class__.__name__}-{code}.pth')
    optimizer = torch.optim.Adam(model.parameters()) if optimizer==None else optimizer
    arrays, indices = model.create_arrays(train, T=T, stride=stride) 
    print(f'Number of samples to train on = {len(indices)}')
    itter = data_batcher(*arrays, batch_size=batch_size, indices=indices)

    arrays_val, indices = model.create_arrays(val, T='sim')
    arrays_val = [array_val[indices] for array_val in arrays_val]

    best_val, best_model, best_optimizer, loss_acc = float('inf'), deepcopy(model), deepcopy(optimizer), float('nan')
    NRMS_val, NRMS_train, time_usage = [], [], 0. #initialize the train and val monitor
    try:
        for it_count, batch in zip(tqdm(range(n_its)), itter):
            if it_count%val_freq==0: ### Validation and printing loop ###
                NRMS_val.append((loss_fun(model, *arrays_val)).detach().numpy()**0.5)
                NRMS_train.append((loss_acc/val_freq)**0.5)
                if NRMS_val[-1]<=best_val:
                    best_val, best_model, best_optimizer = NRMS_val[-1], deepcopy(model), deepcopy(optimizer)
                cloudpickle.dump({'NRMS_val':np.array(NRMS_val),'last_model':model, 'best_model':best_model, \
                                  'best_optimizer':best_optimizer, 'NRMS_train': np.array(NRMS_train),\
                                  'last_optimizer':deepcopy(optimizer.state_dict()), 'samples/sec': (it_count*batch_size/time_usage if time_usage>0 else None)}, \
                                  open(save_filename,'wb'))
                print(f'it {it_count:7,} NRMS loss {NRMS_train[-1]:.5f} NRMS val {NRMS_val[-1]:.5f}{"!!" if NRMS_val[-1]==best_val else "  "} {(it_count*batch_size/time_usage if time_usage>0 else float("nan")):.2f} samps/sec')
                loss_acc = 0.
            ### Train Step ###
            start_t = time.time()
            loss = loss_fun(model, *batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_acc += loss.item()
            time_usage += time.time()-start_t
    except KeyboardInterrupt:
        print('Stopping early due to KeyboardInterrupt')
    model.load_state_dict(best_model.state_dict())
    return cloudpickle.load(open(save_filename,'rb'))

def fit_minimal_implementation(model: nn.Module, train: Input_output_data, val: Input_output_data, n_its: int, 
        T: int = 50, stride: int = 1, batch_size: int = 256, val_freq: int = 250, 
        optimizer: optim.Optimizer = None, loss_fun = compute_NMSE):

    optimizer = torch.optim.Adam(model.parameters()) if optimizer is None else optimizer
    arrays = model.create_arrays(train, T=T, stride=stride)  # Training data with T and stride
    arrays_val = model.create_arrays(val, T=T)  # Validation data
    itter = data_batcher(*arrays, batch_size=batch_size)  # Data batches

    best_val, best_model = float('inf'), deepcopy(model.state_dict())  # Track the best model
    
    for it_count, batch in zip(tqdm(range(n_its)), itter):
        if it_count % val_freq == 0:  # Validation step
            val_loss = loss_fun(model, *arrays_val).detach().numpy()**0.5

            if val_loss <= best_val:
                best_val, best_model = val_loss, deepcopy(model.state_dict())

            print(f'Iter {it_count:7,}, Val Loss: {val_loss:.5f}')
        
        # Training step
        loss = loss_fun(model, *batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.load_state_dict(best_model)  # Load the best model
    return best_model


def get_checkpoint_dir():
    '''A utility function which gets the checkpoint directory for each OS

    It creates a working directory called deepSI-checkpoints 
        in LOCALAPPDATA/deepSI-checkpoints/ for windows
        in ~/.deepSI-checkpoints/ for unix like
        in ~/Library/Application Support/deepSI-checkpoints/ for darwin

    Returns
    -------
    checkpoints_dir
    '''
    import os
    from sys import platform
    if platform == "darwin": #not tested but here it goes
        checkpoints_dir = os.path.expanduser('~/Library/Application Support/deepSI-checkpoints/')
    elif platform == "win32":
        checkpoints_dir = os.path.join(os.getenv('LOCALAPPDATA'),'deepSI-checkpoints/')
    else: #unix like, might be problematic for some weird operating systems.
        checkpoints_dir = os.path.expanduser('~/.deepSI-checkpoints/')#Path('~/.deepSI/')
    if os.path.isdir(checkpoints_dir) is False:
        os.mkdir(checkpoints_dir)
    return checkpoints_dir
