
import numpy as np
from deepSI_lite.SUBNET import SUBNET, SUBNET_CT
import torch


def data_batcher(*arrays, batch_size=256, seed=0):
    rng = np.random.default_rng(seed=seed)
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = rng.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start, end = start + batch_size, end + batch_size


#given optimization
#minimal!
def compute_NMS(*A) -> torch.Tensor:
    model, *xarrays, yarray = A
    yout = model(*xarrays)
    return torch.mean((yout-yarray)**2/model.norm.ystd**2)

import cloudpickle, os
from secrets import token_urlsafe
from copy import deepcopy
from tqdm.auto import tqdm
def fit(model: SUBNET | SUBNET_CT, train, val, n_its, T=50, batch_size=256, stride=1, val_freq=500, optimizer=None, loss_fun=compute_NMS):
    code = token_urlsafe(4).replace('_','0').replace('-','a')
    save_filename = os.path.join(get_checkpoint_dir(), f'{model.__class__.__name__}-{code}.pth')
    optimizer = torch.optim.Adam(model.parameters()) if optimizer==None else optimizer
    arrays = model.create_arrays(train, T=T, stride=stride)
    arrays_val = model.create_arrays(val, T='sim')
    itter = data_batcher(*arrays, batch_size=batch_size)
    best_val, best_model, best_optimizer, loss_acc = float('inf'), deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), float('nan')
    NRMS_val, NRMS_train = [], []
    try:
        for it_count, batch in zip(tqdm(range(n_its)), itter):
            if it_count%val_freq==0:
                NRMS_val.append((loss_fun(model, *arrays_val)).detach().numpy()**0.5)
                NRMS_train.append((loss_acc/val_freq)**0.5)
                if NRMS_val[-1]<=best_val:
                    best_val, best_model, best_optimizer = NRMS_val[-1], deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
                cloudpickle.dump({'NRMS_val':np.array(NRMS_val),'last_model':deepcopy(model.state_dict()), 'best_model':best_model, 'best_optimizer':best_optimizer, \
                                  'NRMS_train':np.array(NRMS_train),'last_optimizer':deepcopy(optimizer.state_dict())}, open(save_filename,'wb'))
                print(f'it {it_count:7,} NRMS loss {NRMS_train[-1]:.4f} NRMS val {NRMS_val[-1]:.4f}{"!!" if NRMS_val[-1]==best_val else "  "}')
                loss_acc = 0.
            loss = loss_fun(model, *batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_acc += loss.item()
    except KeyboardInterrupt:
        print('Stopping early due to KeyboardInterrupt')
    model.load_state_dict(best_model)
    return cloudpickle.load(open(save_filename,'rb'))
    
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
