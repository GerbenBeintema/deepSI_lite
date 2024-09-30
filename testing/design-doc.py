import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import nonlinear_benchmarks
import numpy as np

import torch
train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
s = int(len(train_val)*0.99)
train, val = train_val[:s], train_val[s:]
nx = 10
na = nb = 5

#option 2:
from deepSI_lite.networks import MLP_res_net, RK4
from deepSI_lite.normalization import get_nu_ny_and_auto_norm
from deepSI_lite.models import SUBNET, SUBNET_CT

nu, ny, norm = get_nu_ny_and_auto_norm(train_val)
f = norm.f(MLP_res_net(input_size = [nx , nu], output_size= nx))
h = norm.h(MLP_res_net(input_size = [nx , nu], output_size = ny))
encoder = norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx))

#what about shared parameters? -> define hf function or own structure
model = SUBNET(nu, ny, norm=norm, nx=nx, nb=nb, na=na, f=f, h=h, encoder=encoder, feedthrough=True) 

# CT SUBNET
integrator = RK4()
f_CT = norm.f_CT(MLP_res_net(input_size = [nx , nu], output_size = nx), tau=norm.sampling_time*50)
model_CT = SUBNET_CT(nu, ny, norm=norm, nx=nx, nb=nb, na=na, f_CT=f_CT, h=h, encoder=encoder, integrator=integrator, feedthrough=True)

#simulation example DT
upast, ypast, ufuture, yfuture = model.create_arrays(val, T='sim')
yfuture_sim = []
x = model.encoder(upast, ypast)
for u in ufuture.T:
    y = model.h(x,u)
    yfuture_sim.append(y)
    x = model.f(x,u)
yfuture_sim = torch.stack(yfuture_sim, dim=1) #has a batch dimension

#simulation example CT
yfuture_sim = []
upast, ypast, ufuture, sampling_time, yfuture = model_CT.create_arrays(val, T='sim') #This should return torch arrays
x = model_CT.encoder(upast, ypast)
for u in ufuture.T:
    y = model_CT.h(x,u)
    yfuture_sim.append(y)
    x = model_CT.integrator(model.f, x, u, sampling_time)
yfuture_sim = torch.stack(yfuture_sim, dim=1) #has a batch dimension

#alternative:
upast, ypast, ufuture, yfuture = model.create_arrays(val,T='sim')
yfuture_sim = model(upast, ypast, ufuture, yfuture)[0].detach().numpy()
#alternative CT:
upast, ypast, ufuture, sampling_time, yfuture = model_CT.create_arrays(val,T='sim')
yfuture_sim = model_CT(upast, ypast, ufuture, sampling_time, yfuture)[0].detach().numpy()

#or
val_sim = model.simulate(val)
val_sim = model_CT.simulate(val)

# #non-batched versions of functions

xnext = model.f_unbached(x=torch.randn((nx,)),u=torch.randn(tuple()))
ypred = model.h_unbached(x=torch.randn((nx,)),u=torch.randn(tuple())) if model.feedthrough else model.h_unbached(x=torch.randn((nx,)))

# model.f_unbached(x,u) #should be excluded from 
# model.h_unbached(x,u) #
# model.encoder_unbached(upast, ypast) #

# model_CT.f_CT_unbached(x,u) #should be excluded from 
# model_CT.integrator_unbached(x,u,sampling_time)
# model_CT.h_unbached(x,u) #
# model_CT.encoder_unbached(upast, ypast) #

from deepSI_lite.fitting import fit
train_dict = fit(model, train, val, n_its=1000, T=50, batch_size=128, stride=1, val_freq=100)

