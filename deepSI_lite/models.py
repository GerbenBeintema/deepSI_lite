import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from torch import nn
import torch
from deepSI_lite.networks import MLP_res_net, rk4_integrator
from nonlinear_benchmarks import Input_output_data
import numpy as np
from deepSI_lite.normalization import Norm
from warnings import warn

def past_future_arrays(data : Input_output_data | list, na, nb, T, stride=1, add_sampling_time=False):
    if T=='sim':
        if isinstance(data, (tuple,list)):
            assert all(len(data[0])==len(d) for d in data), "if T='sim' than all given datasets need to have the same lenght (you should create the arrays in for loop instead)"
            T = len(data[0]) - max(na, nb)
        else:
            T = len(data) - max(na, nb)
    
    if isinstance(data, (tuple,list)):
        u, y = np.concatenate([di.u for di in data], dtype=np.float32), np.concatenate([di.y for di in data], dtype=np.float32) #this always creates a copy
    else:
        u, y = data.u.astype(np.float32, copy=False), data.y.astype(np.float32, copy=False)

    def window(x,window_shape=T): 
        x = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_shape,axis=0, writeable=True)
        s = (0,len(x.shape)-1) + tuple(range(1,len(x.shape)-1))
        return x.transpose(s)

    npast = max(na, nb)
    ufuture = window(u[npast:len(u)], window_shape=T)
    yfuture = window(y[npast:len(y)], window_shape=T)
    upast = window(u[npast-nb:len(u)-T], window_shape=nb)
    ypast = window(y[npast-na:len(y)-T], window_shape=na)

    if isinstance(data, (tuple,list)):
        acc_L, ids = 0, []
        for d in data:
            assert len(d.u)>=npast+T, f'some dataset was shorter than the length required by {max(na,nb)+T=} {len(d.u)=}'
            ids.append(np.arange(0,len(d.u)-npast-T+1, stride) + acc_L) #only add ids which are valid for training (no overlap between the different datasets)
            acc_L += len(d.u)
        ids = np.concatenate(ids)
    else:
        ids = np.arange(0, len(data)-npast-T+1, stride)

    s = torch.as_tensor
    if not add_sampling_time:
        return (s(upast), s(ypast), s(ufuture), s(yfuture)), ids #this could return all the valid indicies
    else:
        if isinstance(data, (tuple,list)):
            sampling_time = torch.cat([torch.as_tensor(d.sampling_time,dtype=torch.float32)*torch.ones(len(d)) for d in data])[:len(upast)]
        else:
            sampling_time = torch.as_tensor(data.sampling_time,dtype=torch.float32)*torch.ones(len(upast))
        return (s(upast), s(ypast), s(ufuture), sampling_time, s(yfuture)), ids

def validate_SUBNET_structure(model):
    nx, nu, ny, na, nb = model.nx, model.nu, model.ny, model.na, model.nb
    v = lambda *size: torch.randn(size)
    xtest = v(1,nx)
    utest = v(1) if nu=='scalar' else v(1,nu)
    upast_test =  v(1, nb) if nu=='scalar' else v(1, nb, nu)
    ypast_test = v(1, na) if ny=='scalar' else v(1, na, ny)

    with torch.no_grad():
        if isinstance(model, (SUBNET, SUBNET_CT)):
            f = model.f if isinstance(model, SUBNET) else model.f_CT
            xnext_test = f(xtest, utest)
            assert xnext_test.shape==(1,nx), f'f returned the incorrect shape it should be f(x, u).shape==(nbatch=1, nx) but got {xnext_test.shape}'
            x_encoded = model.encoder(upast_test, ypast_test)
            assert x_encoded.shape==(1,nx), f'encoder returned the incorrect shape it should be model.encoder(upast, ypast).shape==(nbatch=1, nx) but got {x_encoded.shape}'
            y_pred = model.h(xtest, utest) if model.feedthrough else model.h(xtest)
            assert (y_pred.shape==(1,)) if ny=='scalar' else (y_pred.shape==(1,ny)), f'h returned the incorrect shape it should be model.h(x{", u" if model.feedthrough else ""}).shape==(nbatch=1{"" if ny=="scalar" else ", ny"}) but got {y_pred.shape}'
            if isinstance(model, SUBNET_CT):
                xnext_test = model.integrator(model.f_CT, xtest, utest, torch.ones((1,)))
                assert xnext_test.shape==(1,nx), f'integrator returned the incorrect shape it should be model.integrator(model.f_CT, x, u, Ts).shape==(nbatch=1, nx) but got {xnext_test.shape}'
        else:
            raise NotImplementedError(f'model validation of type {model} cannot be validated yet')

class SUBNET(nn.Module):
    def __init__(self, nu, ny, norm : Norm, nx=10, nb=20, na=20, \
                 f=None, h=None, encoder=None, feedthrough=False, validate=True) -> None:
        super().__init__()
        self.nu, self.ny, self.norm, self.nx, self.nb, self.na, self.feedthrough = nu, ny, norm, nx, nb, na, feedthrough
        self.f = f if f is not None else norm.f(MLP_res_net(input_size = [nx , nu], output_size = nx))
        self.h = h if h is not None else norm.h(MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny))
        self.encoder = encoder if encoder is not None else norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx))
        if validate:
            validate_SUBNET_structure(self)

    def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
        return past_future_arrays(data, self.na, self.nb, T=T, stride=stride)

    def forward_simple(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        #is a lot simplier but also about 50% slower
        yfuture_sim = []
        x = self.encoder(upast, ypast)
        for u in ufuture.swapaxes(0,1):
            y = self.h(x,u) if self.feedthrough else self.h(x)
            yfuture_sim.append(y)
            x = self.f(x,u)
        return torch.stack(yfuture_sim, dim=1)

    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        B, T = ufuture.shape[:2]
        x = self.encoder(upast, ypast)
        xfuture = []
        for u in ufuture.swapaxes(0,1): #unroll over time dim
            xfuture.append(x)
            x = self.f(x,u)
        xfuture = torch.stack(xfuture,dim=1) #has shape (Nbatch, Ntime=T, nx)

        #compute output at all the future time indecies at the same time by combining the time and batch dim.
        fl = lambda ar: torch.flatten(ar, start_dim=0, end_dim=1) #conbine batch dim and time dim (Nbatch, Ntime, ...) -> (Nbatch*Ntim, ...)
        yfuture_sim_flat = self.h(fl(xfuture), fl(ufuture)) if self.feedthrough else self.h(fl(xfuture))
        return torch.unflatten(yfuture_sim_flat, dim=0, sizes=(B,T)) #(Nbatch*T, ...) -> (Nbatch, T, ...)

    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        if data.sampling_time!=self.norm.sampling_time:
            warn('It seems that the model is being simulated at a different sampling time as it was trained on.')
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=False)[0])[0].detach().numpy()
        return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

    def f_unbached(self, x, u):
        return self.f(x[None],u[None])[0]
    def h_unbached(self, x, u=None):
        return self.h(x[None], u[None])[0] if self.feedthrough else self.h(x[None])[0]
    def encoder_unbached(self, upast, ypast):
        return self.encoder(upast[None],ypast[None])[0]


class SUBNET_CT(nn.Module):
    #both norm, base_sampling_time have a sample time 
    def __init__(self, nu, ny, norm:Norm, nx=10, nb=20, na=20, f_CT=None, h=None, encoder=None, integrator=None, feedthrough=False, validate=True) -> None:
        super().__init__()
        self.nu, self.ny, self.norm, self.nx, self.nb, self.na, self.feedthrough = nu, ny, norm, nx, nb, na, feedthrough
        self.f_CT = f_CT if f_CT is not None else norm.f_CT(MLP_res_net(input_size = [nx , nu], output_size = nx), tau=norm.sampling_time*50)
        self.h = h if h is not None else norm.h(MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny))
        self.encoder = encoder if encoder is not None else norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx))
        self.integrator = integrator if integrator is not None else rk4_integrator
        if validate:
            validate_SUBNET_structure(self)

    def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
        return past_future_arrays(data, self.na, self.nb, T=T, stride=stride, add_sampling_time=True)

    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, sampling_time : float | torch.Tensor, yfuture: torch.Tensor=None):
        B, T = ufuture.shape[:2]
        x = self.encoder(upast, ypast)
        xfuture = []
        for u in ufuture.swapaxes(0,1):
            xfuture.append(x)
            x = self.integrator(self.f_CT, x, u, sampling_time)
        xfuture = torch.stack(xfuture,dim=1) #has shape (Nbatch, Ntime=T, nx)

        #compute output at all the future time indecies at the same time by combining the time and batch dim.
        fl = lambda ar: torch.flatten(ar, start_dim=0, end_dim=1) #conbine batch dim and time dim 
        yfuture_sim_flat = self.h(fl(xfuture), fl(ufuture)) if self.feedthrough else self.h(fl(xfuture))
        return torch.unflatten(yfuture_sim_flat, dim=0, sizes=(B,T)) #(Nbatch*T) -> (Nbatch, T)
    
    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        if data.sampling_time!=self.norm.sampling_time:
            warn('It seems that the model is being simulated at a different sampling time as it was trained on. The encoder currently assumes that the sampling_time is kept constant')
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=True)[0])[0].detach().numpy()
        return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

    def f_CT_unbached(self, x, u):
        return self.f_CT(x[None],u[None])[0]
    def integrator_unbached(self, f_CT, x, u, sampling_time):
        return self.integrator(f_CT, x[None], u[None], sampling_time[None])[0]
    def h_unbached(self, x, u=None):
        return self.h(x[None], u[None])[0] if self.feedthrough else self.h(x[None])[0]
    def encoder_unbached(self, upast, ypast):
        return self.encoder(upast[None],ypast[None])[0]


class Custom_SUBNET(nn.Module):
    def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
        return past_future_arrays(data, self.na, self.nb, T=T, stride=stride, add_sampling_time=False)

    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=False)[0])[0].detach().numpy()
        return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

class Custom_SUBNET_CT(nn.Module):
    def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
        return past_future_arrays(data, self.na, self.nb, T=T, stride=stride, add_sampling_time=True)

    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=True)[0])[0].detach().numpy()
        return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

def validate_custom_SUBNET_structure(model):
    nu, ny, na, nb = model.nu, model.ny, model.na, model.nb
    for batch_size in [1,2]:
        T = 10
        v = lambda *size: torch.randn(size)
        upast_test =  v(batch_size, nb) if nu=='scalar' else v(batch_size, nb, nu)
        ypast_test = v(batch_size, na) if ny=='scalar' else v(batch_size, na, ny)
        ufuture_test = v(batch_size, T) if nu=='scalar' else v(batch_size, T, nu)

        with torch.no_grad():
            if isinstance(model, Custom_SUBNET):
                yfuture_pred = model(upast_test, ypast_test, ufuture_test)
            else:
                yfuture_pred = model(upast_test, ypast_test, ufuture_test, v(batch_size))
            assert yfuture_pred.shape==((batch_size,T) if ny=='scalar' else (batch_size,T,ny))

#########################
####### SUBNET_LPV ######
#########################

from deepSI_lite.networks import Bilinear
class SUBNET_LPV(Custom_SUBNET):
    def __init__(self, nu, ny, norm:Norm, nx, n_schedual, na, nb, scheduling_net=None, A=None, B=None, C=None, D=None, encoder=None, feedthrough=True):
        if n_schedual.any(10*abs(norm.y0)>norm.ystd) or n_schedual.any(10*abs(norm.umean)>norm.ustd):
            from warnings import warn
            warn('SUBNET_LPV assumes that the data is approximatly zero mean. Not doing so can lead to unintended behaviour.')
        assert isinstance(nu, int) and isinstance(ny, int) and isinstance(n_schedual, int) and feedthrough, 'SUBNET_LPV requires the input, output schedualing parameter to be vectors and feedthrough to be present'
        super().__init__()
        self.nu, self.ny, self.norm, self.nx, self.n_schedual, self.na, self.nb, self.feedthrough = nu, ny, norm, nx, n_schedual, na, nb, feedthrough
        self.A = A if A is not None else Bilinear(n_in=nx, n_out=nx, n_schedual=n_schedual)
        self.B = B if B is not None else Bilinear(n_in=nu, n_out=nx, n_schedual=n_schedual, std_input=norm.ustd)
        self.C = C if C is not None else Bilinear(n_in=nx, n_out=ny, n_schedual=n_schedual, std_output=norm.ystd)
        self.D = D if D is not None else Bilinear(n_in=nu, n_out=ny, n_schedual=n_schedual, std_output=norm.ystd, std_input=norm.ustd)
        self.encoder = encoder if encoder is not None else norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx))
        self.scheduling_net = scheduling_net if scheduling_net is not None else norm.f(MLP_res_net(input_size = [nx , nu], output_size = n_schedual))
        validate_custom_SUBNET_structure(self) #does checks if forward is working as intended
    
    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        mv = lambda A, x: torch.bmm(A, x[:, :, None])[:,:,0] #batched matrix vector multiply
        yfuture_sim = []
        x = self.encoder(upast, ypast)
        for u in ufuture.swapaxes(0,1): #iterate over time
            p = self.scheduling_net(x, u)
            A, B, C, D = self.A(p), self.B(p), self.C(p), self.D(p)
            y = mv(C, x) + mv(D, u)
            x = mv(A, x) + mv(B, u)
            yfuture_sim.append(y)
        return torch.stack(yfuture_sim, dim=1)


##########################
####### CNN_SUBNET #######
##########################

class CNN_SUBNET(SUBNET):
    def __init__(self, nu, ny, norm, nx, nb, na):
        from deepSI_lite.networks import CNN_vec_to_image, CNN_encoder, MLP_res_net
        h = norm.h(CNN_vec_to_image(nx, ny=ny))
        f = norm.f(MLP_res_net(input_size=[nx, nu], output_size=nx))
        encoder = norm.encoder(CNN_encoder(nb, nu, na, ny, nx))
        super().__init__(nu, ny, norm, nx, nb, na, f, h, encoder, validate=False)

###########################
####### pHNN_SUBNET #######
###########################

from deepSI_lite.networks import Ham_converter, ELU_lower_bound, Skew_sym_converter, Sym_pos_semidef_converter, Matrix_converter
class pHNN_SUBNET(Custom_SUBNET_CT):
    def __init__(self, nu, ny, norm, nx, na, nb, Hnet=None, Jnet=None, \
                 Rnet=None, Gnet=None, encoder=None, integrator=None, tau=None):
        super().__init__()
        assert nu==ny
        self.nu, self.ny, self.norm, self.nx, self.na, self.nb = nu, ny, norm, nx, na, nb
        self.Hnet = Ham_converter(ELU_lower_bound(MLP_res_net(nx, 'scalar'))) if Hnet is None else Hnet
        self.Jnet = Skew_sym_converter(MLP_res_net(nx, nx*nx)) if Jnet is None else Jnet
        self.Rnet = Sym_pos_semidef_converter(MLP_res_net(nx, nx*nx)) if Rnet is None else Rnet
        nu_val = 1 if nu=='scalar' else nu
        self.Gnet = Matrix_converter(MLP_res_net(nx, nx*nu_val), nrows=nx, ncols=nu_val) if Gnet is None else Gnet
        self.integrator = rk4_integrator if integrator is None else integrator
        self.encoder = norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx)) if encoder is None else encoder
        self.norm = norm
        self.tau = norm.sampling_time*10 if tau is None else tau
        validate_custom_SUBNET_structure(self)
    
    def get_matricies(self, x):
        with torch.enable_grad():
            if x.requires_grad == False:
                x.requires_grad = True
            H = self.Hnet(x)
            Hsum = H.sum()
            dHdx = torch.autograd.grad(Hsum, x, create_graph=True)[0]

        J_x = self.Jnet(x)
        R_x = self.Rnet(x)
        G_x = self.Gnet(x)
        return J_x, R_x, G_x, dHdx, H

    def forward(self, upast, ypast, ufuture, sampling_time, yfuture=None):
        x = self.encoder(upast, ypast)
        ufuture = (ufuture.view(ufuture.shape[0],ufuture.shape[1],-1)-self.norm.umean)/self.norm.ustd #normalize inputs
        yfuture_sim = []
        for u in ufuture.swapaxes(0,1): #if using a 1-step euler this can be reduced further 
            J_x, R_x, G_x, dHdx, H = self.get_matricies(x) #this can be done outside of the loop for a speedup
            y_hat = torch.einsum('bij,bi->bj', G_x, dHdx) #bij,bi->bj  = A^T @ dHdx
            yfuture_sim.append(y_hat)
            def f_CT(x, u):
                J_x, R_x, G_x, dHdx, H = self.get_matricies(x)
                Gu = torch.einsum('bij,bj->bi', G_x, u) # G_x (Nb, nx, nu) times u (Nb, nu) = (Nb, nx)
                return (torch.einsum('bij,bj->bi', J_x - R_x, dHdx) + Gu)/self.tau
    
            x = self.integrator(f_CT, x, u, sampling_time)
        
        yfuture_sim = torch.stack(yfuture_sim, dim=1)
        yfuture_sim = yfuture_sim[:,:,0] if self.ny=='scalar' else yfuture_sim
        return yfuture_sim*self.norm.ystd + self.norm.ymean
