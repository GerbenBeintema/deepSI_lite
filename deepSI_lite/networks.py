import torch
from torch.nn import Sequential
from torch import nn
class MLP_res_net(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers = 2, n_hidden_nodes = 64, \
                 activation=nn.Tanh, zero_bias=True):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.scalar_output = output_size=='scalar'
        #convert input shape:
        def to_num(s):
            if isinstance(s, int):
                return s
            if s=='scalar':
                return 1
            a = 1
            for si in s:
                a = a*(1 if si=='scalar' else si)
            return a
        if isinstance(input_size, list):
            input_size = sum(to_num(s) for s in input_size)
        
        output_size = 1 if self.scalar_output else output_size
        self.net_res = nn.Linear(input_size, output_size)

        seq = [nn.Linear(input_size,n_hidden_nodes),activation()]
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_hidden_nodes,n_hidden_nodes))
            seq.append(activation())
        seq.append(nn.Linear(n_hidden_nodes,output_size))
        self.net_nonlin = nn.Sequential(*seq)

        if zero_bias:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, val=0) #bias
        
    def forward(self, *ars):
        if len(ars)==1:
            net_in = ars[0]
            net_in = net_in.view(net_in.shape[0], -1) #adds a dim when needed
        else:
            net_in = torch.cat([a.view(a.shape[0], -1) for a in ars],dim=1) #flattens everything
        out = self.net_nonlin(net_in) + self.net_res(net_in)
        return out[:,0] if self.scalar_output else out
    
class RK4(nn.Module):
    def __init__(self, n_RK4_steps=1):
        super().__init__()
        self.n_RK4_steps = n_RK4_steps

    def forward(self, f, x, u, dt):
        dtp = (dt/self.n_RK4_steps)[:,None]
        for _ in range(self.n_RK4_steps): #f(x,u) has shape (nbatch, nx)
            k1 = f(x,u)*dtp
            k2 = f(x+k1*0.5,u)*dtp
            k3 = f(x+k2*0.5,u)*dtp
            k4 = f(x+k3,u)*dtp
            x = x + (k1+2*k2+2*k3+k4)/6
        return x
    

import numpy as np
class Bilinear(nn.Module):
    def __init__(self, n_in, n_out, n_schedual, std_output=None, std_input=None):
        super().__init__()
        scale_fac = (n_in*(n_schedual+1))**0.5*10
        self.Alin = nn.Parameter(torch.randn((n_out, n_in))/scale_fac)
        self.Anlin = nn.Parameter(torch.randn((n_schedual, n_out, n_in))/scale_fac)
        self.std_output = torch.as_tensor(std_output,dtype=torch.float32) if std_output is not None else torch.ones((n_out,), dtype=torch.float32)
        assert self.std_output.shape == (n_out,)
        self.std_input = torch.as_tensor(std_input,dtype=torch.float32) if std_input is not None else torch.ones((n_in,), dtype=torch.float32)
        assert self.std_input.shape == (n_in,), f'{self.std_input.shape} == {(n_in,)}'
    
    def forward(self, p):
        #p (Nb, np) 
        #Anlin (np, n_out, n_in) -> (1, np, n_out, n_in)
        #self.Alin (n_out, n_int) -> (None, n_out, n_in)
        A = (self.Alin[None] + (self.Anlin[None]*p[:,:,None,None]).sum(1)) #nbatch, n_out, n_in
        return self.std_output[:,None]*A/self.std_input[None,:]

class LPV_layer(nn.Module):
    #y = A(p)@x + B(p)@u
    def __init__(self, nu, ny, nx, np):
        super().__init__()
        self.sqeeze_output = ny =='scalar'
        self.nu = 1 if nu=='scalar' else nu
        self.ny = 1 if ny=='scalar' else ny
        self.nx, self.np = nx, np
        self.A = Bilinear(nx, ny, np)
        self.B = Bilinear(nu, ny, np)

    def forward(self, x, u, p):
        # x (Nbatch, nx)
        u = u.view(u.shape[0],-1) #(Nbatch, nu)
        A = self.A(p) #(Nbatch, ny, nx)
        B = self.B(p) #(Nbatch, ny, nu)
        y = torch.matmul(A, x.unsqueeze(2))[:,:,0] + torch.matmul(B, u.unsqueeze(2))[:,:,0]
        return y[:,0] if self.sqeeze_output else y

#problems: 
# pnet depdent on nu?
# what to do with exernally schedualed?

class LPV_SS(nn.Module):
    def __init__(self, nx, nu, ny, np, pnet=None, feedthrough=True):
        super().__init__()
        assert feedthrough==True
        self.nx, self.nu, self.ny, self.np, self.feedthrough = nx, nu, ny, np, feedthrough
        self.pnet = MLP_res_net(input_size=[nx, nu], output_size=np) if pnet is None else pnet
        self.output_LPV = LPV_layer(nu=nu, ny=ny, nx=nx, np=np)
        self.state_LPV = LPV_layer(nu=nu, ny=nx, nx=nx, np=np) #the output of this layer is next state hence ny=nx
    
    def forward(self, x, u):
        p = self.pnet(x)
        y = self.output_LPV(x, u, p=p) #might be depent on feedthrough
        x_next = self.state_LPV(x, u, p=p)
        return y, x_next

