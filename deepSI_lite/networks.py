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
# pnet depdent on u?
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


####################
###  CNN SUBNET ####
####################


class ConvShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', upscale_factor=2, \
        padding_mode='zeros'):
        super(ConvShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels*upscale_factor**2, kernel_size, padding=padding, \
            padding_mode=padding_mode)
    
    def forward(self, X):
        X = self.conv(X) #(N, Cout*upscale**2, H, W)
        return nn.functional.pixel_shuffle(X, self.upscale_factor) #(N, Cin, H*r, W*r)


class Upscale_Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', \
                 upscale_factor=2, main_upscale=ConvShuffle, shortcut=ConvShuffle, \
                 padding_mode='zeros', activation=nn.functional.relu, Ch=0, Cw=0):
        assert isinstance(upscale_factor, int)
        super(Upscale_Conv_block, self).__init__()
        #padding='valid' is weird????
        self.shortcut = shortcut(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, upscale_factor=upscale_factor)
        self.activation = activation
        self.upscale = main_upscale(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, upscale_factor=upscale_factor)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.Ch = Ch
        self.Cw = Cw
        
    def forward(self, X):
        #shortcut
        X_shortcut = self.shortcut(X) # (N, Cout, H*r, W*r)
        
        #main line
        X = self.activation(X) # (N, Cin, H, W)
        X = self.upscale(X)    # (N, Cout, H*r, W*r)
        X = self.activation(X) # (N, Cout, H*r, W*r)
        X = self.conv(X)       # (N, Cout, H*r, W*r)
        
        #combine
        # X.shape[:,Cout,H,W]
        H,W = X.shape[2:]
        H2,W2 = X_shortcut.shape[2:]
        if H2>H or W2>W:
            padding_height = (H2-H)//2
            padding_width = (W2-W)//2
            X = X + X_shortcut[:,:,padding_height:padding_height+H,padding_width:padding_width+W]
        else:
            X = X + X_shortcut
        return X[:,:,self.Ch:,self.Cw:] #slice if needed
        #Nnodes = W*H*N(Cout*4*r**2 + Cin)

class CNN_vec_to_image(nn.Module):
    def __init__(self, nx, ny, nu=-1, features_out = 1, kernel_size=3, padding='same', \
                 upscale_factor=2, feature_scale_factor=2, final_padding=4, main_upscale=ConvShuffle, shortcut=ConvShuffle, \
                 padding_mode='zeros', activation=nn.functional.relu):
        super(CNN_vec_to_image, self).__init__()
        self.feedthrough = nu!=-1
        if self.feedthrough:
            self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
            FCnet_in = nx + np.prod(self.nu, dtype=int)
        else:
            FCnet_in = nx
        
        self.activation  = activation
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        if len(ny)==2:
            self.nchannels = 1
            self.None_nchannels = True
            self.height_target, self.width_target = ny
        else:
            self.None_nchannels = False
            self.nchannels, self.height_target, self.width_target = ny
        
        if self.nchannels>self.width_target or self.nchannels>self.height_target:
            import warnings
            text = f"Interpreting shape of data as (Nnchannels={self.nchannels}, Nheight={self.height_target}, Nwidth={self.width_target}), This might not be what you intended!"
            warnings.warn(text)

        #work backwards
        features_out = int(features_out*self.nchannels)
        self.final_padding = final_padding
        height_now = self.height_target + 2*self.final_padding
        width_now  = self.width_target  + 2*self.final_padding
        features_now = features_out
        
        self.upblocks = []
        while height_now>=2*upscale_factor+1 and width_now>=2*upscale_factor+1:
            
            Ch = (-height_now)%upscale_factor
            Cw = (-width_now)%upscale_factor
            # print(height_now, width_now, features_now, Ch, Cw)
            B = Upscale_Conv_block(int(features_now*feature_scale_factor), int(features_now), kernel_size, padding=padding, \
                 upscale_factor=upscale_factor, main_upscale=main_upscale, shortcut=shortcut, \
                 padding_mode=padding_mode, activation=activation, Cw=Cw, Ch=Ch)
            self.upblocks.append(B)
            features_now *= feature_scale_factor
            #implement slicing 
            
            height_now += Ch
            width_now += Cw
            height_now //= upscale_factor
            width_now //= upscale_factor
        # print(height_now, width_now, features_now)
        self.width0 = width_now
        self.height0 = height_now
        self.features0 = int(features_now)
        
        self.upblocks = nn.Sequential(*list(reversed(self.upblocks)))
        self.FC = MLP_res_net(input_size=FCnet_in,output_size=self.width0*self.height0*self.features0, n_hidden_layers=1)
        self.final_conv = nn.Conv2d(features_out, self.nchannels, kernel_size=3, padding=padding, padding_mode='zeros')
        
    def forward(self, x, u=None):
        if self.feedthrough:
            xu = torch.cat([x,u.view(u.shape[0],-1)],dim=1)
        else:
            xu = x
        X = self.FC(xu).view(-1, self.features0, self.height0, self.width0) 
        X = self.upblocks(X)
        X = self.activation(X)
        Xout = self.final_conv(X)
        if self.final_padding>0:
            Xout = Xout[:,:,self.final_padding:-self.final_padding,self.final_padding:-self.final_padding]
        return Xout[:,0,:,:] if self.None_nchannels else Xout

class ShuffleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', upscale_factor=2, padding_mode='zeros'):
        super(ShuffleConv, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode) #kernal larger?
    
    def forward(self, X):
        X = torch.cat([X]*self.upscale_factor**2,dim=1) #(N, Cin*r**2, H, W)
        X = nn.functional.pixel_shuffle(X, self.upscale_factor)  #(N, Cin, H*r, W*r)
        return self.conv(X)
        
class ClassicUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', upscale_factor=2, padding_mode='zeros'):
        super(ClassicUpConv, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode) #kernal larger?
        self.up = nn.Upsample(size=None,scale_factor=upscale_factor,mode='bicubic',align_corners=False)

    def forward(self, X):
        X = self.up(X) #(N, Cin, H*r, W*r)
        return self.conv(X)

class Down_Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', \
                 downscale_factor=2, padding_mode='zeros', activation=nn.functional.relu):
        assert isinstance(downscale_factor, int)
        super(Down_Conv_block, self).__init__()
        #padding='valid' is weird????
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, stride=downscale_factor)
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', padding_mode='zeros')
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, stride=downscale_factor)
        
    def forward(self, X):
        #shortcut
        X_shortcut = self.shortcut(X) # (N, Cout, H/r, W/r)
        
        #main line
        X = self.activation(X)  # (N, Cin, H, W)
        X = self.conv(X)        # (N, Cout, H, W)
        X = self.activation(X)  # (N, Cout, H, W)
        X = self.downscale(X)   # (N, Cout, H/r, W/r)
        
        #combine
        X = X + X_shortcut
        return X

class CNN_chained_downscales(nn.Module):
    def __init__(self, ny, kernel_size=3, padding='valid', features_ups_factor=1.5, \
                 downscale_factor=2, padding_mode='zeros', activation=nn.functional.relu):

        super(CNN_chained_downscales, self).__init__()
        self.activation  = activation
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        if len(ny)==2:
            self.nchannels = 1
            self.None_nchannels = True
            self.height, self.width = ny
        else:
            self.None_nchannels = False
            self.nchannels, self.height, self.width = ny
        
        #work backwards
        Y = torch.randn((1,self.nchannels,self.height,self.width))
        _, features_now, height_now, width_now = Y.shape
        
        self.downblocks = []
        features_now_base = features_now
        while height_now>=2*downscale_factor+1 and width_now>=2*downscale_factor+1:
            features_now_base *= features_ups_factor
            B = Down_Conv_block(features_now, int(features_now_base), kernel_size, padding=padding, \
                 downscale_factor=downscale_factor, padding_mode=padding_mode, activation=activation)
            
            self.downblocks.append(B)
            with torch.no_grad():
                Y = B(Y)
            _, features_now, height_now, width_now = Y.shape #i'm lazy sorry

        self.width0 = width_now
        self.height0 = height_now
        self.features0 = features_now
        self.nout = self.width0*self.height0*self.features0
        # print('CNN output size=',self.nout)
        self.downblocks = nn.Sequential(*self.downblocks)
        
    def forward(self, Y):
        if self.None_nchannels:
            Y = Y[:,None,:,:]
        return self.downblocks(Y).view(Y.shape[0],-1)
    
class CNN_encoder(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_hidden_nodes=64, n_hidden_layers=2, activation=nn.Tanh, features_ups_factor=1.5):
        super(CNN_encoder, self).__init__()
        self.nx = nx
        self.nu = tuple() if nu=='scalar' else ((nu,) if isinstance(nu,int) else nu)
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        ny = (ny[0]*na, ny[1], ny[2]) if len(ny)==3 else (na, ny[0], ny[1])
        # print('ny=',ny)

        self.CNN = CNN_chained_downscales(ny, features_ups_factor=features_ups_factor) 
        self.net = MLP_res_net(input_size=nb*np.prod(self.nu,dtype=int) + self.CNN.nout, \
            output_size=nx, n_hidden_nodes=n_hidden_nodes, n_hidden_layers=n_hidden_layers, activation=activation)


    def forward(self, upast, ypast):
        #ypast = (samples, na, W, H) or (samples, na, C, W, H) to (samples, na*C, W, H)
        ypast = ypast.view(ypast.shape[0],-1,ypast.shape[-2],ypast.shape[-1])
        # print('ypast.shape=',ypast.shape)
        ypast_encode = self.CNN(ypast)
        # print('ypast_encode.shape=',ypast_encode.shape)
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast_encode.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)


