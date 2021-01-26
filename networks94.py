import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 94

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3

encoded_dim = 16

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, seq_len, **kwargs):
        super().__init__()

        self.project_features = (in_channels != out_channels)
        if self.project_features:
            self.proj = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.bn1 = nn.Identity()#nn.InstanceNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.bn2 = nn.Identity()#nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        y0 = x

        y = self.act(self.bn1(self.conv1(x)))

        y = self.bn2(self.conv2(y))
        
        if self.project_features:
            y0 = self.proj(y0)

        y = self.act(y0 + y)

        return y

class ResBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, 
                kernel_size, stride, padding, **kwargs)
        self.act = nn.ReLU()
        self.bn1 = nn.InstanceNorm1d(out_channels)
        self.bn2 = nn.InstanceNorm1d(out_channels)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, 
            kernel_size, stride, padding, **kwargs)

        self.project_features = (in_channels != out_channels)

        if self.project_features:
            self.proj = nn.ConvTranspose1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, out_act=True):
        y0 = x

        y = self.act(self.bn1(self.conv1(y0)))
        y = self.conv2(y)

        if self.project_features:
            y0 = self.proj(y0)

        y = self.bn2(y + y0)
        if out_act == True:
            y = self.act(y)

        return y

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, receptive_field_size):
        super().__init__()

        self.act = nn.LeakyReLU(0.2)
        self.convs = nn.ModuleList()
        self.n_convs = int(np.log2(receptive_field_size))+1
        self.bns = nn.ModuleList()
        self.pads = nn.ModuleList()
        self.dropout = nn.Identity()#nn.Dropout(0.05)
        for i in range(self.n_convs):
            ic = in_channels
            oc = out_channels
            if i > 0:
                ic = out_channels
            dilation = int(2**i)
            self.convs.append(nn.Conv1d(ic, oc, 3, 1, 0, dilation=dilation).cuda())
            self.bns.append(nn.Identity())#nn.BatchNorm1d(oc).cuda())
            self.pads.append(nn.ConstantPad1d((dilation, dilation), 0.0))
        self.has_out_conv = False
        if in_channels != out_channels:
            self.has_out_conv = True
            self.out_conv = nn.Conv1d(in_channels, out_channels, 1, 1, 0).cuda()
            
    def forward(self, x):
        y = x
        out = []
        for i in range(self.n_convs):
            if i > 0:
                y = self.act(y)
            y = self.dropout(self.bns[i](self.convs[i](self.pads[i](y))))
            out.append(y)
        if self.has_out_conv:
            x = self.out_conv(x)

        ret = x
        for i in range(len(out)):
            ret = ret + out[i]
        return self.act(ret)

class TCNBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, receptive_field_size):
        super().__init__()

        self.act = nn.ReLU()
        self.convs = nn.ModuleList()
        self.n_convs = int(np.log2(receptive_field_size))+1
        self.bns = nn.ModuleList()
        self.pads = nn.ModuleList()
        self.dropout = nn.Identity()#nn.Dropout(gen_dropout)
        for i in reversed(range(self.n_convs)):
            ic = in_channels
            oc = out_channels
            if i < self.n_convs-1:
                ic = out_channels
            dilation = int(2**i)
            self.convs.append(nn.Conv1d(ic, oc, 3, 1, 0, dilation=dilation).cuda())
            self.bns.append(nn.InstanceNorm1d(oc).cuda())
            self.pads.append(nn.ConstantPad1d((dilation, dilation), 0.0))
        self.has_out_conv = False
        if in_channels != out_channels:
            self.has_out_conv = True
            self.out_conv = nn.Conv1d(in_channels, out_channels, 1, 1, 0).cuda()
            
    def forward(self, x):
        y = x
        out = []
        for i in range(self.n_convs):
            if i > 0:
                y = self.act(self.bns[i-1](y))
            y = self.convs[i](self.pads[i](y))
            out.append(y)
        if self.has_out_conv:
            x = self.out_conv(x)
        ret = x
        for i in range(len(out)):
            ret = ret + out[i]
        return self.act(ret)

class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len):
        super().__init__()
        
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.05)

        self.lin0 = nn.Linear(latent_dims, seq_len//64*ngf*12, bias=True)

        self.convu1 = nn.ConvTranspose1d(ngf*12, ngf*11, 4, 2, 1)
        self.convu2 = nn.ConvTranspose1d(ngf*11, ngf*10, 4, 2, 1)
        self.convu3 = nn.ConvTranspose1d(ngf*10, ngf*9, 4, 2, 1)
        self.convu4 = nn.ConvTranspose1d(ngf*9, ngf*8, 4, 2, 1)
        self.convu5 = nn.ConvTranspose1d(ngf*8, ngf*7, 4, 2, 1)
        self.convu6 = nn.ConvTranspose1d(ngf*7, ngf*6, 4, 2, 1)
        self.bnu1 = nn.InstanceNorm1d(ngf*11)
        self.bnu2 = nn.InstanceNorm1d(ngf*10)
        self.bnu3 = nn.InstanceNorm1d(ngf*9)
        self.bnu4 = nn.InstanceNorm1d(ngf*8)
        self.bnu5 = nn.InstanceNorm1d(ngf*7)
        self.bnu6 = nn.InstanceNorm1d(ngf*6)


        self.res6 = ResBlockTranspose(ngf*6, ngf*4, 17, 1, 8)
        self.res7 = ResBlockTranspose(ngf*4, ngf*2, 33, 1, 16)

        #self.resw0 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        #self.resw1 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.resw2 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.convw3 = nn.Conv1d(ngf*2, encoded_dim, 33, 1, 16)
        #self.tcn1 = TCNBlockTranspose(ngf*2, ngf*1, 512)
        #self.tcn2 = TCNBlockTranspose(ngf*1, encoded_dim, 512)

        self.dec = nn.Sequential(
            nn.Conv1d(encoded_dim, encoded_dim*2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(encoded_dim*2, encoded_dim*4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(encoded_dim*4, n_wires, 1, 1, 0, bias=False),
        )

        #self.resp0 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        #self.resp1 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.resp2 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.convp3 = nn.Conv1d(ngf*2, ngf, 33, 1, 16)
        self.convp = nn.Conv1d(ngf*1, n_features, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, self.ngf*12, self.seq_len // 64))

        x = self.act(self.bnu1(self.convu1(x)))
        x = self.act(self.bnu2(self.convu2(x)))
        x = self.act(self.bnu3(self.convu3(x)))
        x = self.act(self.bnu4(self.convu4(x)))
        x = self.act(self.bnu5(self.convu5(x)))
        x = self.act(self.bnu6(self.convu6(x)))

        x = self.res6(x)
        x = self.res7(x)
        
        # x (batch, ngf, len)
        w = self.resw2(x)
        w = self.convw3(x)

        encoded_w = w
        
        dec_w = self.dec(encoded_w)

        sim = F.gumbel_softmax(dec_w, dim=1, hard=True, tau=tau)
        #sim = F.softmax(w, dim=1)

        p = self.resp2(x)
        p = self.convp3(p)
        p = self.convp(p)

        return self.out(p), sim

class Disc(nn.Module):
    def __init__(self, ndf, seq_len):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.dropout = nn.Dropout(0.05)

        self.enc = nn.Sequential(
            nn.Conv1d(n_wires, encoded_dim*4, 1, 1, 0, bias=False),
            nn.ReLU(),
            nn.Conv1d(encoded_dim*4, encoded_dim*2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(encoded_dim*2, encoded_dim, 1, 1, 0),
        )

        self.resw0 = ResBlock(encoded_dim, ndf*2, 33, 1, 16, seq_len)
        self.resw1 = ResBlock(ndf*2, ndf*2, 17, 1, 8, seq_len)

        self.convp = nn.Conv1d(n_features, ndf, 1, 1, 0)
        self.resp0 = ResBlock(ndf, ndf*2, 33, 1, 16, seq_len)
        self.resp1 = ResBlock(ndf*2, ndf*2, 17, 1, 8, seq_len)
        
        self.res1 = ResBlock(ndf*4, ndf*4, 33, 1, 16, seq_len)
        self.res2 = ResBlock(ndf*4, ndf*6, 17, 1, 8, seq_len)

        self.convd1 = nn.Conv1d(ndf*6, ndf*7, 4, 2, 1) # / 2
        self.convd2 = nn.Conv1d(ndf*7, ndf*8, 4, 2, 1) # / 4
        self.convd3 = nn.Conv1d(ndf*8, ndf*9, 4, 2, 1) # / 8
        self.convd4 = nn.Conv1d(ndf*9, ndf*10, 4, 2, 1) # / 16
        self.convd5 = nn.Conv1d(ndf*10, ndf*11, 4, 2, 1) # / 32
        self.convd6 = nn.Conv1d(ndf*11, ndf*12, 4, 2, 1) # / 64

        self.lin0 = nn.Linear(ndf*12 * seq_len // 64, 1, bias=False)

        self.out = nn.Identity()

    
    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ shape is one-hot encoded wire (batch, n_wires, seq_len)

        seq_len = x_.shape[2]
        p_ = x_[:,:n_features,:]
        w_ = x_[:,n_features:,:]

        #norm = w_.norm(dim=2).norm(dim=1)
        #occupancy = w_.sum(dim=2).var(dim=1)

        #norm = norm.repeat((seq_len, 1, 1)).permute(2, 1, 0)
        #occupancy = occupancy.repeat((seq_len, 1, 1)).permute(2, 1, 0)

        # w_ (batch, n_wires, seq_len)
        w = self.enc(w_)
        
        w = self.resw0(w)
        w = self.resw1(w)
        #w = self.resw2(w)
        #w = self.resw3(w)

        p = self.convp(p_)
        p = self.resp0(p)
        p = self.resp1(p)
        #p = self.resp2(p)
        #p = self.resp3(p)
        
        x = torch.cat([p, w], dim=1)

        x = self.res1(x)
        x = self.res2(x)
        #x = self.res3(x)
        #x = self.res4(x)

        x = self.act(self.convd1(x))
        x = self.act(self.convd2(x))
        x = self.act(self.convd3(x))
        x = self.act(self.convd4(x))
        x = self.act(self.convd5(x))
        x = self.act(self.convd6(x))

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
