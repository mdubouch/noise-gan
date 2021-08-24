import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 117

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3

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
    def __init__(self, ngf, latent_dims, seq_len, encoded_dim):
        super().__init__()
        
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.05)

        self.lin0 = nn.Linear(latent_dims, seq_len//256*ngf*6, bias=True)

        self.ups = nn.Upsample(scale_factor=4)

        self.conv1 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.conv2 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.conv3 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.conv4 = nn.Conv1d(ngf*6, ngf*4, 9, 1, 4)
        self.bn1 = nn.InstanceNorm1d(ngf*6)
        self.bn2 = nn.InstanceNorm1d(ngf*6)
        self.bn3 = nn.InstanceNorm1d(ngf*6)
        self.bn4 = nn.InstanceNorm1d(ngf*4)

        self.conv7 = nn.Conv1d(ngf*4, ngf*1, 9, 1, 4)
        self.bn7 = nn.InstanceNorm1d(ngf*1)

        self.conv8 = nn.Conv1d(ngf, n_features+encoded_dim, 9, 1, 4)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, self.ngf*6, self.seq_len // 256))

        x = self.act(self.bn1(self.conv1(x)))
        x = self.ups(x)
        
        x = self.act(self.bn2(self.conv2(x)))
        x = self.ups(x)

        x = self.act(self.bn3(self.conv3(x)))
        x = self.ups(x)
        
        x = self.act(self.bn4(self.conv4(x)))
        x = self.ups(x)

        x = self.act(self.bn7(self.conv7(x)))
        
        # x (batch, ngf, len)
        x = self.conv8(x)

        p = x[:,:n_features]
        w = x[:,n_features:]

        return self.out(p), self.out(w)

class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.dropout = nn.Dropout(0.05)

        self.conv1 = nn.Conv1d(n_features+encoded_dim, ndf*8, 513, 32, 256)
        self.conv2 = nn.Conv1d(ndf*8, ndf*8, 1, 1, 0)
        self.conv3 = nn.Conv1d(ndf*8, ndf*8, 1, 1, 0)

        self.lin0 = nn.Linear(ndf*8 * seq_len // 32, 1, bias=True)

        self.out = nn.Identity()

    
    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        x = x_

        # w_ (batch, encoded_dim, seq_len)

        x0 = self.act(self.conv1(x))
        x = self.act(self.conv2(x0))
        x = self.act(x0 + self.conv3(x))

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


class AE(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(n_wires, encoded_dim*2),
            nn.Tanh(),
            nn.Linear(encoded_dim*2, encoded_dim),
            nn.Tanh()
        )

        self.dec_net = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim*2),
            nn.ReLU(),
            nn.Linear(encoded_dim*2, n_wires)
        )
    def enc(self, x):
        return self.enc_net(x.permute(0,2,1)).permute(0,2,1)
    def dec(self, x):
        return self.dec_net(x.permute(0,2,1)).permute(0,2,1)
    def forward(self, x):
        y = self.dec_net(self.enc_net(x.permute(0, 2, 1))).permute(0,2,1)
        return y


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
