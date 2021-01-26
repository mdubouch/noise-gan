import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 34

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()

        self.project_features = (in_channels != out_channels)
        if self.project_features:
            self.proj = nn.utils.spectral_norm(
                    nn.Conv1d(in_channels, out_channels, 1, 1, 0))

        self.conv1 = nn.utils.spectral_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs))
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, **kwargs))
        self.dropout = nn.Dropout(0.05)



    def forward(self, x):
        y0 = x
        y = self.dropout(self.act(self.conv1(y0)))

        if self.project_features:
            y0 = self.proj(y0)
        y = y0 + self.conv2(y)

        return self.act(y)

class ResBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, in_channels, 
                kernel_size, stride, padding, **kwargs)
        self.act = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.ConvTranspose1d(in_channels, out_channels, 
            kernel_size, stride, padding, **kwargs)
        self.dropout = nn.Dropout(0.05)

        self.project_features = (in_channels != out_channels)

        if self.project_features:
            self.proj = nn.ConvTranspose1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, out_act=True):
        y0 = x
        y = self.dropout(self.act(self.bn1(self.conv1(y0))))

        if self.project_features:
            y0 = self.proj(y0)
        y = y0 + self.bn2(self.conv2(y))
        if out_act == True:
            y = self.act(y)

        return y

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, receptive_field_size):
        super().__init__()

        self.act = nn.LeakyReLU(0.2)
        self.convs = []
        self.n_convs = int(np.log2(receptive_field_size))+1
        self.bns = []
        self.pads = []
        self.dropout = nn.Dropout(0.05)
        for i in range(self.n_convs):
            ic = in_channels
            oc = out_channels
            if i > 0:
                ic = out_channels
            dilation = int(2**i)
            self.convs.append(nn.utils.spectral_norm(
                nn.Conv1d(ic, oc, 3, 1, 0, dilation=dilation).cuda()))
            self.bns.append(nn.Identity())#nn.BatchNorm1d(oc).cuda())
            self.pads.append(nn.ConstantPad1d((dilation, dilation), 0.0))
            print(self.convs[i])
        self.has_out_conv = False
        if in_channels != out_channels:
            self.has_out_conv = True
            self.out_conv = nn.utils.spectral_norm(
                    nn.Conv1d(in_channels, out_channels, 1, 1, 0).cuda())
            
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

        self.act = nn.LeakyReLU(0.2)
        self.convs = []
        self.n_convs = int(np.log2(receptive_field_size))+1
        self.bns = []
        self.pads = []
        self.dropout = nn.Dropout(0.05)
        for i in reversed(range(self.n_convs)):
            ic = in_channels
            oc = out_channels
            if i < self.n_convs-1:
                ic = out_channels
            dilation = int(2**i)
            self.convs.append(nn.Conv1d(ic, oc, 3, 1, 0, dilation=dilation).cuda())
            self.bns.append(nn.BatchNorm1d(oc).cuda())
            self.pads.append(nn.ConstantPad1d((dilation, dilation), 0.0))
        print(self.convs)
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



class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len):
        super().__init__()
        
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.LeakyReLU(0.2, False)

        self.dropout = nn.Dropout(0.05)

        self.lin0 = nn.Linear(latent_dims, seq_len//64*ngf*6, bias=True)

        self.convu1 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 2
        self.convu2 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 4
        self.convu3 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 8
        self.convu4 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 16
        self.convu5 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 32
        self.convu6 = nn.ConvTranspose1d(ngf*6, ngf*4, 4, 2, 1) # * 64
        
        self.bnu1 = nn.BatchNorm1d(ngf*6)
        self.bnu2 = nn.BatchNorm1d(ngf*6)
        self.bnu3 = nn.BatchNorm1d(ngf*6)
        self.bnu4 = nn.BatchNorm1d(ngf*6)
        self.bnu5 = nn.BatchNorm1d(ngf*6)
        self.bnu6 = nn.BatchNorm1d(ngf*4)


        #self.res4 = ResBlockTranspose(ngf*4, ngf*4, 65, 1, 32)
        #self.res5 = ResBlockTranspose(ngf*4, ngf*4, 129, 1, 64)
        #self.res6 = ResBlockTranspose(ngf*4, ngf*4, 17, 1, 8)
        #self.res7 = ResBlockTranspose(ngf*4, ngf*2, 33, 1, 16)
        self.tcn1 = TCNBlockTranspose(ngf*4, ngf*2, 1024)

        #self.resw0 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        #self.resw1 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        #self.resw2 = ResBlockTranspose(ngf*2, ngf*2, 1, 1, 0)
        #self.resw3 = ResBlockTranspose(ngf*2, ngf, 1, 1, 0)
        self.tcnw1 = TCNBlockTranspose(ngf*2, ngf*2, 256)
        self.tcnw2 = TCNBlockTranspose(ngf*2, ngf*2, 512)
        self.tcnw3 = TCNBlockTranspose(ngf*2, ngf, 1024)
        self.convw = nn.ConvTranspose1d(ngf, n_wires, 1, 1, 0, bias=False)

        #self.resp0 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        #self.resp1 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        #self.resp2 = ResBlockTranspose(ngf*2, ngf*2, 1, 1, 0)
        #self.resp3 = ResBlockTranspose(ngf*2, ngf, 1, 1, 0)
        self.tcnp1 = TCNBlockTranspose(ngf*2, ngf*2, 256)
        self.tcnp2 = TCNBlockTranspose(ngf*2, ngf*2, 512)
        self.tcnp3 = TCNBlockTranspose(ngf*2, ngf, 1024)
        self.convp = nn.ConvTranspose1d(ngf, n_features, 1, 1, 0, bias=False)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x0 = self.act(self.lin0(z).view(-1, self.ngf*6, self.seq_len // 64))

        x1 = self.act(self.bnu1(self.convu1(self.dropout(x0))))
        x2 = self.act(F.interpolate(x0, scale_factor=4)  + self.bnu2(self.convu2(self.dropout(x1))))

        x3 = self.act(self.bnu3(self.convu3(self.dropout(x2))))
        x4 = self.act(F.interpolate(x2, scale_factor=4) + self.bnu4(self.convu4(self.dropout(x3))))

        x5 = self.act(self.bnu5(self.convu5(self.dropout(x4))))
        x6 = self.act(self.bnu6(self.convu6(self.dropout(x5))))

        #x = self.res4(x6)
        #x = self.res5(x)
        #x = self.res6(x6)
        #x = self.res7(x)
        x = self.tcn1(x6)
        
        # x (batch, ngf, len)
        #w = self.resw0(x)
        #w = self.resw1(w)
        #w = self.resw2(x)
        #w = self.resw3(w)
        w = self.tcnw1(x)
        w = self.tcnw2(w)
        w = self.tcnw3(w)
        w = self.convw(w)
        
        sim = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        #p = self.resp0(x)
        #p = self.resp1(p)
        #p = self.resp2(x)
        #p = self.resp3(p)
        p = self.tcnp1(x)
        p = self.tcnp2(p)
        p = self.tcnp3(p)
        p = self.convp(p)

        return self.out(p), sim

class Disc(nn.Module):
    def __init__(self, ndf, seq_len):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.convw = nn.utils.spectral_norm(nn.Conv1d(n_wires, ndf, 1, 1, 0, bias=False))
        self.tcnw1 = TCNBlock(ndf, ndf*2, 1024)
        self.tcnw2 = TCNBlock(ndf*2, ndf*2, 512)
        self.tcnw3 = TCNBlock(ndf*2, ndf*2, 256)
        #self.resw0 = ResBlock(ndf, ndf*2, 1, 1, 0, padding_mode='circular')
        #self.resw1 = ResBlock(ndf*2, ndf*2, 1, 1, 0, padding_mode='circular')
        #self.resw2 = ResBlock(ndf*2, ndf*2, 25, 1, 12, padding_mode='circular')
        #self.resw3 = ResBlock(ndf*2, ndf*2, 17, 1, 8, padding_mode='circular')

        self.convp = nn.utils.spectral_norm(nn.Conv1d(n_features, ndf, 1, 1, 0, bias=False))
        self.tcnp1 = TCNBlock(ndf, ndf*2, 1024)
        self.tcnp2 = TCNBlock(ndf*2, ndf*2, 512)
        self.tcnp3 = TCNBlock(ndf*2, ndf*2, 256)
        #self.resp0 = ResBlock(ndf, ndf*2, 1, 1, 0, padding_mode='circular')
        #self.resp1 = ResBlock(ndf*2, ndf*2, 1, 1, 0, padding_mode='circular')
        #self.resp2 = ResBlock(ndf*2, ndf*2, 25, 1, 12, padding_mode='circular')
        #self.resp3 = ResBlock(ndf*2, ndf*2, 17, 1, 8, padding_mode='circular')
        

        #self.tcn1 = TCNBlock(ndf*4, ndf*6, 1024)
        self.res1 = ResBlock(ndf*4, ndf*6, 17, 1, 8, padding_mode='zeros')
        self.res2 = ResBlock(ndf*6, ndf*6, 17, 1, 8, padding_mode='zeros')
        self.res3 = ResBlock(ndf*6, ndf*6, 17, 1, 8, padding_mode='zeros')
        #self.res4 = ResBlock(ndf*6, ndf*6, 25, 1, 12, padding_mode='circular')
        self.conv1 = nn.utils.spectral_norm(
                nn.Conv1d(ndf*6, ndf*6, 17, 1, 8, padding_mode='zeros'))
        self.convd1 = nn.utils.spectral_norm(
                nn.Conv1d(ndf*6, ndf*6, 3, 2, 1, padding_mode='zeros'))

        self.conv2 = nn.utils.spectral_norm(
            nn.Conv1d(ndf*6, ndf*6, 17, 1, 8, padding_mode='zeros'))
        self.convd2 = nn.utils.spectral_norm(
            nn.Conv1d(ndf*6, ndf*6, 3, 2, 1, padding_mode='zeros'))

        self.conv3 = nn.utils.spectral_norm(
            nn.Conv1d(ndf*6, ndf*6, 17, 1, 8, padding_mode='zeros'))
        self.convd3 = nn.utils.spectral_norm(
            nn.Conv1d(ndf*6, ndf*6, 3, 2, 1, padding_mode='zeros'))

        self.conv4 = nn.utils.spectral_norm(
            nn.Conv1d(ndf*6, ndf*6, 17, 1, 8, padding_mode='zeros'))
        self.convd4 = nn.utils.spectral_norm(
            nn.Conv1d(ndf*6, ndf*6, 3, 2, 1, padding_mode='zeros'))

        self.lin0 = nn.utils.spectral_norm(nn.Linear(ndf*6 * seq_len // 16, 1, bias=True))

        self.out = nn.Identity()

    
    def forward(self, p_, w_): # p_ shape is (batch, features, seq_len), w_ shape is one-hot encoded wire (batch, 4986, seq_len)

        #norm = w_.norm(dim=2).norm(dim=1)
        #occupancy = w_.sum(dim=2).var(dim=1)

        #norm = norm.repeat((seq_len, 1, 1)).permute(2, 1, 0)
        seq_len = p_.shape[2]
        #occupancy = occupancy.repeat((seq_len, 1, 1)).permute(2, 1, 0)

        w = self.convw(w_)
        w = self.tcnw1(w)
        w = self.tcnw2(w)
        w = self.tcnw3(w)
        #w = self.resw0(w)
        #w = self.resw1(w)
        #w = self.resw2(w)
        #w = self.resw3(w)

        p = self.convp(p_)
        p = self.tcnp1(p)
        p = self.tcnp2(p)
        p = self.tcnp3(p)
        #p = self.resp0(p)
        #p = self.resp1(p)
        #p = self.resp2(p)
        #p = self.resp3(p)
        
        x = torch.cat([p, w], dim=1)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        pool = nn.AvgPool1d(2)

        x0 = x
        x = self.act(self.conv1(x0))
        x = self.act(pool(x0) + self.convd1(x))

        x0 = x
        x = self.act(self.conv2(x0))
        x = self.act(pool(x0) + self.convd2(x))

        x0 = x
        x = self.act(self.conv3(x0))
        x = self.act(pool(x0) + self.convd3(x))

        x0 = x
        x = self.act(self.conv4(x0))
        x = self.act(pool(x0) + self.convd4(x))

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
