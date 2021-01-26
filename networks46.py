import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = 46

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
                    nn.Conv1d(in_channels, out_channels, 1, 1, 0, **kwargs))

        self.conv1 = nn.utils.spectral_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs))
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.utils.spectral_norm(
                nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, **kwargs))

    def forward(self, x):
        y0 = x

        y = self.act(self.conv1(x))

        y = self.conv2(y)
        
        if self.project_features:
            y0 = self.proj(y0)

        y = self.act(y0 + y)

        return y

class ResBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, 
                kernel_size, stride, padding, **kwargs)
        self.act = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, 
            kernel_size, stride, padding, **kwargs)

        self.project_features = (in_channels != out_channels)

        if self.project_features:
            self.proj = nn.ConvTranspose1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, out_act=True):
        y0 = x
        y = self.conv1(self.act(self.bn1(y0)))
        y = self.conv2(self.act(self.bn2(y)))

        if self.project_features:
            y0 = self.proj(y0)

        y = y + y0
        if out_act == True:
            y = self.act(y)

        return y

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
        self.conv1 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu2 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 4
        self.conv2 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu3 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 8
        self.conv3 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu4 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 16
        self.conv4 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu5 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 32
        self.conv5 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu6 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 64
        self.conv6 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.bnu1 = nn.BatchNorm1d(ngf*6)
        self.bnu2 = nn.BatchNorm1d(ngf*6)
        self.bnu3 = nn.BatchNorm1d(ngf*6)
        self.bnu4 = nn.BatchNorm1d(ngf*6)
        self.bnu5 = nn.BatchNorm1d(ngf*6)
        self.bnu6 = nn.BatchNorm1d(ngf*6)
        self.bn1 = nn.BatchNorm1d(ngf*6)
        self.bn2 = nn.BatchNorm1d(ngf*6)
        self.bn3 = nn.BatchNorm1d(ngf*6)
        self.bn4 = nn.BatchNorm1d(ngf*6)
        self.bn5 = nn.BatchNorm1d(ngf*6)
        self.bn6 = nn.BatchNorm1d(ngf*6)


        self.res6 = ResBlockTranspose(ngf*6, ngf*4, 17, 1, 8)
        self.res7 = ResBlockTranspose(ngf*4, ngf*2, 33, 1, 16)

        #self.resw0 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.resw1 = ResBlockTranspose(ngf*2, ngf*2, 129, 1, 64)
        self.resw2 = ResBlockTranspose(ngf*2, ngf*4, 257, 1, 128)
        self.resw3 = ResBlockTranspose(ngf*4, ngf*8, 513, 1, 256)
        self.convw = nn.Conv1d(ngf*8, n_wires, 1, 1, 0)

        #self.resp0 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.resp1 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.resp2 = ResBlockTranspose(ngf*2, ngf*2, 33, 1, 16)
        self.resp3 = ResBlockTranspose(ngf*2, ngf*2, 513, 1, 256)
        self.convp = nn.Conv1d(ngf*2, n_features, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, self.ngf*6, self.seq_len // 64))

        x0 = x
        x = self.act(self.bnu1(self.convu1(x0)))
        x = self.bn1(self.conv1(x))
        x = self.act(x + F.interpolate(x0, scale_factor=2, mode='linear'))

        x0 = x
        x = self.act(self.bnu2(self.convu2(x0)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + F.interpolate(x0, scale_factor=2, mode='linear'))

        x0 = x
        x = self.act(self.bnu3(self.convu3(x0)))
        x = self.bn3(self.conv3(x))
        x = self.act(x + F.interpolate(x0, scale_factor=2, mode='linear'))

        x0 = x
        x = self.act(self.bnu4(self.convu4(x0)))
        x = self.bn4(self.conv4(x))
        x = self.act(x + F.interpolate(x0, scale_factor=2, mode='linear'))

        x0 = x
        x = self.act(self.bnu5(self.convu5(x0)))
        x = self.bn5(self.conv5(x))
        x = self.act(x + F.interpolate(x0, scale_factor=2, mode='linear'))

        x0 = x
        x = self.act(self.bnu6(self.convu6(x0)))
        x = self.bn6(self.conv6(x))
        x = self.act(x + F.interpolate(x0, scale_factor=2, mode='linear'))

        x = self.res6(x, out_act=False)
        x = self.res7(x, out_act=False)
        
        # x (batch, ngf, len)
        w = self.resw1(x, out_act=False)
        w = self.resw2(x, out_act=False)
        w = self.resw3(w, out_act=True)
        w = self.convw(w)
        
        sim = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p = self.resp1(x, out_act=False)
        p = self.resp2(x, out_act=False)
        p = self.resp3(p, out_act=True)
        p = self.convp(p)

        return self.out(p), sim

class Disc(nn.Module):
    def __init__(self, ndf, seq_len):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.dropout = nn.Dropout(0.05)

        self.convw = nn.Conv1d(2, ndf*8, 1, 1, 0)
        self.resw0 = ResBlock(ndf*8, ndf*4, 513, 1, 256, padding_mode='zeros')
        self.resw1 = ResBlock(ndf*4, ndf*2, 257, 1, 128, padding_mode='zeros')
        self.resw2 = ResBlock(ndf*2, ndf*2, 129, 1, 64, padding_mode='zeros')

        self.convp = nn.Conv1d(n_features, ndf*1, 1, 1, 0)
        self.resp0 = ResBlock(ndf*1, ndf*2, 513, 1, 256, padding_mode='zeros')
        self.resp1 = ResBlock(ndf*2, ndf*2, 33, 1, 16, padding_mode='zeros')
        self.resp2 = ResBlock(ndf*2, ndf*2, 17, 1, 8, padding_mode='zeros')
        

        self.res1 = ResBlock(ndf*4, ndf*6, 17, 1, 8, dilation=1, padding_mode='zeros')
        self.res2 = ResBlock(ndf*6, ndf*6, 17, 1, 16, dilation=2, padding_mode='zeros')
        self.res3 = ResBlock(ndf*6, ndf*6, 17, 1, 32, dilation=4, padding_mode='zeros')
        self.res4 = ResBlock(ndf*6, ndf*6, 17, 1, 64, dilation=8, padding_mode='zeros')
        self.res5 = ResBlock(ndf*6, ndf*6, 17, 1, 128, dilation=16, padding_mode='zeros')

        self.lin0 = nn.Linear(ndf*6 * seq_len // 1, 1, bias=True)

        self.out = nn.Identity()

    
    def forward(self, p_, w_): # p_ shape is (batch, features, seq_len), w_ shape is one-hot encoded wire (batch, 4986, seq_len)

        #norm = w_.norm(dim=2).norm(dim=1)
        #occupancy = w_.sum(dim=2).var(dim=1)

        #norm = norm.repeat((seq_len, 1, 1)).permute(2, 1, 0)
        seq_len = p_.shape[2]
        #occupancy = occupancy.repeat((seq_len, 1, 1)).permute(2, 1, 0)

        w = self.convw(w_)
        w = self.resw0(w)
        w = self.resw1(w)
        w = self.resw2(w)
        #w = self.resw3(w)

        p = self.convp(p_)
        p = self.resp0(p)
        p = self.resp1(p)
        p = self.resp2(p)
        #p = self.resp3(p)
        
        x = torch.cat([p, w], dim=1)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
