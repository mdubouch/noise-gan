import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = 50

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
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, 
            kernel_size, stride, padding, **kwargs)

        self.project_features = (in_channels != out_channels)

        if self.project_features:
            self.proj = nn.ConvTranspose1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, out_act=True):
        y0 = x

        y = self.act(self.bn1(self.conv1(y0)))
        y = self.bn2(self.conv2(y))

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
        self.convu2 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 4
        self.convu3 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 8
        self.convu4 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 16
        self.convu5 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 32
        self.convu6 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 64
        self.bnu1 = nn.BatchNorm1d(ngf*6)
        self.bnu2 = nn.BatchNorm1d(ngf*6)
        self.bnu3 = nn.BatchNorm1d(ngf*6)
        self.bnu4 = nn.BatchNorm1d(ngf*6)
        self.bnu5 = nn.BatchNorm1d(ngf*6)
        self.bnu6 = nn.BatchNorm1d(ngf*6)

        self.conv6 = nn.Conv1d(ngf*6, ngf*4, 17, 1, 8)
        self.conv7 = nn.Conv1d(ngf*4, ngf*2, 33, 1, 16)
        self.bn6 = nn.BatchNorm1d(ngf*4)
        self.bn7 = nn.BatchNorm1d(ngf*2)

        #self.resw0 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        #self.resw1 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.convw2 = nn.Conv1d(ngf*2, ngf*2, 17, 1, 8)
        self.convw3 = nn.Conv1d(ngf*2, ngf*2, 513, 1, 256)
        self.convw = nn.Conv1d(ngf*2, n_wires, 1, 1, 0)

        #self.resp0 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        #self.resp1 = ResBlockTranspose(ngf*2, ngf*2, 17, 1, 8)
        self.convp2 = nn.Conv1d(ngf*2, ngf*2, 17, 1, 8)
        self.convp3 = nn.Conv1d(ngf*2, ngf*2, 33, 1, 16)
        self.convp = nn.Conv1d(ngf*2, n_features, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, self.ngf*6, self.seq_len // 64))

        x0 = x
        x = self.bnu1(self.convu1(x0))
        x = self.bnu2(self.convu2(self.act(x)))
        x = self.act(x + F.interpolate(x0, scale_factor=4, mode='linear'))

        x0 = x
        x = self.bnu3(self.convu3(x0))
        x = self.bnu4(self.convu4(self.act(x)))
        x = self.act(x + F.interpolate(x0, scale_factor=4, mode='linear'))

        x0 = x
        x = self.bnu5(self.convu5(x0))
        x = self.bnu6(self.convu6(self.act(x)))
        x = self.act(x + F.interpolate(x0, scale_factor=4, mode='linear'))

        x = self.act(self.bn6(self.conv6(x)))
        x = self.act(self.bn7(self.conv7(x)))
        
        # x (batch, ngf, len)
        w = self.act(self.convw2(x))
        w = self.convw3(w)
        w = self.convw(w)
        
        sim = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p = self.act(self.convp2(x))
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

        self.resw0 = nn.Conv1d(2, ndf*2, 513, 1, 256, padding_mode='zeros')
        self.resw1 = nn.utils.spectral_norm(nn.Conv1d(ndf*2, ndf*2, 17, 1, 8, padding_mode='zeros'))

        self.resp0 = nn.Conv1d(n_features, ndf*2, 33, 1, 16, padding_mode='zeros')
        self.resp1 = nn.utils.spectral_norm(nn.Conv1d(ndf*2, ndf*2, 17, 1, 8, padding_mode='zeros'))

        self.res1 = nn.utils.spectral_norm(nn.Conv1d(ndf*4, ndf*4, 33, 1, 16, padding_mode='zeros'))
        self.res2 = nn.utils.spectral_norm(nn.Conv1d(ndf*4, ndf*6, 17, 1, 8, padding_mode='zeros'))
        self.res3 = nn.utils.spectral_norm(nn.Conv1d(ndf*6, ndf*6, 17, 1, 8, padding_mode='zeros'))
        self.res4 = nn.utils.spectral_norm(nn.Conv1d(ndf*6, ndf*6, 17, 1, 8, padding_mode='zeros'))

        self.lin0 = nn.Linear(ndf*6 * seq_len // 1, 1, bias=True)

        self.out = nn.Identity()

    
    def forward(self, p_, w_): # p_ shape is (batch, features, seq_len), w_ shape is one-hot encoded wire (batch, 4986, seq_len)

        #norm = w_.norm(dim=2).norm(dim=1)
        #occupancy = w_.sum(dim=2).var(dim=1)

        #norm = norm.repeat((seq_len, 1, 1)).permute(2, 1, 0)
        seq_len = p_.shape[2]
        #occupancy = occupancy.repeat((seq_len, 1, 1)).permute(2, 1, 0)

        w = self.act(self.resw0(w_))
        w = self.act(self.resw1(w))
        #w = self.resw2(w)
        #w = self.resw3(w)

        p = self.act(self.resp0(p_))
        p = self.act(self.resp1(p))
        #p = self.resp2(p)
        #p = self.resp3(p)
        
        x = torch.cat([p, w], dim=1)

        x = self.act(self.res1(x))
        x = self.act(self.res2(x))
        x = self.act(self.res3(x))
        x = self.act(self.res4(x))
        #x = self.res3(x)
        #x = self.res4(x)

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
