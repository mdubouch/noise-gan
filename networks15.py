import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = 15

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()

        self.project_features = (in_channels != out_channels)
        if self.project_features:
            self.proj = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

        self.conv1 = nn.utils.spectral_norm(
                nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, **kwargs))
        self.act = nn.LeakyReLU(0.02)
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, **kwargs))



    def forward(self, x):
        y0 = x
        if self.project_features:
            y0 = self.act(self.proj(y0))

        y = self.act(self.conv1(y0))
        y = self.act(y0 + self.conv2(y))

        return y

class ResBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, in_channels, 
                kernel_size, stride, padding, **kwargs)
        self.act = nn.LeakyReLU(0.02)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.ConvTranspose1d(in_channels, in_channels, 
            kernel_size, stride, padding, **kwargs)

        self.project_features = (in_channels != out_channels)

        if self.project_features:
            self.proj = nn.ConvTranspose1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, out_act=True):
        y = self.act(self.bn1(self.conv1(x)))
        y = x + self.bn2(self.conv2(y))
        if self.project_features:
            y = self.proj(self.act(y))

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

        self.dropout = nn.Dropout(0.02)

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

        self.conv4 = nn.ConvTranspose1d(ngf*4, ngf*4, 5, 1, 2)
        self.conv5 = nn.ConvTranspose1d(ngf*4, ngf*4, 11, 1, 5)
        self.conv6 = nn.ConvTranspose1d(ngf*4, ngf*4, 17, 1, 8)
        self.conv7 = nn.ConvTranspose1d(ngf*4, ngf*4, 33, 1, 16)
        self.conv8 = nn.ConvTranspose1d(ngf*4, ngf*2, 49, 1, 24)

        self.bn4 = nn.BatchNorm1d(ngf*4)
        self.bn5 = nn.BatchNorm1d(ngf*4)
        self.bn6 = nn.BatchNorm1d(ngf*4)
        self.bn7 = nn.BatchNorm1d(ngf*4)
        self.bn8 = nn.BatchNorm1d(ngf*2)

        self.resw0 = ResBlockTranspose(ngf*2, ngf*2, 3, 1, 1)
        self.resw1 = ResBlockTranspose(ngf*2, ngf*2, 5, 1, 2)
        self.resw2 = ResBlockTranspose(ngf*2, ngf*2, 7, 1, 3)
        self.resw3 = ResBlockTranspose(ngf*2, n_wires, 513, 1, 256)

        self.resp0 = ResBlockTranspose(ngf*2, ngf*2, 3, 1, 1)
        self.resp1 = ResBlockTranspose(ngf*2, ngf*2, 5, 1, 2)
        self.resp2 = ResBlockTranspose(ngf*2, ngf*2, 7, 1, 3)
        self.resp3 = ResBlockTranspose(ngf*2, n_features, 513, 1, 256)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x0 = self.lin0(z).view(-1, self.ngf*6, self.seq_len // 64)

        x1 = self.act(self.bnu1(self.convu1(self.dropout(x0))))
        x2 = self.act(F.interpolate(x0, scale_factor=4)  + self.bnu2(self.convu2(self.dropout(x1))))

        x3 = self.act(self.bnu3(self.convu3(self.dropout(x2))))
        x4 = self.act(F.interpolate(x2, scale_factor=4) + self.bnu4(self.convu4(self.dropout(x3))))

        x5 = self.act(self.bnu5(self.convu5(self.dropout(x4))))
        x6 = self.act(self.bnu6(self.convu6(self.dropout(x5))))

        x7 = self.act(self.bn4(self.conv4(self.dropout(x6))))
        x8 = self.act(x6 + self.bn5(self.conv5(self.dropout(x7))))

        x9 = self.act(self.bn6(self.conv6(self.dropout(x8))))
        x10 = self.act(x8 + self.bn7(self.conv7(self.dropout(x9))))

        x11 = self.act(self.bn8(self.conv8(self.dropout(x10))))
        
        
        # x (batch, ngf, len)
        w = self.resw0(x11)
        w = self.resw1(w)
        w = self.resw2(w)
        w = self.resw3(w, out_act=False)
        
        sim = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p = self.resp0(x11)
        p = self.resp1(p)
        p = self.resp2(p)
        p = self.resp3(p, out_act=False)

        return self.out(p), sim

class Disc(nn.Module):
    def __init__(self, ndf, seq_len):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.dropout = nn.Dropout(0.01)


        self.resw0 = ResBlock(n_wires, ndf*2, 513, 1, 256, padding_mode='circular')
        self.resw1 = ResBlock(ndf*2, ndf*2, 7, 1, 3, padding_mode='circular')
        self.resw2 = ResBlock(ndf*2, ndf*2, 5, 1, 2, padding_mode='circular')
        self.resw3 = ResBlock(ndf*2, ndf*2, 3, 1, 1, padding_mode='circular')

        self.resp0 = ResBlock(n_features, ndf*2, 513, 1, 256, padding_mode='circular')
        self.resp1 = ResBlock(ndf*2, ndf*2, 7, 1, 3, padding_mode='circular')
        self.resp2 = ResBlock(ndf*2, ndf*2, 5, 1, 2, padding_mode='circular')
        self.resp3 = ResBlock(ndf*2, ndf*2, 3, 1, 1, padding_mode='circular')
        

        self.res1 = ResBlock(ndf*4, ndf*6, 17, 1, 8, padding_mode='circular')
        self.res2 = ResBlock(ndf*6, ndf*6, 11, 1, 5, padding_mode='circular')
        self.res3 = ResBlock(ndf*6, ndf*6, 5, 1, 2, padding_mode='circular')

        self.lin0 = nn.Linear(ndf*6 * seq_len // 1, 1, bias=True)

        self.out = nn.Identity()

    
    def forward(self, p_, w_): # p_ shape is (batch, features, seq_len), w_ shape is one-hot encoded wire (batch, 4986, seq_len)

        #norm = w_.norm(dim=2).norm(dim=1)
        #occupancy = w_.sum(dim=2).var(dim=1)

        #norm = norm.repeat((seq_len, 1, 1)).permute(2, 1, 0)
        seq_len = p_.shape[2]
        #occupancy = occupancy.repeat((seq_len, 1, 1)).permute(2, 1, 0)

        w = self.resw0(w_)
        w = self.resw1(w)
        w = self.resw2(w)
        w = self.resw3(w)

        p = self.resp0(p_)
        p = self.resp1(p)
        p = self.resp2(p)
        p = self.resp3(p)
        
        x = torch.cat([p, w], dim=1)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
