import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 102

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3

class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len, encoded_dim):
        super().__init__()
        
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.PReLU()

        self.dropout = nn.Dropout(0.05)

        self.lin0 = nn.Linear(latent_dims, seq_len//8*ngf*8, bias=True)


        class Res(nn.Module):
            def __init__(self, channels, n_layers, *args, **kwargs):
                super().__init__()
                self.convs = nn.ModuleList(
                    [nn.Conv1d(channels, channels, *args, **kwargs) for i in range(n_layers)]
                )
                self.norms = nn.ModuleList(
                    [nn.InstanceNorm1d(channels) for i in range(n_layers)])
                self.act = nn.PReLU()
                self.n_layers = n_layers
            def forward(self, x):
                y = x
                for i in range(self.n_layers):
                    y = self.norms[i](self.convs[i](y))
                    if i != self.n_layers-1:
                        y = self.act(y)
                return self.act(x + y)

        self.res0 = Res(ngf*8, 2, 9, 1, 4)

        self.convu1 = nn.ConvTranspose1d(ngf*8, ngf*4, 4, 2, 1) # * 2
        self.res1 = Res(ngf*4, 2, 9, 1, 4)
        self.convu2 = nn.ConvTranspose1d(ngf*4, ngf*2, 4, 2, 1) # * 4
        self.res2 = Res(ngf*2, 2, 9, 1, 4)
        self.convu3 = nn.ConvTranspose1d(ngf*2, ngf*1, 4, 2, 1) # * 8
        self.res3 = Res(ngf*1, 2, 9, 1, 4)

        # Split down the middle
        
        self.convw = nn.Conv1d(ngf, encoded_dim, 1, 1, 0)

        self.convp = nn.Conv1d(ngf, n_features, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        ngf=self.ngf
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, self.ngf*8, self.seq_len // 8))

        x = self.res0(x)
        
        x = self.act(self.convu1(x))
        x = self.res1(x)

        x = self.act(self.convu2(x))
        x = self.res2(x)

        x = self.act(self.convu3(x))
        x = self.res3(x)

        # w (batch, ngf, len)
        w = self.convw(x)

        p = self.convp(x)

        return self.out(p), self.out(w)

class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        self.dropout = nn.Dropout(0.05)

        class Res(nn.Module):
            def __init__(self, channels, n_layers, *args, **kwargs):
                super().__init__()
                self.convs = nn.ModuleList(
                    [nn.Conv1d(channels, channels, *args, **kwargs) for i in range(n_layers)]
                )
                self.act = nn.LeakyReLU(0.2)
                self.n_layers = n_layers
            def forward(self, x):
                y = x
                for i in range(self.n_layers):
                    y = self.convs[i](y)
                    if i != self.n_layers-1:
                        y = self.act(y)
                return self.act(x + y)

        self.convw = nn.Conv1d(encoded_dim, ndf, 1, 1, 0) 
        self.convp = nn.Conv1d(n_features, ndf, 1, 1, 0)

        self.res1 = Res(ndf*2, 2, 9, 1, 4)
        self.conv1 = nn.Conv1d(ndf*2, ndf*4, 9, 1, 4)

        self.res2 = Res(ndf*4, 2, 9, 1, 4)
        self.conv2 = nn.Conv1d(ndf*4, ndf*6, 9, 1, 4)

        self.res3 = Res(ndf*6, 2, 9, 1, 4)
        self.conv3 = nn.Conv1d(ndf*6, ndf*8, 9, 1, 4)

        self.res4 = Res(ndf*8, 2, 9, 1, 4)
        self.conv4 = nn.Conv1d(ndf*8, ndf*8, 9, 1, 4)

        self.out = nn.Identity()

    
    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        p_ = x_[:,:n_features,:]
        w_ = x_[:,n_features:,:]

        # w_ (batch, encoded_dim, seq_len)
        
        w = self.convw(w_)
        p = self.convp(p_)
        
        x = torch.cat([p, w], dim=1)

        x = self.res1(x)
        x = self.conv1(x)

        x = self.res2(x)
        x = self.conv2(x)

        x = self.res3(x)
        x = self.conv3(x)

        x = self.res4(x)
        x = self.conv4(x)

        return self.out(x.mean(dim=2).mean(dim=1))


class AE(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_wires, encoded_dim*4, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(encoded_dim*4, encoded_dim*2, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(encoded_dim*2, encoded_dim, 1, 1, 0),
            nn.Tanh()
        )

        self.dec = nn.Sequential(
            nn.Conv1d(encoded_dim, encoded_dim*2, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(encoded_dim*2, encoded_dim*4, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(encoded_dim*4, n_wires, 1, 1, 0),
        )

    def forward(self, x):
        y = self.dec(self.enc(x))
        return y



def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
