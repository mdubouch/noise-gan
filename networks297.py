import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

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
        self.act = nn.ReLU()

        self.lin0 = nn.Linear(latent_dims, seq_len//64*512, bias=True)

        class GBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.convp = nn.ConvTranspose1d(in_channels, out_channels, 1, 1, 0)
                self.convu = nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1)
                self.conv1 = nn.ConvTranspose1d(out_channels, out_channels, 3, 1, 1)
                self.bnu = nn.InstanceNorm1d(out_channels)
                self.bn1 = nn.InstanceNorm1d(out_channels)
                self.act = nn.ReLU()

            def forward(self, x):
                y0 = F.interpolate(self.convp(x), scale_factor=2, mode='linear')
                y = self.act(self.bnu(self.convu(x)))
                y = self.act(y0 + self.bn1(self.conv1(y)))
                return y

        self.gb1 = GBlock(512, 512)
        self.gb2 = GBlock(512, 512)
        self.gb3 = GBlock(512, 512)

        self.gbp1 = GBlock(512, 256)
        self.gbp2 = GBlock(256, 128)
        self.gbp3 = GBlock(128, 64)
        self.convp = nn.ConvTranspose1d(64, n_features, 1, 1, 0)

        self.gbw1 = GBlock(512, 256)
        self.gbw2 = GBlock(256, 128)
        self.gbw3 = GBlock(128, 64)
        self.convw1 = nn.ConvTranspose1d(64, 64, 3, 1, 1)
        self.convw2 = nn.ConvTranspose1d(64, 64, 3, 1, 1)
        self.convw = nn.ConvTranspose1d(64, encoded_dim, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, 512, self.seq_len // 64))

        x = self.gb1(x)
        x = self.gb2(x)
        x = self.gb3(x)
        p = x
        w = x

        p = self.gbp1(p)
        p = self.gbp2(p)
        p = self.gbp3(p)
        p = self.convp(p)

        w = self.gbw1(w)
        w = self.gbw2(w)
        w64 = self.gbw3(w)
        w = self.act(self.convw1(w64))
        w = self.act(w64 + self.convw2(w))
        w = self.convw(w)

        return self.out(p), self.out(w)

class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        class DResBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv1d(channels, channels, 3, 1, 1)
                self.conv2 = nn.Conv1d(channels, channels, 3, 1, 1)
                self.act = nn.LeakyReLU(0.2)

            def forward(self, x):
                x0 = x
                y = self.act(self.conv1(x0))
                y = self.act(x0 + self.conv2(y))
                return y

        class DBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.convd = nn.Conv1d(in_channels, out_channels, 4, 2, 1)
                self.act = nn.LeakyReLU(0.2)

            def forward(self, x):
                y = self.act(self.convd(x))
                return y

        self.convi = nn.Conv1d(n_features+encoded_dim, 128, 3, 1, 1)

        self.resw1 = DResBlock(128)
        self.convw1 = nn.Conv1d(128, 256, 3, 1, 1)
        self.resw2 = DResBlock(256)
        self.convw2 = nn.Conv1d(256, 512, 3, 1, 1)
        self.resw3 = DResBlock(512)
        self.convw3 = nn.Conv1d(512, 1024, 3, 1, 1)


        #self.db1 = DBlock(1024, 1024)
        #self.db2 = DBlock(1024, 1024)
        #self.db3 = DBlock(1024, 1024)
        
        self.lin0 = nn.Linear(1024 * seq_len // 1, 1, bias=True)
        #self.lin1 = nn.Linear(512, 1, bias=True)

        self.out = nn.Identity()

    
    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        x = x_
        p = x[:,:n_features]
        w = x[:,n_features:]
        xy = x[:,-2:]
        #dist = ((xy - nn.ConstantPad1d((1, 0), 0.0)(xy[:,:,:-1]))**2).sum(dim=1).unsqueeze(1)

        
        x = torch.cat([p, w], dim=1)
        x = self.act(self.convi(x))

        x = self.resw1(x)
        x = self.act(self.convw1(x))
        x = self.resw2(x)
        x = self.act(self.convw2(x))
        x = self.resw3(x)
        x = self.act(self.convw3(x))

        x = self.lin0(x.flatten(1,2))
        #x = self.lin1(self.act(x))
        
        return self.out(x).squeeze(1)


class VAE(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        class Enc(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.LeakyReLU(0.2)
                self.lin1 = nn.Conv1d(n_wires, hidden_size, 1, 1, 0, bias=False)
                self.lin2 = nn.Conv1d(hidden_size, encoded_dim, 1, 1, 0, bias=False)
                self.out = nn.Tanh()
            def forward(self, x):
                y = self.act(self.lin1(x))
                y = self.lin2(y)
                return self.out(y)


        class Dec(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.ReLU()
                self.lin1 = nn.Conv1d(encoded_dim, hidden_size, 1, 1, 0, bias=False)
                self.lin2 = nn.Conv1d(hidden_size, n_wires, 1, 1, 0, bias=False)
            def forward(self, x):
                y = self.lin2(self.act(self.lin1(x)))
                return y
        self.enc_net = Enc(encoded_dim*2)
        self.dec_net = Dec(encoded_dim*2)

    def enc(self, x):
        return self.enc_net(x)
    def dec(self, x):
        return self.dec_net(x)
    def forward(self, x):
        y = self.dec_net(self.enc_net(x))
        return y


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
