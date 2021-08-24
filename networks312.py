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
        self.encoded_dim = encoded_dim

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

        self.gb1 = nn.Linear(seq_len//64 * 512, seq_len//1 * (n_features+encoded_dim))

        self.out = nn.Tanh()
        
    def forward(self, z):
        # z: random point in latent space
        x = self.act(self.lin0(z))

        x = self.gb1(x).view(-1, n_features+self.encoded_dim, self.seq_len // 1)

        return self.out(x)

class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        class DBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.convd = nn.Conv1d(in_channels, out_channels, 4, 2, 1)
                self.act = nn.LeakyReLU(0.2)

            def forward(self, x):
                y = self.act(self.convd(x))
                return y

        self.convp = nn.Conv1d(n_features, 64, 1, 1, 0)
        self.convw = nn.Conv1d(encoded_dim, 64, 1, 1, 0)

        self.dbp1 = DBlock(64, 128)
        self.dbp2 = DBlock(128, 256)
        self.dbp3 = DBlock(256, 512)

        self.dbw1 = DBlock(64, 128)
        self.dbw2 = DBlock(128, 256)
        self.dbw3 = DBlock(256, 512)

        self.db1 = DBlock(512, 512)
        self.db2 = DBlock(512, 512)
        self.db3 = DBlock(512, 512)
        
        self.lin0 = nn.Linear(512 * seq_len // 64, 512, bias=True)
        self.lin1 = nn.Linear(512, 1, bias=True)

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

        p = self.act(self.convp(p))
        w = self.act(self.convw(w))

        p = self.dbp1(p)
        p = self.dbp2(p)
        p = self.dbp3(p)
        w = self.dbw1(w)
        w = self.dbw2(w)
        w = self.dbw3(w)

        #x = F.dropout(torch.cat([p, w], dim=1), p=0.5)
        x = p + w

        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)

        x = self.lin0(x.flatten(1,2))
        x = self.lin1(self.act(x))
        
        return self.out(x).squeeze(1)


class VAE(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        class Enc(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.LeakyReLU(0.2)
                self.lin1 = nn.Linear(n_wires, encoded_dim)
                self.out = nn.Tanh()
            def forward(self, x):
                return self.out(self.lin1(x))


        class Dec(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.ReLU()
                self.lin1 = nn.Linear(encoded_dim, n_wires)
            def forward(self, x):
                return self.lin1(x)
        self.enc_net = Enc(encoded_dim*2)
        self.dec_net = Dec(encoded_dim*2)

    def enc(self, x):
        return self.enc_net(x.permute(0, 2, 1)).permute(0,2,1)
    def dec(self, x):
        return self.dec_net(x.permute(0, 2, 1)).permute(0,2,1)
    def forward(self, x):
        y = self.dec_net(self.enc_net(x))
        return y


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())