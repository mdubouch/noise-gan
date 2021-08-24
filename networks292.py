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

        self.dropout = nn.Dropout(0.05)

        self.lin0 = nn.Linear(latent_dims, seq_len//32*512, bias=True)

        class GBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.convu = nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1)
                self.conv1 = nn.ConvTranspose1d(out_channels, out_channels, 3, 1, 1)
                self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, 3, 1, 1)
                self.bnu = nn.InstanceNorm1d(out_channels)
                self.bn1 = nn.InstanceNorm1d(out_channels)
                self.bn2 = nn.InstanceNorm1d(out_channels)
                self.act = nn.ReLU()

            def forward(self, x):
                y0 = self.act(self.bnu(self.convu(x)))
                y = self.act(self.bn1(self.conv1(y0)))
                y = self.act(y0 + self.bn2(self.conv2(y)))
                return y
        self.gb1 = GBlock(512, 512)
        self.gb2 = GBlock(512, 512)
        self.gb3 = GBlock(512, 256)
        self.gb4 = GBlock(256, 128)
        self.gb5 = GBlock(128, 64)

        self.convp = nn.ConvTranspose1d(64, n_features, 1, 1, 0)
        self.convw = nn.ConvTranspose1d(64, encoded_dim, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, 512, self.seq_len // 32))

        x = self.gb1(x) 
        x = self.gb2(x) 
        x = self.gb3(x) 
        x = self.gb4(x) 
        x = self.gb5(x) 

        p = self.convp(x)
        w = self.convw(x)

        return self.out(p), self.out(w)
class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        self.dropout = nn.Dropout(0.05)

        self.convp1 = nn.Conv1d(n_features+encoded_dim, 128, 1, 1, 0)
        self.convp2 = nn.Conv1d(128, 128, 1, 1, 0)
        self.convp3 = nn.Conv1d(128, 128, 1, 1, 0)
        self.convp4 = nn.Conv1d(128, 128, 1, 1, 0)

        self.conv1 = nn.Conv1d(128, 256, 4, 2, 1)
        self.conv2 = nn.Conv1d(256, 512, 4, 2, 1)
        self.conv3 = nn.Conv1d(512, 512, 4, 2, 1)
        self.conv4 = nn.Conv1d(512, 512, 4, 2, 1)
        
        self.lin0 = nn.Linear(512 * seq_len // 16, 128, bias=True)
        self.lin1 = nn.Linear(128, 1, bias=True)

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

        p = torch.cat([p, w], dim=1)

        p = self.act(self.convp1(p))
        p0 = p
        p = self.act(self.convp2(p))
        p = self.act(p0 + self.convp3(p))
        p = self.act(self.convp4(p))

        x = p

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

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
                self.lin1 = nn.Conv1d(n_wires, encoded_dim, 1, 1, 0, bias=False)
                self.out = nn.Tanh()
            def forward(self, x):
                y = self.lin1(x)
                return self.out(y)


        class Dec(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.ReLU()
                self.lin1 = nn.Conv1d(encoded_dim, n_wires, 1, 1, 0, bias=False)
            def forward(self, x):
                y = self.lin1(x)
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