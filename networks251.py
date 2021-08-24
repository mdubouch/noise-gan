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

        self.lin0 = nn.Linear(latent_dims, seq_len//8*512, bias=False)

        self.convu1 = nn.ConvTranspose1d(512, 512, 4, 2, 1)
        self.bnu1 = nn.InstanceNorm1d(512)
        self.convu2 = nn.ConvTranspose1d(512, 512, 4, 2, 1)
        self.bnu2 = nn.InstanceNorm1d(512)
        self.convu3 = nn.ConvTranspose1d(512, 512, 4, 2, 1)
        self.bnu3 = nn.Identity()#nn.InstanceNorm1d(128)

        self.convp = nn.ConvTranspose1d(512, n_features, 1, 1, 0,  bias=True)
        self.convw = nn.ConvTranspose1d(512, encoded_dim, 1, 1, 0, bias=True)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, 512, self.seq_len // 8))

        x = self.act(self.bnu1(self.convu1(x)))
        x = self.act(self.bnu2(self.convu2(x)))
        x = self.act(self.bnu3(self.convu3(x)))

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

        #self.convp1 = nn.Conv1d(n_features, ndf*2, 33, 8, 16) # // 8
        #self.convp2 = nn.Conv1d(ndf*2, ndf*2, 3, 1, 1)

        #self.convw1 = nn.Conv1d(encoded_dim, ndf*4, 33, 8, 16)
        #self.convw2 = nn.Conv1d(ndf*4, ndf*4, 3, 1, 1)

        #self.convxy1 = nn.Conv1d(2, ndf*2, 33, 8, 16)
        #self.convxy2 = nn.Conv1d(ndf*2, ndf*2, 3, 1, 1)

        self.convp = nn.Conv1d(n_features, 512, 1, 1, 0, bias=True)
        self.convw = nn.Conv1d(encoded_dim, 512, 1, 1, 0, bias=True)

        self.conv2 = nn.Conv1d(1024, 512, 4, 2, 1)
        self.conv3 = nn.Conv1d(512, 512, 4, 2, 1)
        self.conv4 = nn.Conv1d(512, 512, 4, 2, 1)
        
        self.lin0 = nn.Linear(512 * seq_len // 8, 1, bias=True)

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

        p = self.convp(p)
        w = self.convw(w)

        x = torch.cat([p, w], dim=1)

        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


class VAE(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        class Enc(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.LeakyReLU(0.2)
                self.lin1 = nn.Linear(n_wires, encoded_dim, bias=False)
                self.out = nn.Tanh()
            def forward(self, x):
                y = self.lin1(x)
                return self.out(y)


        class Dec(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.ReLU()
                self.lin1 = nn.Linear(encoded_dim, n_wires, bias=False)
            def forward(self, x):
                y = self.lin1(x)
                return y
        self.enc_net = Enc(encoded_dim*2)
        self.dec_net = Dec(encoded_dim*2)

    def enc(self, x):
        return self.enc_net(x.permute(0,2,1)).permute(0,2,1)
    def dec(self, x):
        return self.dec_net(x.permute(0,2,1)).permute(0,2,1)
    def forward(self, x):
        y = self.dec_net(self.enc_net(x.permute(0, 2, 1))).permute(0,2,1)
        return y


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
