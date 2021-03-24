import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 156

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

        self.lin0 = nn.Linear(latent_dims, seq_len//2048*ngf*8, bias=True)

        self.ups = nn.Upsample(scale_factor=4)

        self.conv1 = nn.ConvTranspose1d(ngf*8, ngf*8, 32, 4, 14)
        self.conv2 = nn.ConvTranspose1d(ngf*8, ngf*8, 32, 8, 12)
        self.conv3 = nn.ConvTranspose1d(ngf*8, ngf*8, 32, 8, 12)
        self.conv4 = nn.ConvTranspose1d(ngf*8, ngf*8, 32, 8, 12)
        self.bn1 = nn.InstanceNorm1d(ngf*8)
        self.bn2 = nn.InstanceNorm1d(ngf*8)
        self.bn3 = nn.InstanceNorm1d(ngf*8)
        self.bn4 = nn.InstanceNorm1d(ngf*8)

        self.conv7 = nn.Conv1d(ngf*8, ngf*8, 1, 1, 0)
        self.bn7 = nn.InstanceNorm1d(ngf*8)

        self.convp1 = nn.ConvTranspose1d(ngf*8, ngf*2, 513, 1, 256)
        self.convp2 = nn.ConvTranspose1d(ngf*2, ngf*1, 1, 1, 0)
        self.convp3 = nn.ConvTranspose1d(ngf*1, n_features, 1, 1, 0)

        self.convw1 = nn.ConvTranspose1d(ngf*8, ngf*4, 513, 1, 256)
        self.convw2 = nn.ConvTranspose1d(ngf*4, ngf*1, 1, 1, 0)
        self.convw3 = nn.ConvTranspose1d(ngf*1, encoded_dim, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, self.ngf*8, self.seq_len // 2048))

        x = self.act(self.bn1(self.conv1(x)))
        
        x = self.act(self.bn2(self.conv2(x)))

        x = self.act(self.bn3(self.conv3(x)))
        
        x = self.act(self.bn4(self.conv4(x)))
        
        # x (batch, ngf, len)
        x = self.act(self.bn7(self.conv7(x)))

        p = self.act(self.convp1(x))
        p = self.act(self.convp2(p))
        p = self.convp3(p)

        w = self.act(self.convw1(x))
        w = self.act(self.convw2(w))
        w = self.convw3(w)

        return self.out(p), self.out(w)

class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.dropout = nn.Dropout(0.05)

        self.convp1 = nn.Conv1d(n_features, ndf*2, 33, 8, 16) # // 8
        self.convp2 = nn.Conv1d(ndf*2, ndf*2, 1, 1, 0)
        self.convp3 = nn.Conv1d(ndf*2, ndf*4, 33, 8, 16)
        self.convp4 = nn.Conv1d(ndf*4, ndf*4, 1, 1, 0)

        self.convw1 = nn.Conv1d(encoded_dim, ndf*4, 33, 8, 16)
        self.convw2 = nn.Conv1d(ndf*4, ndf*4, 1, 1, 0)
        self.convw3 = nn.Conv1d(ndf*4, ndf*8, 33, 8, 16)
        self.convw4 = nn.Conv1d(ndf*8, ndf*8, 1, 1, 0)

        self.convxy1 = nn.Conv1d(2, ndf*2, 33, 8, 16)
        self.convxy2 = nn.Conv1d(ndf*2, ndf*2, 1, 1, 0)
        self.convxy3 = nn.Conv1d(ndf*2, ndf*4, 33, 8, 16)
        self.convxy4 = nn.Conv1d(ndf*4, ndf*4, 1, 1, 0)
        
        self.convdist1 = nn.Conv1d(1, ndf*2, 33, 8, 16)
        self.convdist2 = nn.Conv1d(ndf*2, ndf*2, 1, 1, 0)
        self.convdist3 = nn.Conv1d(ndf*2, ndf*2, 33, 8, 16)
        self.convdist4 = nn.Conv1d(ndf*2, ndf*2, 1, 1, 0)

        self.conv1 = nn.Conv1d(ndf*18, ndf*12, 1, 1, 0)
        self.conv2 = nn.Conv1d(ndf*12, ndf*8, 1, 1, 0)

        self.lin0 = nn.Linear(ndf*8 * seq_len // 8 // 8, 1, bias=True)

        self.out = nn.Identity()

    
    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        x = x_
        p = x[:,:n_features]
        w = x[:,n_features:-2]
        xy = x[:,-2:]
        dist = ((xy - nn.ConstantPad1d((1, 0), 0.0)(xy[:,:,:-1]))**2).sum(dim=1).unsqueeze(1)

        p = self.act(self.convp1(p))
        p = self.act(self.convp2(p))
        p = self.act(self.convp3(p))
        p = self.convp4(p)

        w = self.act(self.convw1(w))
        w = self.act(self.convw2(w))
        w = self.act(self.convw3(w))
        w = self.convw4(w)

        xy = self.act(self.convxy1(xy))
        xy = self.act(self.convxy2(xy))
        xy = self.act(self.convxy3(xy))
        xy = self.convxy4(xy)

        dist = self.act(self.convdist1(dist))
        dist = self.act(self.convdist2(dist))
        dist = self.act(self.convdist3(dist))
        dist = self.convdist4(dist)

        print('w %.2f %.2f'% (w.mean().item(), w.var().item()))
        print('p %.2f %.2f'% (p.mean().item(), p.var().item()))
        print('xy %.2f %.2f'% (xy.mean().item(), xy.var().item()))
        print('dist %.2f %.2f'% (dist.mean().item(), dist.var().item()))

        x = self.act(torch.cat([p, w, xy, dist], dim=1))

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


class VAE(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        class Enc(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.Tanh()
                self.lin1 = nn.Linear(n_wires, hidden_size)
                self.lin_mu = nn.Linear(hidden_size, encoded_dim)
                self.lin_logvar = nn.Linear(hidden_size, encoded_dim)
            def forward(self, x):
                y = self.act(self.lin1(x)) 
                mu = self.lin_mu(y)
                logvar = self.lin_logvar(y)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                out = mu + eps * std
                self.kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(),
                    dim=1), dim=0)
                return self.act(out)


        class Dec(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.ReLU()
                self.lin1 = nn.Linear(encoded_dim, hidden_size)
                self.lin2 = nn.Linear(hidden_size, n_wires)
            def forward(self, x):
                y = self.lin2(self.act(self.lin1(x)))
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
