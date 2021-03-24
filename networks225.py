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

        self.lin0 = nn.Linear(latent_dims, seq_len//1*ngf*1, bias=True)


        self.conv0 = nn.Conv1d(ngf*1, ngf*1, 1, 1, 0)
        self.bn0 = nn.InstanceNorm1d(ngf*1)
        self.conv1 = nn.ConvTranspose1d(ngf*1, ngf*1, 1, 1, 0)
        self.bn1 = nn.InstanceNorm1d(ngf*1)

        self.convp = nn.Conv1d(ngf, n_features, 1, 1, 0)
        self.convw = nn.Conv1d(ngf, encoded_dim, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, self.ngf*1, self.seq_len // 1))

        # x (batch, ngf, len)
        x = self.act(self.bn0(self.conv0(x)))
        x = self.act(self.bn1(self.conv1(x)))

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

        self.convp = nn.Conv1d(n_features, ndf*1, 1, 1, 0)
        self.convw = nn.Conv1d(encoded_dim, ndf*1, 1, 1, 0)
        self.conv2 = nn.Conv1d(ndf*2, ndf*1, 1, 1, 0)
        self.conv3 = nn.Conv1d(ndf*1, ndf*4, 1, 1, 0)
        
        self.lin0 = nn.Linear(ndf*4 * seq_len // 1, 1, bias=False)

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

        #p = self.act(self.convp1(p))
        #p = self.convp2(p)

        #w = self.act(self.convw1(w))
        #w = self.convw2(w)

        #xy = self.act(self.convxy1(xy))
        #xy = self.convxy2(xy)

        p = self.act(self.convp(p))
        w = self.act(self.convw(w))

        x = torch.cat([p, w], dim=1)
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        #dist = self.act(self.convdist1(dist))
        #dist = self.act(self.convdist2(dist))
        #dist = self.act(self.convdist3(dist))
        #dist = self.convdist4(dist)

        #print('w %.2f %.2f'% (w.mean().item(), w.var().item()))
        #print('p %.2f %.2f'% (p.mean().item(), p.var().item()))
        #print('xy %.2f %.2f'% (xy.mean().item(), xy.var().item()))
        #print('dist %.2f %.2f'% (dist.mean().item(), dist.var().item()))

        #x = self.act(torch.cat([p, w, xy], dim=1))

        x = self.lin0(x.flatten(1,2))
        
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
                y = self.lin1(x)
                return self.out(y)


        class Dec(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.ReLU()
                self.lin1 = nn.Linear(encoded_dim, n_wires)
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
