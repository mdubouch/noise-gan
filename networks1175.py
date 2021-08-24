import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of continuous features (E, t, dca)
n_features = 3
geom_dim = 2

class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len, encoded_dim, n_wires):
        super().__init__()
        
        self.latent_dims = latent_dims
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        n512 = 128
        self.n512 = n512
        n256 = n512 // 2
        n128 = n512 // 4
        n64  = n512 // 8
        n32  = n512 // 16
        n16  = n512 // 32

        class Simple(nn.Module):
            def __init__(self, in_c, out_c, *args, **kwargs):
                super().__init__()
                self.conv = nn.ConvTranspose1d(in_c, out_c, *args, **kwargs)
                self.norm = nn.BatchNorm1d(out_c)
                self.act = nn.ReLU()
            def forward(self, x):
                return self.act(self.norm(self.conv(x)))
        class Res(nn.Module):
            def __init__(self, in_c, out_c, k_s, stride, *args, **kwargs):
                super().__init__()
                self.s1 = Simple(in_c, out_c, k_s, stride, *args, **kwargs)
                self.s2 = Simple(out_c, out_c, 3, 1, 1)
                self.s3 = Simple(out_c, out_c, 3, 1, 1)
                self.conv4 = nn.ConvTranspose1d(out_c, out_c, 3, 1, 1)
                self.norm4 = nn.BatchNorm1d(out_c)
                if in_c != out_c:
                    self.convp = nn.ConvTranspose1d(in_c, out_c, 1, 1, 0)
                else:
                    self.convp = nn.Identity()
                self.interp = nn.Upsample(scale_factor=stride, mode='linear')
                self.act = nn.ReLU()
            def forward(self, x):
                y0 = self.convp(self.interp(x))
                y = self.s1(x)
                y = self.s2(y)
                y = self.s3(y)
                y = self.act(self.norm4(y0 + self.conv4(y)))
                return y

        self.out = nn.Tanh()

        self.max_its = 200000
        self.temp_min = 1e-3
        self.gen_it = 0

        self.lins = nn.Sequential(
                nn.Linear(latent_dims, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, seq_len*8)
        )
        self.convp = nn.ConvTranspose1d(8, n_features, 1, 1, 0)
        self.convw = nn.ConvTranspose1d(8, n_wires, 1, 1, 0)
        
    def forward(self, z):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = self.lins(z).view(z.shape[0], -1, self.seq_len)

        p = self.convp(x)
        w = self.convw(x)
        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        self.gen_it += 1
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        return self.out(p), wg


class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim, n_wires):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        n512 = 512
        n256 = n512//2
        n128 = n512//4
        n64 = n512//8
        n32 = n512//16
        nproj = 8


        class Simple(nn.Module):
            def __init__(self, in_c, out_c, *args, **kwargs):
                super().__init__()
                self.conv = nn.Conv1d(in_c, out_c, *args, **kwargs)
                self.act = nn.LeakyReLU(0.2)
            def forward(self, x):
                return self.act(self.conv(x))
        class Res(nn.Module):
            def __init__(self, in_c, out_c, k_s, stride, *args, **kwargs):
                super().__init__()
                self.s1 = Simple(in_c, in_c, 3, 1, 1)
                self.s2 = Simple(in_c, in_c, 3, 1, 1)
                self.s3 = Simple(in_c, in_c, 3, 1, 1)
                self.conv4 = nn.Conv1d(in_c, out_c, k_s, stride, *args, **kwargs)
                self.act = nn.LeakyReLU(0.2)
                if in_c != out_c:
                    self.convp = nn.Conv1d(in_c, out_c, 1, 1, 0)
                else:
                    self.convp = nn.Identity()
                self.interp = nn.AvgPool1d(stride)
            def forward(self, x):
                y0 = self.convp(self.interp(x))
                y = self.s1(x)
                y = self.s2(y)
                y = self.s3(y)
                y = self.act(y0 + self.conv4(y))
                return y

        self.convw0 = nn.Conv1d(n_wires, nproj, 1, 1, 0, bias=False)

        self.lins = nn.Sequential(
                nn.Linear((n_features+nproj)*seq_len, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.LeakyReLU(0.2),
        )
        self.lin0 = nn.Linear(256+1, 1)

        self.out = nn.Identity()

    def forward(self, x_, w_): 
        seq_len = x_.shape[2]

        p = x_
        wg = w_

        occupancy = wg.sum(dim=2).var(dim=1).unsqueeze(1).unsqueeze(2) / 0.015 * 2 - 1.
        #print('occ', occupancy.mean().item())

        w0 = self.convw0(wg)
        x = torch.cat([p, w0], dim=1)

        x = self.lins(x.flatten(1,2))

        #print('x', x[:,0].mean().item(), x[:,0].std().item())

        x = self.lin0(torch.cat([x, occupancy.squeeze(2)], dim=1))

        return self.out(x)

def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
