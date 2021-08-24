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

        self.lin0 = nn.Linear(latent_dims, 128 * 64)
        self.s1 = Simple(128, 128, 3, 2, 1, output_padding=1)
        self.s2 = Simple(128, 64, 3, 2, 1, output_padding=1)


        self.pbranch = nn.Sequential(
                nn.ConvTranspose1d(64, 64, 3, 1, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 64, 3, 1, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.ConvTranspose1d(64, n_features, 3, 1, 1)
        )
        self.convw = nn.ConvTranspose1d(64, n_wires, 1, 1, 0, bias=True)
        self.out = nn.Tanh()

        self.max_its = 3000
        self.temp_min = 1.0
        self.gen_it = 3000
        
    def forward(self, z):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = self.act(self.lin0(z).reshape(-1, 128, 64))

        x = self.s1(x)
        x = self.s2(x)

        w = self.convw(x)
        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p = self.pbranch(x)
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
        nproj = 16


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

        self.w0 = nn.Conv1d(nproj+geom_dim, 64, 65, 1, 32)
        self.p0 = nn.Conv1d(n_features, 64, 65, 1, 32)

        self.s1 = Simple(128, 256, 3, 2, 1)
        self.s2 = Simple(256+1, 512, 3, 2, 1)
        self.lin0 = nn.Linear(512, 1)

        self.out = nn.Identity()

    def forward(self, x_, xy_, w_): 
        seq_len = x_.shape[2]

        p = x_
        wg = w_
        xy = xy_

        occupancy = wg.sum(dim=2).var(dim=1).unsqueeze(1).unsqueeze(2) / 16.0 * 2.0 - 1.0

        w0 = self.convw0(wg)
        w = torch.cat([w0, xy], dim=1)
        w = self.act(self.w0(w))

        p = self.act(self.p0(p))

        x = torch.cat([p, w], dim=1)

        x = self.s1(x)

        xocc = torch.cat([x, occupancy.expand(-1, 1, x.shape[2])], dim=1)
        x = self.s2(xocc)

        x = self.lin0(x.mean(2)).squeeze()
        
        return self.out(x)

def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
