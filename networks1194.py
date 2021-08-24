import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of continuous features (E, t, dca)
n_features = 3

class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len, embedding_dim, n_wires):
        super().__init__()
        
        self.latent_dims = latent_dims
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.ReLU()

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

        self.lin0 = nn.Linear(latent_dims, 256 * 4)

        self.convs = nn.Sequential(
                nn.ConvTranspose1d(256, 128, 4, 2, 1),
                nn.InstanceNorm1d(128),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 64, 4, 2, 1),
                nn.InstanceNorm1d(64),
                nn.ReLU(),
        )
        self.convp = nn.ConvTranspose1d(64, n_features, 1, 1, 0)
        self.convw = nn.ConvTranspose1d(64, embedding_dim, 1, 1, 0)
        
    def forward(self, z):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = self.lin0(z).view(z.shape[0], -1, 4)

        x = self.convs(x)
        p = self.convp(x)
        w = self.convw(x)

        return torch.tanh(p), w


class Disc(nn.Module):
    def __init__(self, ndf, seq_len, embedding_dim, n_wires):
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

        self.convp = nn.Conv1d(n_features, 32, 1, 1, 0)
        self.convw = nn.Conv1d(embedding_dim, 32, 1, 1, 0)
        self.normw = nn.LayerNorm((32,seq_len), elementwise_affine=False)
        self.normp = nn.LayerNorm((32,seq_len), elementwise_affine=False)

        self.convs = nn.Sequential(
                nn.Conv1d(64, 64, 3, 1, 1),
                nn.LayerNorm((64,seq_len), elementwise_affine=False),
                nn.LeakyReLU(0.2),
                nn.Conv1d(64, 128, 3, 2, 1),
                nn.LayerNorm((128,seq_len//2), elementwise_affine=False),
                nn.LeakyReLU(0.2),
                nn.Conv1d(128, 256, 3, 2, 1),
                nn.LayerNorm((256,seq_len//4), elementwise_affine=False),
                nn.LeakyReLU(0.2),
        )

        self.lin0 = nn.Linear(256, 1)

        self.out = nn.Identity()

    def forward(self, x_):
        seq_len = x_.shape[2]

        p = self.act(self.normp(self.convp(x_[:,:n_features])))
        w = self.act(self.normw(self.convw(x_[:,n_features:])))
        x = torch.cat([p, w], dim=1)
        x = self.convs(x)
        
        x = self.lin0(x.mean(dim=2))

        return self.out(x)

def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
