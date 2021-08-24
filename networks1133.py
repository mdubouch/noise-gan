import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of continuous features (E, t, dca)
n_features = 5
geom_dim = 3

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
                return self.norm(self.act(self.conv(x)))
        class Res(nn.Module):
            def __init__(self, in_c, out_c, k_s, stride, *args, **kwargs):
                super().__init__()
                self.conv1 = nn.ConvTranspose1d(in_c, out_c, k_s, stride, *args, **kwargs)
                self.norm1 = nn.BatchNorm1d(out_c)
                if in_c != out_c:
                    self.conv2 = nn.ConvTranspose1d(in_c, out_c, 1, 1, 0)
                else:
                    self.conv2 = nn.Identity()
                self.interp = nn.Upsample(scale_factor=stride)
                self.act = nn.ReLU()
            def forward(self, x):
                y0 = self.conv2(self.interp(x))
                y = self.norm1(self.act(y0 + self.conv1(x)))
                return y

        self.s1 = Res(latent_dims, 512, 5, 4, 1, output_padding=1)
        self.s2 = Res(512, 512, 3, 2, 1, output_padding=1)
        self.s3 = Res(512, 256, 3, 2, 1, output_padding=1)
        self.s4 = Res(256, 128, 3, 2, 1, output_padding=1)
        self.s5 = Res(128, 128, 3, 2, 1, output_padding=1)
        self.s6 = Res(128, 128, 3, 2, 1, output_padding=1)
        self.s7 = Res(128, 128, 3, 2, 1, output_padding=1)

        self.sw1 = Res(128, 128, 3, 2, 1, output_padding=1)
        self.sw2 = Res(128, 128, 3, 2, 1, output_padding=1)
        self.sw3 = Res(128, 64, 3, 2, 1, output_padding=1)

        self.sp1 = Res(128, 128, 3, 2, 1, output_padding=1)
        self.sp2 = Res(128, 128, 3, 2, 1, output_padding=1)
        self.sp3 = Res(128, 64, 3, 2, 1, output_padding=1)

        self.convp = nn.ConvTranspose1d(64, n_features, 1, 1, 0, bias=True)
        self.convw = nn.ConvTranspose1d(64, n_wires, 1, 1, 0, bias=True)
        self.out = nn.Tanh()

        self.max_its = 3000
        self.temp_min = 1.0
        self.gen_it = 3000
        
    def forward(self, z):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = z.reshape(-1, self.latent_dims, 1)

        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.s6(x)
        x = self.s7(x)

        w = self.sw1(x)
        w = self.sw2(w)
        w = self.sw3(w)

        p = self.sp1(x)
        p = self.sp2(p)
        p = self.sp3(p)

        w = self.convw(w)
        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p = self.convp(p)
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
        nproj = 256


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
                self.conv1 = nn.Conv1d(in_c, out_c, k_s, stride, *args, **kwargs)
                self.act = nn.LeakyReLU(0.2)
                if in_c != out_c:
                    self.conv2 = nn.Conv1d(in_c, out_c, 1, 1, 0)
                else:
                    self.conv2 = nn.Identity()
                self.interp = nn.AvgPool1d(stride)
            def forward(self, x):
                y0 = self.conv2(self.interp(x))
                y = self.act(y0 + self.conv1(x))
                return y

        self.convw0 = nn.Conv1d(n_wires, nproj, 1, 1, 0, bias=False)

        self.s1 = Res(n_features+nproj+geom_dim, 64, 3, 1, 1, padding_mode='circular')
        self.s2 = Res(64, 128, 3, 2, 1)
        self.s3 = Res(128, 128, 3, 2, 1)
        self.s4 = Res(128, 128, 3, 2, 1)
        self.s5 = Res(128, 128, 3, 2, 1)
        self.s6 = Res(128, 128, 3, 2, 1)
        self.s7 = Res(128, 256, 3, 2, 1)
        self.s8 = Res(256, 512, 3, 2, 1)
        self.s9 = Res(512, 512, 3, 2, 1)
        self.s10 = Res(512, 512, 3, 2, 1)
        self.lin0 = nn.Linear(512, 1)

        self.out = nn.Identity()

    def forward(self, x_, xy_, w_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        #dist = ((xy - nn.ConstantPad1d((1, 0), 0.0)(xy[:,:,:-1]))**2).sum(dim=1).unsqueeze(1)
        p = x_
        #xy = x[:,n_features:n_features+geom_dim]
        wg = w_

        #xy = torch.tensordot(wg, wire_sphere+torch.randn_like(wire_sphere) * 0.01, dims=[[1], [1]]).permute(0,2,1)
        xy = xy_


        w0 = self.convw0(wg)

        x = torch.cat([w0, xy, p], dim=1)

        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.s6(x)
        x = self.s7(x)
        x = self.s8(x)
        x = self.s9(x)
        x = self.s10(x)

        x = self.lin0(x.mean(2)).squeeze()
        
        return self.out(x)

def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
