import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of continuous features (E, t, dca)
n_features = 3
geom_dim = 3

def wire_hook(grad):
    print('wg %.2e %.2e' % (grad.abs().mean().item(), grad.std().item()))
    return grad
class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len, encoded_dim, n_wires):
        super().__init__()
        
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.ReLU()

        n512 = 128
        self.lin0 = nn.Linear(latent_dims, seq_len//128*n512, bias=True)
        self.dropout = nn.Dropout(0.1)

        self.n512 = n512
        n256 = n512 // 2
        n128 = n512 // 4
        n64  = n512 // 8
        n32  = n512 // 16
        n16  = n512 // 32

        class Simple(nn.Module):
            def __init__(self, in_c, out_c, *args, **kwargs):
                super().__init__()
                self.norm = nn.BatchNorm1d(in_c)
                self.conv = nn.ConvTranspose1d(in_c, out_c, *args, **kwargs)
                self.act = nn.ReLU()
            def forward(self, x):
                return self.act(self.conv(self.norm(x)))

        self.convu1 = Simple(n512, n512, 5, 4, 1, output_padding=1)
        self.convu2 = Simple(n512, n512//2, 5, 4, 1, output_padding=1)
        self.convu3 = Simple(n512//2, n512//4, 3, 2, 1, output_padding=1)
        self.convu4 = Simple(n512//4, n512//8, 3, 2, 1, output_padding=1)
        self.convu5 = Simple(n512//8, n512//16, 3, 2, 1, output_padding=1)

        self.convw2 = Simple(n512//16, n512//16, 7, 1, 3)
        self.convw1 = nn.Conv1d(n512//16, n_wires, 1, 1, 0)

        self.convp1 = nn.Conv1d(n512//16, n_features, 7, 1, 3)


        self.out = nn.Tanh()

        self.max_its = 3000
        self.temp_min = 1.0
        self.gen_it = 3000
        
    def forward(self, z):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = self.act(self.lin0(z).reshape(-1, self.n512, self.seq_len // 128))

        x = self.convu1(x)
        x = self.convu2(x)
        x = self.convu3(x)
        x = self.convu4(x)
        x = self.convu5(x)

        w = self.convw2(x)
        w = self.convw1(w)

        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p = self.convp1(x)

        return self.out(p), wg

def xy_hook(grad):
    print('xy %.2e %.2e' % (grad.abs().mean().item(), grad.std().item()))
    return grad
class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim, n_wires):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        n512 = 256
        nproj = 4


        class ResBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv1d(channels, channels, 3, 1, 1, padding_mode='circular')
                self.conv2 = nn.Conv1d(channels, channels, 3, 1, 1, padding_mode='circular')
                self.act = nn.LeakyReLU(0.2)
            def forward(self, x):
                y = self.act(self.conv1(x))
                y = self.conv2(y)
                y = self.act(y + x) 
                return y
        class ResBlockDown(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv1d(channels, channels*2, 3, 2, 1, padding_mode='circular')
                self.conv2 = nn.Conv1d(channels*2, channels*2, 3, 1, 1, padding_mode='circular')
                self.convp = nn.Conv1d(channels, channels*2, 1, 2, 0, bias=False)
                self.act = nn.LeakyReLU(0.2)
            def forward(self, x):
                y = self.act(self.conv1(x))
                y = self.conv2(y)
                xp = self.convp(x)
                y = self.act(y + xp)
                return y


        self.convw0 = nn.Conv1d(n_wires, nproj, 1, 1, 0, bias=False)

        self.conv1 = nn.Conv1d(nproj+geom_dim+n_features, n512//8, 7, 1, 3, padding_mode='circular')
        self.conv2 = nn.Conv1d(n512//8, n512//4, 3, 1, 3, dilation=3, padding_mode='circular')
        self.conv3 = nn.Conv1d(n512//4, n512//2, 3, 1, 9, dilation=9, padding_mode='circular')
        self.conv4 = nn.Conv1d(n512//2, n512, 3, 1, 27, dilation=27, padding_mode='circular')
        self.conv5 = nn.Conv1d(n512,    n512, 3, 1, 81, dilation=81, padding_mode='circular')
        self.conv6 = nn.Conv1d(n512,    n512, 3, 1, 243, dilation=243, padding_mode='circular')

        self.lin0 = nn.Linear(n512, 1)

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

        x = torch.cat([w0, p, xy], dim=1)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = self.lin0(x.mean(2)).squeeze()
        
        return self.out(x)

def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
