import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of continuous features (E, t, dca)
n_features = 5
geom_dim = 3

def wire_hook(grad):
    print('wg %.2e %.2e' % (grad.abs().mean().item(), grad.std().item()))
    return grad
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
                self.norm = nn.BatchNorm1d(in_c)
                self.conv = nn.ConvTranspose1d(in_c, out_c, *args, **kwargs)
                self.act = nn.ReLU()
            def forward(self, x):
                return self.act(self.conv(self.norm(x)))
        class Double(nn.Module):
            def __init__(self, in_c, out_c, *args, **kwargs):
                super().__init__()
                self.norm = nn.BatchNorm1d(in_c)
                self.conv = nn.ConvTranspose1d(in_c, out_c, *args, **kwargs)
                self.norm2 = nn.BatchNorm1d(out_c)
                self.conv2 = nn.ConvTranspose1d(out_c, out_c, 3, 1, 1)
                self.act = nn.ReLU()
            def forward(self, x):
                return self.act(self.conv2(self.norm2(self.act(self.conv(self.norm(x))))))

        self.lin0 = nn.Linear(latent_dims, 32)
        self.lin1 = nn.Linear(32, 64)
        self.lin2 = nn.Linear(64, seq_len*32)

        self.convp = nn.ConvTranspose1d(32, n_features, 1, 1, 0, bias=True)
        self.convw = nn.ConvTranspose1d(32, n_wires, 1, 1, 0, bias=True)
        self.out = nn.Tanh()

        self.max_its = 3000
        self.temp_min = 1.0
        self.gen_it = 3000
        
    def forward(self, z):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = self.act(self.lin0(z))#.reshape(-1, self.n512, 4))
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x)).reshape(-1, 32, self.seq_len)

        w = self.convw(x)
        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p = self.convp(x)
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
        nproj = 4

        self.convw0 = nn.Conv1d(n_wires, nproj, 1, 1, 0, bias=False)

        self.lin0 = nn.Linear(seq_len * (n_features+nproj+geom_dim), 128)
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 1)

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

        x = self.act(self.lin0(x.flatten(1,2)))
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.lin3(x)

        #x = self.lin0(x.mean(2)).squeeze()
        
        return self.out(x)

def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
