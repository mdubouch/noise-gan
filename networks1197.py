import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of continuous features (E, t, dca)
n_features = 3
geom_dim = 2

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

        n512 = 512
        self.lin0 = nn.Linear(latent_dims, seq_len//32*n512, bias=True)
        self.dropout = nn.Dropout(0.1)

        self.n512 = n512
        n256 = n512 // 2
        n128 = n512 // 4
        n64  = n512 // 8
        n32  = n512 // 16
        n16  = n512 // 32

        #class GBlock(nn.Module):
        #    def __init__(self, in_c, out_c, k_s, stride, padding, *args, **kwargs):
        #        super().__init__()
        #        self.bn0 = nn.InstanceNorm1d(in_c)
        #        self.conv = nn.ConvTranspose1d(in_c, out_c, k_s, stride, padding, *args, **kwargs)
        #        self.bn = nn.InstanceNorm1d(out_c)
        #        self.convp = nn.ConvTranspose1d(in_c + out_c, out_c, 1, 1, 0)
        #        self.act = nn.ReLU()
        #    def forward(self, x):
        #        y = self.bn0(x)
        #        y = self.conv(y)
        #        y = self.bn(self.act(y))
        #        #x0 = F.interpolate(x, size=y.shape[2], mode='nearest')
        #        y = self.act(self.conv1(torch.cat([x0, y], dim=1)))
        #        return y
        class ResBlockUp(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv1 = nn.ConvTranspose1d(in_c, out_c, 3, 2, 1, output_padding=1)
                self.conv2 = nn.ConvTranspose1d(out_c, out_c, 3, 1, 1)
                self.convp = nn.ConvTranspose1d(in_c, out_c, 2, 2, 0, bias=False)
                self.bn1 = nn.InstanceNorm1d(out_c)
                self.bn2 = nn.InstanceNorm1d(out_c)
                self.act = nn.ReLU()
            def forward(self, x):
                y = self.bn1(self.act(self.conv1(x)))
                y = self.conv2(y)
                xp = self.convp(x)
                y = self.bn2(self.act(xp + y))
                return y

        self.convu1 = ResBlockUp(n512, n512)
        self.convu2 = ResBlockUp(n512, n512)
        self.convu3 = ResBlockUp(n512, n512//2)

        # Common branch
        self.convu4 = ResBlockUp(n512//2, n512//4)
        self.convu5 = ResBlockUp(n512//4, n512//8)

        # W branch
        self.convuw1 = ResBlockUp(n512//2, n512//4)
        self.convuw2 = ResBlockUp(n512//4, n512//8)

        # P branch
        self.convup1 = ResBlockUp(n512//2, n512//4)
        self.convup2 = ResBlockUp(n512//4, n512//8)

        self.convw2 = nn.Conv1d(n512//4, n512//8, 7, 1, 3)
        self.convw1 = nn.Conv1d(n512//8, n_wires, 1, 1, 0)

        self.convp2 = nn.Conv1d(n512//4, n512//8, 3, 1, 1)
        self.convp1 = nn.Conv1d(n512//8, n_features, 7, 1, 3)

        self.out = nn.Tanh()

        self.max_its = 3000
        self.temp_min = 1.0
        self.gen_it = 3000
        
    def forward(self, z):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = self.act(self.lin0(z).reshape(-1, self.n512, self.seq_len // 32))

        x = self.convu1(x)
        x = self.convu2(x)
        x0 = self.convu3(x)

        # Common
        x = self.convu4(x0)
        x = self.convu5(x)

        # W
        w = self.convuw1(x0)
        w = self.convuw2(w)

        # P
        p = self.convup1(x0)
        p = self.convup2(p)

        w0 = torch.cat([x, w], dim=1)
        w = self.act(self.convw2(w0))
        w = self.convw1(w)

        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        #print(tau)
        wg = torch.softmax(w, dim=1)#F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p0 = torch.cat([x, p], dim=1)
        p = self.act(self.convp2(p0))
        p = self.convp1(p)

        #return torch.cat([self.out(p), xy], dim=1), wg
        return torch.tanh(p), wg

def xy_hook(grad):
    print('xy %.2e %.2e' % (grad.abs().mean().item(), grad.std().item()))
    return grad
class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim, n_wires):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        n768 = 256
        n512 = 256
        n256 = 256
        n128 = 128
        n64  = 64
        nproj = 8


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

        self.conv1 = nn.Conv1d(nproj+geom_dim+n_features, n64, 7, 1, 3, padding_mode='circular')
        self.conv2 = nn.Conv1d(n64, n64*2, 3, 1, 1, dilation=3, padding_mode='circular')
        self.conv3 = nn.Conv1d(n64*2, n64*4, 3, 1, 1, dilation=9, padding_mode='circular')
        self.conv4 = nn.Conv1d(n64*4, n64*8, 3, 1, 1, dilation=27, padding_mode='circular')
        self.conv5 = nn.Conv1d(n64*8, n64*8, 3, 1, 1, dilation=81, padding_mode='circular')
        self.conv6 = nn.Conv1d(n64*8, n64*8, 3, 1, 1, dilation=243, padding_mode='circular')
        self.conv7 = nn.Conv1d(n64*8, n64*8, 3, 1, 1, dilation=400, padding_mode='circular')

        self.lin0 = nn.Linear(n64*8, 1)
        self.dropout = nn.Dropout(0.2)

        self.out = nn.Identity()
        self.padleft = nn.ConstantPad1d((1, 0), 0.)

    def forward(self, x_):
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        #dist = ((xy - nn.ConstantPad1d((1, 0), 0.0)(xy[:,:,:-1]))**2).sum(dim=1).unsqueeze(1)
        p = x_[:,:n_features]
        xy = x_[:,n_features:n_features+geom_dim]
        wg = x_[:,n_features+geom_dim:]

        w0 = self.convw0(wg)

        x = torch.cat([w0, p, xy], dim=1)

        x = self.act(self.conv1(x))

        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x0 = self.conv4(x)
        x = self.act(self.conv5(self.act(x0)))
        x = self.act(self.conv6(x))
        x = self.act(self.conv7(x + F.interpolate(x0, size=x.shape[2], mode='area')))

        x = self.lin0(x.mean(2)).squeeze()
        
        return self.out(x)


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
