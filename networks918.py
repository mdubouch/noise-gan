import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3
geom_dim = 4

def wire_hook(grad):
    print('wg %.2e %.2e' % (grad.abs().mean().item(), grad.std().item()))
    return grad
class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len, encoded_dim):
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

        class GBlock(nn.Module):
            def __init__(self, in_c, out_c, k_s, stride, padding):
                super().__init__()
                self.bn0 = nn.InstanceNorm1d(in_c)
                self.conv = nn.ConvTranspose1d(in_c, out_c, k_s, stride, padding)
                self.bn = nn.InstanceNorm1d(out_c)
                self.conv1 = nn.ConvTranspose1d(in_c + out_c, out_c, 1, 1, 0)
                self.act = nn.ReLU()
            def forward(self, x):
                y = self.bn0(x)
                y = self.conv(y)
                y = self.bn(self.act(y))
                x0 = F.interpolate(x, size=y.shape[2], mode='linear')
                y = self.act(self.conv1(torch.cat([x0, y], dim=1)))
                return y

        self.convu1 = GBlock(n512, n512, 2, 2, 0)
        self.convu2 = GBlock(n512, n512, 2, 2, 0)
        self.convu3 = GBlock(n512, n512, 2, 2, 0)
        self.convu4 = GBlock(n512, n256, 2, 2, 0)
        self.convu5 = GBlock(n256, n128, 2, 2, 0)

        #self.conv1 = nn.Conv1d(n128, n64, 17, 1, 8)
        #self.bn1 = nn.InstanceNorm1d(n64)
        #self.conv2 = nn.Conv1d(n64, n64, 9, 1, 4)
        #self.bn2 = nn.InstanceNorm1d(n64)
        self.gb1 = GBlock(n128, n64, 9, 1, 4)
        self.gb2 = GBlock(n64, n64, 17, 1, 8)

        self.gbw3 = GBlock(n64, n64, 17, 1, 8)
        self.gbw2 = GBlock(n64, n64, 3, 1, 1)
        self.convw1 = nn.Conv1d(n64, n_wires, 1, 1, 0)

        self.gbp2 = GBlock(n64, n64, 3, 1, 1)
        self.convp1 = nn.Conv1d(n64, n_features, 1, 1, 0)

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
        x = self.convu3(x)
        x = self.convu4(x)
        x = self.convu5(x)

        x = self.gb1(x)
        x = self.gb2(x)

        w = self.gbw3(x)
        w = self.gbw2(w)
        w = self.convw1(w)

        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        #print(tau)
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        p = self.gbp2(x)
        p = self.convp1(p)

        #return torch.cat([self.out(p), xy], dim=1), wg
        return self.out(p), wg

def xy_hook(grad):
    print('xy %.2e %.2e' % (grad.abs().mean().item(), grad.std().item()))
    return grad
class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        n768 = 512
        n512 = 512
        n256 = 384
        n128 = 256 
        n64  = 64


        class DBlock(nn.Module):
            def __init__(self, in_c, out_c, k_s, stride, pad, *args, **kwargs):
                super().__init__()
                self.conv1 = nn.Conv1d(in_c, out_c, k_s, stride, pad, *args, **kwargs)
                self.conv2 = nn.Conv1d(in_c+out_c, out_c, 1, 1, 0)
                self.act = nn.LeakyReLU(0.2)
            def forward(self, x):
                y = self.act(self.conv1(x))
                x = F.interpolate(x, size=y.shape[2], mode='linear')
                y = self.act(self.conv2(torch.cat([x, y], dim=1)))
                return y


        self.convw0 = nn.Conv1d(n_wires, n64, 1, 1, 0, bias=False)
        self.convw1 = DBlock(n64, n64*2, 3, 1, 1, padding_mode='circular')

        self.convg0 = nn.Conv1d(geom_dim, n64, 1, 1, 0, bias=False)
        self.convg1 = DBlock(n64, n64*2, 3, 1, 1, padding_mode='circular')

        self.convp0 = nn.Conv1d(n_features, n64, 1, 1, 0, bias=False)
        self.convp1 = DBlock(n64, n64*2, 3, 1, 1, padding_mode='circular')

        self.cv0 = nn.Conv1d(n64*6, n64*6, 3, 1, 1, padding_mode='circular')
        self.conv1 = DBlock(n64*6, n64*8, 5, 4, 1, padding_mode='circular')
        self.cv1 = nn.Conv1d(n64*8, n64*8, 3, 1, 1, padding_mode='circular')
        self.conv2 = DBlock(n64*8, n64*8, 5, 4, 1, padding_mode='circular')
        self.cv2 = nn.Conv1d(n64*8, n64*8, 3, 1, 1, padding_mode='circular')
        self.conv3 = DBlock(n64*8, n64*8, 5, 4, 1, padding_mode='circular')
        self.cv3 = nn.Conv1d(n64*8, n64*8, 3, 1, 1, padding_mode='circular')

        self.lin0 = nn.Linear(n64*8, 1)
        self.dropout = nn.Dropout(0.1)

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

        p0 = self.convp0(p)
        w0 = self.convw0(wg)
        g0 = self.convg0(xy)

        w0 = self.convw1(w0)
        g0 = self.convg1(g0)
        p0 = self.convp1(p0)

        x0 = torch.cat([p0, w0, g0], dim=1)

        x = self.act(self.cv0(self.dropout(x0)))
        x = self.conv1(self.dropout(x))
        x = self.act(self.cv1(x))
        x = self.conv2(x)
        x = self.act(self.cv2(x))
        x = self.conv3(x)
        x = self.act(self.cv3(x))

        x = self.lin0(x.mean(2)).squeeze()
        
        return self.out(x)


class VAE(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        class Enc(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.LeakyReLU(0.2)
                self.lin1 = nn.Linear(n_wires, hidden_size)
                self.lin2 = nn.Linear(hidden_size, encoded_dim)
                self.out = nn.Tanh()
            def forward(self, x):
                x = self.act(self.lin1(x))
                return self.out(self.lin2(x))


        class Dec(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.act = nn.ReLU()
                self.lin1 = nn.Linear(encoded_dim, hidden_size)
                self.lin2 = nn.Linear(hidden_size, n_wires)
            def forward(self, x):
                x = self.act(self.lin1(x))
                return self.lin2(x)
        self.enc_net = Enc(512)
        self.dec_net = Dec(512)

    def enc(self, x):
        return self.enc_net(x.permute(0, 2, 1)).permute(0,2,1)
    def dec(self, x):
        return self.dec_net(x.permute(0, 2, 1)).permute(0,2,1)
    def forward(self, x):
        y = self.dec_net(self.enc_net(x))
        return y


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
