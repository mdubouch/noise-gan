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
            def __init__(self, in_c, out_c, k_s, stride, padding, *args, **kwargs):
                super().__init__()
                self.bn0 = nn.InstanceNorm1d(in_c)
                self.conv = nn.ConvTranspose1d(in_c, out_c, k_s, stride, padding, *args, **kwargs)
                self.bn = nn.InstanceNorm1d(out_c)
                self.conv1 = nn.ConvTranspose1d(in_c + out_c, out_c, 1, 1, 0)
                self.act = nn.ReLU()
            def forward(self, x):
                y = self.bn0(x)
                y = self.conv(y)
                y = self.bn(self.act(y))
                x0 = F.interpolate(x, size=y.shape[2], mode='nearest')
                y = self.act(self.conv1(torch.cat([x0, y], dim=1)))
                return y

        self.convu1 = GBlock(n512, n512, 3, 2, 1, output_padding=1)
        self.convu2 = GBlock(n512, n512, 3, 2, 1, output_padding=1)
        self.convu3 = GBlock(n512, n512, 3, 2, 1, output_padding=1)

        # Common branch
        self.convu4 = GBlock(n512, n256, 3, 2, 1, output_padding=1)
        self.convu5 = GBlock(n256, n128, 3, 2, 1, output_padding=1)

        # W branch
        self.convuw1 = GBlock(n512, n256, 3, 2, 1, output_padding=1)
        self.convuw2 = GBlock(n256, n128, 3, 2, 1, output_padding=1)

        # P branch
        self.convup1 = GBlock(n512, n256, 3, 2, 1, output_padding=1)
        self.convup2 = GBlock(n256, n128, 3, 2, 1, output_padding=1)

        self.gbw2 = nn.Conv1d(n128*2, n64, 3, 1, 1)
        self.convw1 = nn.Conv1d(n64, n_wires, 1, 1, 0)

        self.gbp2 = nn.Conv1d(n128*2, n64, 3, 1, 1)
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

        # Concatenate Common and W
        w0 = torch.cat([x, w], dim=1)
        w = self.gbw2(self.dropout(w0))
        w = self.convw1(w)

        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        #print(tau)
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)

        # Concatenate Common and P
        p0 = torch.cat([x, p], dim=1)
        p = self.gbp2(self.dropout(p0))
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

        n768 = 256
        n512 = 256
        n256 = 256
        n128 = 128
        n64  = 64
        nproj = 16


        class DBlock(nn.Module):
            def __init__(self, in_c, out_c, k_s, stride, pad, *args, **kwargs):
                super().__init__()
                self.conv1 = nn.Conv1d(in_c, out_c, k_s, stride, pad, *args, **kwargs)
                self.conv2 = nn.Conv1d(in_c+out_c, out_c, 1, 1, 0)
                self.act = nn.LeakyReLU(0.2)
            def forward(self, x):
                y = self.act(self.conv1(x))
                x = F.interpolate(x, size=y.shape[2], mode='nearest')
                y = self.act(self.conv2(torch.cat([x, y], dim=1)))
                return y


        self.convw0 = nn.Conv1d(n_wires, nproj, 1, 1, 0)
        self.convw1 = nn.Conv1d(nproj, n64, 3, 1, 1, padding_mode='circular')
        self.convw2 = DBlock(n64, n128, 3, 2, 1, padding_mode='circular')
        self.convw3 = DBlock(n128, n256, 3, 2, 1, padding_mode='circular')

        self.convg0 = nn.Conv1d(geom_dim, nproj, 1, 1, 0)
        self.convg1 = nn.Conv1d(nproj, n64, 3, 1, 1, padding_mode='circular')
        self.convg2 = DBlock(n64, n128, 3, 2, 1, padding_mode='circular')
        self.convg3 = DBlock(n128, n256, 3, 2, 1, padding_mode='circular')

        self.convp0 = nn.Conv1d(n_features, nproj, 1, 1, 0)
        self.convp1 = nn.Conv1d(nproj, n64, 3, 1, 1, padding_mode='circular')
        self.convp2 = DBlock(n64, n128, 3, 2, 1, padding_mode='circular')
        self.convp3 = DBlock(n128, n256, 3, 2, 1, padding_mode='circular')

        # Concatenated branch
        ncat = nproj*3
        self.conv1 = nn.Conv1d(ncat, ncat*2, 3, 1, 1, padding_mode='circular')
        self.conv2 = DBlock(ncat*2, ncat*4, 3, 2, 1, padding_mode='circular')
        self.conv3 = DBlock(ncat*4, ncat*8, 3, 2, 1, padding_mode='circular')

        # Common branch
        ncom = n256*3 + ncat*8
        self.convc1 = nn.Conv1d(ncom, ncom, 3, 2, 1)
        self.convc2 = nn.Conv1d(ncom, ncom, 3, 2, 1)
        self.convc3 = nn.Conv1d(ncom, ncom, 3, 2, 1)
        self.lin0 = nn.Linear(ncom, 1)
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
        p = self.act(self.convp1(self.dropout(p0)))
        p = self.act(self.convp2(p))
        p = self.act(self.convp3(p))

        w0 = self.convw0(wg)
        w = self.act(self.convw1(self.dropout(w0)))
        w = self.act(self.convw2(w))
        w = self.act(self.convw3(w))

        g0 = self.convg0(xy)
        g = self.act(self.convg1(self.dropout(g0)))
        g = self.act(self.convg2(g))
        g = self.act(self.convg3(g))

        cat0 = torch.cat([p0, w0, g0], dim=1)
        cat = self.act(self.conv1(self.dropout(cat0)))
        cat = self.act(self.conv2(cat))
        cat = self.act(self.conv3(cat))

        x = torch.cat([p, w, g, cat], dim=1)
        x = self.dropout(x)
        x = self.act(self.convc1(x))
        x = self.act(self.convc2(x))
        x = self.act(self.convc3(x))

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
