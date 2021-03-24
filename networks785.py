import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3
geom_dim = 1

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
        self.lin0 = nn.Linear(latent_dims, seq_len//16*n512, bias=True)
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
                self.bn0 = nn.BatchNorm1d(in_c)
                self.conv = nn.ConvTranspose1d(in_c, out_c, k_s, stride, padding)
                self.bn = nn.BatchNorm1d(out_c)
                self.conv1 = nn.ConvTranspose1d(in_c + out_c, out_c, 1, 1, 0)
                self.act = nn.ReLU()
            def forward(self, x):
                y = self.bn0(x)
                y = self.conv(y)
                y = self.bn(self.act(y))
                x0 = F.interpolate(x, size=y.shape[2], mode='linear')
                y = self.act(self.conv1(torch.cat([x0, y], dim=1)))
                return y

        self.convw4 = GBlock(n512, n256, 12, 4, 4)
        self.convw3 = GBlock(n256, n128, 12, 4, 4)
        self.convw2 = GBlock(n128, n64, 3, 1, 1)
        self.convw1 = nn.Conv1d(n64, n_wires, 1, 1, 0)
        #self.linw1 = nn.Linear(n512, n256)
        #self.linw2 = nn.Linear(n256, n128)
        #self.linw3 = nn.Linear(n128, n_wires)

        #self.convp4 = nn.ConvTranspose1d(n512, n256, 12, 4, 4)
        #self.bnp4 = nn.BatchNorm1d(n256)
        #self.convp3 = nn.ConvTranspose1d(n256, n128, 12, 4, 4)
        #self.bnp3 = nn.BatchNorm1d(n512+n128)
        #self.convp2 = nn.ConvTranspose1d(n512+n128, n64, 3, 1, 1)
        #self.bnp2 = nn.BatchNorm1d(n64)
        self.convp4 = GBlock(n512, n256, 12, 4, 4)
        self.convp3 = GBlock(n256, n128, 12, 4, 4)
        self.convp2 = GBlock(n128, n64, 3, 1, 1)
        self.convp1 = nn.Conv1d(n64, n_features, 1, 1, 0)

        #self.conv1 = nn.ConvTranspose1d(n128, n128, 32, 2, 15)
        #self.bn1 = nn.BatchNorm1d(n128)

        #self.convw1 = nn.ConvTranspose1d(n128, n_wires, 1, 1, 0, bias=True)

        #self.convp1 = nn.ConvTranspose1d(n128, n_features, 1, 1, 0)

        self.out = nn.Tanh()

        self.max_its = 3000
        self.temp_min = 1.0
        self.gen_it = 3000
        
    def forward(self, z, wire_to_xy):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = self.act(self.lin0(z).reshape(-1, self.n512, self.seq_len // 16))

        #x = self.act(self.bnu1(self.convu1(x)))
        #x = self.act(self.bnu2(self.convu2(x)))
        #x = self.act(self.bnu3(self.convu3(x)))
        #x = self.act(self.bnu4(self.convu4(x)))
        #x = self.act(self.bnu5(self.convu5(x)))
        #x = self.act(self.bnu6(self.convu6(x)))

        #x = self.act(self.bn1(self.conv1(x)))
        #w = self.act(self.linw1(self.dropout(x.permute(0,2,1))))
        #w = self.act(self.linw2(w))
        #w = self.linw3(w).permute(0,2,1)
        w = self.convw4(x)
        w = self.convw3(w)
        w = self.convw2(w)
        w = self.convw1(w)
        #print(w.unsqueeze(0).shape)
        #print((w.unsqueeze(0) - wire_to_xy.view(n_wires, 1, geom_dim, 1)).shape)
        # w: (b, 2, seq)
        # wire_to_xy: (2, n_wires)
        #print(wire_to_xy.unsqueeze(0).unsqueeze(2).shape)
        #print(w.unsqueeze(3).shape)
        #import matplotlib.pyplot as plt
        #import matplotlib.lines as lines
        #plt.figure()
        #plt.scatter(w[:,0,:].detach().cpu(), w[:,1,:].detach().cpu(), s=1)
        #_l = lines.Line2D(w[:,0,:].detach().cpu(), w[:,1,:].detach().cpu(), linewidth=0.1, color='gray', alpha=0.7)
        #plt.gca().add_line(_l)
        #plt.gca().set_aspect(1.0)
        #plt.savefig('test.png')
        #plt.close()

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(w[0,:,0].detach().cpu())
        #plt.savefig('testw.png')
        #plt.close()
        #wdist = torch.norm(w.unsqueeze(3) - wire_to_xy.unsqueeze(0).unsqueeze(2), dim=1)
        #print(wdist.shape)
        ##print(1/wdist)
        #plt.figure()
        #plt.plot(wdist[0,0,:].detach().cpu())
        #plt.savefig('test.png')
        #plt.close()
        #self.gen_it += 1
        tau = 1. / ((1./self.temp_min)**(self.gen_it / self.max_its))
        #print(tau)
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=tau)
        #wg = F.softmax(w, dim=1)
        #print(wg.shape)
        #exit(1)
        #wg.register_hook(wire_hook)
        #xy = torch.tensordot(wg, wire_to_xy, dims=[[1],[1]]).permute(0,2,1)

        #p = self.act(self.bnp4(self.convp4(self.act(x))))
        #p = self.convp3(p)
        #p = torch.cat([p, F.interpolate(x, size=p.shape[2])], dim=1)
        #p = self.act(self.bnp3(p))
        #p = self.act(self.bnp2(self.convp2(p)))
        p = self.convp4(x)
        p = self.convp3(p)
        p = self.convp2(p)
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

        n512 = 4096
        n256 = n512 // 2
        n128 = n256 // 2
        n64  = n128 // 2

        self.conv0 = nn.Conv1d(geom_dim, n64, 1, 1, 0)
        self.conv1 = nn.Conv1d(n64, n128, 9, 1, 4)
        self.bn1 = nn.LayerNorm((n128))
        self.lin2 = nn.Linear(n128, 1)
        #self.conv0 = nn.Conv1d(geom_dim, n512, 1, 1, 0)
        #self.lin1 = nn.Linear(n512, n256)
        #self.lin2 = nn.Linear(n256, n128)
        ##self.conv1 = nn.Conv1d(n64, n128, 17, 2, 8, padding_mode='circular')
        ##self.conv2 = nn.Conv1d(n128, n256, 9, 2, 4, padding_mode='circular')
        ##self.conv3 = nn.Conv1d(n256, n512, 5, 2, 2, padding_mode='circular')
        ##self.conv4 = nn.Conv1d(n512, n512, 5, 4, 2, padding_mode='circular')
        ##self.conv5 = nn.Conv1d(n512, n512, 5, 4, 2, padding_mode='circular')
        #self.dropout=nn.Dropout(0.1)
        #self.lin3 = nn.Linear(n128, 1)

        #self.conv1 = nn.Conv1d(n64*1, n128, 3, 3, 1)

        #self.db1 = DBlock(n256)
        #self.db2 = DBlock(n256)
        #self.db3 = DBlock(n256)
        #self.conv2 = nn.Conv1d(256, 512, 3, 2, 1)
        #self.conv3 = nn.Conv1d(512, 1024, 3, 2, 1)
        #self.conv4 = nn.Conv1d(1024, 2048, 3, 2, 1)

        #self.lin0 = nn.Linear(256 * seq_len // 1, 1, bias=True)
        #self.lin0 = nn.Linear(seq_len//4*512, 1)
        #self.convf = nn.utils.spectral_norm(nn.Conv1d(n512, 1, 3, 1, 1, padding_mode='circular'))

        #self.lin0 = nn.Linear(n128, 1)
        #self.lin0 = nn.utils.spectral_norm(nn.Linear(n512*seq_len//32, 128))
        #self.lin1 = nn.utils.spectral_norm(nn.Linear(128, 1))

        self.out = nn.Identity()

    def forward(self, x_, w_, wire_sphere): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        x = x_
        #dist = ((xy - nn.ConstantPad1d((1, 0), 0.0)(xy[:,:,:-1]))**2).sum(dim=1).unsqueeze(1)
        p = x
        #xy = x[:,n_features:n_features+geom_dim]
        wg = w_
        #pxy = x[:,:n_features+geom_dim]
        #print(wire0)
        #print('mean %.2e %.2e' % (p.mean().item(), xy.mean().item()))
        #print('std %.2e %.2e' % (p.std().item(), xy.std().item()))

        #print('xy1 %.2e %.2e' % (xy.mean().item(), xy.std().item()))
        #print('p %.2e %.2e' %( p.abs().mean().item(), p.std().item()))
        #print('xy %.2e %.2e' %( xy.abs().mean().item(), xy.std().item()))
        #print('xy2 %.2e %.2e' % (xy.mean().item(), xy.std().item()))

        #x = torch.cat([p, xy], dim=1)
        #w0 = self.convw0(wg)
        #xy = torch.tensordot(wg, wire_sphere, dims=[[1], [1]]).permute(0,2,1)
        xy = w_

        x = xy
        #x0 = self.conv0(x)
        #x0 = torch.cat([x0 , w0], dim=1)
        #x0 = w0

        x = self.conv0(x)
        x = self.act(self.conv1(x).permute(0,2,1))

        x = self.lin2(x.sum(dim=1)).squeeze()

        #print(x.shape)

        #x = self.lin0(x.mean(dim=1))
        
        return self.out(x)#.squeeze(1)


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
