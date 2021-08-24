import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3
geom_dim = 3

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

        n512 = 2048
        self.lin0 = nn.Linear(latent_dims, seq_len//64*n512, bias=True)
        self.bn0 = nn.BatchNorm1d(n512)

        self.n512 = n512
        self.convu1 = nn.ConvTranspose1d(n512, n512, 3, 2, 1, output_padding=1)
        self.bnu1 = nn.BatchNorm1d(n512)
        n256 = n512 // 2
        self.convu2 = nn.ConvTranspose1d(n512, n512, 3, 2, 1, output_padding=1)
        self.bnu2 = nn.BatchNorm1d(n512)
        n128 = n512 // 4
        self.convu3 = nn.ConvTranspose1d(n512, n256, 3, 2, 1, output_padding=1)
        self.bnu3 = nn.BatchNorm1d(n256)

        n64 = n512 // 8
        self.conv1 = nn.Conv1d(n256, n128, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(n128)

        n32 = n512 // 16
        n16 = n512 // 32
        self.convw1 = nn.ConvTranspose1d(n128, n64, 5, 2, 2, output_padding=1)
        self.bnw1 = nn.BatchNorm1d(n64)
        self.convw2 = nn.ConvTranspose1d(n64, n32, 5, 2, 2, output_padding=1)
        self.bnw2 = nn.BatchNorm1d(n32)
        self.convw3 = nn.ConvTranspose1d(n32, n16, 2, 2, 0)
        self.bnw3 = nn.BatchNorm1d(n16)
        self.convw4 = nn.Conv1d(n16, n_wires, 1, 1, 0, bias=True)

        self.convp1 = nn.ConvTranspose1d(n128, n64, 3, 2, 1, output_padding=1)
        self.bnp1 = nn.BatchNorm1d(n64)
        self.convp2 = nn.ConvTranspose1d(n64, n32, 3, 2, 1, output_padding=1)
        self.bnp2 = nn.BatchNorm1d(n32)
        self.convp3 = nn.ConvTranspose1d(n32, n16, 2, 2, 0)
        self.bnp3 = nn.BatchNorm1d(n16)
        self.convp4 = nn.Conv1d(n16, n_features, 1, 1, 0)

        self.out = nn.Tanh()

        self.max_its = 3000
        self.temp_min = 0.75
        self.gen_it = 3000
        
    def forward(self, z, wire_to_xy):
        #print('latent space %.2e %.2e' % (z.mean().item(), z.std().item()))
        # z: random point in latent space
        x = self.act(self.bn0(self.lin0(z).view(-1, self.n512, self.seq_len // 64)))

        x = self.act(self.bnu1(self.convu1(x)))
        x = self.act(self.bnu2(self.convu2(x)))
        x = self.act(self.bnu3(self.convu3(x)))

        x = self.act(self.bn1(self.conv1(x)))

        w = self.act(self.bnw1(self.convw1(x)))
        w = self.act(self.bnw2(self.convw2(w)))
        w = self.act(self.bnw3(self.convw3(w)))
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

        w = self.convw4(w)
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
        #wg = F.softmax(w / 10., dim=1)
        #print(wg.shape)
        #exit(1)
        #wg.register_hook(wire_hook)
        #xy = torch.tensordot(wg, wire_to_xy, dims=[[1],[1]]).permute(0,2,1)

        p = self.act(self.bnp1(self.convp1(x)))
        p = self.act(self.bnp2(self.convp2(p)))
        p = self.act(self.bnp3(self.convp3(p)))
        p = self.convp4(p)

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

        class DBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.utils.spectral_norm(nn.Conv1d(channels, channels, 3, 1, 1))
                self.conv2 = nn.utils.spectral_norm(nn.Conv1d(channels, channels, 3, 1, 1))
                self.act = nn.LeakyReLU(0.2)
                self.pool = nn.MaxPool1d(2)

            def forward(self, x):
                y = self.conv1(self.act(x))
                y = x + self.conv2(self.act(y))
                y = self.pool(y)
                return y

        n64 = 16
        n128 = n64  * 2
        n256 = n128 * 2
        n512 = n256 * 2

        self.conv0 = nn.utils.spectral_norm(nn.Conv1d(geom_dim, n64, 64, 2, 0))

        self.conv1 = nn.utils.spectral_norm(nn.Conv1d(n64, n128, 32, 2, 0))

        self.conv2 = nn.utils.spectral_norm(nn.Conv1d(n128, n256, 16, 2, 0))
        self.conv3 = nn.utils.spectral_norm(nn.Conv1d(n256, n512, 8, 2, 0))
        self.conv4 = nn.utils.spectral_norm(nn.Conv1d(n512, n512, 8, 2, 0))
        self.conv5 = nn.utils.spectral_norm(nn.Conv1d(n512, n512, 8, 2, 0))
        #self.db1 = DBlock(n256)
        #self.db2 = DBlock(n256)
        #self.db3 = DBlock(n256)
        #self.conv2 = nn.Conv1d(256, 512, 3, 2, 1)
        #self.conv3 = nn.Conv1d(512, 1024, 3, 2, 1)
        #self.conv4 = nn.Conv1d(1024, 2048, 3, 2, 1)

        #self.lin0 = nn.Linear(256 * seq_len // 1, 1, bias=True)
        #self.lin0 = nn.Linear(seq_len//4*512, 1)
        #self.convf = nn.utils.spectral_norm(nn.Conv1d(n512, 1, 3, 1, 1, padding_mode='circular'))

        self.lin0 = nn.utils.spectral_norm(nn.Linear(n512, 1))
        #self.lin0 = nn.utils.spectral_norm(nn.Linear(n512*seq_len//32, 128))
        #self.lin1 = nn.utils.spectral_norm(nn.Linear(128, 1))

        self.out = nn.Identity()

    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        x = x_
        #dist = ((xy - nn.ConstantPad1d((1, 0), 0.0)(xy[:,:,:-1]))**2).sum(dim=1).unsqueeze(1)
        p = x[:,:n_features]
        xy = x[:,n_features:n_features+geom_dim]
        wg = x[:,n_features+geom_dim:]
        pxy = x[:,:n_features+geom_dim]
        p = p
        xy = xy
        wg = wg
        #print(wire0)
        #print('mean %.2e %.2e' % (p.mean().item(), xy.mean().item()))
        #print('std %.2e %.2e' % (p.std().item(), xy.std().item()))

        #print('xy1 %.2e %.2e' % (xy.mean().item(), xy.std().item()))
        print('p %.2e %.2e' %( p.abs().mean().item(), p.std().item()))
        print('xy %.2e %.2e' %( xy.abs().mean().item(), xy.std().item()))
        #print('xy2 %.2e %.2e' % (xy.mean().item(), xy.std().item()))

        #x = torch.cat([p, xy], dim=1)
        x = xy
        x0 = self.conv0(x)

        x1 = self.conv1(self.act(x0))
        x2 = self.conv2(self.act(x1))
        x3 = self.conv3(self.act(x2))

        x = self.act(self.conv4(self.act(x3)))
        x = self.act(self.conv5(x))

        x = self.lin0(x.mean(2))
        
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
