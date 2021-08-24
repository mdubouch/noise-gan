import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__version__ = 205

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3

class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len, encoded_dim):
        super().__init__()
        
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.ReLU()

        self.lin0 = nn.Linear(latent_dims, seq_len//8*1024, bias=True)

        class GBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.convp = nn.ConvTranspose1d(in_channels, out_channels, 1, 1, 0)
                self.convu = nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1)
                self.conv1 = nn.ConvTranspose1d(out_channels, out_channels, 3, 1, 1)
                self.bnu = nn.BatchNorm1d(out_channels)
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.act = nn.ReLU()

            def forward(self, x):
                y0 = F.interpolate(self.convp(x), scale_factor=2, mode='nearest')
                y = self.act(self.bnu(self.convu(x)))
                y = self.act(y0 + self.bn1(self.conv1(y)))
                return y

        self.conv1 = nn.ConvTranspose1d(1024, 768, 2, 2, 0)
        self.conv2 = nn.ConvTranspose1d(768, 512, 2, 2, 0)
        self.conv3 = nn.ConvTranspose1d(512, 256, 2, 2, 0)
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.InstanceNorm1d(256)

        self.convw1 = nn.ConvTranspose1d(256, 512, 1, 1, 0)
        self.bnw1 = nn.InstanceNorm1d(512)
        self.convw2 = nn.ConvTranspose1d(512, 1024, 1, 1, 0)
        self.bnw2 = nn.InstanceNorm1d(1024)
        self.convw3 = nn.ConvTranspose1d(1024, n_wires, 1, 1, 0)

        #self.bnp0 = nn.BatchNorm1d(n_wires)
        self.convwp = nn.ConvTranspose1d(1024, 32, 1, 1, 0)
        self.convp1 = nn.ConvTranspose1d(2, 32, 1, 1, 0)
        self.bnp1 = nn.InstanceNorm1d(64)
        self.convp2 = nn.ConvTranspose1d(64, n_features, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, wire_to_xy):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, 1024, self.seq_len // 8))

        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))

        w = self.act(self.bnw1(self.convw1(x)))
        wf = self.act(self.bnw2(self.convw2(w)))
        w = self.convw3(wf)
        wg = F.gumbel_softmax(w, dim=1, hard=False, tau=1/3)
        xy = torch.tensordot(wg, wire_to_xy, dims=[[1],[1]]).permute(0,2,1)

        p = self.act(self.bnp1(torch.cat([self.convwp(wf), self.convp1(xy)], dim=1)))
        p = self.convp2(p)

        return torch.cat([self.out(p), xy], dim=1), wg

class Disc(nn.Module):
    def __init__(self, ndf, seq_len, encoded_dim):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2)

        class DBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.convd = nn.Conv1d(in_channels, out_channels, 3, 2, 1)
                self.convp = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
                self.act = nn.LeakyReLU(0.2)

            def forward(self, x):
                y0 = self.convp(F.interpolate(x, scale_factor=0.5, mode='nearest'))
                y = self.act(y0 + self.convd(x))
                return y

        self.conv0 = nn.Conv1d(n_features+2, 128, 17, 1, 8)
        self.conv1 = nn.Conv1d(128, 256, 5, 1, 2)
        self.conv2 = nn.Conv1d(256, 512, 5, 1, 2)
        self.conv3 = nn.Conv1d(512, 512, 3, 1, 1)
        #self.db1 = DBlock(256, 512)
        #self.db2 = DBlock(512, 1024)
        #self.conv2 = nn.Conv1d(256, 512, 3, 2, 1)
        #self.conv3 = nn.Conv1d(512, 1024, 3, 2, 1)
        #self.conv4 = nn.Conv1d(1024, 2048, 3, 2, 1)

        #self.lin0 = nn.Linear(256 * seq_len // 1, 1, bias=True)
        self.convf = nn.Conv1d(512, 1, 1, 1, 0)

        self.out = nn.Identity()

    
    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        x = x_
        #dist = ((xy - nn.ConstantPad1d((1, 0), 0.0)(xy[:,:,:-1]))**2).sum(dim=1).unsqueeze(1)
        p = x[:,:n_features]
        w = x[:,n_features:]

        #x = torch.cat([p, w], dim=1)
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        #x = self.lin0(x.flatten(1,2))
        x = self.convf(x)
        x = x.mean(2)
        
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
