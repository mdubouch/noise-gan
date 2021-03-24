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

        self.lin0 = nn.Linear(latent_dims, seq_len//64*1024, bias=True)

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

        self.gb1 = GBlock(1024, 768)
        self.gb2 = GBlock(768, 512)
        self.gb3 = GBlock(512, 384)
        self.gb4 = GBlock(384, 256)
        self.gb5 = GBlock(256, 256)
        self.gb6 = GBlock(256, 256)

        self.convw1 = nn.ConvTranspose1d(256, n_wires, 1, 1, 0)

        #self.bnp0 = nn.BatchNorm1d(n_wires)
        self.convwp = nn.ConvTranspose1d(256, 64, 1, 1, 0)
        self.convp1 = nn.ConvTranspose1d(2, 64, 3, 1, 1)
        self.bnp1 = nn.BatchNorm1d(64)
        self.convp2 = nn.ConvTranspose1d(64, 32, 3, 1, 1)
        self.bnp2 = nn.BatchNorm1d(32)
        self.convp3 = nn.ConvTranspose1d(32, n_features, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, wire_to_xy):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, 1024, self.seq_len // 64))

        x = self.gb1(x)
        x = self.gb2(x)
        x = self.gb3(x)
        x = self.gb4(x)
        x = self.gb5(x)
        x = self.gb6(x)

        w = self.convw1(x)
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=2/3)
        xy = torch.tensordot(wg, wire_to_xy, dims=[[1],[1]]).permute(0,2,1)

        p = self.act(self.bnp1(self.convwp(x) + self.convp1(xy)))
        p = self.act(self.bnp2(self.convp2(p)))
        p = self.convp3(p)

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
                self.convd = nn.Conv1d(in_channels, out_channels, 4, 2, 1)
                self.act = nn.LeakyReLU(0.2)

            def forward(self, x):
                y = self.act(self.convd(x))
                return y

        self.conv0 = nn.Conv1d(2, 64, 1, 1, 0)
        self.conv1 = nn.Conv1d(64, 128, 1, 1, 0)

        self.conv2 = nn.Conv1d(128, 128, 1, 1, 0)
        self.conv3 = nn.Conv1d(128, 256, 1, 1, 0)
        self.conv4 = nn.Conv1d(256, 256, 1, 1, 0)
        self.conv5 = nn.Conv1d(256, 512, 1, 1, 0)
        self.conv6 = nn.Conv1d(512, 512, 1, 1, 0)

        self.lin0 = nn.Linear(512 * seq_len // 1, 1, bias=True)

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
        x = self.act(self.conv0(w))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))


        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


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
