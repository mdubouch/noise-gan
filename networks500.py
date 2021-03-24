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

        self.lin0 = nn.Linear(latent_dims, seq_len//32*1024, bias=True)

        self.convu1 = nn.ConvTranspose1d(1024, 512, 4, 4, 0)
        self.bnu1 = nn.BatchNorm1d(512)
        self.convu2 = nn.ConvTranspose1d(512, 512, 2, 2, 0)
        self.bnu2 = nn.BatchNorm1d(512)
        self.convu3 = nn.ConvTranspose1d(512, 512, 2, 2, 0)
        self.bnu3 = nn.BatchNorm1d(512)
        self.convu4 = nn.ConvTranspose1d(512, 256, 2, 2, 0)
        self.bnu4 = nn.BatchNorm1d(256)

        self.conv1 = nn.Conv1d(256, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(128)

        self.convw = nn.Conv1d(128, n_wires, 1, 1, 0, bias=False)
        self.convp = nn.Conv1d(128, n_features, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, wire_to_xy):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, 1024, self.seq_len // 32))

        x = self.act(self.bnu1(self.convu1(x)))
        x = self.act(self.bnu2(self.convu2(x)))
        x = self.act(self.bnu3(self.convu3(x)))
        x = self.act(self.bnu4(self.convu4(x)))

        x = self.act(self.bn1(self.conv1(x)))

        w = self.convw(x)
        p = self.convp(x)
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=1.0)
        xy = torch.tensordot(wg, wire_to_xy, dims=[[1],[1]]).permute(0,2,1)

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
                self.act = nn.LeakyReLU(0.2)

            def forward(self, x):
                y = self.act(self.convd(x))
                return y

        self.convp = nn.Conv1d(n_features+2, 128, 129, 1, 64, bias=False)
        #self.convw = nn.Conv1d(n_wires, 32, 1, 1, 0, bias=False)
        self.conv1 = nn.Conv1d(128, 192, 17, 1, 8)
        self.conv2 = nn.Conv1d(192, 256, 9, 1, 4)
        self.db1 = DBlock(256, 256)
        self.db2 = DBlock(256, 512)
        self.db3 = DBlock(512, 512)
        self.db4 = DBlock(512, 512)
        #self.conv2 = nn.Conv1d(256, 512, 3, 2, 1)
        #self.conv3 = nn.Conv1d(512, 1024, 3, 2, 1)
        #self.conv4 = nn.Conv1d(1024, 2048, 3, 2, 1)

        #self.lin0 = nn.Linear(256 * seq_len // 1, 1, bias=True)
        #self.lin0 = nn.Linear(seq_len//4*512, 1)
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
        xy = x[:,n_features:n_features+2]
        wg = x[:,n_features+2:]
        pxy = x[:,:n_features+2]


        #x = torch.cat([p, w], dim=1)
        #x = self.act(self.conv0(pxy))
        #pxy = self.convpxy(pxy)
        #w = self.convw(wg)
        p = self.convp(torch.cat([p, xy + torch.randn_like(xy) * 0.01], dim=1))
        #x = torch.cat([p, w], dim=1)
        x = p

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        
        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.db4(x)

        #x = self.lin0(x.flatten(1,2))
        x = self.convf(x).mean(2)
        
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
