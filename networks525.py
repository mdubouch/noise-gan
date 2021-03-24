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

        self.lin0 = nn.Linear(latent_dims, seq_len//16*512, bias=True)
        self.bn0 = nn.BatchNorm1d(512)

        self.convu1 = nn.ConvTranspose1d(512, 256, 4, 4, 0)
        self.bnu1 = nn.BatchNorm1d(256)
        self.convu2 = nn.ConvTranspose1d(256, 128, 4, 4, 0)
        self.bnu2 = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv1d(128, 64, 17, 1, 8)
        self.bn1 = nn.BatchNorm1d(64)

        self.convw1 = nn.Conv1d(64, 32, 3, 1, 1)
        self.bnw1 = nn.BatchNorm1d(32)
        self.convw2 = nn.Conv1d(32, n_wires, 1, 1, 0, bias=False)

        self.convp1 = nn.Conv1d(64, 32, 3, 1, 1)
        self.bnp1 = nn.BatchNorm1d(32)
        self.convp2 = nn.Conv1d(32, n_features, 1, 1, 0)
        #self.linp = nn.Linear(512, n_features)

        self.out = nn.Tanh()
        
    def forward(self, z, wire_to_xy):
        # z: random point in latent space
        x = self.act(self.bn0(self.lin0(z).view(-1, 512, self.seq_len // 16)))

        x = self.act(self.bnu1(self.convu1(x)))
        x = self.act(self.bnu2(self.convu2(x)))

        x = self.act(self.bn1(self.conv1(x)))

        w = self.act(self.bnw1(self.convw1(x)))
        w = self.convw2(w)
        wg = F.gumbel_softmax(w, dim=1, hard=True, tau=1.0)
        #xy = torch.tensordot(wg, wire_to_xy, dims=[[1],[1]]).permute(0,2,1)

        p = self.act(self.bnp1(self.convp1(x)))
        p = self.convp2(p)

        #return torch.cat([self.out(p), xy], dim=1), wg
        return self.out(p), wg

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

        self.convp1 = nn.Conv1d(n_features, 64, 1, 1, 0)
        self.convp2 = nn.Conv1d(64, 64, 3, 1, 1)
        self.convp3 = nn.Conv1d(64, 64, 3, 1, 1)
        self.convxy1 = nn.Conv1d(3, 64, 1, 1, 0)
        self.convxy2 = nn.Conv1d(64, 64, 9, 1, 4)
        self.convw1 = nn.Conv1d(n_wires, 64, 1, 1, 0)

        self.conv1 = nn.Conv1d(192, 192, 17, 1, 8)
        self.conv2 = nn.Conv1d(192, 192, 9, 1, 4)
        self.db1 = DBlock(192, 256)
        self.db2 = DBlock(256, 512)
        #self.conv2 = nn.Conv1d(256, 512, 3, 2, 1)
        #self.conv3 = nn.Conv1d(512, 1024, 3, 2, 1)
        #self.conv4 = nn.Conv1d(1024, 2048, 3, 2, 1)

        #self.lin0 = nn.Linear(256 * seq_len // 1, 1, bias=True)
        #self.lin0 = nn.Linear(seq_len//4*512, 1)
        self.convf = nn.Conv1d(512, 1, 3, 1, 1, padding_mode='circular')

        self.out = nn.Identity()

    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ is AE-encoded wire (batch, encoded_dim, seq_len)

        seq_len = x_.shape[2]
        x = x_
        #dist = ((xy - nn.ConstantPad1d((1, 0), 0.0)(xy[:,:,:-1]))**2).sum(dim=1).unsqueeze(1)
        p = x[:,:n_features]
        xy = x[:,n_features:n_features+3]
        wg = x[:,n_features+3:]

        #wg.register_hook(hook)
        #print(wg.argmax(dim=1)[0,0])
        #xy = self.wireproj(wg, wire_to_xy)
        #print(xy.shape)
        #print(xy[0,:,0])

        #x = self.act(self.conv0(pxy))
        #pxy = self.convpxy(pxy)
        #w = self.convw(wg)
        xy = self.act(self.convxy1(xy + torch.randn_like(xy) * 0.01))
        xy = self.act(self.convxy2(xy))
        p = self.act(self.convp1(p + torch.randn_like(p) * 0.01))
        p = self.act(self.convp2(p))
        w = self.act(self.convw1(wg))
        #p = self.convp(x)
        #xy = self.convxy(xy
        x = torch.cat([p, xy, w], dim=1)

#        self.conv1.weight.register_hook(printhook)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        
        #self.db2.convd.weight.register_hook(printhook)
        x = self.db1(x)
        x = self.db2(x)


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
