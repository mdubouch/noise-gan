import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = 9

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3

class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len):
        super().__init__()
        
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.LeakyReLU(0.2, False)

        self.dropout = nn.Dropout(0.05)

        self.lin0 = nn.Linear(latent_dims, seq_len//64*ngf*6, bias=True)

        self.convu1 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 2
        self.convu2 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 4
        self.convu3 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 8
        self.convu4 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 16
        self.convu5 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 32
        self.convu6 = nn.ConvTranspose1d(ngf*6, ngf*4, 4, 2, 1) # * 64
        
        self.bnu1 = nn.BatchNorm1d(ngf*6)
        self.bnu2 = nn.BatchNorm1d(ngf*6)
        self.bnu3 = nn.BatchNorm1d(ngf*6)
        self.bnu4 = nn.BatchNorm1d(ngf*6)
        self.bnu5 = nn.BatchNorm1d(ngf*6)
        self.bnu6 = nn.BatchNorm1d(ngf*4)

        self.conv4 = nn.ConvTranspose1d(ngf*4, ngf*4, 17, 1, 8)
        self.conv5 = nn.ConvTranspose1d(ngf*4, ngf*4, 17, 1, 8)
        self.conv6 = nn.ConvTranspose1d(ngf*4, ngf*2, 17, 1, 8)

        self.bn4 = nn.BatchNorm1d(ngf*4)
        self.bn5 = nn.BatchNorm1d(ngf*4)
        self.bn6 = nn.BatchNorm1d(ngf*2)

        self.convw0 = nn.ConvTranspose1d(ngf*2, ngf*2, 513, 1, 256)
        self.bnw0 = nn.BatchNorm1d(ngf*2)
        self.convw1 = nn.ConvTranspose1d(ngf*2, ngf*2, 513, 1, 256)
        self.bnw1 = nn.BatchNorm1d(ngf*2)
        self.convw2 = nn.ConvTranspose1d(ngf*2, ngf*1, 513, 1, 256)
        self.convw = nn.ConvTranspose1d(ngf*1, n_wires, 1, 1, 0, bias=False)

        self.convp0 = nn.ConvTranspose1d(ngf*2, ngf*2, 17, 1, 8)
        self.bnp0 = nn.BatchNorm1d(ngf*2)
        self.convp1 = nn.ConvTranspose1d(ngf*2, ngf*2, 17, 1, 8)
        self.bnp1 = nn.BatchNorm1d(ngf*2)
        self.convp2 = nn.ConvTranspose1d(ngf*2, ngf*1, 17, 1, 8)
        self.convp = nn.ConvTranspose1d(ngf*1, n_features, 1, 1, 0, bias=False)


        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x0 = self.act(self.lin0(z).view(-1, self.ngf*6, self.seq_len // 64))

        x1 = self.act(self.bnu1(self.convu1(self.dropout(x0))))
        x2 = self.act(F.interpolate(x0, scale_factor=4)  + self.bnu2(self.convu2(self.dropout(x1))))

        x3 = self.act(self.bnu3(self.convu3(self.dropout(x2))))
        x4 = self.act(F.interpolate(x2, scale_factor=4) + self.bnu4(self.convu4(self.dropout(x3))))

        x5 = self.act(self.bnu5(self.convu5(self.dropout(x4))))
        x6 = self.act(self.bnu6(self.convu6(self.dropout(x5))))

        x7 = self.act(self.bn4(self.conv4(self.dropout(x6))))
        x8 = self.act(x6 + self.bn5(self.conv5(self.dropout(x7))))
        x9 = self.act(self.bn6(self.conv6(self.dropout(x8))))
        
        
        # x (batch, ngf, len)
        w0 = self.act(self.bnw0(self.convw0(self.dropout(x9))))
        w1 = self.act(x9 + self.bnw1(self.convw1(self.dropout(w0))))
        w2 = self.act(self.convw2(self.dropout(w1)))
        w =  self.convw((w2 + embed_space_noise * torch.randn_like(w2))).permute(0,2,1)
        
        sim = F.gumbel_softmax(w, dim=2, hard=True, tau=tau)

        p0 = self.act(self.bnp0(self.convp0(self.dropout(x9))))
        p1 = self.act(x9 + self.bnp1(self.convp1(self.dropout(p0))))
        p2 = self.act(self.convp2(self.dropout(p1)))
        p = self.convp(p2)

        return self.out(p), sim.permute(0,2,1)

class Disc(nn.Module):
    def __init__(self, ndf, seq_len):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.dropout = nn.Dropout(0.01)


        self.convw = nn.Conv1d(n_wires, ndf*1, 1, 1, 0, bias=False)
        self.convw0 = nn.utils.spectral_norm(nn.Conv1d(ndf*1, ndf*2, 513, 1, 256, padding_mode='circular'))
        self.convw1 = nn.utils.spectral_norm(nn.Conv1d(ndf*2, ndf*2, 513, 1, 256, padding_mode='circular'))
        self.convw2 = nn.utils.spectral_norm(nn.Conv1d(ndf*2, ndf*2, 513, 1, 256, padding_mode='circular'))

        
        self.convp = nn.Conv1d(n_features, ndf*1, 1, 1, 0, bias=False)
        self.convp0 = nn.utils.spectral_norm(nn.Conv1d(ndf*1, ndf*2, 17, 1, 8, padding_mode='circular'))
        self.convp1 = nn.utils.spectral_norm(nn.Conv1d(ndf*2, ndf*2, 17, 1, 8, padding_mode='circular'))
        self.convp2 = nn.utils.spectral_norm(nn.Conv1d(ndf*2, ndf*2, 17, 1, 8, padding_mode='circular'))


        self.conv1 = nn.utils.spectral_norm(nn.Conv1d(ndf*4, ndf*4, 17, 1, 8, padding_mode='circular'))
        self.conv2 = nn.utils.spectral_norm(nn.Conv1d(ndf*4, ndf*4, 17, 1, 8, padding_mode='circular'))
        self.conv3 = nn.utils.spectral_norm(nn.Conv1d(ndf*4, ndf*6, 17, 1, 8, padding_mode='circular'))

        self.lin0 = nn.Linear(ndf*6 * seq_len // 1, 1, bias=True)

        self.out = nn.Identity()

    
    def forward(self, p_, w_): # p_ shape is (batch, features, seq_len), w_ shape is one-hot encoded wire (batch, 4986, seq_len)

        #norm = w_.norm(dim=2).norm(dim=1)
        #occupancy = w_.sum(dim=2).var(dim=1)

        #norm = norm.repeat((seq_len, 1, 1)).permute(2, 1, 0)
        seq_len = p_.shape[2]
        #occupancy = occupancy.repeat((seq_len, 1, 1)).permute(2, 1, 0)

        w = self.convw(w_)
        w0 = self.act(self.convw0(w))
        w1 = self.act(self.convw1(self.dropout(w0)))
        w2 = w0 + self.convw2(self.dropout(w1))
        w = w2

        p = self.convp(p_)
        p0 = self.act(self.convp0(p))
        p1 = self.act(self.convp1(self.dropout(p0)))
        p2 = p0 + self.convp2(self.dropout(p1))
        p = p2
        
        
        x = torch.cat([p, w], dim=1)

        x1 = self.act(self.conv1(self.dropout(self.act(x))))
        x2 = self.act(x + self.conv2(self.dropout(x1)))
        x3 = self.conv3(self.dropout(x2))
        

        x = self.lin0(self.act(x3).flatten(1,2))
        
        return self.out(x).squeeze(1)


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
