import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = 89

# Number of wires in the CDC
n_wires = 3606
# Number of continuous features (E, t, dca)
n_features = 3

encoded_dim = 16

class LayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        return self.ln(x.permute(0,2,1)).permute(0,2,1)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, seq_len, **kwargs):
        super().__init__()

        self.project_features = (in_channels != out_channels)
        if self.project_features:
            self.proj = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.bn1 = nn.Identity()#nn.InstanceNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.bn2 = nn.Identity()#nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        y0 = x

        y = self.act(self.bn1(self.conv1(x)))

        y = self.bn2(self.conv2(y))
        
        if self.project_features:
            y0 = self.proj(y0)

        y = self.act(y0 + y)

        return y

class ResBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, 
                kernel_size, stride, padding, **kwargs)
        self.act = nn.ReLU()
        self.bn1 = nn.InstanceNorm1d(out_channels)
        self.bn2 = nn.InstanceNorm1d(out_channels)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, 
            kernel_size, stride, padding, **kwargs)

        self.project_features = (in_channels != out_channels)

        if self.project_features:
            self.proj = nn.ConvTranspose1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, out_act=True):
        y0 = x

        y = self.act(self.bn1(self.conv1(y0)))
        y = self.conv2(y)

        if self.project_features:
            y0 = self.proj(y0)

        y = self.bn2(y + y0)
        if out_act == True:
            y = self.act(y)

        return y


class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len):
        super().__init__()
        
        self.ngf = ngf
        self.seq_len = seq_len

        self.version = __version__
        
        # Input: (B, latent_dims, 1)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.05)

        self.lin0 = nn.Linear(latent_dims, seq_len//64*ngf*6, bias=True)

        self.convu1 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 2
        self.conv1 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu2 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 4
        self.conv2 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu3 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 8
        self.conv3 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu4 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 16
        self.conv4 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu5 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 32
        self.conv5 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.convu6 = nn.ConvTranspose1d(ngf*6, ngf*6, 4, 2, 1) # * 64
        self.conv6 = nn.Conv1d(ngf*6, ngf*6, 9, 1, 4)
        self.bnu1 = nn.InstanceNorm1d(ngf*6)
        self.bnu2 = nn.InstanceNorm1d(ngf*6)
        self.bnu3 = nn.InstanceNorm1d(ngf*6)
        self.bnu4 = nn.InstanceNorm1d(ngf*6)
        self.bnu5 = nn.InstanceNorm1d(ngf*6)
        self.bnu6 = nn.InstanceNorm1d(ngf*6)
        self.bn1 = nn.InstanceNorm1d(ngf*6)
        self.bn2 = nn.InstanceNorm1d(ngf*6)
        self.bn3 = nn.InstanceNorm1d(ngf*6)
        self.bn4 = nn.InstanceNorm1d(ngf*6)
        self.bn5 = nn.InstanceNorm1d(ngf*6)
        self.bn6 = nn.InstanceNorm1d(ngf*6)


        self.res6 = nn.Conv1d(ngf*6, ngf*4, 17, 1, 8)
        self.res7 = nn.Conv1d(ngf*4, ngf*2, 33, 1, 16)
        self.conv7_diff = nn.Conv1d(ngf*4, ngf*2, 33, 1, 16)
        self.conv8 = nn.Conv1d(ngf*4, ngf*2, 33, 1, 16)

        self.convw2_diff = nn.Conv1d(ngf*2, ngf*1, 17, 1, 8)
        self.convw2 = nn.Conv1d(ngf*2, ngf*1, 17, 1, 8)
        self.convw3 = nn.Conv1d(ngf*2, ngf*1, 17, 1, 8)
        self.convw4 = nn.Conv1d(ngf*1, encoded_dim, 33, 1, 16)

        self.dec = nn.Sequential(
            nn.Conv1d(encoded_dim, encoded_dim*2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(encoded_dim*2, encoded_dim*4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(encoded_dim*4, n_wires, 1, 1, 0, bias=False),
        )

        self.resp2 = nn.Conv1d(ngf*2, ngf*2, 17, 1, 8)
        self.convp3 = nn.Conv1d(ngf*2, ngf, 33, 1, 16)
        self.convp = nn.Conv1d(ngf*1, n_features, 1, 1, 0)

        self.out = nn.Tanh()
        
    def forward(self, z, embed_space_noise=0.0, tau=1.0):
        # z: random point in latent space
        x = self.act(self.lin0(z).view(-1, self.ngf*6, self.seq_len // 64))

        x0 = x
        x = self.act(self.bnu1(self.convu1(x0)))
        x = self.bn1(self.conv1(x) + F.interpolate(x0, scale_factor=2, mode='linear'))
        x = self.act(x)

        x0 = x
        x = self.act(self.bnu2(self.convu2(x0)))
        x = self.bn2(self.conv2(x) + F.interpolate(x0, scale_factor=2, mode='linear'))
        x = self.act(x)

        x0 = x
        x = self.act(self.bnu3(self.convu3(x0)))
        x = self.bn3(self.conv3(x) + F.interpolate(x0, scale_factor=2, mode='linear'))
        x = self.act(x)

        x0 = x
        x = self.act(self.bnu4(self.convu4(x0)))
        x = self.bn4(self.conv4(x) + F.interpolate(x0, scale_factor=2, mode='linear'))
        x = self.act(x)

        x0 = x
        x = self.act(self.bnu5(self.convu5(x0)))
        x = self.bn5(self.conv5(x) + F.interpolate(x0, scale_factor=2, mode='linear'))
        x = self.act(x)

        x0 = x
        x = self.act(self.bnu6(self.convu6(x0)))
        x = self.bn6(self.conv6(x) + F.interpolate(x0, scale_factor=2, mode='linear'))
        x = self.act(x)

        x = self.act(self.res6(x))
        x1 = self.act(self.res7(x))
        xd = self.act(self.conv7_diff(x - nn.ConstantPad1d((1, 0), 0.0)(x[:,:,:-1])))
        x = self.act(self.conv8(torch.cat([x1, xd], dim=1)))
        
        # x (batch, ngf, len)
        w = self.act(self.convw2(x))
        w_diff = self.act(self.convw2_diff(x - nn.ConstantPad1d((1, 0), 0.0)(x[:,:,:-1])))
        w = self.act(self.convw3(torch.cat([w, w_diff], dim=1)))
        w = self.convw4(w)

        encoded_w = w
        
        dec_w = self.dec(encoded_w)

        sim = F.gumbel_softmax(dec_w, dim=1, hard=True, tau=tau)
        #sim = F.softmax(w, dim=1)

        p = self.act(self.resp2(x))
        p = self.act(self.convp3(p))
        p = self.convp(p)

        return self.out(p), sim

class Disc(nn.Module):
    def __init__(self, ndf, seq_len):
        super().__init__()

        self.version = __version__
        
        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.dropout = nn.Dropout(0.05)

        self.enc = nn.Sequential(
            nn.Conv1d(n_wires, encoded_dim*4, 1, 1, 0, bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(encoded_dim*4, encoded_dim*2, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(encoded_dim*2, encoded_dim, 1, 1, 0),
            nn.Tanh()
        )

        self.resw0 = nn.Conv1d(encoded_dim, ndf*1, 33, 1, 16)
        self.resw1 = nn.Conv1d(ndf*1, ndf*1, 17, 1, 8)
        self.resw1_diff = nn.Conv1d(ndf*1, ndf*1, 17, 1, 8)
        self.resw2 = nn.Conv1d(ndf*2, ndf*2, 17, 1, 8)

        self.convp = nn.Conv1d(n_features, ndf, 1, 1, 0)
        self.resp0 = nn.Conv1d(ndf, ndf*2, 33, 1, 16)
        self.resp1 = nn.Conv1d(ndf*2, ndf*2, 17, 1, 8)
        

        self.res1 = nn.Conv1d(ndf*4, ndf*8, 33, 1, 16)
        self.res1_diff = nn.Conv1d(ndf*4, ndf*8, 33, 1, 16)
        self.res2 = nn.Conv1d(ndf*16, ndf*16, 17, 1, 8)

        self.lin0 = nn.Linear(ndf*16 * seq_len // 1, 1, bias=False)

        self.out = nn.Identity()

    
    def forward(self, x_): 
        # x_ is concatenated tensor of p_ and w_, shape (batch, features+n_wires, seq_len) 
        # p_ shape is (batch, features, seq_len), 
        # w_ shape is one-hot encoded wire (batch, n_wires, seq_len)

        seq_len = x_.shape[2]
        p_ = x_[:,:n_features,:]
        w_ = x_[:,n_features:,:]

        #norm = w_.norm(dim=2).norm(dim=1)
        #occupancy = w_.sum(dim=2).var(dim=1)

        #norm = norm.repeat((seq_len, 1, 1)).permute(2, 1, 0)
        #occupancy = occupancy.repeat((seq_len, 1, 1)).permute(2, 1, 0)

        # w_ (batch, n_wires, seq_len)
        w = self.enc(w_)
        
        x = self.act(self.resw0(w))
        w = self.act(self.resw1(x))
        w_diff = self.act(self.resw1_diff(x - nn.ConstantPad1d((1, 0), 0.0)(x[:,:,:-1])))
        w = self.act(self.resw2(torch.cat([w, w_diff], dim=1)))
        #w = self.resw2(w)
        #w = self.resw3(w)

        p = self.convp(p_)
        p = self.act(self.resp0(p))
        p = self.act(self.resp1(p))
        #p = self.resp2(p)
        #p = self.resp3(p)
        
        x = torch.cat([p, w], dim=1)

        x1 = self.act(self.res1(x))
        xd = self.act(self.res1_diff(x - nn.ConstantPad1d((1, 0), 0.0)(x[:,:,:-1])))
        x = self.act(self.res2(torch.cat([x1, xd], dim=1)))
        #x = self.res3(x)
        #x = self.res4(x)

        x = self.lin0(x.flatten(1,2))
        
        return self.out(x).squeeze(1)


def get_n_params(model):
    return sum(p.reshape(-1).shape[0] for p in model.parameters())
