import torch
import torch.nn as nn

class Gen(nn.Module):
    def __init__(self, ngf, latent_dims, seq_len, n_features):
        super().__init__()

        k = ngf
        self.k = k

        # Input: (B, latent_dims, 1)
        self.act = nn.LeakyReLU(0.2, True)

        self.lin0 = nn.Linear(latent_dims, n_features * seq_len)

        #self.lins6 = nn.Linear(seq_len // 64, seq_len // 16)
        self.linf6 = nn.Linear(n_features, ngf*2)
        self.bn6 = nn.InstanceNorm1d(ngf*2)


        self.linf7 = nn.Linear(ngf*2, ngf*4)
        #self.lins7 = nn.Linear(seq_len // 16, seq_len // 8)
        self.bn7 = nn.InstanceNorm1d(ngf*4)

        self.linf8 = nn.Linear(ngf*4, ngf*8)
        #self.lins8 = nn.Linear(seq_len // 8, seq_len)
        self.bn8 = nn.InstanceNorm1d(ngf*8)

        self.linf9 = nn.Linear(ngf*8, ngf*8)
        self.bn9 = nn.InstanceNorm1d(ngf*8)

        self.linf10 = nn.Linear(ngf*8, ngf*8)
        self.bn10 = nn.InstanceNorm1d(ngf*8)

        self.linf11 = nn.Linear(ngf*8, n_features)

        self.out = nn.Tanh()

    def forward(self, z):
        # z: random point in latent space
        # stage: length of sequence returned by generator
        # alpha: blending factor (0 = upsampled image from previous layer; 1 = full image from new layer)

        x = self.act(self.lin0(z)) # (B, C*8 * L/64)

        x = x.view(-1, seq_len, n_features)

        #x = self.bn6(self.act(self.linf6(self.act(self.lins6(x)).permute(0,2,1))).permute(0,2,1)).permute(0,2,1)
        #x = self.bn7(self.act(self.lins7(self.act(self.linf7(x)).permute(0,2,1)))).permute(0,2,1)
        #x = self.bn8(self.act(self.lins8(self.act(self.linf8(x)).permute(0,2,1)))).permute(0,2,1)
        x = self.bn6(self.act(self.linf6(x)).permute(0,2,1)).permute(0,2,1)
        x = self.bn7(self.act(self.linf7(x)).permute(0,2,1)).permute(0,2,1)
        x = self.bn8(self.act(self.linf8(x)).permute(0,2,1)).permute(0,2,1)

        x = self.bn9(self.act(self.linf9(x)).permute(0,2,1)).permute(0,2,1)

        x = self.bn10(self.act(self.linf10(x)).permute(0,2,1)).permute(0,2,1)

        x = self.linf11(x)

        return self.out(x.permute(0,2,1))

class Disc(nn.Module):
    def __init__(self, ndf, seq_len, n_features):
        super().__init__()

        k = ndf
        self.k = k


        # (B, n_features, 256)
        self.act = nn.LeakyReLU(0.2, True)

        self.linf1 = nn.Linear(n_features, ndf*12)

        self.linf2 = nn.utils.spectral_norm(nn.Linear(ndf*12, ndf*8))
        #self.lins2 = nn.Linear(seq_len, seq_len // 8)
        self.bn2 = nn.InstanceNorm1d(ndf*8)

        self.linf3 = nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4))
        self.bn3 = nn.InstanceNorm1d(ndf*4)

        self.linf4 = nn.utils.spectral_norm(nn.Linear(ndf*4, ndf*2))
        self.bn4 = nn.InstanceNorm1d(ndf*2)

        self.lin0 = nn.utils.spectral_norm(nn.Linear(seq_len * ndf*2, 1))

        self.out = nn.Identity()


    def forward(self, x_): # x shape is (batch, features, seq_len)
        x = x_

        x = self.act(self.linf1(x.permute(0, 2, 1)))
        #x = self.bn2(self.act(self.lins2(self.act(self.linf2(x)).permute(0,2,1))))
        x = self.bn2(self.act(self.linf2(x)).permute(0,2,1)).permute(0,2,1)

        x = self.bn3(self.act(self.linf3(x)).permute(0,2,1)).permute(0,2,1)

        x = self.bn4(self.act(self.linf4(x)).permute(0,2,1))

        x = x.view(-1, seq_len * ndf*2)

        x = self.lin0(x)

        return self.out(x.squeeze())


