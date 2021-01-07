import networks
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser('Train CDC GAN')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--latent-dims', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=2048)
parser.add_argument('--job-id', type=int)
parser.add_argument('--gfx', type=bool, default=False)
args = parser.parse_args()
output_dir = 'output_%d/' % (args.job_id)
print('Evaluating job %d in %s' % (args.job_id, output_dir))

ngf = args.ngf
ndf = args.ndf
latent_dims = args.latent_dims
seq_len = args.sequence_length

def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

import dataset
data = dataset.Data()

import geom_util
gu = geom_util.GeomUtil(data.get_cdc_tree())

# Initialize networks
gen = to_device(networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len))
disc = to_device(networks.Disc(ndf=ndf, seq_len=seq_len))
torchsummary.summary(gen, input_size=(latent_dims,))
torchsummary.summary(disc, input_size=[(3, seq_len), (gu.cum_n_wires[-1], seq_len)])

optimizer_gen = torch.optim.Adam(gen.parameters(),  lr=1e-4, betas=(0.5, 0.999))
optimizer_disc = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
disciminator_losses = []
generator_losses = []
tau = 0
n_epochs = 0

# Load network states
def load_states(path):
    print('Loading GAN states from %s...' % (path))
    states = torch.load(path)
    disc.load_state_dict(states['disc'])
    optimizer_disc.load_state_dict(states['d_opt'])
    global discriminator_losses
    discriminator_losses = states['d_loss']
    gen.load_state_dict(states['gen'])
    optimizer_gen.load_state_dict(states['g_opt'])
    global generator_losses
    generator_losses = states['g_loss']
    global tau
    tau = states['tau']
    global n_epochs
    n_epochs = states['n_epochs']
    global data
    data.qt = states['qt']
    data.minmax = states['minmax']
    print('OK')
load_states('output_%d/states_%d.pt' % (args.job_id, args.job_id))

def sample_fake(batch_size, tau):
    noise = to_device(torch.randn((batch_size, latent_dims), requires_grad=True))
    sample = gen(noise, 0.0, tau)
    return sample

print('tau:', tau)
p, w = sample_fake(32, tau)
print(p.shape, w.shape)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

inv_p = data.inv_preprocess(p.permute(0, 2, 1).flatten(0, 1))
print(inv_p.shape)

fw = torch.argmax(w, dim=1).flatten().detach().cpu()
print(fw.shape)
plt.figure(figsize=(6,6))
plt.scatter(gu.wire_x[fw], gu.wire_y[fw], s=inv_p[:,0] * 1e3, c=inv_p[:,2], cmap='inferno',
        )
plt.savefig(output_dir+'gen_scatter.png', dpi=120)

plt.figure()
plt.hist(np.log10(inv_p[:,0]).cpu(), bins=50)
plt.savefig(output_dir+'gen_edep.png', dpi=120)

data.load()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(gu.wire_x[fw], gu.wire_y[fw], s=inv_p[:,0] * 1e3, c=inv_p[:,2], cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max())
ax[0].set_aspect(1.0)

ax[1].scatter(data.dbg_z-7650, data.dbg_y, s=data.edep*1e3, c=data.doca, cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max())
ax[1].set_aspect(1.0)
plt.savefig(output_dir+'comp_scatter.png', dpi=120)

plt.figure()
plt.hist(np.log10(data.edep), bins=50, alpha=0.7, density=True)
plt.hist(np.log10(inv_p[:,0].cpu()), bins=50, alpha=0.7, density=True)
plt.savefig(output_dir+'comp_edep.png', dpi=120)
