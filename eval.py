#!/usr/bin/python3
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
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    states = torch.load(path, map_location=device)
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
load_states('output_%d/states.pt' % (args.job_id))

gen.eval()

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

# Load in the training data for comparisons
print("Loading training data")
data.load()
print("OK")
###########

# Scatterplot comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(gu.wire_x[fw], gu.wire_y[fw], s=inv_p[:,0] * 1e3, c=inv_p[:,2], cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max())
ax[0].set_aspect(1.0)

ax[1].scatter(data.dbg_z-7650, data.dbg_y, s=data.edep*1e3, c=data.doca, cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max())
ax[1].set_aspect(1.0)
plt.savefig(output_dir+'comp_scatter.png', dpi=120)

# Feature histogram comparison
plt.figure()
plt.hist(np.log10(data.edep), bins=50, alpha=0.7, density=True)
plt.hist(np.log10(inv_p[:,0].cpu()), bins=50, alpha=0.7, density=True)
plt.xlabel('log(Edep [MeV])')
plt.savefig(output_dir+'comp_edep.png', dpi=120)

plt.figure()
plt.hist(np.log10(data.t), bins=50, alpha=0.7, density=True)
plt.hist(np.log10(inv_p[:,1].cpu()), bins=50, alpha=0.7, density=True)
plt.xlabel('log(t [ns])')
plt.savefig(output_dir+'comp_t.png', dpi=120)

plt.figure()
plt.hist(data.doca, bins=50, alpha=0.7, density=True)
plt.hist(inv_p[:,2].cpu(), bins=50, alpha=0.7, density=True)
plt.xlabel('doca [mm]')
plt.savefig(output_dir+'comp_doca.png', dpi=120)

# Loss plots
plt.figure()
n_critic = len(discriminator_losses) // len(generator_losses)
plt.plot(np.linspace(0, n_epochs // n_critic, num=len(discriminator_losses)), discriminator_losses,
        label='Discriminator', alpha=0.7)
plt.plot(np.linspace(0, n_epochs, num=len(generator_losses)), generator_losses, alpha=0.7,
        label='Generator')
plt.ylabel('WGAN-GP loss')
plt.xlabel('Epoch')
plt.savefig(output_dir+'losses.png', dpi=120)


# Sample real sequences of 2048 hits from the training set.
# Real data in evaluation is not pre-processed, so it does not need to be inv_preprocessed.
def sample_real(batch_size):
    idx = np.random.randint(0, data.edep.size - seq_len, size=(batch_size,))
    start = idx
    stop = idx + 2048
    slices = np.zeros((batch_size, 2048), dtype=np.int64)
    for i in range(batch_size):
        slices[i] = np.r_[start[i]:stop[i]] 
    edep = data.edep[slices]
    t = data.t[slices]
    doca = data.doca[slices]
    w = data.wire[slices]
    #one_hot_w = F.one_hot(w, num_classes=gu.cum_n_wires[-1]).squeeze(1).permute(0, 2, 1)
    return np.array([edep, t, doca]), w

_p, _w = sample_real(12)
print(_p.shape, _w.shape)
from matplotlib.patches import Ellipse
def only_walls(ax):
    # Draw walls
    inner = Ellipse((0, 0), 488*2, 488*2, facecolor=(0, 0, 0, 0), edgecolor='gray')
    outer = Ellipse((0, 0), 786*2, 786*2, facecolor=(0, 0, 0, 0), edgecolor='gray')

    ax.add_patch(inner)
    ax.add_patch(outer);
    
    ax.set(xlim=(-800,800), ylim=(-800,800), xlabel='x [mm]', ylabel='y [mm]')

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12,12))
n_samples=4
gs1 = gridspec.GridSpec(n_samples, n_samples)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
plt.tight_layout()
for i in range(n_samples):
    for j in range(n_samples):
        p, w = sample_real(1)
        p = p.squeeze()
        w = w.squeeze()
        #plt.title('G4')
        ax = plt.subplot(gs1[i*4 + j])
        only_walls(ax)
        ax.set_xlim(-800,800)
        ax.set_ylim(-800,800)
        ax.set_aspect(1)
        ax.scatter(gu.wire_x[w], gu.wire_y[w], s=p[0] * 1e2+0, alpha=0.7, c=p[2], cmap='inferno', vmin=data.doca.min(), vmax=data.doca.max())
        ax.axis('off')
plt.savefig(output_dir+'grid_real.png', dpi=240)

# Same for fake samples
fig = plt.figure(figsize=(12,12))
n_samples=4
gs1 = gridspec.GridSpec(n_samples, n_samples)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
plt.tight_layout()
for i in range(n_samples):
    for j in range(n_samples):
        p, w = sample_fake(1, tau)
        p = data.inv_preprocess(p.permute(0,2,1).flatten(0,1))
        p = p.squeeze().detach().cpu().T
        w = torch.argmax(w, dim=1).squeeze().detach().cpu()
        #plt.title('G4')
        ax = plt.subplot(gs1[i*4 + j])
        only_walls(ax)
        ax.set_xlim(-800,800)
        ax.set_ylim(-800,800)
        ax.set_aspect(1)
        ax.scatter(gu.wire_x[w], gu.wire_y[w], s=p[0] * 1e2+0, alpha=0.7, c=p[2], cmap='inferno', vmin=data.doca.min(), vmax=data.doca.max())
        ax.axis('off')
plt.savefig(output_dir+'grid_fake.png', dpi=240)

# Activated wires per sequence histogram

n_fake_uq = []
n_real_uq = []
n_seq = 256
plt.figure()
for i in range(n_seq):
    fake_p, fake_w = sample_fake(1, tau)
    real_p, real_w = sample_real(1)

    fw = torch.argmax(fake_w, dim=1).flatten().detach().cpu()
    rw = real_w.squeeze()

    n_fake_uq.append(np.unique(fw).size)
    n_real_uq.append(np.unique(rw).size)
#print(n_fake_uq)

plt.hist(n_real_uq, bins=50, alpha=0.7, label='G4', range=[0,800]);
plt.hist(n_fake_uq, bins=50, alpha=0.7, label='GAN', range=[0, 800]);
#plt.xlim(0, 1200)
plt.legend()
plt.title('Number of activated wires per sequence')
plt.savefig(output_dir+'activated_wires.png', dpi=120)
