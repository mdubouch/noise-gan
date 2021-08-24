#!/usr/bin/python3
import argparse
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser('Train CDC GAN', argparse.SUPPRESS)
parser.add_argument('--ngf', type=int)
parser.add_argument('--ndf', type=int)
parser.add_argument('--latent-dims', '--ld', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=2048)
parser.add_argument('--job-id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--net-version', type=int)
parser.add_argument('--dataset', type=str, default='dataset')
parser.add_argument('--enc-dim', type=int, default=4)
parser.add_argument('--seed', type=int, default=1339)
parser.add_argument('--dpi', type=int, default=120)
parser.add_argument('--gfx', type=bool, default=False)
args = parser.parse_args()
output_dir = 'output_%d/' % (args.job_id)

import glob
def find_last_epoch():
    last_epoch = 0
    for save_file in glob.glob(output_dir+'states_*.pt'):
        idx = int(save_file.split('/')[1].split('_')[1].split('.')[0])
        if idx > last_epoch:
            last_epoch = idx
    return last_epoch

if args.epoch is None:
    args.epoch = find_last_epoch()
if args.ndf is None or args.ngf is None:
    log_file = open(output_dir+'output.log', 'rt')
    contents = log_file.read()
    log_file.close()
    g_s = contents.rfind("ngf=")
    g_s = contents.find('=', g_s) + 1
    g_e = contents.find('\n', g_s)
    if g_s == -1:
        print("Couldn't find ngf")
        exit(1)
    args.ngf = int(contents[g_s:g_e])
    d_s = contents.rfind("ndf=")
    d_s = contents.find('=', d_s) + 1
    d_e = contents.find('\n', d_s)
    if d_s == -1:
        print("Couldn't find ndf")
        exit(1)
    args.ndf = int(contents[d_s:d_e])
if args.net_version is None:
    log_file = open(output_dir+'output.log', 'rt')
    contents = log_file.read()
    log_file.close()
    n_s = contents.rfind('networks=')
    if n_s == -1:
        print("Couldn't find networks version")
        exit(1)
    n_s = contents.find('=', n_s) + 1
    n_e = contents.find('\n', n_s)
    args.net_version = int(contents[n_s:n_e])


print('Evaluating job %d in %s at epoch %d' % (args.job_id, output_dir, args.epoch))

ngf = args.ngf
ndf = args.ndf
print('ngf:', ngf)
print('ndf:', ndf)
latent_dims = args.latent_dims
seq_len = args.sequence_length
encoded_dim = args.enc_dim

def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

import importlib
print('Importing dataset from %s.py' % (args.dataset))
dataset = importlib.import_module(args.dataset)
data = dataset.Data()

import geom_util
gu = geom_util.GeomUtil(data.get_cdc_tree())

networks = importlib.import_module('networks%s' % (args.net_version))
print('Evaluating with networks version %d' % (args.net_version))
# Initialize networks
gen = to_device(networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len, encoded_dim=encoded_dim,
    n_wires=gu.n_wires))
disc = to_device(networks.Disc(ndf=ndf, seq_len=seq_len, encoded_dim=encoded_dim,
    n_wires=gu.n_wires))
print('Gen summary:')
#torchsummary.summary(gen, input_size=(latent_dims,))
print('Disc summary:')
#torchsummary.summary(disc, input_size=(3+16, seq_len))
print('generator params:      {:,}'.format(networks.get_n_params(gen)))
print('discriminator params:  {:,}'.format(networks.get_n_params(disc)))

disciminator_losses = []
generator_losses = []
gradient_penalty = []
validation_losses = []
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
    global discriminator_losses
    discriminator_losses = states['d_loss']
    gen.load_state_dict(states['gen'])
    global generator_losses
    generator_losses = states['g_loss']
    global tau
    tau = states['tau']
    global n_epochs
    n_epochs = states['n_epochs']
    global data
    data.qt = states['qt']
    data.minmax = states['minmax']
    global gradient_penalty
    if 'gradient_penalty' in states:
        gradient_penalty = states['gradient_penalty']
    global validation_losses
    if 'validation_loss' in states:
        validation_losses = states['validation_loss']
    print('OK')
load_states('output_%d/states_%d.pt' % (args.job_id, args.epoch))

wire_to_xy = torch.tensor([gu.wire_x, gu.wire_y], device='cuda', dtype=torch.float32)
wire_to_xy_norm = (wire_to_xy - wire_to_xy.min() + 10) / (wire_to_xy.max() - wire_to_xy.min() + 10)

# Load in the training data for comparisons
print("Loading training data")
data.load()
print("OK")

n_features=3
gen.eval()
def sample_fake(batch_size, tau):
    noise = to_device(torch.randn((batch_size, latent_dims)))
    p, dec_w = gen(noise)

    return p, dec_w

p, w = sample_fake(1, tau)


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = args.dpi

inv_p = data.inv_preprocess(p.permute(0, 2, 1).flatten(0, 1))

###########
def get_wire_weights():
    wire_counts = np.bincount(data.wire, minlength=gu.n_wires)
    print(wire_counts.shape)
    return torch.tensor(1 / (wire_counts + 1e-1), device='cuda', dtype=torch.float)
wire_weights = get_wire_weights()

fw = torch.argmax(w, dim=1).flatten().detach().cpu()
plt.figure()
plt.scatter(gu.wire_x[fw], gu.wire_y[fw], s=1+inv_p[:,0] * 1e3, c=inv_p[:,2], cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8
        )
# Draw lines between consecutive hits
import matplotlib.lines as lines
scatter_l = lines.Line2D(gu.wire_x[fw], gu.wire_y[fw], linewidth=0.2, color='gray', alpha=0.7)
ax = plt.gca()
ax.set_aspect(1.0)
ax.add_line(scatter_l)
gu.draw_cdc(ax)
plt.savefig(output_dir+'gen_scatter.png', dpi=240)
plt.close()

# No lines
plt.figure()
plt.scatter(gu.wire_x[fw], gu.wire_y[fw], s=1+inv_p[:,0] * 1e3, c=inv_p[:,2], cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8
        )
ax = plt.gca()
ax.set_aspect(1.0)
gu.draw_cdc(ax)
plt.savefig(output_dir+'gen_scatter_nolines.png', dpi=240)
plt.close()

# Draw only lines
plt.figure()
scatter_l = lines.Line2D(gu.wire_x[fw], gu.wire_y[fw], linewidth=0.1, color='gray', alpha=0.7)
ax = plt.gca()
ax.set_aspect(1.0)
ax.add_line(scatter_l)
gu.draw_cdc(ax)
plt.savefig(output_dir+'gen_scatter_onlylines.png', dpi=240)
plt.close()

x = gu.wire_x[fw]
y = gu.wire_y[fw]
fake_dist = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
t = inv_p[:,1]
fake_time_diff = t[1:] - t[:-1]

plt.figure()
plt.hist(np.log10(inv_p[:,0]).cpu(), bins=50)
plt.savefig(output_dir+'gen_edep.png')
plt.close()



# Features and correlations
n_feat = 3
fig, axes = plt.subplots(n_feat, n_feat, figsize=(12,12))
axis_labels = [r'log($E$ [MeV])', r'log($t$ [ns])', 'DCA [mm]']
axis_ranges = [[np.log10(data.edep).min(), np.log10(data.edep).max()], 
    [np.log10(data.t).min(), np.log10(data.t).max()], [data.doca.min(), data.doca.max()]]
n_seqs = 4
n_its = data.edep.size // (n_seqs * seq_len)
n_elements = n_its * n_seqs * seq_len
inv_p = np.zeros((n_feat, n_elements))

for i in range(n_its):
    gen.eval()
    fake_p, fake_w = sample_fake(n_seqs, tau)
    fake_p = fake_p.permute(0, 2, 1).flatten(0, 1)
    _inv_p = data.inv_preprocess(fake_p).cpu().numpy().T

    inv_p[0, i*n_seqs*seq_len:(i+1)*n_seqs*seq_len] = np.log10(_inv_p[0])
    inv_p[1, i*n_seqs*seq_len:(i+1)*n_seqs*seq_len] = np.log10(_inv_p[1])
    inv_p[2, i*n_seqs*seq_len:(i+1)*n_seqs*seq_len] = _inv_p[2]

for i in range(n_feat):
    for j in range(n_feat):
        ax = axes[i][j]
        if j == 0:
            ax.set_ylabel(axis_labels[i], fontsize='x-large')
        if i == n_feat - 1:
            ax.set_xlabel(axis_labels[j], fontsize='x-large')
            
        if j == i:
            ax.hist(inv_p[i], bins=50, range=axis_ranges[i], color='orange')
        else:
            _x = inv_p[j]#.flatten().cpu().detach().numpy()
            _y = inv_p[i]#.flatten().cpu().detach().numpy()
            
            if i > j:
                ax.hist2d(_x, _y, bins=50, range=[axis_ranges[j], axis_ranges[i]])
            else:
                ax.remove()
            #    ax.hist2d(_x, _y, bins=50, range=[[-1, 1], [-1, 1]], norm=mcolors.PowerNorm(0.5))
plt.savefig(output_dir+'feature_matrix_fake.png', bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(n_feat, n_feat, figsize=(12,12))
#plt.subplots_adjust(wspace=0.075, hspace=0.075)
axis_labels = [r'log($E$ [MeV])', r'log($t$ [ns])', 'DCA [mm]']
axis_ranges = [[np.log10(data.edep).min(), np.log10(data.edep).max()], 
    [np.log10(data.t).min(), np.log10(data.t).max()], [data.doca.min(), data.doca.max()]]
inv_p = np.array([np.log10(data.edep), np.log10(data.t), data.doca])
for i in range(n_feat):
    for j in range(n_feat):
        ax = axes[i][j]
        if j == 0:
            ax.set_ylabel(axis_labels[i], fontsize='x-large')
        if i == n_feat - 1:
            ax.set_xlabel(axis_labels[j], fontsize='x-large')
            
        if j == i:
            ax.hist(inv_p[i], bins=50, range=axis_ranges[i])
        else:
            _x = inv_p[j]#.flatten().cpu().detach().numpy()
            _y = inv_p[i]#.flatten().cpu().detach().numpy()
            
            if i > j:
                ax.hist2d(_x, _y, bins=50, range=[axis_ranges[j], axis_ranges[i]])
            else:
                ax.remove()
            #    ax.hist2d(_x, _y, bins=50, range=[[-1, 1], [-1, 1]], norm=mcolors.PowerNorm(0.5))
plt.savefig(output_dir+'feature_matrix_real.png', bbox_inches='tight')
plt.close()

#############
# Scatterplot comparison
p, w = sample_fake(1, tau)

fw = torch.argmax(w, dim=1).flatten().detach().cpu()
inv_p = data.inv_preprocess(p.permute(0, 2, 1).flatten(0, 1))
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title('GAN')
ax[0].scatter(gu.wire_x[fw], gu.wire_y[fw], s=1+inv_p[:,0] * 1e3, c=inv_p[:,2], cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8)
ax[0].set_aspect(1.0)
gu.draw_cdc(ax[0])

#print("WARNING: We're using the first sequence everytime (l404)")
first_idx = np.random.randint(0, data.edep.size - seq_len)
#first_idx = first_idx - first_idx % 2048
last_idx = first_idx + seq_len
ax[1].set_title('G4')
ax[1].scatter(gu.wire_x[data.wire[first_idx:last_idx]], gu.wire_y[data.wire[first_idx:last_idx]], s=1+data.edep[first_idx:last_idx]*1e3, c=data.doca[first_idx:last_idx], cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8)
ax[1].set_aspect(1.0)
gu.draw_cdc(ax[1])
plt.savefig(output_dir+'comp_scatter.png')
plt.close()

# Real data scatter plot (event display)
plt.figure()
plt.scatter(gu.wire_x[data.wire[first_idx:last_idx]], gu.wire_y[data.wire[first_idx:last_idx]], s=1+data.edep[first_idx:last_idx]*1e3, c=data.doca[first_idx:last_idx], cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8)
scatter_l = lines.Line2D(gu.wire_x[data.wire[first_idx:last_idx]], gu.wire_y[data.wire[first_idx:last_idx]],
        linewidth=0.2, color='gray', alpha=0.7)
ax = plt.gca()
gu.draw_cdc(ax)
ax.set_aspect(1.0)
ax.add_line(scatter_l)
plt.savefig(output_dir+'real_scatter.png', dpi=240)
plt.close()

# No lines
plt.figure()
plt.scatter(gu.wire_x[data.wire[first_idx:last_idx]], gu.wire_y[data.wire[first_idx:last_idx]], s=1+data.edep[first_idx:last_idx]*1e3, c=data.doca[first_idx:last_idx], cmap='inferno',
        vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8)
ax = plt.gca()
gu.draw_cdc(ax)
ax.set_aspect(1.0)
plt.savefig(output_dir+'real_scatter_nolines.png', dpi=240)
plt.close()

# Only lines
plt.figure()
scatter_l = lines.Line2D(gu.wire_x[data.wire[first_idx:last_idx]], 
        gu.wire_y[data.wire[first_idx:last_idx]],
        linewidth=0.1, color='gray', alpha=0.7)
ax = plt.gca()
gu.draw_cdc(ax)
ax.set_aspect(1.0)
ax.add_line(scatter_l)
plt.savefig(output_dir+'real_scatter_onlylines.png', dpi=240)
plt.close()

p, w = sample_fake(8, tau)#data.wire.size // seq_len, tau)
fw = torch.argmax(w, dim=1).flatten().detach().cpu()
fx = gu.wire_x[fw]
fy = gu.wire_y[fw]
inv_p = data.inv_preprocess(p.permute(0, 2, 1).flatten(0, 1))
ft = inv_p[:,1]
fake_time_diff = ft[1:] - ft[:-1]
# Distance distribution comparison
plt.figure()
# Real
x = gu.wire_x[data.wire]
y = gu.wire_y[data.wire]
fake_dist = np.sqrt((fx[1:] - fx[:-1])**2 + (fy[1:] - fy[:-1])**2)
real_dist = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
#_range = [min(real_dist.min(), fake_dist.min()), max(real_dist.max(), fake_dist.max())]
lrd = np.log10(real_dist[real_dist>0])
lfd = np.log10(fake_dist[fake_dist>0])
_range = [min(lrd.min(), lfd.min()), max(lrd.max(), lfd.max())]
plt.hist(np.log10(real_dist[real_dist>0]), bins=150, alpha=0.7, density=True, range=_range)
plt.hist(np.log10(fake_dist[fake_dist>0]), bins=150, alpha=0.7, density=True, range=_range)
plt.savefig(output_dir+'comp_dist.png')
plt.xlabel('Distance [mm]')
plt.close()

# Time difference distribution comparison
plt.figure()
t = data.t[first_idx:last_idx]
t_diff = t[1:] - t[:-1]
_range = [min(t_diff.min(), fake_time_diff.min().item()),
        max(t_diff.max(), fake_time_diff.max().item())]
plt.hist(np.log10(np.abs(t_diff)+1e-8), bins=50, alpha=0.7, density=True)#, range=_range)
plt.hist(np.log10(np.abs(fake_time_diff)+1e-8), bins=50, alpha=0.7, density=True)#, range=_range)
plt.yscale('log')
plt.xlabel('Time difference [ns]')
plt.savefig(output_dir+'comp_time_diff.png')
plt.close()


##########
# Feature histogram comparison

fw = torch.argmax(w, dim=1).flatten().detach().cpu()
inv_p = data.inv_preprocess(p.permute(0, 2, 1).flatten(0, 1))

plt.figure()
_min = np.log10(data.edep).min()
_max = np.log10(data.edep).max()
plt.hist(np.log10(data.edep), bins=50, alpha=0.7, density=True, label='G4', range=[_min, _max])
plt.hist(np.log10(inv_p[:,0].cpu()), bins=50, alpha=0.7, density=True,
        label='GAN', range=[_min, _max])
plt.xlabel('log(Edep [MeV])')
plt.legend()
plt.savefig(output_dir+'comp_edep.png')
plt.close()

plt.figure()
_min = np.log10(data.t).min()
_max = np.log10(data.t).max()
plt.hist(np.log10(data.t), bins=50, alpha=0.7, density=True, label='G4', range=[_min, _max])
plt.hist(np.log10(inv_p[:,1].cpu()), bins=50, alpha=0.7, density=True,
        label='GAN', range=[_min, _max])
plt.xlabel('log(t [ns])')
plt.legend()
plt.savefig(output_dir+'comp_t.png')
plt.close()

plt.figure()
_min = data.doca.min()
_max = data.doca.max()
plt.hist(data.doca, bins=50, alpha=0.7, density=True, label='G4', range=[_min, _max])
plt.hist(inv_p[:,2].cpu(), bins=50, alpha=0.7, density=True, label='GAN', range=[_min, _max])
plt.xlabel('doca [mm]')
plt.legend()
plt.savefig(output_dir+'comp_doca.png')
plt.close()

# Loss plots
plt.figure()
if len(generator_losses) > 0:
    n_critic = len(discriminator_losses) // len(generator_losses)
else:
    n_critic = 1
plt.plot(np.linspace(0, n_epochs, num=len(discriminator_losses)), discriminator_losses,
        label='Discriminator', alpha=0.7)
plt.plot(np.linspace(0, n_epochs, num=len(generator_losses)), generator_losses, alpha=0.7,
        label='Generator')
plt.ylabel('WGAN-GP loss')
plt.xlabel('Epoch')
#plt.ylim(-200, 200)
plt.legend()
plt.savefig(output_dir+'losses.png')
plt.close()

# Critic score (-(D loss - lambda_gp * GP))
if len(gradient_penalty) > 0:
    plt.figure()
    plt.plot(np.linspace(0, n_epochs, num=len(discriminator_losses)), 
            -(np.array(discriminator_losses) - 10 * np.array(gradient_penalty)))
    plt.ylabel('Critic score')
    plt.xlabel('Epoch')
    plt.savefig(output_dir+'critic_score.png')
    plt.close()

# GP loss plot
plt.figure()
plt.plot(np.linspace(0, n_epochs, num=len(gradient_penalty)), gradient_penalty)
plt.ylabel('Gradient penalty')
plt.xlabel('Epoch')
plt.savefig(output_dir+'gp.png')
plt.close()

# Validation loss plot
if len(validation_losses) > 0:
    plt.figure()
    plt.plot(np.linspace(0, n_epochs, 
        num=len(discriminator_losses)), discriminator_losses, alpha=0.7)
    plt.plot(np.linspace(0, n_epochs, 
        num=len(validation_losses)), validation_losses, alpha=0.7)
    plt.savefig(output_dir + 'val_loss.png')
    plt.close()

# Sample real sequences of 2048 hits from the training set.
# Real data in evaluation is not pre-processed, so it does not need to be inv_preprocessed.
def sample_real(batch_size, idx=None):
    if idx is None:
        idx = np.random.randint(0, data.edep.size - seq_len, size=(batch_size,))
        idx = idx - (idx % 2048)
    else:
        idx = np.array(idx)
    start = idx
    stop = idx + seq_len
    slices = np.zeros((batch_size, seq_len), dtype=np.int64)
    for i in range(batch_size):
        slices[i] = np.r_[start[i]:stop[i]] 
    edep = data.edep[slices]
    t = data.t[slices]
    doca = data.doca[slices]
    w = data.wire[slices]
    #one_hot_w = F.one_hot(w, num_classes=gu.cum_n_wires[-1]).squeeze(1).permute(0, 2, 1)
    return np.array([edep, t, doca]), w

_p, _w = sample_real(12)

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12,12))
n_samples=4
gs1 = gridspec.GridSpec(n_samples, n_samples, figure=fig, wspace=0.025, hspace=0.05)
for i in range(n_samples):
    for j in range(n_samples):
        p, w = sample_real(1, idx=[(i*n_samples + j) * seq_len % data.wire.size])
        p = p.squeeze()
        w = w.squeeze()
        #plt.title('G4')
        ax = plt.subplot(gs1[i*4 + j])
        gu.draw_cdc(ax)
        ax.set_aspect(1)
        ax.scatter(gu.wire_x[w], gu.wire_y[w], s=1+p[0] * 1e2+0, alpha=0.7, c=p[2], cmap='inferno', vmin=data.doca.min(), vmax=data.doca.max())
        ax.axis('off')
plt.savefig(output_dir+'grid_real.png', bbox_inches=None, dpi=240)
plt.close()

# Same for fake samples
fig = plt.figure(figsize=(12,12))
n_samples=4
gs1 = gridspec.GridSpec(n_samples, n_samples, figure=fig, wspace=0.025, hspace=0.05)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
for i in range(n_samples):
    for j in range(n_samples):
        #z = to_device(torch.zeros(1, latent_dims))
        #z[:,:latent_dims//2] = 1.0 * (i / (n_samples-1) * 2 - 1)
        #z[:,latent_dims//2:] = 1.0 * (j / (n_samples-1) * 2 - 1)
        z = to_device(torch.randn(1, latent_dims))
        p, w = gen(z)
        p = data.inv_preprocess(p.permute(0,2,1).flatten(0,1))
        p = p.squeeze().detach().cpu().T
        w = torch.argmax(w, dim=1).squeeze().detach().cpu()
        #plt.title('G4')
        ax = plt.subplot(gs1[i*4 + j])
        gu.draw_cdc(ax)
        ax.set_aspect(1)
        ax.scatter(gu.wire_x[w], gu.wire_y[w], s=1+p[0] * 1e2+0, alpha=0.7, c=p[2], cmap='inferno', vmin=data.doca.min(), vmax=data.doca.max())
        ax.axis('off')
plt.savefig(output_dir+'grid_fake.png', dpi=240)
plt.close()

# Activated wires per sequence histogram
n_seq = 256
n_fake_uq = np.zeros(n_seq, dtype=int)
n_real_uq = np.zeros(n_seq, dtype=int)
plt.figure()
for i in range(n_seq):
    fake_p, fake_w = sample_fake(1, tau)
    real_p, real_w = sample_real(1)

    fw = torch.argmax(fake_w, dim=1).flatten().detach().cpu()
    rw = real_w.squeeze()

    n_fake_uq[i] = np.unique(fw).size
    n_real_uq[i] = np.unique(rw).size

_range = [min(n_real_uq.min(), n_fake_uq.min()), max(n_real_uq.max(), n_fake_uq.max())]
plt.hist(n_real_uq, bins=50, alpha=0.7, label='G4', range=_range)
plt.hist(n_fake_uq, bins=50, alpha=0.7, label='GAN', range=_range)
plt.legend()
plt.title('Number of activated wires per sequence')
plt.savefig(output_dir+'activated_wires.png')
plt.close()

# Wire, Radius and Theta plots
n_seqs = 16
n = data.wire.size // seq_len // n_seqs
fake_wire = np.zeros(n_seqs * seq_len * n, dtype=int)
fake_edep = np.zeros(n_seqs * seq_len * n)
fake_t = np.zeros(n_seqs * seq_len * n)
fake_doca = np.zeros(n_seqs * seq_len * n)
for i in range(n):
    with torch.no_grad():
        gen.eval()
        
        latent_var = to_device(torch.randn((n_seqs, latent_dims)))
        x, w = gen(latent_var)
        p = x[:,:n_features]

        fake_wire[i*seq_len*n_seqs:(i+1)*seq_len*n_seqs] = torch.argmax(w, dim=1).cpu().flatten()
        inv_p = data.inv_preprocess(p.permute(0,2,1).flatten(0,1)) 
        fake_edep[i*seq_len*n_seqs:(i+1)*seq_len*n_seqs] = inv_p[:,0]
        fake_t[i*seq_len*n_seqs:(i+1)*seq_len*n_seqs] = inv_p[:,1]
        fake_doca[i*seq_len*n_seqs:(i+1)*seq_len*n_seqs] = inv_p[:,2]


plt.figure()
plt.hist(data.wire, bins=200, alpha=0.7, density=True, label='G4');
plt.hist(fake_wire, bins=200, alpha=0.7, density=True, label='GAN');
plt.legend()
plt.savefig(output_dir+'comp_wire.png')
plt.close()


plt.figure()
fake_radius = np.sqrt(gu.wire_x[fake_wire]**2 + gu.wire_y[fake_wire]**2)
real_radius = np.sqrt(gu.wire_x[data.wire]**2 + gu.wire_y[data.wire]**2)
real_layer = (np.round((gu.n_layers-1) * (real_radius - real_radius.min()) / (real_radius.max() - real_radius.min()))).astype(int)
fake_layer = (np.round((gu.n_layers-1) * (fake_radius - real_radius.min()) / (real_radius.max() - real_radius.min()))).astype(int)
plt.hist(real_layer, bins=gu.n_layers, range=[0,gu.n_layers], rwidth=0.8, align='left',
        alpha=0.7, label='G4', density=True)
plt.hist(fake_layer, bins=gu.n_layers, range=[0,gu.n_layers], rwidth=0.8, align='left',
        alpha=0.7, label='GAN', density=True)
plt.legend()
plt.xticks(np.arange(0, gu.n_layers))
plt.xlabel('Layer ID')
plt.savefig(output_dir+'comp_layer.png', dpi=240)
plt.close()
#plt.figure()
#plt.hist(real_radius, alpha=0.7, density=True, label='G4', bins=50)
#        #range=[real_radius.min(), real_radius.max()], label='G4')
#plt.hist(fake_radius, alpha=0.7, density=True, label='GAN', bins=50);
#        #range=[real_radius.min(), real_radius.max()], label='GAN')
#plt.legend()
#plt.savefig(output_dir+'comp_radius.png')

plt.figure()
fake_theta = np.arctan2(gu.wire_y[fake_wire], gu.wire_x[fake_wire])
real_theta = np.arctan2(gu.wire_y[data.wire], gu.wire_x[data.wire])
plt.hist(real_theta, bins=50, alpha=0.7, density=True, label='G4');
plt.hist(fake_theta, bins=50, alpha=0.7, density=True, label='GAN');
plt.legend()
plt.savefig(output_dir+'comp_theta.png')
plt.close()

# Edep per layer comparison
plt.figure()
fake_edep_pl = np.zeros(gu.n_layers)
real_edep_pl = np.zeros(gu.n_layers)
for i in range(gu.n_layers):
    fake_edep_pl[i] = fake_edep[fake_layer == i].sum()    
    real_edep_pl[i] = data.edep[real_layer == i].sum()

plt.bar(np.arange(0, gu.n_layers), real_edep_pl, alpha=0.7, width=0.8, label='G4')
plt.bar(np.arange(0, gu.n_layers), fake_edep_pl, alpha=0.7, width=0.8, label='GAN')
plt.xticks(np.arange(0, gu.n_layers))
plt.legend()
plt.savefig(output_dir+'comp_edep_per_layer.png')
plt.close()

print('Eval of job %d at epoch %d with networks %d done.' % (args.job_id, args.epoch, args.net_version))
