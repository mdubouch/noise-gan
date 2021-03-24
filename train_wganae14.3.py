#!/usr/bin/python3
#$ -P P_comet
#$ -j y
#$ -cwd
#$ -M m.dubouchet18@imperial.ac.uk
#$ -m be
#$ -q mc_gpu_long
#$ -pe multicores_gpu 4
#$ -l sps=1,GPU=1,GPUtype=V100

import os
import sys
sys.path.append(os.getcwd())

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


parser = argparse.ArgumentParser('Train CDC GAN')
parser.add_argument('--n-epochs', type=int, default=1)
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--latent-dims', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=2048)
parser.add_argument('--net-version', type=int)
parser.add_argument('--log', type=str, default='info')
parser.add_argument('--gfx', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--continue-from-epoch', type=int)
parser.add_argument('--continue-from-job', type=int)
args = parser.parse_args()
job_id = int(os.getenv('JOB_ID', default='0'))
output_dir = 'output_%d/' % (job_id)
print('Outputting to %s' % (output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.basicConfig(filename=output_dir+'output.log', level=getattr(logging, args.log.upper()), format='%(asctime)s %(message)s')

n_epochs = args.n_epochs
ngf = args.ngf
ndf = args.ndf
logging.info('ndf=%d' % (ndf))
logging.info('ngf=%d' % (ngf))
latent_dims = args.latent_dims
seq_len = args.sequence_length
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    logging.info('Running on GPU: %s' % (torch.cuda.get_device_name()))
else:
    logging.info('Running on CPU')

def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

print('Import networks version %d' % (args.net_version))
logging.info('networks=%d' % (args.net_version))
import importlib
networks = importlib.import_module('networks%d' % (args.net_version))
print('Importing networks from "%s"...' % (networks.__name__))
gen = to_device(networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len, 
    encoded_dim=16))
logging.info(gen)
disc = to_device(networks.Disc(ndf=ndf, seq_len=seq_len, encoded_dim=16))
logging.info(disc)
ae = to_device(networks.AE(encoded_dim=16))
print('generator params: %d' % (networks.get_n_params(gen)))
print('discriminator params: %d' % (networks.get_n_params(disc)))
print('AE params: %d' % (networks.get_n_params(ae)))
logging.info('generator params: %d' % (networks.get_n_params(gen)))
logging.info('discriminator params: %d' % (networks.get_n_params(disc)))
logging.info('AE params: %d' % (networks.get_n_params(ae)))

#print('Importing geometry...')
#import geom_util as gu
#logging.info('cumulative wires {0}'.format(gu.cum_n_wires))

print('Importing dataset...')
import dataset
data = dataset.Data()
data.load()
logging.info('pot %d  bunches %d', data.n_pot, data.n_bunches)
logging.info('dtypes {0}'.format(data.data.dtype))
logging.info('shape {0}'.format(data.data.shape))

import geom_util
gu = geom_util.GeomUtil(data.get_cdc_tree())
gu.validate_wire_pos()

print(data.get_cdc_tree().shape, data.get_cdc_tree().dtype)
import matplotlib
if args.gfx:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['savefig.facecolor'] = 'white'
plt.figure(figsize=(6,6))
plt.scatter(gu.wire_x, gu.wire_y, s=1, c=gu.layer)
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.savefig(output_dir+'wire_position.png', dpi=120)
plt.clf()

print('Pre-processing...')
train_minmax = data.preprocess()
data.diagnostic_plots(output_dir)

train_loader, train_dataset, n_chunks = data.chunk(seq_len, batch_size=32)
print(train_dataset[0:4][0].shape)

def sample_real(batch_size):
    idx = np.random.choice(np.arange(n_chunks), size=batch_size)
    p, w = train_dataset[idx]
    one_hot_w = F.one_hot(w, num_classes=gu.cum_n_wires[-1]).squeeze(1).permute(0, 2, 1)
    # Return shape is (batch, feature, seq)
    return p, one_hot_w
def sample_fake(batch_size, tau):
    noise = to_device(torch.randn((batch_size, latent_dims), requires_grad=True))
    sample = gen(noise, 0.0, tau)
    return sample

_p, _w = sample_real(2)
print(_p.shape, _w.shape)

__f = sample_fake(2, 1.0)
print(__f[0].shape, __f[1].shape)

tau = 10
discriminator_losses = []
generator_losses = []
occupancy_losses = []
gradient_pen_hist = []
ae_losses = []
dist_losses = []
start_epoch = 0

optimizer_gen = torch.optim.Adam(gen.parameters(),  lr=1e-4, betas=(0.0, 0.9))
optimizer_disc = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.0, 0.9))
optimizer_ae = torch.optim.AdamW(ae.parameters(), lr=1e-4)

noise_level = 0.00
def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d): # or isinstance(m, nn.Linear) # or isinstance(m, nn.BatchNorm1d):# or isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0., 0.008)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

#gen.apply(weight_init);
#disc.apply(weight_init);

if args.continue_from_epoch is not None:
    path = ''
    if args.continue_from_job is not None:
        path = 'output_%d/states_%d.pt' % (args.continue_from_job, args.continue_from_epoch)
    else:
        path = output_dir+'states_%d.pt' % (args.continue_from_epoch)
    print('Loading GAN states from %s...' % (path))
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    states = torch.load(path, map_location=device)
    disc.load_state_dict(states['disc'])
    optimizer_disc.load_state_dict(states['d_opt'])
    discriminator_losses = states['d_loss']
    gen.load_state_dict(states['gen'])
    optimizer_gen.load_state_dict(states['g_opt'])
    generator_losses = states['g_loss']
    tau = states['tau']
    start_epoch = states['n_epochs']
    print('Starting from', start_epoch)
    #data.qt = states['qt']
    #data.minmax = states['minmax']
    occupancy_losses = states['occupancy_loss']
    if 'gradient_penalty' in states:
        gradient_pen_hist = states['gradient_penalty']
    if 'ae_loss' in states:
        ae_losses = states['ae_loss']
    if 'dist_loss' in states:
        dist_losses = states['dist_loss']
    print('OK')

def add_noise(x, noise_level, clamp_min, clamp_max):
    #return torch.clamp(x + torch.randn_like(x) * noise_level, clamp_min, clamp_max)
    return x

print('Training begin')
import time
import torch.autograd as autograd
from tqdm import tqdm

def save_states(epoch):
    states = { 'disc': disc.state_dict(), 'd_opt': optimizer_disc.state_dict(), 
            'd_loss': discriminator_losses, 'gen': gen.state_dict(), 
            'g_opt': optimizer_gen.state_dict(), 'g_loss': generator_losses, 
            'tau': tau, 'n_epochs': epoch, 'qt': data.qt, 'minmax': data.minmax,
            'occupancy_loss': occupancy_losses, 'gradient_penalty': gradient_pen_hist,
            'ae': ae.state_dict(),
            'ae_loss': ae_losses, 'dist_loss': dist_losses }

    torch.save(states, output_dir + 'states_%d.pt' % (epoch))
    print("Saved after epoch %d to" % (epoch), output_dir + '/states_%d.pt' % (epoch))

wire_to_xy = torch.tensor([gu.wire_x, gu.wire_y], device='cuda', dtype=torch.float32)
wire_to_xy = wire_to_xy / wire_to_xy.max()
print('wire_to_xy:', wire_to_xy.shape)
real_dist_matrix = torch.cdist(wire_to_xy.T, wire_to_xy.T)
print(real_dist_matrix)
print(real_dist_matrix.shape)
print(wire_to_xy[0].max(), wire_to_xy[1].max())
print(wire_to_xy[0].min(), wire_to_xy[1].min())
print(torch.norm(wire_to_xy, dim=0))

def concatenate_p_w(p, w):
    return torch.cat([p, w], dim=1)

# Implement "Gradient Penalty" for WGAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
def gradient_penalty(disc, interpolates_p, interpolates_w):
    interp_x = concatenate_p_w(interpolates_p, interpolates_w)
    d_interpolates = disc(interp_x).squeeze()
    grad_outputs_x = to_device(torch.ones(d_interpolates.shape, requires_grad=False))


    gradients_x = autograd.grad(outputs=d_interpolates,
                              inputs=interp_x,
                              grad_outputs=grad_outputs_x,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True
    )[0]

    gradients_x = gradients_x.reshape(gradients_x.shape[0], -1) + 1e-8
    gradient_pen = ((gradients_x.norm(2, dim=1) - 1)**2).mean()
    return gradient_pen

gen.train()
disc.train()
ae_loss_fn = nn.BCEWithLogitsLoss()
lambda_gp = 10
n_critic = 5
for e in range(start_epoch, start_epoch + n_epochs):
    logging.info('Epoch %d' % (e))
    print('Epoch %d' % (e))
    for i, (real_p, real_w) in enumerate(train_loader):


        disc.train()
        gen.train()

        # AE step: compute loss between dec(enc(x)) and x
        # x is a sample of wires in one-hot format
        # enc(x) is disc.enc(x), dec(x) is gen.dec(x)
        real_w_ohe = to_device(F.one_hot(real_w, 
            num_classes=gu.cum_n_wires[-1]).squeeze(1).permute(0, 2, 1).float()).requires_grad_(True)
        optimizer_ae.zero_grad()
        enc = ae.enc(real_w_ohe)
        dec = ae.dec(enc)

        tgt = real_w_ohe.permute(0,2,1).flatten(0, 1)
        pred = dec.permute(0,2,1).flatten(0,1)

        ae_loss = ae_loss_fn(pred, tgt)

        all_wires = F.one_hot(torch.arange(gu.n_wires), 
                num_classes=gu.n_wires).float().cuda().requires_grad_(True)

        enc = ae.enc(all_wires.view(gu.n_wires, gu.n_wires, 1)).squeeze()
        # enc (n_wires, enc_dim)
        fake_dist_matrix = enc.view(gu.n_wires, 1, -1) - enc.view(1, gu.n_wires, -1)
        fake_dist_matrix = torch.sqrt(1e-7 + (fake_dist_matrix**2).sum(dim=2))

        dist_loss = torch.mean((fake_dist_matrix - real_dist_matrix)**2)

        enc_loss = ae_loss + 5.0 * dist_loss

        enc_loss.backward()
        optimizer_ae.step()

        ae_losses.append(ae_loss.item())
        dist_losses.append(dist_loss.item())

        # Pre-train the AE for the first epoch
        if e < 5:
            continue

        # Critic optimization step
        optimizer_disc.zero_grad()

        # Take loss between real samples and objective 1.0
        real_p = to_device(real_p).requires_grad_(True)
        enc_real_w = ae.enc(real_w_ohe)

        real_x = concatenate_p_w(real_p, enc_real_w)
        out_real = disc(real_x)
        D_loss_real = out_real

        fake_p, fake_w = sample_fake(real_p.shape[0], tau)

        fake_x = concatenate_p_w(fake_p, fake_w).detach()
        out_fake = disc(fake_x)
        D_loss_fake = out_fake
        real_enc_w = ae.enc(real_w_ohe)
        fake_enc_w = fake_w
        eps = to_device(torch.rand((real_p.shape[0], 1, 1)))
        interpolates_p = (eps * real_p + (1-eps) * fake_p).requires_grad_(True)
        interpolates_enc_w = (eps * real_enc_w + (1-eps) * fake_enc_w).requires_grad_(True)
        interpolates_w = interpolates_enc_w
        gp = gradient_penalty(disc, interpolates_p, interpolates_w)
        gradient_pen_hist.append(gp.item())
        
        D_loss = -torch.mean(D_loss_real) + torch.mean(D_loss_fake) + lambda_gp * gp
        discriminator_losses.append(D_loss.item())
        D_loss.backward()
        optimizer_disc.step()


        if (i % n_critic == 0):
            # Generator update
            disc.train()
            gen.train()
            optimizer_gen.zero_grad()
            fake_hits = sample_fake(real_p.shape[0], tau)
            fake_p = fake_hits[0]
            fake_w = fake_hits[1]
            fake_x = concatenate_p_w(fake_p, fake_w)
            #fake_wx = torch.tensordot(wire_to_xy, fake_w, dims=([1], [1])).permute(1, 0, 2)
            out_fake = disc(fake_x)

            G_loss = -torch.mean(out_fake)            

            generator_losses.append(G_loss.item())

            G_loss.backward()
            optimizer_gen.step()

            if (tau > 1e-1):
                tau *= 0.99#95

        if (noise_level > 1e-4):
            noise_level *= 0.999
        logging.info('noise level %f' % (noise_level))
        logging.info('tau %f' % (tau))
    if ((e+1) % 5) == 0:
        save_states(e+1)



print('Done')

print('Saving models...')



print(start_epoch + n_epochs)
save_states(start_epoch + n_epochs)
