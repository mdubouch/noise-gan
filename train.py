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
import importlib
networks = importlib.import_module('networks%d' % (args.net_version))
print('Importing networks from "%s"...' % (networks.__name__))
gen = to_device(networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len))
logging.info(gen)
disc = to_device(networks.Disc(ndf=ndf, seq_len=seq_len))
logging.info(disc)
logging.info('generator params: %d' % (networks.get_n_params(gen)))
logging.info('discriminator params: %d' % (networks.get_n_params(disc)))

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
plt.figure(figsize=(6,6))
plt.scatter(gu.wire_x, gu.wire_y, s=1, c=gu.layer)
plt.savefig(output_dir+'wire_position.png', dpi=120)
plt.clf()

print('Pre-processing...')
train_minmax = data.preprocess()
data.diagnostic_plots(output_dir)

train_loader, train_dataset, n_chunks = data.chunk(seq_len, batch_size=12)
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

optimizer_gen = torch.optim.Adam(gen.parameters(),  lr=1e-4, betas=(0.5, 0.999))
optimizer_disc = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

noise_level = 0.1
def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d): # or isinstance(m, nn.Linear) # or isinstance(m, nn.BatchNorm1d):# or isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0., 0.008)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

gen.apply(weight_init);
disc.apply(weight_init);

def add_noise(x, noise_level, clamp_min, clamp_max):
    return torch.clamp(x + torch.randn_like(x) * noise_level, clamp_min, clamp_max)

print('Training begin')
import time
import torch.autograd as autograd
from tqdm import tqdm

def save_states(epoch):
    states = { 'disc': disc.state_dict(), 'd_opt': optimizer_disc.state_dict(), 
            'd_loss': discriminator_losses, 'gen': gen.state_dict(), 
            'g_opt': optimizer_gen.state_dict(), 'g_loss': generator_losses, 
            'tau': tau, 'n_epochs': epoch, 'qt': data.qt, 'minmax': data.minmax }

    torch.save(states, output_dir + 'states_%d.pt' % (epoch+1))
    print("Saved after epoch %d to" % (epoch), output_dir + '/states_%d.pt' % (epoch+1))

# Implement "Gradient Penalty" for WGAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
def gradient_penalty(disc, real_p, real_w, fake_p, fake_w):
    eps = to_device(torch.rand((1,)))

    interpolates_p = (eps * real_p + (1-eps) * fake_p).requires_grad_(True)
    #interpolates_w = (eps * real_w + (1-eps) * fake_w).requires_grad_(True)
    interpolates_w = real_w
    #if (eps.mean() < 0):
        #interpolates_w = fake_w
    d_interpolates = disc(interpolates_p, interpolates_w).squeeze()
    grad_outputs_p = to_device(torch.ones(d_interpolates.shape, requires_grad=False))
    grad_outputs_w = to_device(torch.ones(d_interpolates.shape, requires_grad=False))


    gradients_p = autograd.grad(outputs=d_interpolates,
                              inputs=interpolates_p,
                              grad_outputs=grad_outputs_p,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True
    )[0]
    gradients_w = autograd.grad(outputs=d_interpolates,
                              inputs=interpolates_w,
                              grad_outputs=grad_outputs_w,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True,
                              allow_unused=True
    )[0]
    gradients_p = gradients_p.reshape(gradients_p.shape[0], -1) + 1e-8
    gradients_w = gradients_w.reshape(gradients_w.shape[0], -1) + 1e-8
    gradient_pen = ((gradients_p.norm(2, dim=1) - 1)**2).mean() + 0.1 * ((gradients_w.norm(2, dim=1) - 1)**2).mean()
    return gradient_pen

gen.train()
disc.train()
lambda_gp = 10
n_critic = 2
for e in range(n_epochs):
    logging.info('Epoch %d' % (e))
    for i, (real_p, real_w) in enumerate(train_loader):

        optimizer_disc.zero_grad()

        disc.train()
        gen.train()

        # Take loss between real samples and objective 1.0
        real_p = to_device(real_p)
        real_w = to_device(real_w)
        real_w = F.one_hot(real_w, num_classes=gu.cum_n_wires[-1]).squeeze(1).permute(0, 2, 1).float().requires_grad_(True)

        out_real = disc(add_noise(real_p, noise_level, -1, 1), real_w)
        D_loss_real = out_real

        fake_p, fake_w = sample_fake(real_p.shape[0], tau)
        out_fake = disc(fake_p.detach(), fake_w.detach())
        D_loss_fake = out_fake

        gp = gradient_penalty(disc, add_noise(real_p, noise_level, -1, 1), real_w, 
                              fake_p, fake_w)
        
        D_loss = -torch.mean(D_loss_real) + torch.mean(D_loss_fake) + lambda_gp * gp
        discriminator_losses.append(D_loss.item())
        D_loss.backward()
        optimizer_disc.step()

        if (i % n_critic == 0):
            # Generator update
            disc.eval()
            gen.train()
            optimizer_gen.zero_grad()
            fake_hits = sample_fake(real_p.shape[0], tau)
            fake_p = fake_hits[0]
            fake_w = fake_hits[1]
            out_fake = disc(*fake_hits)

            
            var_loss = torch.mean((fake_p.var(2).mean(0) - real_p.var(2).mean(0))**2)
            G_loss = -torch.mean(out_fake)            

            generator_losses.append(G_loss.item())

            G_loss.backward()
            optimizer_gen.step()

            if (tau > 1e-4):
                tau *= 0.9995

        if (noise_level > 1e-4):
            noise_level *= 0.999
    if ((e+1) % 5) == 0:
        save_states(e)



print('Done')

print('Saving models...')



print(len(discriminator_losses), len(train_loader))
n_epochs = len(discriminator_losses) // len(train_loader)
save_states(n_epochs)
