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
parser.add_argument('--embedding', type=str, required=True)
parser.add_argument('--net-version', type=int)
parser.add_argument('--dataset', type=str, default='dataset')
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--log', type=str, default='info')
parser.add_argument('--gfx', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--save-every', type=int, default=1)
parser.add_argument('--continue-from-epoch', '--cfe', type=int)
parser.add_argument('--continue-from-job', '--cfj', type=int)
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

import importlib
print('Importing dataset...')
print('Importing dataset from %s.py' % (args.dataset))
dataset = importlib.import_module(args.dataset)

#smplr = dataset.SequenceSampler(np.arange(1000) // 3, 128)
#for s in smplr:
#    #print(s)
#    pass
#exit(0)

data = dataset.Data()
data.load()
logging.info('pot %d  bunches %d', data.n_pot, data.n_bunches)
logging.info('dtypes {0}'.format(data.data.dtype))
logging.info('shape {0}'.format(data.data.shape))

import geom_util
gu = geom_util.GeomUtil(data.get_cdc_tree())
gu.validate_wire_pos()

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
        
# Load up the embedding model
from training_state import TrainingState
emb_ts = torch.load(args.embedding, map_location=device)['state']
embed = emb_ts.model
embedding_dim = embed.emb.embedding_dim

print('Import networks version %d' % (args.net_version))
logging.info('networks=%d' % (args.net_version))
networks = importlib.import_module('networks%d' % (args.net_version))
print('Importing networks from "%s"...' % (networks.__name__))
gen = networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len, 
    embedding_dim=embedding_dim, n_wires=gu.n_wires).to(device)
logging.info(gen)
disc = networks.Disc(ndf=ndf, seq_len=seq_len, embedding_dim=embedding_dim, n_wires=gu.n_wires).to(device)
logging.info(disc)
print('Generator params:     {:,}'.format(networks.get_n_params(gen)))
print('Discriminator params: {:,}'.format(networks.get_n_params(disc)))
logging.info('generator params: %d' % (networks.get_n_params(gen)))
logging.info('discriminator params: %d' % (networks.get_n_params(disc)))

logging.info('%s %s' % (data.get_cdc_tree().shape, data.get_cdc_tree().dtype))
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

train_loader, train_dataset, n_chunks = data.chunk(seq_len, batch_size=args.batch_size)
logging.info('%s' % (train_dataset[0:4][0].shape,))


def sample_fake(batch_size):
    noise = torch.randn((batch_size, latent_dims), requires_grad=True).to(device)
    sample = gen(noise)
    return sample

#_p, _w = sample_real(2)
#print(_p.shape, _w.shape)

__f = sample_fake(2)
#print(__f.shape)

# Initialisation of saved variables and lists
tau = 1
discriminator_losses = []
generator_losses = []
gradient_pen_hist = []
start_epoch = 0
validation_losses = []

optimizer_gen = torch.optim.Adam(list(gen.parameters()),
        lr=1e-4, betas=(0.9, 0.999))
optimizer_disc = torch.optim.Adam(list(disc.parameters()),
        lr=1e-4, betas=(0.9, 0.999))

def weight_init_relu(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    #elif classname.find('Norm') != -1:
    #    nn.init.ones_(m.weight)
    #    nn.init.zeros_(m.bias)

    #elif classname.find('Linear') != -1:
    #    nn.init.normal_(m.weight, 0.0, 0.02)
    #    if m.bias is not None:
    #        nn.init.zeros_(m.bias)

def weight_init_leakyrelu(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

if args.continue_from_epoch is not None:
    path = ''
    if args.continue_from_job is not None:
        path = 'output_%d/states_%d.pt' % (args.continue_from_job, args.continue_from_epoch)
    else:
        path = output_dir+'states_%d.pt' % (args.continue_from_epoch)
    print('Loading GAN states from %s...' % (path))
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
    if 'validation_loss' in states:
        validation_losses = states['validation_loss']
    if 'gradient_penalty' in states:
        gradient_pen_hist = states['gradient_penalty']
    print('OK')

else:
    pass
    disc.apply(weight_init_leakyrelu)
    gen.apply(weight_init_relu)


print('Training begin')
import time
import torch.autograd as autograd


def save_states(epoch):
    states = { 'disc': disc.state_dict(), 'd_opt': optimizer_disc.state_dict(), 
            'd_loss': discriminator_losses, 'gen': gen.state_dict(), 
            'g_opt': optimizer_gen.state_dict(), 'g_loss': generator_losses, 
            'tau': tau, 'n_epochs': epoch, 'qt': data.qt, 'minmax': data.minmax,
            'gradient_penalty': gradient_pen_hist, 'validation_loss': validation_losses}

    torch.save(states, output_dir + 'states_%d.pt' % (epoch))
    print("Saved after epoch %d (%d gen its) to" % (epoch, len(generator_losses)), output_dir + '/states_%d.pt' % (epoch))

def wire_hook(grad):
    print('%.2e' % (grad.abs().mean().item()))
    return grad

# Implement "Gradient Penalty" for WGAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
def gradient_penalty(disc, interp):
    interp.requires_grad_()


    d_interpolates = disc(interp).squeeze()
    grad_outputs = torch.ones(d_interpolates.shape, requires_grad=False, device='cuda')

    gradients = autograd.grad(outputs=d_interpolates,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True, # IMPORTANT! Allows to compute gradient with respect to gradient
            only_inputs=False
            )[0]

    gradients_pen = (gradients.pow(2).flatten(1, 2).sum(dim=1).pow(0.5) - 1)**2

    gradient_pen = torch.mean(gradients_pen)

    return gradient_pen

def get_wire_weights():
    wire_counts = np.bincount(data.wire, minlength=gu.n_wires)
    logging.info('{}'.format(wire_counts.shape))
    return torch.tensor(1 / (wire_counts + 1e-1), device='cuda', dtype=torch.float)
wire_weights = get_wire_weights()
logging.info('{}'.format(wire_weights))
logging.info('{}'.format(wire_weights[1600]))
logging.info('{}'.format(wire_weights.shape))

gen.train()
disc.train()
lambda_gp = 10
n_critic = 5
critic_count = 0
for e in range(start_epoch, start_epoch + n_epochs):
    logging.info('Epoch %d, generator iterations %d' % (e, len(generator_losses)))
    print('Epoch %d, generator iterations %d' % (e, len(generator_losses)))
    for i, (real_p, real_w) in enumerate(train_loader):
        if i % 10 == 0:
            print("it %d" % (i))

        # real_p (batch, n_features, seq_len)
        # real_w (batch, 1, seq_len)

        real_p = real_p.cuda().permute(0,2,1)
        real_w_emb = embed.emb(real_w.cuda()).squeeze().permute(0, 2, 1)

        # Critic optimization step
        optimizer_disc.zero_grad()

        # Weight clipping
        #for p in disc.parameters():
        #    p.data.clamp_(-0.01, 0.01)


        out_real = disc(torch.cat([real_p, real_w_emb], dim=1))
        fake_p, fake_wg = sample_fake(real_p.shape[0])

        out_fake = disc(torch.cat([fake_p, fake_wg], dim=1))

        eps = torch.rand((real_p.shape[0], 1, 1), device='cuda')

        interpolates_p = eps * real_p + (1-eps) * fake_p
        interpolates_w = eps * real_w_emb + (1-eps) * fake_wg
        
        gp = gradient_penalty(disc, torch.cat([interpolates_p, interpolates_w], dim=1))
        gradient_pen_hist.append(gp.item())

        D_loss = -out_real.mean() + out_fake.mean() + lambda_gp * gp
        D_loss.backward()
         
        discriminator_losses.append(D_loss.item())
        optimizer_disc.step()

        critic_count += 1
        if (critic_count % n_critic == 0):
            critic_count = 0

            # Generator update
            optimizer_gen.zero_grad()
            fake_p, fake_wg = sample_fake(real_p.shape[0])

            out_fake = disc(torch.cat([fake_p, fake_wg], dim=1))
            G_loss = -out_fake.mean()
            generator_losses.append(G_loss.item())
            G_loss.backward()

            optimizer_gen.step()

    # Calculate validation loss once every epoch
    if (e+1) % 100 == 0:
        _val_loss_values = []
        for val_p, val_w in data.test_loader:
            print('test')
            val_p = val_p.cuda().permute(0,2,1)
            val_w_emb = embed.emb(val_w.cuda()).squeeze().permute(0,2,1)

            out_real = disc(torch.cat([val_p, val_w_emb], dim=1))

            fake_p, fake_wg = sample_fake(val_p.shape[0])

            eps = torch.rand((val_p.shape[0], 1, 1), device='cuda')
            interpolates_p = eps * val_p + (1-eps) * fake_p
            interpolates_w = eps * val_w_emb + (1-eps) * fake_wg
            gp = gradient_penalty(disc, torch.cat([interpolates_p, interpolates_w], dim=1))

            D_loss = -out_real.mean() + out_fake.mean() + lambda_gp * gp

            _val_loss_values.append(D_loss.item())

        validation_losses.append(np.mean(_val_loss_values))


    if ((e+1) % args.save_every) == 0:
        save_states(e+1)



print('Done')

print('Saving models...')

plt.figure()
plt.plot(np.linspace(0, n_epochs, num=len(discriminator_losses)), discriminator_losses, alpha=0.7)
plt.plot(np.linspace(0, n_epochs, num=len(validation_losses)), validation_losses, alpha=0.7)
plt.savefig(output_dir + 'val_loss.png')
plt.close()



print(start_epoch + n_epochs)
save_states(start_epoch + n_epochs)
