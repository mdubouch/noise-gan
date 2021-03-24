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
parser.add_argument('--enc-dim', type=int, default=4)
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--no-pretrain', action='store_true')
parser.add_argument('--log', type=str, default='info')
parser.add_argument('--gfx', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1337)
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
encoded_dim = args.enc_dim
torch.manual_seed(args.seed)
np.random.seed(args.seed)
pretrain_epochs = args.pretrain

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
    encoded_dim=encoded_dim))
logging.info(gen)
disc = to_device(networks.Disc(ndf=ndf, seq_len=seq_len, encoded_dim=encoded_dim))
logging.info(disc)
ae = to_device(networks.VAE(encoded_dim=encoded_dim))
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
import dataset_altered as dataset
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
pretrain_losses = []
pretrain_dist_losses = []
pretrain_acc = []
start_epoch = 0

from adabelief_pytorch import AdaBelief
optimizer_gen = AdaBelief(list(gen.parameters()) + list(ae.dec_net.parameters()),
        lr=2e-4, betas=(0.5, 0.999), eps=1e-12, weight_decay=0.0, rectify=False,
        fixed_decay=False, amsgrad=False)
optimizer_disc = AdaBelief(list(disc.parameters()) + list(ae.enc_net.parameters()),
        lr=2e-4, betas=(0.5, 0.999), eps=1e-12, weight_decay=0.0, rectify=False,
        fixed_decay=False, amsgrad=False)
print(optimizer_disc)
optimizer_ae = torch.optim.Adam(ae.parameters())

noise_level = 0.00
def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d): # or isinstance(m, nn.Linear) # or isinstance(m, nn.BatchNorm1d):# or isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0., 0.08)
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
    ae.load_state_dict(states['ae'])
    optimizer_ae.load_state_dict(states['ae_opt'])
    if 'gradient_penalty' in states:
        gradient_pen_hist = states['gradient_penalty']
    if 'ae_loss' in states:
        ae_losses = states['ae_loss']
    if 'dist_loss' in states:
        dist_losses = states['dist_loss']
    print('OK')

if pretrain_epochs == 0 and args.no_pretrain == False:
    if args.pretrained is not None:
        path = pretrained
    else:
        path = 'ae_states_v11.pt'

    print('Loading pretrained autoencoder from %s...' % (path))
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    states = torch.load(path, map_location=device)
    if args.continue_from_epoch is None:
        ae.load_state_dict(states['ae'])
        optimizer_ae.load_state_dict(states['ae_opt'])
    pretrain_losses = states['pretrain_loss']
    pretrain_dist_losses = states['pretrain_dist_loss']
    pretrain_acc = states['pretrain_acc']
    print('OK')

def add_noise(x, noise_level, clamp_min, clamp_max):
    #return torch.clamp(x + torch.randn_like(x) * noise_level, clamp_min, clamp_max)
    return x

print('Training begin')
import time
import torch.autograd as autograd

def save_states(epoch):
    states = { 'disc': disc.state_dict(), 'd_opt': optimizer_disc.state_dict(), 
            'd_loss': discriminator_losses, 'gen': gen.state_dict(), 
            'g_opt': optimizer_gen.state_dict(), 'g_loss': generator_losses, 
            'tau': tau, 'n_epochs': epoch, 'qt': data.qt, 'minmax': data.minmax,
            'occupancy_loss': occupancy_losses, 'gradient_penalty': gradient_pen_hist,
            'ae': ae.state_dict(), 'ae_opt': optimizer_ae.state_dict(),
            'ae_loss': ae_losses, 'dist_loss': dist_losses,
            'pretrain_loss': pretrain_losses, 'pretrain_dist_loss': pretrain_dist_losses,
            'pretrain_acc': pretrain_acc }

    torch.save(states, output_dir + 'states_%d.pt' % (epoch))
    print("Saved after epoch %d to" % (epoch), output_dir + '/states_%d.pt' % (epoch))

wire_to_xy = torch.tensor([gu.wire_x, gu.wire_y], device='cuda', dtype=torch.float32)
wire_to_xy = wire_to_xy / wire_to_xy.max()
# wire_to_xy (2, 3606)
real_dist_matrix = torch.cdist(wire_to_xy.T, wire_to_xy.T)

def concatenate_p_w_xy(p, w, xy):
    return torch.cat([p, w], dim=1)#, xy], dim=1)

# Implement "Gradient Penalty" for WGAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
def gradient_penalty(disc, interpolates_p, interpolates_w, interpolates_xy):
    interp_x = concatenate_p_w_xy(interpolates_p, interpolates_w, interpolates_xy)
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

ae.train()
ae_loss_fn = nn.CrossEntropyLoss()
# Pre-train AE
print('Pretraining AE...')
norm_losses = []
kld_losses = []
pretrain_noise_lvl = 0.05
for e in range(pretrain_epochs):
    # Make the batch size 1, so we get all wires every time
    bsize = 1024
    wires = torch.tensor(np.random.choice(np.arange(gu.n_wires), size=bsize)).cuda()
    n_its = gu.n_wires // bsize + 1
    for i in range(n_its):
        optimizer_ae.zero_grad()
        # wid (256,)
        wid_ohe = F.one_hot(wires, num_classes=gu.n_wires).float().requires_grad_(True)
        # wid_ohe (3606, 3606)
        enc = ae.enc_net(wid_ohe)
        # enc (256, enc_dim)
        fake_dist_matrix = enc.view(bsize, 1, encoded_dim) - enc.view(1, bsize, encoded_dim)
        # fake_dist_matrix (256, enc_dim, enc_dim)
        fake_dist_matrix = torch.sqrt(1e-16 + (fake_dist_matrix**2).sum(dim=2))

        #print(wire_to_xy.shape)
        r_w = wire_to_xy[:,wires].T
        #print(r_w.shape)
        real_dist_m = r_w.view(bsize, 1, 2) - r_w.view(1, bsize, 2)
        real_dist_m = torch.sqrt(1e-16 + (real_dist_m**2).sum(dim=2))
        #print(real_dist_m.shape)
        #print(real_dist_matrix_sub)
        #print(real_dist_matrix_sub.max())
        #print(fake_dist_matrix.max())
        dist_loss = nn.MSELoss()(fake_dist_matrix, real_dist_m)
        #norm = torch.norm(enc, dim=1)
        #norm_loss = torch.mean((norm - 1)**2)
        #norm_losses.append(norm_loss.item())

        encdec = ae.dec_net(enc)
        # encdec (256, 3606)
        ae_loss = ae_loss_fn(encdec, wires)
        #print(ae_loss, dist_loss)
        loss = ae_loss
        #print(ae.enc_net.kl_loss)
        loss.backward()
        optimizer_ae.step()
        pretrain_losses.append(ae_loss.item())
        pretrain_dist_losses.append(dist_loss.item())

        choice = torch.argmax(encdec, dim=1)
        hit = (wires == choice).sum().float()
        acc = hit / wires.shape[0]
        pretrain_acc.append(acc.item())

    #if pretrain_noise_lvl > 0.001:
        #pretrain_noise_lvl *= 0.999
    

    if e % 200 == 0:
        print('Epoch', e)
        print('cross entropy loss:', np.mean(pretrain_losses[-100:]))
        print('accuracy:', np.mean(pretrain_acc[-100:]))
        print('KL loss:', np.mean(kld_losses[-100:]))
        print('dist loss:', np.mean(pretrain_dist_losses[-100:]))
        print('norm loss:', np.mean(norm_losses[-100:]))
        print('noise_lvl =', pretrain_noise_lvl)
        #print(ae.enc_net(F.one_hot(torch.tensor([0]), num_classes=gu.n_wires).float().cuda()))
        #print(ae.enc_net(F.one_hot(torch.tensor([1]), num_classes=gu.n_wires).float().cuda()))
        #print(ae.enc_net(F.one_hot(torch.tensor([3605]), num_classes=gu.n_wires).float().cuda()))
        all_wires = F.one_hot(torch.arange(gu.n_wires), 
            num_classes=gu.n_wires).float().cuda().requires_grad_(True)

        enc = ae.enc_net(all_wires)
        enc_norm = torch.norm(enc, dim=1)
        #print(enc_norm.shape)
        #print(enc_norm)
        print('norm mean:', enc_norm.mean())
        print('norm std:', enc_norm.std())
        print('norm min / max:', enc_norm.min().item(), enc_norm.max().item())

print('OK')

if pretrain_epochs > 0:
    save_states(0)

gen.train()
disc.train()
lambda_gp = 10
n_critic = 4
for e in range(start_epoch, start_epoch + n_epochs):
    logging.info('Epoch %d' % (e))
    print('Epoch %d' % (e))
    for i, (real_p, real_w) in enumerate(train_loader):

        disc.train()
        gen.train()

        # real_p (batch, 3, seq_len)
        # real_w (batch, 1, seq_len)

        real_w_ohe = F.one_hot(real_w.cuda(), 
            num_classes=gu.cum_n_wires[-1]).squeeze(1).permute(0, 2, 1).float().requires_grad_(True)
        # real_w_ohe (batch, 3606, seq_len)

        # Critic optimization step
        optimizer_disc.zero_grad()
        # Weight clipping
        #for p in disc.parameters():
            #p.data.clamp_(-0.01, 0.01)

        # Take loss between real samples and objective 1.0
        real_p = to_device(real_p).requires_grad_(True)
        real_enc_w = ae.enc(real_w_ohe)
        real_xy = torch.tensordot(real_w_ohe, wire_to_xy, dims=[[1], [1]]).permute(0, 2, 1)
        #print(real_xy.shape)
        # real_xy (batch, 2, seq_len)
        #print(real_xy[5,:,5])
        #print(wire_to_xy[:,real_w[5,0,5]]) OK!

        real_x = concatenate_p_w_xy(real_p, real_enc_w, real_xy)
        out_real = disc(real_x)

        fake_p, fake_w = sample_fake(real_p.shape[0], tau)
        fake_dec_w = F.gumbel_softmax(ae.dec(fake_w), dim=1, hard=True, tau=tau)
        fake_xy = torch.tensordot(fake_dec_w, wire_to_xy, dims=[[1], [1]]).permute(0, 2, 1)
        fake_enc_w = ae.enc(fake_dec_w)

        fake_x = concatenate_p_w_xy(fake_p, fake_enc_w, fake_xy).detach()
        out_fake = disc(fake_x)

        eps = to_device(torch.rand((real_p.shape[0], 1, 1)))
        interpolates_p = (eps * real_p + (1-eps) * fake_p).requires_grad_(True)
        interpolates_enc_w = (eps * real_enc_w + (1-eps) * fake_enc_w).requires_grad_(True)
        interpolates_w = interpolates_enc_w
        interpolates_xy = 0
        #interpolates_dec_w = F.gumbel_softmax(ae.dec(interpolates_w), dim=1, hard=True, tau=tau)
        #interpolates_xy = torch.tensordot(interpolates_dec_w, 
                #wire_to_xy, dims=[[1], [1]]).permute(0, 2, 1).requires_grad_(True)
        gp = gradient_penalty(disc, interpolates_p, interpolates_w, interpolates_xy)
        gradient_pen_hist.append(gp.item())

        #print('real score:', torch.mean(out_real).item())
        #print('fake score:', torch.mean(out_fake).item())
        #print('delta:', torch.mean(out_fake).item() - torch.mean(out_real).item())
        
        D_loss = -torch.mean(out_real) + torch.mean(out_fake) + lambda_gp * gp
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
            #fake_enc_w = fake_w
            fake_dec_w = F.gumbel_softmax(ae.dec(fake_w), dim=1, hard=True, tau=tau)
            fake_xy = torch.tensordot(fake_dec_w, wire_to_xy, dims=[[1], [1]]).permute(0, 2, 1)
            fake_enc_w = ae.enc(fake_dec_w)
            fake_x = concatenate_p_w_xy(fake_p, fake_enc_w, fake_xy)
            #fake_wx = torch.tensordot(wire_to_xy, fake_w, dims=([1], [1])).permute(1, 0, 2)
            out_fake = disc(fake_x)

            G_loss = -torch.mean(out_fake)            

            generator_losses.append(G_loss.item())

            G_loss.backward()
            optimizer_gen.step()

            if (tau > 5e-1):
                tau *= 0.99#5

        if (noise_level > 1e-4):
            noise_level *= 0.999
        logging.info('noise level %f' % (noise_level))
        logging.info('tau %f' % (tau))
    if ((e+1) % 100) == 0:
        save_states(e+1)



print('Done')

print('Saving models...')



print(start_epoch + n_epochs)
save_states(start_epoch + n_epochs)
