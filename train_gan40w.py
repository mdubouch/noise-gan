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
print('generator params:     %d' % (networks.get_n_params(gen)))
print('discriminator params: %d' % (networks.get_n_params(disc)))
print('AE params:            %d' % (networks.get_n_params(ae)))
logging.info('generator params: %d' % (networks.get_n_params(gen)))
logging.info('discriminator params: %d' % (networks.get_n_params(disc)))
logging.info('AE params: %d' % (networks.get_n_params(ae)))

#print('Importing geometry...')
#import geom_util as gu
#logging.info('cumulative wires {0}'.format(gu.cum_n_wires))

print('Importing dataset...')
import dataset_altered_one as dataset
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

wire_to_xy = torch.tensor([gu.wire_x, gu.wire_y], device='cuda', dtype=torch.float32)
wire_to_r = torch.sqrt(wire_to_xy[0]**2 + wire_to_xy[1]**2)
print(wire_to_r.min(), wire_to_r.max())
wire_to_xy_norm = (wire_to_xy - wire_to_xy * (wire_to_r.min()-10) / wire_to_r) / (wire_to_r.max() - wire_to_r.min() + 20)
wire_to_xy_norm.requires_grad_(False)
print(wire_to_xy_norm.min(), wire_to_xy_norm.max())
wire_to_r_norm = wire_to_xy_norm.norm(dim=0, keepdim=True)#(wire_to_r - wire_to_r.min() + 10) / (wire_to_r.max() - wire_to_r.min() + 10)
print(wire_to_r_norm.min(), wire_to_r_norm.max())
wire_to_rth = torch.zeros_like(wire_to_xy_norm)
wire_to_rth[0] = wire_to_r_norm * 2 - 1.0
wire_to_rth[1] = torch.atan2(wire_to_xy[1], wire_to_xy[0]) / np.pi / 2
geom_dim=3
wire_sphere = torch.zeros((geom_dim, gu.n_wires), device='cuda')

# wire cylinder: points lie on the surface of a cylinder close to z=0, such that
# their relative neighborhoods are respected and their distance to the origin is approximately 1
for i in range(gu.n_layers):
    start = np.sum(gu.n_wires_per_layer[:i])
    n_wires = gu.n_wires_per_layer[i]
    end = start+n_wires
    theta_dist = np.sqrt((gu.wire_x[start+1:end] - gu.wire_x[start:end-1])**2 + \
            (gu.wire_y[start+1:end] - gu.wire_y[start:end-1])**2)
    if i > 0:
        r_dist = np.sqrt(gu.wire_x[start]**2 + gu.wire_y[start]**2) - \
                    np.sqrt(gu.wire_x[start-1]**2 + gu.wire_y[start-1]**2)
    # So average distance is ~16mm
    # Cylinder surface: 
    # x = r cos phi
    # y = r sin phi
    # z = z
    # We want the average distance in phi between wires to be equal to that between layers
    # Avg distance in phi between wires: 
    # d = r * 2pi/n_wires
    # r = 1
    # avg n_wires = 225
    # => d = 0.0246
    # 16 layers -> max displacement from z=0 is d * 8 = 0.197
    # which has a distance to center of sqrt(r**2 + z**2) = 1.019
    #phi0 = np.arctan2(gu.wire_y[start], gu.wire_x[start])
    #phi = phi0 + torch.arange(n_wires) * 2 * np.pi / n_wires
    #x = torch.cos(phi)
    #y = torch.sin(phi)
    #z = -0.2 + i / (gu.n_layers-1) * 0.4
    #z = np.sqrt(1 - np.sqrt(x**2 + y**2 + z**2))
    x = gu.wire_x[start:end]
    y = gu.wire_y[start:end]
    max_r = wire_to_r.max().cpu() + 200
    R = max_r
    z = np.sqrt(R**2 - (x**2 + y**2))
    wire_sphere[0,start:end] = torch.from_numpy(x)
    wire_sphere[1,start:end] = torch.from_numpy(y)
    wire_sphere[2,start:end] = torch.from_numpy(np.sqrt(x**2+y**2))
    #wire_sphere[3,start:end] = torch.from_numpy(np.arctan2(y, x))
    #wire_sphere[4,start:end] = torch.from_numpy(np.arange(start, end))
    #wire_sphere[5,start:end] = torch.from_numpy(np.ones_like(x) * i)
    #wire_sphere[3,start:end] = torch.from_numpy(np.arange(start, end))
    #wire_sphere[2,start:end] = torch.from_numpy(np.zeros_like(x))
    #wire_sphere[3,start:end] = torch.from_numpy(np.zeros_like(x))


    #print('aaa', wire_sphere[3, start:end])
    #wire_sphere[:,start:end] /= wire_sphere[:,start:end].norm(dim=0, keepdim=True)
print(wire_sphere.norm(dim=0))
data_std = data.train_dataset.tensors[0].permute(1,0,2).flatten(1,2).std(dim=1)
print('DATA STD', data_std)
data_std = data_std.mean()
wire_sphere = wire_sphere# / (wire_sphere.std(dim=1, keepdim=True)) * data_std
print('VAR', wire_sphere.std(dim=1))
print('NORM', wire_sphere.norm(dim=0))
#wire_sphere = wire_sphere / wire_sphere.std(dim=1, keepdim=True) - wire_sphere.mean(dim=1, keepdim=True)
#wire_sphere[2] += torch.randn_like(wire_sphere[2]) * 0.01
#wire_sphere[3] += torch.randn_like(wire_sphere[2]) * 0.01
wire_sphere = (wire_sphere - wire_sphere.mean(dim=1, keepdim=True)) / wire_sphere.std(dim=1, keepdim=True)
#wire_sphere[0] = wire_sphere[0] + 2*wire_sphere.norm(dim=0)
    
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
x = wire_sphere[0].cpu()
y = wire_sphere[1].cpu()
z = wire_sphere[2].cpu()
#z = np.zeros_like(x)
ax.scatter(x, y, z, s=1, c=gu.layer)
#ax.set_zlim(-1, 1)

#for i in range(gu.n_layers):
#    idx = gu.cum_n_wires[i] - gu.n_wires_per_layer[i]
#    ax.text(x[idx], y[idx], z[idx], str(i))

plt.savefig(output_dir+'wire_cylinder.png', dpi=120)
plt.close()


plt.figure(figsize=(8,8))
plt.plot(wire_to_rth.norm(dim=0).cpu(), marker='.', linewidth=0)
plt.savefig(output_dir+'wire_rth_norm.png', dpi=60)
plt.close()
plt.figure(figsize=(8,8))
plt.plot(wire_sphere.norm(dim=0).cpu(), marker='.', linewidth=0)
plt.savefig(output_dir+'wire_sphere_norm.png', dpi=60)
plt.close()

# Just in case we need to make sure we have the right CDC.
plt.figure(figsize=(8,8))
plt.scatter(wire_to_xy_norm[0].cpu(), wire_to_xy_norm[1].cpu(), s=1)
plt.savefig(output_dir+'wire_norm_scatter.png', dpi=60)
plt.close()
plt.figure(figsize=(8,8))
plt.scatter(wire_to_xy[0].cpu(), wire_to_xy[1].cpu(), s=1)
plt.savefig(output_dir+'wire_scatter.png', dpi=60)
plt.close()


# wire_to_xy (2, 3606)
real_dist_matrix = torch.cdist(wire_to_xy.T, wire_to_xy.T)

def sample_real(batch_size):
    idx = np.random.choice(np.arange(n_chunks), size=batch_size)
    p, w = train_dataset[idx]
    one_hot_w = F.one_hot(w, num_classes=gu.n_wires).squeeze(1).permute(0, 2, 1)
    # Return shape is (batch, feature, seq)
    return p, one_hot_w
def sample_fake(batch_size, tau):
    noise = to_device(torch.randn((batch_size, latent_dims), requires_grad=True))
    sample = gen(noise, wire_sphere)
    return sample

#_p, _w = sample_real(2)
#print(_p.shape, _w.shape)

__f = sample_fake(2, 1.0)
#print(__f.shape)

tau = 2/3
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

optimizer_gen = torch.optim.Adam(list(gen.parameters()),# + list(ae.parameters()),
        lr=1e-4, betas=(0.9, 0.999))
optimizer_disc = torch.optim.Adam(list(disc.parameters()),
        lr=1e-4, betas=(0.9, 0.999))
optimizer_ae_pretrain = torch.optim.Adam(ae.parameters(),
        lr=2e-3, betas=(0.9, 0.999))
optimizer_ae = torch.optim.Adam(ae.parameters(),
        lr=2e-4, betas=(0.9, 0.999))

def weight_init_relu(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    #elif classname.find('Norm') != -1:
    #    nn.init.ones_(m.weight)
    #    nn.init.zeros_(m.bias)

    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def weight_init_leakyrelu(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        #nn.init.normal_(m.weight, 0.0, 0.02)
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

else:
    disc.apply(weight_init_leakyrelu)
    gen.apply(weight_init_relu)
    #ae.enc_net.apply(weight_init_leakyrelu)
    #ae.dec_net.apply(weight_init_relu)

if pretrain_epochs == 0 and args.no_pretrain == False:
    if args.pretrained is not None:
        path = args.pretrained
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

# Implement "Gradient Penalty" for WGAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
def gradient_penalty(disc, interp_p, interp_w):
    interp_p.requires_grad_()
    interp_w.requires_grad_()
    d_interpolates = disc(interp_p, interp_w, wire_sphere).squeeze()
    grad_outputs_w = torch.ones(d_interpolates.shape, requires_grad=False, device='cuda')

    gradients_w = autograd.grad(outputs=d_interpolates,
            inputs=interp_w,
            grad_outputs=grad_outputs_w,
            create_graph=True,
            only_inputs=True
            )[0]
#    gradients_p = autograd.grad(outputs=d_interpolates,
#            inputs=interp_p,
#            grad_outputs=grad_outputs_w,
#            create_graph=True, # IMPORTANT! Allows to compute gradient with respect to gradient
#            only_inputs=True
#            )[0]

    #gradients_x = gradients_x.reshape(gradients_x.shape[0], -1) + 1e-8
    #gradients_x = gradients_x + 1e-16

    #gradients_p_pen = (gradients_p.pow(2).flatten(1, 2).sum(dim=1).pow(0.5) - 1)**2
    #print(gradients_p_pen.mean())
    gradients_w_pen = (gradients_w.pow(2).flatten(1, 2).sum(dim=1).pow(0.5) - 1)**2
    #print(gradients_w_pen.mean())

    gradient_pen = gradients_w_pen.mean() #+ gradients_w_pen

    return gradient_pen

def get_wire_weights():
    wire_counts = np.bincount(data.wire, minlength=gu.n_wires)
    print(wire_counts.shape)
    return torch.tensor(1 / (wire_counts + 1e-1), device='cuda', dtype=torch.float)
wire_weights = get_wire_weights()
print(wire_weights)
print(wire_weights[1600])
print(wire_weights.shape)

ae.train()
# Pre-train AE
print('Pretraining AE...')
for e in range(pretrain_epochs):
    # Make the batch size 1, so we get all wires every time
    bsize = 1024
    n_its = gu.n_wires // bsize + 1
    wires = torch.tensor(np.random.choice(np.arange(gu.n_wires), 
        size=(n_its, bsize), replace=True)).cuda()

    batch_wid_ohe = F.one_hot(wires, num_classes=gu.n_wires).float().requires_grad_(True)        

    code_losses = []
    for i in range(n_its):
        optimizer_ae_pretrain.zero_grad()
        wid_ohe = batch_wid_ohe[i]
        # wid_ohe (bsize, n_wires)

        enc = ae.enc_net(wid_ohe)
        # enc (bsize, enc_dim)

        encdec = ae.dec_net(enc)
        ae_loss = F.cross_entropy(encdec, wires[i], weight=wire_weights)
        #reenc = ae.enc_net(F.gumbel_softmax(encdec, dim=1, hard=True))
        #code_loss = F.mse_loss(reenc, enc)
        #code_losses.append(code_loss.item())
        loss = ae_loss# + 2 * code_loss
        loss.backward()
        optimizer_ae_pretrain.step()
        pretrain_losses.append(ae_loss.item())

        choice = torch.argmax(encdec, dim=1)
        hit = (choice == wires[i]).sum().float()
        acc = hit / bsize
        pretrain_acc.append(acc.item())

    

    if e % 200 == 0:
        print('Epoch', e)
        print('cross entropy loss:', np.mean(pretrain_losses[-100:]))
        print('accuracy:', np.mean(pretrain_acc[-100:]))
        #print('code loss:', np.mean(code_losses))
        #print(ae.enc_net(F.one_hot(torch.tensor([0]), num_classes=gu.n_wires).float().cuda()))
        #print(ae.enc_net(F.one_hot(torch.tensor([1]), num_classes=gu.n_wires).float().cuda()))
        #print(ae.enc_net(F.one_hot(torch.tensor([3605]), num_classes=gu.n_wires).float().cuda()))

print('OK')

if pretrain_epochs > 0:
    save_states(0)

def gumbel(x):
    return F.gumbel_softmax(x, dim=1, tau=tau, hard=True)

def wire_hook(grad):
    return grad


gen.train()
disc.train()
lambda_gp = 10
n_critic = 5
critic_count = 0
for e in range(start_epoch, start_epoch + n_epochs):
    logging.info('Epoch %d' % (e))
    print('Epoch %d' % (e))
    for i, (real_p, real_w) in enumerate(train_loader):

        # real_p (batch, 3, seq_len)
        # real_w (batch, 1, seq_len)
        real_p = real_p.cuda()

        real_w_ohe = F.one_hot(real_w.cuda(), 
            num_classes=gu.cum_n_wires[-1]).squeeze(1).permute(0, 2, 1).float()
        # real_w_ohe (batch, 3606, seq_len)

        # Critic optimization step
        optimizer_disc.zero_grad()
        # Weight clipping
        #for p in disc.parameters():
        #    p.data.clamp_(-0.01, 0.01)

        real_xy = wire_sphere[:,real_w].squeeze(2).permute(1, 0, 2)

        #print(real_xy[0,:,0])

        real_xy = torch.tensordot(real_w_ohe, wire_sphere, dims=[[1], [1]]).permute(0,2,1)
        #real_x = torch.cat([real_p, real_w_ohe], dim=1).requires_grad_()
        out_real = disc(real_p, real_xy, wire_sphere)
        #D_loss_real = F.binary_cross_entropy_with_logits(out_real, torch.ones_like(out_real))
        print('out real %.2e' % out_real.mean())

        #with torch.no_grad():
        fake_p, fake_wg = sample_fake(real_p.shape[0], tau)
        #fake_wg.register_hook(wire_hook)
        fake_xy = torch.tensordot(fake_wg, wire_sphere, dims=[[1], [1]]).permute(0,2,1)
        #print('fk',fake_xy[0,:,0])
        #fake_x = torch.cat([fake_p, fake_wg], dim=1)

        out_fake = disc(fake_p, fake_xy, wire_sphere)
        print('out fake %.2e' % out_fake.mean())
        #D_loss_fake = F.binary_cross_entropy_with_logits(out_fake, torch.zeros_like(out_fake))
        #D_loss_fake.backward()
        #print('D loss fake', D_loss_fake.item())

        eps = torch.rand((real_p.shape[0], 1, 1), device='cuda')
        seq_stop = (eps[0] * seq_len).long()
        ##interpolates_p = (eps * real_p + (1-eps) * fake_p).requires_grad_(True)
        ##interpolates_x = torch.cat([real_x[:,:,:seq_stop], fake_x[:,:,seq_stop:]], dim=2)
        reverse = torch.rand(1)
        if (reverse > 0.5):
            p1 = real_p[:,:,:seq_stop]
            p2 = fake_p[:,:,seq_stop:].detach()
            wg1 = real_w_ohe[:,:,:seq_stop]
            wg2 = fake_wg[:,:,seq_stop:].detach()
        else:
            p1 = fake_p[:,:,:seq_stop].detach()
            p2 = real_p[:,:,seq_stop:]
            wg1 = fake_wg[:,:,:seq_stop].detach()
            wg2 = real_w_ohe[:,:,seq_stop:]

        interpolates_p =  torch.cat([p1, p2], dim=2)
        interpolates_w = torch.cat([wg1, wg2], dim=2)
        interpolates_xy = torch.tensordot(interpolates_w, wire_sphere, dims=[[1], [1]]).permute(0,2,1)

        #interpolates_x = torch.cat([interpolates_p, interpolates_w], dim=1)
        gp = gradient_penalty(disc, interpolates_p, interpolates_xy)
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

            fake_p, fake_wg = sample_fake(real_p.shape[0], tau)
            # Figure out what the avg gradient is in inner and outer layers
            
            #fake_xy = wireproj(fake_wg)
            fake_xy = torch.tensordot(fake_wg, wire_sphere, dims=[[1], [1]]).permute(0,2,1)
            #print('fk',fake_xy[0,:,0])
            #fake_x = torch.cat([fake_p, fake_wg], dim=1)

            out_fake = disc(fake_p, fake_xy, wire_sphere)

            G_loss = -out_fake.mean()

            generator_losses.append(G_loss.item())

            #print(fake_wg.grad, fake_xy.grad, fake_p.grad)
            G_loss.backward()
            #print(fake_wg.shape)
            #print(fake_wg.grad, fake_xy.grad, fake_p.grad)
            #inner_grad = fake_wg[:,0].grad.abs().mean().item()
            #print(inner_grad)
            #outer_grad = fake_wg[:,-1].grad.abs().mean().item()
            #print(outer_grad)
            #print('convwp grad %.2e %.2e' % 
            #        (gen.convpw.weight.grad.mean().item(), gen.convpw.weight.grad.std().item()))
            #print('convp3 grad %.2e %.2e' % 
            #        (gen.convp3.weight.grad.mean().item(), gen.convp3.weight.grad.std().item()))
            #print('convwp weight %.2e %.2e' % 
            #        (gen.convpw.weight.mean().item(), gen.convpw.weight.std().item()))
            #print('convp3 weight %.2e %.2e' % 
            #        (gen.convp3.weight.mean().item(), gen.convp3.weight.std().item()))

            optimizer_gen.step()

            #if (tau > 1):
                #tau *= 0.99#5

        logging.info('tau %f' % (tau))
    if ((e+1) % 500) == 0:
        save_states(e+1)



print('Done')

print('Saving models...')



print(start_epoch + n_epochs)
save_states(start_epoch + n_epochs)
