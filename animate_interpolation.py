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
parser.add_argument('--sequence-length', '--sl', type=int, default=2048)
parser.add_argument('--num-dims', '--nd', type=int, default=1, 
        help="Number of dimensions to interpolate through")
parser.add_argument('--n-frames-per-dim', type=int, default=25)
parser.add_argument('--job-id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--net-version', type=str)
parser.add_argument('--only-real', action='store_true')
parser.add_argument('--interpolation-mode', type=str, default='slerp', help='interpolation function: either lerp or slerp (default)')
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
    args.net_version = contents[n_s:n_e]


print('Animating latent-space interp of job %d at epoch %d' % (args.job_id, args.epoch))

ngf = args.ngf
ndf = args.ndf
print('ngf:', ngf)
print('ndf:', ndf)
latent_dims = args.latent_dims
seq_len = args.sequence_length
encoded_dim = args.enc_dim

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

import importlib
dataset = importlib.import_module(args.dataset)
data = dataset.Data()

import geom_util
gu = geom_util.GeomUtil(data.get_cdc_tree())

networks = importlib.import_module('networks%s' % (args.net_version))
print('Animating with networks version %s' % (args.net_version))
# Initialize networks
gen = networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len, encoded_dim=encoded_dim,
        n_wires=gu.n_wires).to(device)
torchsummary.summary(gen, input_size=(latent_dims,))
print('generator params:', networks.get_n_params(gen))

n_epochs = 0

# Set up output directory
output_dir = output_dir + '/latent_anim/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load network states
def load_states(path):
    print('Loading GAN states from %s...' % (path))
    states = torch.load(path, map_location=device)
    gen.load_state_dict(states['gen'])

    global n_epochs
    n_epochs = states['n_epochs']
    print('OK')

print('Loading dataset')
data.load()
data.preprocess()

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

def sample_fake(z):
    with torch.no_grad():
        p, dec_w = gen(z.to(device))
    w = dec_w.argmax(dim=1).flatten().cpu()
    return p.cpu(), w

load_states('output_%d/states_%d.pt' % (args.job_id, args.epoch))
# Generate one sample
gen.eval()

def slerp(p0, p1, t):
    omega = torch.acos((p0 / p0.norm()).dot(p1 / p1.norm()))
    z = torch.sin((1-t) * omega) / torch.sin(omega) * p0 + \
            torch.sin(t * omega) / torch.sin(omega) * p1
    return z.view(1, latent_dims).cuda()
def lerp(p0, p1, t):
    z = (1-t) * p0 + t * p1
    return z.view(1, latent_dims).cuda()

if args.interpolation_mode == 'lerp':
    interp_func = lerp
else:
    interp_func = slerp

print('Interp function:', args.interpolation_mode)

# Starting point
z0 = torch.randn(latent_dims)
z0[:args.num_dims] = -3.0
z1 = z0.clone()
for j in range(args.num_dims):
    print("Dim %d" % j)
    # End point
    z0 = z1
    z1 = z0.clone()
    z1[j] = +3.0
    n_frames = args.n_frames_per_dim
    for i in range(n_frames):
        plt.figure(figsize=(6,6))
        z = interp_func(z0, z1, i / (n_frames-1))
        p, w = sample_fake(z)
        p = data.inv_preprocess(p.permute(0,2,1).flatten(0,1))
    
        ax = plt.gca()
        gu.draw_cdc(ax)
    
        x = gu.wire_x[w]
        y = gu.wire_y[w]
        plt.scatter(x, y, s=1+p[:,0]*1e3, c=p[:,2], cmap='inferno',
                vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8
                )
        plt.text(0, 0, 'Latent dim %d' % j, 
                horizontalalignment='center', verticalalignment='center')
        ax.set_aspect(1.0)
        plt.savefig(output_dir + 'frame_%03d.png' % (j*args.n_frames_per_dim + i), dpi=120)
        plt.close()

print('Latent space interp animation of job %d at epoch %d with networks %s done.' % (args.job_id, args.epoch, args.net_version))
