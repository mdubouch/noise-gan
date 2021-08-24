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
parser.add_argument('--only-real', action='store_true')
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


print('Time-animating job %d at epoch %d' % (args.job_id, args.epoch))

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
print('Animating with networks version %d' % (args.net_version))
# Initialize networks
gen = networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len, encoded_dim=encoded_dim,
        n_wires=gu.n_wires).to(device)
torchsummary.summary(gen, input_size=(latent_dims,))
print('generator params:', networks.get_n_params(gen))

n_epochs = 0

# Set up output directory
output_dir = output_dir + '/time_anim/'
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
from matplotlib.patches import Ellipse

def sample_real(batch_size):
    idx = np.random.randint(0, data.edep.size - seq_len, size=(batch_size,))
    idx = idx - (idx % 2048)
    start = idx
    stop = idx + seq_len
    slices = np.zeros((batch_size, seq_len), dtype=np.int64)
    for i in range(batch_size):
        slices[i] = np.r_[start[i]:stop[i]] 
    edep = data.edep[slices]
    t = data.t[slices]
    doca = data.doca[slices]
    w = data.wire[slices]
    return torch.FloatTensor([edep, t, doca]).permute(1, 0, 2), torch.LongTensor(w.flatten())
def sample_fake(z):
    with torch.no_grad():
        p, dec_w = gen(z.to(device))
    w = dec_w.argmax(dim=1).flatten().cpu()
    return p.cpu(), w

load_states('output_%d/states_%d.pt' % (args.job_id, args.epoch))
# Generate one sample
gen.eval()
p, w = sample_fake(torch.randn(1, latent_dims))
print(p.shape)
p = data.inv_preprocess(p.permute(0,2,1).flatten(0,1))

rl, rl_w = sample_real(1)
print(rl.shape)
rl = rl.permute(0,2,1).flatten(0,1)
#rl = data.inv_preprocess(rl.permute(0,2,1).flatten(0,1))

plt.figure()
plt.hist(np.log10(rl[:,1]), bins=50);
plt.savefig(output_dir+'t_hist.png', dpi=120)
plt.close()


n_frames = 50
#max_time = 1e7 # ns
max_time = data.t.max() # ns
times = np.geomspace(data.t.min(), data.t.max(), num=n_frames)
for i in range(n_frames):
    if args.only_real:
        plt.figure(figsize=(6,6))
    else:
        plt.figure(figsize=(12,6))
        plt.subplot(121)
    ax = plt.gca()
    gu.draw_cdc(ax)

    cur_time = times[i]
    if args.only_real == False:
        _t = p[:,1]
        _w = w[_t <= cur_time]
        _x = gu.wire_x[_w]
        _y = gu.wire_y[_w]
        _e = p[:,0][_t <= cur_time]
        _doca = p[:,2][_t <= cur_time]
        plt.scatter(_x, _y, s=1+_e*1e3, c=_doca, cmap='inferno',
                vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8
                )
        ax.set_aspect(1.0)
        ax = plt.subplot(122)
        gu.draw_cdc(ax)

    _t = rl[:,1]
    _w = rl_w[_t <= cur_time]
    _x = gu.wire_x[_w]
    _y = gu.wire_y[_w]
    _e = rl[:,0][_t <= cur_time]
    _doca = rl[:,2][_t <= cur_time]
    plt.scatter(_x, _y, s=1+_e*1e3, c=_doca, cmap='inferno',
            vmin=data.doca.min(), vmax=data.doca.max(), alpha=0.8
            )
    ax.set_aspect(1.0)

    if args.only_real:
        plt.savefig(output_dir + 'frame_real_%03d.png' % (i), dpi=120)
    else:
        plt.savefig(output_dir + 'frame_%03d.png' % (i), dpi=120)
    plt.close()

print('Time-animation of job %d at epoch %d with networks %d done.' % (args.job_id, args.epoch, args.net_version))
