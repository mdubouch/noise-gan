import argparse
import logging
import os


parser = argparse.ArgumentParser('Train CDC GAN')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--latent-dims', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=2048)
parser.add_argument('--log', type=str, default='info')
parser.add_argument('--output-dir', type=str, default='output/')
args = parser.parse_args()
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.basicConfig(filename='output.log', level=getattr(logging, args.log.upper()), format='%(asctime)s %(message)s')

ngf = args.ngf
ndf = args.ndf
latent_dims = args.latent_dims
seq_len = args.sequence_length

print('Importing networks...')
import networks
gen = networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len)
logging.info(gen)
disc = networks.Disc(ndf=ndf, seq_len=seq_len)
logging.info(disc)

print('Importing geometry...')
import geom_util as gu
logging.info('cumulative wires {0}'.format(gu.cum_n_wires))

print('Importing dataset...')
import data
logging.info('pot %d  bunches %d', data.n_pot, data.n_bunches)
logging.info('dtypes {0}'.format(data.tree.dtype))
logging.info('shape {0}'.format(data.tree.shape))

print('Pre-processing...')
import torch

# Set up the data, throw away the invalid ones
wire = data.tree['wire']
event_id = data.tree['event_id']
layer = data.tree['layer']
edep = data.tree['edep']
doca = data.tree['doca']
t = data.tree['t']
dbg_x = data.tree['x']
dbg_y = data.tree['y']
dbg_z = data.tree['z']
track_id = data.tree['track_id']
pid = data.tree['pid']

_select = (wire>=0) * (edep>1e-6)
layer = layer[_select]
event_id = event_id[_select]
t = t[_select]
dbg_x = dbg_x[_select]
dbg_y = dbg_y[_select]
dbg_z = dbg_z[_select]
track_id = track_id[_select]
pid = pid[_select]
doca = doca[_select]
wire = wire[_select]

edep = edep[(data.tree['wire']>=0) * (edep>1e-6)]

logging.info('Size wire %d pid %d doca %d' % (wire.size, pid.size, doca.size))
# Some diagnostic plots
import matplotlib.pyplot as plt
import numpy as np
plt.hist(np.log10(edep), bins=50)
plt.savefig(output_dir+'train_edep.png')
plt.clf()
plt.hist(np.log10(t), bins=50)
plt.savefig(output_dir+'train_t.png')
plt.clf()
plt.hist(doca, bins=50)
plt.savefig(output_dir+'train_doca.png')
plt.clf()

plt.figure(figsize=(6,6))
plt.scatter(dbg_z, dbg_y, s=edep*1e3, c=doca, cmap='inferno')
plt.savefig(output_dir+'train_scatter.png')
plt.clf()

train = np.array([edep, t, doca], dtype=np.float32).T
logging.info('train shape {0}'.format(train.shape))
import sklearn.preprocessing as skp
g_qt = skp.QuantileTransformer(output_distribution='normal', n_quantiles=5000)
g_minmax = skp.MinMaxScaler(feature_range=(-1, 1))
train_qt = g_qt.fit_transform(train)

plt.hist(train_qt[:,0], bins=50);
plt.savefig(output_dir+'qt_edep.png')
plt.clf()

train_minmax = g_minmax.fit_transform(train_qt)

__inv_train = g_qt.inverse_transform(g_minmax.inverse_transform(train_minmax))
plt.hist(np.log10(__inv_train[:,0]), bins=50, alpha=0.7)
plt.hist(np.log10(edep), bins=50, alpha=0.7)
plt.savefig(output_dir+'inv_transform.png')
plt.clf()

def inv_transform(tensor):
    inv_tensor = g_qt.inverse_transform(g_minmax.inverse_transform(tensor.detach().cpu().numpy()))
    return torch.tensor(inv_tensor)

train_minmax_torch = torch.from_numpy(train_minmax).T
chunk_size = seq_len
chunk_stride = 256
train_minmax_chunked = train_minmax_torch.unfold(1, chunk_size, chunk_stride) # (feature, batch, seq)
n_chunks = train_minmax_chunked.shape[1]
n_features = train_minmax_chunked.shape[0]

wire_torch = torch.from_numpy(wire).long().unsqueeze(0)
wire_chunked = wire_torch.unfold(1, chunk_size, chunk_stride) # (feature, batch, seq)

logging.info('Continuous features shape: {0}   Discrete features shape: {1}'.format(train_minmax_chunked.shape, wire_chunked.shape))

train_dataset = torch.utils.data.TensorDataset(train_minmax_chunked.permute(1, 0, 2), wire_chunked.permute(1, 0, 2))
batch_size=12
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

print('Training begin')
import time
from tqdm import tqdm
#for i in tqdm(range(100)):
#    time.sleep(0.01) 

print('Done')
