#!/usr/bin/python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
parser = argparse.ArgumentParser('Plot CDC embedding network')
parser.add_argument('input', type=str)
args = parser.parse_args()

import uproot3 as uproot
print('Loading CDC geometry...')
file = uproot.open('cdc_geom_hvro.root')
cdc_tree = file['cdc_geom/wires'].array()

import geom_util
gu = geom_util.GeomUtil(cdc_tree)
gu.validate_wire_pos()

xhv = cdc_tree['xhv']
yhv = cdc_tree['yhv']
xro = cdc_tree['xro']
yro = cdc_tree['yro']
x0 = (xhv + xro) / 2
y0 = (yhv + yro) / 2

n_wires = xhv.size

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
#eps = torch.rand(1)
## Linear interpolate between HV and RO positions to get a position at arbitrary z
#sampled_x = (1-eps) * xhv + eps * xro
#sampled_y = (1-eps) * yhv + eps * yro
#sampled_pos = torch.cat([sampled_x.unsqueeze(0), sampled_y.unsqueeze(0)], dim=0)
#
#dist_matrix = torch.sqrt((sampled_pos.unsqueeze(1) - sampled_pos.unsqueeze(2)).pow(2).sum(dim=0) + 1e-16)
#plt.imshow(dist_matrix.numpy(), interpolation='nearest')
#plt.savefig('dist_matrix.png', dpi=120)

# Find nearest wires
#n_samples = 10
#indices = torch.randint(0, n_wires, (n_samples,))
#nearest = (1/dist_matrix[indices]).topk(18, dim=1).indices[:,1:]

#cmap = cm.get_cmap('rainbow')
#plt.figure(figsize=(9,9))
#plt.scatter(x0, y0, s=0.1, color='gray')
#for i in range(n_samples):
#    plt.scatter(x0[indices[i]], y0[indices[i]], s=5, color=cmap(i/n_samples))
#    plt.scatter(x0[nearest[i]], y0[nearest[i]], s=1, color=cmap(i/n_samples))
#plt.savefig('embedding_nearest.png')

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

from training_state import TrainingState
print('Initialising model...')
ts = torch.load(args.input+'/cdc_embedding.pt')['state']
print("Plotting at iteration %d" % ts.its)

import os
os.chdir(args.input)

xro = torch.tensor(xro)
yro = torch.tensor(yro)
real_dist = torch.sqrt((xro.unsqueeze(1) - xro.unsqueeze(0))**2 + \
        (yro.unsqueeze(1) - yro.unsqueeze(0))**2)
print(real_dist.shape)

# Plot some embedding wire positions
wires = torch.arange(0, n_wires).cuda()
emb_wires = ts.model.emb(wires).detach().cpu()
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.scatter(xro, yro, s=1, c=real_dist[0], cmap='rainbow')
plt.gca().set_aspect(1.0)
plt.subplot(122)
#plt.scatter(emb_wires[:,0], emb_wires[:,1], s=1, c=wires.float().cpu(), cmap='rainbow')
plt.scatter(emb_wires[:,0], emb_wires[:,1], s=1, c=real_dist[0], cmap='rainbow')
plt.gca().set_aspect(1.0)
plt.savefig('emb_wires.png')

plt.figure(figsize=(12,6))
real_theta = np.arctan2(yro, xro)
#real_theta = np.sin(real_theta/2) * np.cos(real_theta/2)
plt.subplot(121)
plt.scatter(xro, yro, s=1, c=real_theta, cmap='hsv')
plt.gca().set_aspect(1.0)
plt.subplot(122)
#plt.scatter(emb_wires[:,0], emb_wires[:,1], s=1, c=wires.float().cpu(), cmap='rainbow')
plt.scatter(emb_wires[:,0], emb_wires[:,1], s=1, c=real_theta, cmap='hsv')
plt.gca().set_aspect(1.0)
plt.savefig('emb_wires_theta.png')

print(emb_wires.shape)
emb_dist = torch.sqrt((emb_wires.unsqueeze(1) - emb_wires.unsqueeze(0)).pow(2).sum(dim=2))
emb_sim = torch.tensordot(emb_wires, emb_wires, dims=[[1], [1]]) / emb_wires.norm(dim=1).unsqueeze(1) / emb_wires.norm(dim=1).unsqueeze(0)
print(emb_dist.shape)
print(emb_sim.shape)
w0mat = torch.stack([real_dist[2000].float() / real_dist.std(), emb_sim[2000].float() / emb_sim.std()])
#w0mat = torch.stack([real_dist[0].float() / real_dist.std(), emb_dist[0].float() / emb_dist.std()])
print(w0mat.shape)
plt.figure()
plt.imshow(w0mat.cpu(), interpolation='none', aspect=200)
plt.savefig('emb_matrix.png', dpi=240)

from sklearn.decomposition import PCA
pca = PCA(2)
y = pca.fit_transform(emb_wires)
print(y.shape)
plt.figure()
plt.scatter(y[:,0], y[:,1], c=real_dist[0], cmap='hsv', s=1)
plt.savefig('emb_pca.png')


os.chdir('../')
print('OK')
