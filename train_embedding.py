#!/usr/bin/python3
#$ -P P_comet
#$ -j y
#$ -cwd
#$ -q mc_gpu_long
#$ -pe multicores_gpu 4
#$ -l sps=1,GPU=1,GPUtype=V100

import os
import sys
sys.path.append(os.getcwd())

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
parser = argparse.ArgumentParser('Train CDC embedding network')
parser.add_argument('--n-epochs', type=int, default=1)
parser.add_argument('--embedding-dim', type=int, default=16)
parser.add_argument('--context-size', type=int, default=6)
parser.add_argument('--cff', '--continue-from-file', type=str, default=None)
args = parser.parse_args()

import os
job_id = int(os.getenv('JOB_ID', default='0'))
output_dir = 'embedding_%d/' % (job_id)
print('Outputting to %s' % (output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

import cdc_embedding
print('Initialising model...')
context_size = args.context_size
embedding_dim = args.embedding_dim
model = cdc_embedding.CDCEmbedding(context_size, embedding_dim, n_wires).to(device)


from training_state import TrainingState
ts = TrainingState(context_size, embedding_dim, n_wires, device=device)

if args.cff is not None:
    ts = torch.load(args.cff)['state']

# Move to output directory once all the data and modules have been imported
os.chdir(output_dir)

# Create a distance matrix for a discrete set of z values so we don't have to
# compute it at each iteration.
print('Calculating distance matrices...')
n_z_vals = 100
z_vals = torch.linspace(0, 1, n_z_vals).unsqueeze(0)
xhv = torch.tensor(xhv).unsqueeze(1)
xro = torch.tensor(xro).unsqueeze(1)
yhv = torch.tensor(yhv).unsqueeze(1)
yro = torch.tensor(yro).unsqueeze(1)
s_x = (1-z_vals) * xhv + z_vals * xro
assert(s_x.shape == (n_wires, n_z_vals))
s_y = (1-z_vals) * yhv + z_vals * yro
assert(s_y.shape == (n_wires, n_z_vals))
s_pos = torch.cat([s_x.unsqueeze(0), s_y.unsqueeze(0)], dim=0)
assert(s_pos.shape == (2, n_wires, n_z_vals))

dist_matrix = torch.sqrt((s_pos.unsqueeze(1) - s_pos.unsqueeze(2)).pow(2).sum(dim=0) + 1e-16)
assert(dist_matrix.shape == (n_wires, n_wires, n_z_vals))

# Build the training and test datasets from random target/context pairs
print('Building dataset...')
torch.manual_seed(1337)
dataset_size = 100000
tgts = torch.randint(0, n_wires, (dataset_size,))
z = torch.randint(0, n_z_vals, (dataset_size,))

# Out of the context_size*2 closest wires, pick context_size of them at random for each
# data sample.
rand_choice = torch.zeros((dataset_size, context_size*2-1)).uniform_().argsort(dim=1)[:,:context_size]+1
contexts = (1/dist_matrix[tgts,:,z]).topk(context_size*2, dim=1).indices[torch.arange(dataset_size).unsqueeze(1), rand_choice]

plt.figure(figsize=(9,9))
plt.scatter(x0, y0, s=0.1, color='gray')
cmap = cm.get_cmap('rainbow')
n_samples=20
for i in range(n_samples):
    plt.scatter(x0[tgts[i]], y0[tgts[i]], s=5, color=cmap(i/n_samples))
    plt.scatter(x0[contexts[i]], y0[contexts[i]], s=1, color=cmap(i/n_samples))
plt.savefig('embedding_nearest.png')

train_test_split = 0.7
train_tgts = tgts[:int(train_test_split * tgts.shape[0])].to(device)
train_contexts = contexts[:int(train_test_split * tgts.shape[0])].to(device)
test_tgts = tgts[int(train_test_split * tgts.shape[0]):].to(device)
test_contexts = contexts[int(train_test_split * tgts.shape[0]):].to(device)
torch.seed()

def diagnostic_plots(ts):
    plt.figure()
    plt.plot(np.linspace(0, ts.its, num=len(ts.losses)), ts.losses, alpha=0.8)
    plt.plot(np.linspace(0, ts.its, num=len(ts.test_losses)), ts.test_losses, alpha=0.8)
    plt.savefig('embedding_losses.png')
    plt.close()
    plt.figure()
    plt.plot(np.linspace(0, ts.its, num=len(ts.accuracy)), ts.accuracy, alpha=0.8)
    plt.plot(np.linspace(0, ts.its, num=len(ts.test_accuracy)), ts.test_accuracy, alpha=0.8)
    plt.ylim(0, 1)
    plt.savefig('embedding_accuracy.png')
    plt.close()

print('Training start')
n_its = args.n_epochs
batch_size=64
for i in range(n_its):
    ts.model.train()
    ts.optim.zero_grad()
    
    idx = torch.randint(0, train_tgts.shape[0], (batch_size,))
    tgt = train_tgts[idx]
    context = train_contexts[idx]
    if i == 0:
        assert(context.shape == (batch_size, context_size))

    pred = ts.model(context)
    loss = F.cross_entropy(pred, tgt)
    ts.losses.append(loss.item())
    if (i+1) % 100 == 0:
        print('%d / %d, %.3f' % (i+1, n_its, loss.item()))

    loss.backward()
    ts.optim.step()

    pred_hard = pred.argmax(dim=1)
    acc = (pred_hard == tgt).sum() / batch_size
    ts.accuracy.append(acc.item())

    ts.its += 1

    if (i+1) % 10 == 0:
        ts.model.eval()
        with torch.no_grad():
            idx = torch.randint(0, test_tgts.shape[0], (batch_size,))
            tgt = test_tgts[idx]
            context = test_contexts[idx]

            pred = ts.model(context)
            loss = F.cross_entropy(pred, tgt)
            ts.test_losses.append(loss.item())
            pred_hard = pred.argmax(dim=1)
            acc = (pred_hard == tgt).sum() / batch_size
            ts.test_accuracy.append(acc.item())
    if (i+1) % 100 == 0:
        diagnostic_plots(ts)
    if (i+1) % 1000 == 0:
        print('Saving model as "%s"' % (output_dir+'cdc_embedding.pt'))
        ts.save('cdc_embedding.pt')


diagnostic_plots(ts)

print('Saving model as "%s"' % (output_dir+'cdc_embedding.pt'))
ts.save('cdc_embedding.pt')

os.chdir('../')
print('OK')
