#!/usr/bin/python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
parser = argparse.ArgumentParser('Train CDC embedding network')
parser.add_argument('--n-epochs', type=int, default=1)
parser.add_argument('--embedding-dim', type=int, default=16)
parser.add_argument('--context-size', type=int, default=6)
parser.add_argument('--output', '-o', type=str, default='cdc_embedding.pt')
parser.add_argument('--cff', '--continue-from-file', type=str, default=None)
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

import cdc_embedding
print('Initialising model...')
context_size = args.context_size
embedding_dim = args.embedding_dim
model = cdc_embedding.CDCEmbedding(context_size, embedding_dim, n_wires).to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
if args.cff is not None:
    save_file = torch.load(args.cff)
    model.load_state_dict(save_file['model'])
    optim.load_state_dict(save_file['optim'])

# Create a distance matrix for a discrete set of z values so we don't have to
# compute it at each iteration.
print('Calculating distance matrices...')
n_z_vals = 1
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

perm_z = torch.randperm(n_z_vals)
train_z = perm_z[:n_z_vals - n_z_vals // 10]
#test_z = perm_z[n_z_vals - n_z_vals // 10:]

losses = []
test_losses = []
accuracy = []
test_accuracy = []

print('Training start')
n_its = args.n_epochs
batch_size=16
for i in range(n_its):
    model.train()
    optim.zero_grad()
    # Pick a random z value from the train set
    z = train_z[torch.randint(0, train_z.shape[0], (1,))]
    d_mat = dist_matrix[:,:,z]
    
    tgt = torch.randint(0, n_wires, (batch_size,)).to(device)
    context = (1/d_mat[tgt]).topk(context_size+1, dim=1).indices[:,1:].squeeze().to(device)
    if i == 0:
        assert(context.shape == (batch_size, context_size))

    pred = model(context)
    loss = F.cross_entropy(pred, tgt)
    losses.append(loss.item())
    if i % 100 == 0:
        print(i, loss.item())
        pred_hard = pred.argmax(dim=1)
        print(pred_hard.shape)
        acc = (pred_hard == tgt).sum() / batch_size
        accuracy.append(acc.item())

    loss.backward()
    optim.step()

    #if i % 100 == 0:
    #    model.eval()
    #    with torch.no_grad():
    #        z = test_z[torch.randint(0, test_z.shape[0], (1,))]
    #        d_mat = dist_matrix[:,:,z]
    #        
    #        tgt = torch.randint(0, n_wires, (batch_size,)).to(device)
    #        context = (1/d_mat[tgt]).topk(context_size+1, dim=1).indices[:,1:].squeeze().to(device)
    #        pred = model(context)
    #        loss = F.cross_entropy(pred, tgt)
    #        test_losses.append(loss.item())
    #        pred_hard = pred.argmax(dim=1)
    #        acc = (pred_hard == tgt).sum() / batch_size
    #        test_accuracy.append(acc.item())



plt.figure()
plt.plot(np.linspace(0, n_its, num=len(losses)), losses)
plt.plot(np.linspace(0, n_its, num=len(test_losses)), test_losses)
plt.savefig('embedding_losses.png')
plt.figure()
plt.plot(np.linspace(0, n_its, num=len(accuracy)), accuracy)
plt.plot(np.linspace(0, n_its, num=len(test_accuracy)), test_accuracy)
plt.savefig('embedding_accuracy.png')

print('Saving model as "%s"' % args.output)
torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, args.output)
print('OK')
