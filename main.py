import uproot
import numpy as np
from data import get_noise_data, get_chanmap


data, n_pot = get_noise_data()
n_bunches = n_pot / 16e6

print(n_bunches)
print(data.dtype)
print(data.shape)

chan_tree = get_chanmap()
print(chan_tree.keys())

# Define everything in terms of only sense wires
xro = chan_tree['xro'].array()
yro = chan_tree['yro'].array()
map_layerid = chan_tree['LayerID'].array()
map_wire = chan_tree['wire'].array()

issense = chan_tree['isSenseWire'].array()

xro = xro[issense==True]
yro = yro[issense==True]
map_layerid = map_layerid[issense==True]
map_wire = map_wire[issense==True]

n_wires_per_layer = np.zeros(np.unique(map_layerid).size, dtype=int)
for l in np.unique(map_layerid):
    n_wires_per_layer[l] = map_wire[(map_layerid==l)].size

cum_n_wires = np.concatenate([[0], np.cumsum(n_wires_per_layer)])
print(cum_n_wires)

def wire_abs_to_rel(wire_idx):
    layer = (wire_idx[:, np.newaxis] >= cum_n_wires).sum(axis=1) - 1
    wire = wire_idx - (cum_n_wires[layer])

    return (layer, wire)


def wire_rel_to_abs(layer_idx, wire_idx):
    return cum_n_wires[layer_idx] + wire_idx

map_wire_abs = wire_rel_to_abs(map_layerid, map_wire)
n_wires = np.unique(map_wire_abs).size
map_wire_abs[np.arange(0, n_wires)] = np.arange(0, n_wires)

def wire_pos(wire_abs_idx):
    cond = (wire_abs_idx[:, np.newaxis] == map_wire_abs)
    big_xro = np.tile(xro, (wire_abs_idx.size, 1))
    big_yro = np.tile(yro, (wire_abs_idx.size, 1))

    arr = (big_xro[cond], big_yro[cond])

    return arr

wire_x, wire_y = wire_pos(np.arange(0, map_wire_abs.size))


from model import Gen, Disc

ngf = 16
ndf = 32
latent_dims = 128
seq_len = 2048

embed_dims = 13
n_features = 3 + embed_dims

gen = Gen(ngf, latent_dims, seq_len, n_features)
disc = Disc(ndf, seq_len, n_features)

print(gen)
print(disc)
