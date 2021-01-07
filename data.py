import uproot3 as uproot
import torch
import logging
import numpy as np
import sklearn.preprocessing as skp

def get_noise_data():

    file = uproot.open('reconstructible_toy+cdc.root')
    n_pot = 990678399
    n_bunches = n_pot / 16e6
    
    data = file['noise/noise'].array()
    cdc_tree = file['cdc_geom/wires'].array()

    return data, n_pot, cdc_tree

def get_chanmap():
    f_chanmap = uproot.open('chanmap_20180416.root')
    chan_tree = f_chanmap['t']
    return chan_tree

tree, n_pot, cdc_tree = get_noise_data()
n_bunches = n_pot / 16e6

def get_cdc_tree():
    return cdc_tree

# Set up the data, throw away the invalid ones
wire = tree['wire']
event_id = tree['event_id']
layer = tree['layer']
edep = tree['edep']
doca = tree['doca']
t = tree['t']
dbg_x = tree['x']
dbg_y = tree['y']
dbg_z = tree['z']
track_id = tree['track_id']
pid = tree['pid']

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

edep = edep[(tree['wire']>=0) * (edep>1e-6)]

logging.info('Size wire %d pid %d doca %d' % (wire.size, pid.size, doca.size))

# Format data into tensor
train = np.array([edep, t, doca], dtype=np.float32).T
logging.info('train shape {0}'.format(train.shape))


g_qt = skp.QuantileTransformer(output_distribution='normal', n_quantiles=5000)
g_minmax = skp.MinMaxScaler(feature_range=(-1, 1))

def preprocess():
    train_qt = g_qt.fit_transform(train)
    train_minmax = g_minmax.fit_transform(train_qt)
    return train_minmax

def inv_preprocess(tensor):
    inv_tensor = g_qt.inverse_transform(g_minmax.inverse_transform(tensor.detach().cpu().numpy()))
    return torch.tensor(inv_tensor)





def diagnostic_plots(train_minmax, output_dir):
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
    
    
    __inv_train = inv_preprocess(torch.from_numpy(train_minmax))
    plt.hist(np.log10(__inv_train[:,0]), bins=50, alpha=0.7)
    plt.hist(np.log10(edep), bins=50, alpha=0.7)
    plt.savefig(output_dir+'inv_transform.png')
    plt.clf()


def chunk(train_minmax, seq_len, batch_size):

    train_minmax_torch = torch.from_numpy(train_minmax).T
    chunk_size = seq_len
    chunk_stride = 256
    train_minmax_chunked = train_minmax_torch.unfold(1, 
            chunk_size, chunk_stride) # (feature, batch, seq)
    n_chunks = train_minmax_chunked.shape[1]
    n_features = train_minmax_chunked.shape[0]
    
    wire_torch = torch.from_numpy(wire).long().unsqueeze(0)
    wire_chunked = wire_torch.unfold(1, 
            chunk_size, chunk_stride) # (feature, batch, seq)
    
    logging.info('Continuous features shape: {0}   Discrete features shape: {1}'.format(train_minmax_chunked.shape, wire_chunked.shape))
    
    train_dataset = torch.utils.data.TensorDataset(train_minmax_chunked.permute(1, 0, 2), 
            wire_chunked.permute(1, 0, 2))
    train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader, train_dataset, n_chunks

