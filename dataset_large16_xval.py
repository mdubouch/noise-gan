import uproot3 as uproot
import torch
import logging
import numpy as np
import sklearn.preprocessing as skp

class Data():
    def __init__(self):
        self.file = uproot.open('reconstructible_toy+cdc.root')
        self.n_pot = 990678399
        self.n_bunches = self.n_pot / 16e6
        self.cdc_tree = self.file['cdc_geom/wires']
        self.data_tree = self.file['noise/noise']

        self.qt = skp.QuantileTransformer(output_distribution='normal', n_quantiles=5000)
        self.minmax = skp.MinMaxScaler(feature_range=(-1, 1))

    def get_cdc_tree(self):
        return self.cdc_tree.array()

    def load(self):
        self.data = self.data_tree.array()
        tree = self.data
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

        self.layer = layer[_select]
        self.event_id = event_id[_select]
        self.t = t[_select]
        self.dbg_x = dbg_x[_select]
        self.dbg_y = dbg_y[_select]
        self.dbg_z = dbg_z[_select]
        self.track_id = track_id[_select]
        self.pid = pid[_select]
        self.doca = doca[_select]
        self.wire = wire[_select]

        self.edep = edep[(tree['wire']>=0) * (edep>1e-6)]

        # Hardcoded. Should be seq_len instead of 2048...
        self.test_edep = self.edep[-1*2048:]
        self.test_t = self.t[-1*2048:]
        self.test_doca = self.doca[-1*2048:]
        self.test_wire = self.wire[-1*2048:]

        self.edep = self.edep[:16*2048]
        self.t = self.t[:16*2048]
        self.doca = self.doca[:16*2048]
        self.wire = self.wire[:16*2048]

        logging.info('Train size wire %d pid %d doca %d' % (self.wire.size, self.pid.size, self.doca.size))

        # Format data into tensor
        self.train = np.array([self.edep, self.t, self.doca], dtype=np.float32).T
        self.test = np.array([self.test_edep, self.test_t, self.test_doca], dtype=np.float32).T
        logging.info('train shape {0}'.format(self.train.shape))
        logging.info('test shape {0}'.format(self.test.shape))

    def preprocess(self):
        self.qt.fit(self.train)
        self.train_qt = self.qt.transform(self.train)
        self.test_qt = self.qt.transform(self.test)

        self.minmax.fit(self.train_qt)
        self.train_minmax = self.minmax.transform(self.train_qt)
        self.test_minmax = self.minmax.transform(self.test_qt)
        
    def inv_preprocess(self, tensor):
        inv_tensor = self.qt.inverse_transform(
                self.minmax.inverse_transform(tensor.detach().cpu().numpy()))
        return torch.tensor(inv_tensor)


    def diagnostic_plots(self, output_dir):
        # Some diagnostic plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.hist(np.log10(self.edep), bins=50)
        plt.savefig(output_dir+'train_edep.png')
        plt.clf()
        plt.hist(np.log10(self.t), bins=50)
        plt.savefig(output_dir+'train_t.png')
        plt.clf()
        plt.hist(self.doca, bins=50)
        plt.savefig(output_dir+'train_doca.png')
        plt.clf()
        
        #plt.figure(figsize=(6,6))
        #plt.scatter(dbg_z, dbg_y, s=edep*1e3, c=doca, cmap='inferno')
        #plt.savefig(output_dir+'train_scatter.png')
        #plt.clf()
        
        
        __inv_train = self.inv_preprocess(torch.from_numpy(self.train_minmax))
        plt.figure(figsize=(18,6))
        plt.subplot(131)
        plt.hist(np.log10(__inv_train[:,0]), bins=50, alpha=0.7)
        plt.hist(np.log10(self.edep), bins=50, alpha=0.7)
        plt.subplot(132)
        plt.hist(np.log10(__inv_train[:,1]), bins=50, alpha=0.7)
        plt.hist(np.log10(self.t), bins=50, alpha=0.7)
        plt.subplot(133)
        plt.hist(__inv_train[:,2], bins=50, alpha=0.7)
        plt.hist(self.doca, bins=50, alpha=0.7)
        plt.savefig(output_dir+'inv_transform.png')
        plt.close()


    def _chunk(self, continuous_features, discrete_features, seq_len, batch_size):
        data_torch = torch.from_numpy(continuous_features).T
        chunk_size = seq_len
        chunk_stride = 1
        data_chunked = data_torch.unfold(1, 
                chunk_size, chunk_stride) # (feature, batch, seq)
        n_chunks = data_chunked.shape[1]
        n_features = data_chunked.shape[0]
        
        wire_torch = torch.from_numpy(discrete_features).long().unsqueeze(0)
        wire_chunked = wire_torch.unfold(1, 
                chunk_size, chunk_stride) # (feature, batch, seq)

        logging.info('Continuous features shape: {0}   Discrete features shape: {1}'.format(data_chunked.shape, wire_chunked.shape))
        
        dataset = torch.utils.data.TensorDataset(data_chunked.permute(1, 0, 2), 
                wire_chunked.permute(1, 0, 2))
        loader = torch.utils.data.DataLoader(dataset, 
                batch_size=batch_size, shuffle=True, pin_memory=True)
    
        return loader, dataset, n_chunks

    def chunk(self, seq_len, batch_size):
        self.train_loader, self.train_dataset, self.n_chunks = self._chunk(self.train_minmax, 
                self.wire, seq_len, batch_size)
        self.test_loader, self.test_dataset, self.n_test_chunks = self._chunk(self.test_minmax, 
                self.test_wire, seq_len, batch_size)
        print('TRAIN CHUNKS:', self.n_chunks, ', TEST CHUNKS:', self.n_test_chunks)
        return self.train_loader, self.train_dataset, self.n_chunks
    
