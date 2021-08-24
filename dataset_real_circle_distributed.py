import uproot3 as uproot
import torch
import logging
import numpy as np
import sklearn.preprocessing as skp

class Data():
    def __init__(self):
        self.file = uproot.open('reconstructible_mc5a02_rconsthits_geom.root')
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

        # Define unique track ID from original event_id and track_id
        uq_tid, idx, inv, cnts = np.unique([self.event_id, self.track_id], axis=1,
                return_counts=True, return_inverse=True, return_index=True)
        n_uq_tracks = uq_tid.shape[1]
        # Sequential unique track_id (goes from 0 to n_uq_tracks-1)
        uq_tid_seq = np.arange(0, n_uq_tracks)
        tid_seq = np.zeros_like(self.track_id, dtype=int)
        tid_seq = uq_tid_seq[inv]
        # Track progression
        prog = np.zeros_like(self.track_id, dtype=float)
        # Indices to data for start of every track
        newtrack_idx = idx
        for i in range(newtrack_idx.size):
            first = np.sum(cnts[:i])
            last = first + cnts[i] 
            prog[first:last] = np.arange(0, cnts[i]) / cnts[i]

        progx = np.cos(prog * 2*np.pi)
        progy = np.sin(prog * 2*np.pi)
        print(progx.shape)

        self.test_edep = self.edep[-1*2048:]
        self.test_t = self.t[-1*2048:]
        self.test_doca = self.doca[-1*2048:]
        self.test_wire = self.wire[-1*2048:]
        self.test_progx = progx[-1*2048:]
        self.test_progy = progy[-1*2048:]

        self.edep = self.edep[:-1*2048]
        self.t = self.t[:-1*2048]
        self.doca = self.doca[:-1*2048]
        self.wire = self.wire[:-1*2048]
        self.progx = progx[:-1*2048]
        self.progy = progy[:-1*2048]

        logging.info('Train size wire %d pid %d doca %d' % (self.wire.size, self.pid.size, self.doca.size))

        # Format data into tensor
        # The data in the train and test arrays will be preprocessed
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

        self.train_final = np.concatenate([self.train_minmax,
            self.progx[:,np.newaxis], self.progy[:,np.newaxis]], axis=1)
        self.test_final = np.concatenate([self.test_minmax, 
            self.test_progx[:,np.newaxis], self.test_progy[:,np.newaxis]], axis=1)
        logging.info('%s' % (self.train_final.shape,))
        
    def inv_preprocess(self, tensor):
        # Hard-coded 3 for the three continuous features we preprocess
        inv_tensor = self.qt.inverse_transform(
                self.minmax.inverse_transform(tensor[:,:3].detach().cpu().numpy()))
        ret = torch.cat([torch.tensor(inv_tensor), tensor[:,3:].detach().cpu()], dim=1)
        return ret


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
        plt.hist(self.progx, bins=50)
        plt.savefig(output_dir+'train_progx.png')
        plt.clf()
        plt.hist(self.progy, bins=50)
        plt.savefig(output_dir+'train_progy.png')
        
        __inv_train = self.inv_preprocess(torch.from_numpy(self.train_minmax))
        plt.hist(np.log10(__inv_train[:,0]), bins=50, alpha=0.7)
        plt.hist(np.log10(self.edep), bins=50, alpha=0.7)
        plt.savefig(output_dir+'inv_transform.png')
        plt.clf()


    def _chunk(self, continuous_features, discrete_features, seq_len, batch_size, 
            num_replicas, rank):
        
        # Add circle for track encoding

        data_torch = torch.from_numpy(continuous_features).float().T
        chunk_size = seq_len
        chunk_stride = seq_len
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
        sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank)
        loader = torch.utils.data.DataLoader(dataset, 
                batch_size=batch_size, shuffle=False, pin_memory=True, 
                sampler=sampler)
    
        return loader, dataset, n_chunks

    def chunk(self, seq_len, batch_size, num_replicas, rank):
        self.train_loader, self.train_dataset, self.n_chunks = self._chunk(self.train_final, 
                self.wire, seq_len, batch_size, num_replicas, rank)
        self.test_loader, self.test_dataset, self.n_test_chunks = self._chunk(self.test_final, 
                self.test_wire, seq_len, batch_size, num_replicas, rank)
        print('Training samples: %d, test samples: %d' % (self.n_chunks, self.n_test_chunks))
        return self.train_loader, self.train_dataset, self.n_chunks


