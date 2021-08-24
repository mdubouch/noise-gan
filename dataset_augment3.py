import uproot3 as uproot
import torch
import logging
import numpy as np
import sklearn.preprocessing as skp
from torch.utils.data import Sampler

class TestSequenceSampler(Sampler):
    def __init__(self, dataset, seq_len):
        self.seq_len = seq_len
        self.size = len(dataset) // seq_len

    def __iter__(self):
        for i in range(self.size):
            indices = torch.arange(i*self.seq_len, (i+1)*self.seq_len)
            yield indices

    def __len__(self):
        return self.size

class SequenceSampler(Sampler):
    # Samples a fixed-length sequence of hits by randomly arranging events
    # in the original hit dataset.
    def __init__(self, event_ids, seq_len):
        self.event_ids = torch.tensor(event_ids, device='cuda:0').unsqueeze(0)
        self.seq_len = seq_len

        self.uq_events, self.uq_idx, self.uq_inv = np.unique(event_ids, return_index=True,
                return_inverse=True)
        n = self.uq_events.size
        self.uq_events = torch.tensor(self.uq_events, device='cuda:0')
        print('unique events:', n)
        k = seq_len
        # Number of combinations is n choose k
        self.n_combinations = \
                np.math.factorial(n) // np.math.factorial(k) // np.math.factorial(n - k)
        if (self.n_combinations > 100): # this just sets the epoch timing. In theory we just
            # have too many possible combinations of events so the dataset is enormous.
            self.n_combinations = 100
        self.generator = torch.Generator(device='cuda:0')

    def __iter__(self):
        # Draw a random combination of event ids and return the corresponding hit
        # sequence, cropped to seq_len.
        #print(self.n_combinations)
        print('init seed')
        self.generator.manual_seed(self.generator.initial_seed())
        for i in range(self.n_combinations):
            #print('start')
            indices = torch.randint(0, self.uq_events.shape[0], size=(self.seq_len,), 
                    device='cuda:0', generator=self.generator)
            hits = (self.uq_events[indices].unsqueeze(1) == self.event_ids).flatten()
            #print(hits.shape) # (seq_len, n_events)
            #print(hits.sum(axis=1))
            #print(indices)
            sequence = torch.where(hits)[0] % self.event_ids.shape[1]
            #print(sequence)
            #print(sequence.shape)
            #print(sequence)
            #print('yield')
            yield sequence[:self.seq_len]

    def __len__(self):
        # Length is the number of seq_len-long combinations by event of the initial dataset
        return self.n_combinations

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

    def load(self, test_size=4096):
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

        self.test_edep = self.edep[-test_size:]
        self.test_t = self.t[-test_size:]
        self.test_doca = self.doca[-test_size:]
        self.test_wire = self.wire[-test_size:]
        self.test_event_id = self.event_id[-test_size:]

        self.edep = self.edep[:-test_size]
        self.t = self.t[:-test_size]
        self.doca = self.doca[:-test_size]
        self.wire = self.wire[:-test_size]
        self.event_id = self.event_id[:-test_size]

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

        self.train_final = self.train_minmax
        self.test_final = self.test_minmax
        logging.info('%s' % (self.train_final.shape,))
        
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
        
        __inv_train = self.inv_preprocess(torch.from_numpy(self.train_minmax))
        plt.hist(np.log10(__inv_train[:,0]), bins=50, alpha=0.7)
        plt.hist(np.log10(self.edep), bins=50, alpha=0.7)
        plt.savefig(output_dir+'inv_transform.png')
        plt.clf()


    def _chunk(self, continuous_features, discrete_features, seq_len, batch_size):
        
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
        loader = torch.utils.data.DataLoader(dataset, 
                batch_size=batch_size, shuffle=True, pin_memory=True)
    
        return loader, dataset, n_chunks

    def chunk(self, seq_len, batch_size):
        print(self.train_final.shape, self.wire[:,np.newaxis].shape)
        self.train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.train_final, dtype=torch.float32), 
                torch.tensor(self.wire[:,np.newaxis], dtype=torch.int64))
        train_sampler = SequenceSampler(self.event_id, seq_len)
        print('Train sequences: %d' % len(train_sampler))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                sampler=train_sampler, pin_memory=True)

        self.n_chunks = len(self.train_dataset)

        self.test_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.test_final, dtype=torch.float32),
                torch.tensor(self.test_wire[:,np.newaxis], dtype=torch.int64))
        test_sampler = TestSequenceSampler(self.test_dataset, seq_len)
        print('Test sequences: %d' % len(test_sampler))
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size,
                sampler=test_sampler, pin_memory=True)

        print('Training samples: %d, test samples: %d' % 
                (len(self.train_dataset), len(self.test_dataset)))

        return self.train_loader, self.train_dataset, self.n_chunks


