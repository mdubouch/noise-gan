import uproot
import numpy as np

class GeomUtil():
    def __init__(self, cdc_tree):
        self.wire_x = cdc_tree['z'] - 7650
        self.wire_y = cdc_tree['y']
        self.n_wires = cdc_tree['z'].size
        self.id = cdc_tree['id']
        self.layer = cdc_tree['layer']
        uq, cnts = np.unique(self.layer, return_counts=True)
        self.n_wires_per_layer = cnts
        self.cum_n_wires = np.cumsum(self.n_wires_per_layer)

        self.n_layers = uq.size
        assert(self.n_wires_per_layer.sum() == self.cum_n_wires[-1])
    
    def wire_pos(self, wire_idx):
        # wire_idx is array of indices
        cond = (wire_idx[:, np.newaxis] == self.id)
        big_x = np.tile(self.wire_x, (wire_idx.size, 1))
        big_y = np.tile(self.wire_y, (wire_idx.size, 1))

        arr = (big_x[cond], big_y[cond])

        return arr

    def validate_wire_pos(self):
        x, y = self.wire_pos(np.arange(0, self.cum_n_wires[-1]))
        assert((x == self.wire_x).all())
        assert((y == self.wire_y).all())

