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

    def draw_cdc(self, ax, margin=50):
        r = np.sqrt(self.wire_x**2 + self.wire_y**2)
        max_r = r.max()
        min_r = r.min()
        r_layer1 = r[self.cum_n_wires[0]]
        layer_spacing = r_layer1 - min_r
        min_r = min_r - layer_spacing*2
        max_r = max_r + layer_spacing*2

        from matplotlib.patches import Ellipse
        inner = Ellipse((0, 0), min_r*2, min_r*2, facecolor=(0, 0, 0, 0), edgecolor='gray')
        outer = Ellipse((0, 0), max_r*2, max_r*2, facecolor=(0, 0, 0, 0), edgecolor='gray')

        ax.add_patch(inner)
        ax.add_patch(outer);

        max_r = max_r+margin
        min_r = min_r-margin

        ax.set(xlim=(-max_r,max_r), ylim=(-max_r,max_r), xlabel='x [mm]', ylabel='y [mm]')

