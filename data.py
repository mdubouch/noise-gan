import uproot

def get_noise_data():

    file = uproot.open('reconstructible_mc5a02_rconsthits.root')
    n_pot = 990678399
    n_bunches = n_pot / 16e6
    
    data = file['noise/noise'].array()

    return data, n_pot

def get_chanmap():
    f_chanmap = uproot.open('chanmap_20180416.root')
    chan_tree = f_chanmap['t']
    return chan_tree

tree, n_pot = get_noise_data()
n_bunches = n_pot / 16e6
