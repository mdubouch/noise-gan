import argparse

parser = argparse.ArgumentParser('Train CDC GAN')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--latent-dims', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=2048)
args = parser.parse_args()

ngf = args.ngf
ndf = args.ndf
latent_dims = args.latent_dims
seq_len = args.sequence_length

print('Importing networks...')
import networks
gen = networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len)
print(gen)
disc = networks.Disc(ndf=ndf, seq_len=seq_len)
print(disc)

print('Importing geometry...')
import geom_util as gu
print('cumulative wires', gu.cum_n_wires)

print('Importing dataset...')
import data
print('pot', data.n_pot, ' bunches', data.n_bunches)
print('dtypes', data.tree.dtype)
print('shape', data.tree.shape)

print('Pre-processing...')

print('Training begin')
import time
from tqdm import tqdm
for i in tqdm(range(100)):
    time.sleep(0.01) 

print('Done')
