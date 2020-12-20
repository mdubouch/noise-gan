import argparse
import logging


parser = argparse.ArgumentParser('Train CDC GAN')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--latent-dims', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=2048)
parser.add_argument('--log', type=str, default='info')
args = parser.parse_args()

logging.basicConfig(filename='output.log', level=getattr(logging, args.log.upper()), format='%(asctime)s %(message)s')

ngf = args.ngf
ndf = args.ndf
latent_dims = args.latent_dims
seq_len = args.sequence_length

print('Importing networks...')
import networks
gen = networks.Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len)
logging.info(gen)
disc = networks.Disc(ndf=ndf, seq_len=seq_len)
logging.info(disc)

print('Importing geometry...')
import geom_util as gu
logging.info('cumulative wires {0}'.format(gu.cum_n_wires))

print('Importing dataset...')
import data
logging.info('pot %d  bunches %d', data.n_pot, data.n_bunches)
logging.info('dtypes {0}'.format(data.tree.dtype))
logging.info('shape {0}'.format(data.tree.shape))

print('Pre-processing...')

print('Training begin')
import time
from tqdm import tqdm
for i in tqdm(range(100)):
    time.sleep(0.01) 

print('Done')
