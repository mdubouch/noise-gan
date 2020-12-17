from networks import *
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

gen = Gen(ngf=ngf, latent_dims=latent_dims, seq_len=seq_len)
print(gen)
disc = Disc(ndf=ndf, seq_len=seq_len)
print(disc)
