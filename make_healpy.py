import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import sys
import os


if len(sys.argv) != 2:
    sys.exit('Usage: python %s <numpy_filename>' % sys.argv[0])

fn = sys.argv[1]

if fn.startswith('NH'):
    mi = 1e18
    cbtext = 'log($N_{H}$ [$cm^{-2}$])'
if fn.startswith('Ne'):
    mi = 1e0
    cbtext = 'log(DM [$pc \cdot cm^{-3}$])'
ma = np.inf

struct = np.load(fn)
n_theta = struct.shape[0]
n_phi = struct.shape[1]
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
ttheta, pphi = np.meshgrid(theta, phi, indexing = 'ij')
struct.clip(mi,ma, out=struct)

nside = 100
pixel_indices = hp.ang2pix(nside, ttheta, pphi)
m = np.ones(hp.nside2npix(nside))
m[pixel_indices] = struct
m.clip(mi, ma, out=m)

fig = plt.figure()
hp.mollview(m, fig=fig, norm='log', title='',  rot=(180, 0, 0), flip='geo')
cb = fig.get_axes()[1]
ticks = np.logspace(np.log10(m.min()), np.log10(m.max()), 4, endpoint=True)
cb.set_xticks(ticks)
logticks = ['%2.1f' % tick for tick in np.log10(ticks)]
cb.set_xticklabels(logticks)
cb.text( 0.5, -4.0, cbtext, transform=cb.transAxes, ha="center", va="bottom")
name = os.path.splitext(os.path.basename(fn))[0]
plt.savefig('%s.png' % name)
