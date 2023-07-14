"""
Code for converting Sam's theta-phi numpy arrays to a healpy image
"""
import healpy as hp
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import sys
import os

# Code expects a numpy file describing an array of Nx2N size with values of
# theta, phi for NH (neutral hydrogen) or Ne (dispersion measure)

if len(sys.argv) != 2:
    sys.exit('Usage: python %s <numpy_filename>' % sys.argv[0])

fn = sys.argv[1]
name = os.path.splitext(os.path.basename(fn))[0]

# Key to hydrogen column density or dispersion measure based on filename
if name.startswith('NH'):
    mi = 1e18
    ma = np.inf
    cbtext = 'log($N_{H}$ [$cm^{-2}$])'
    cmap = mpl.cm.viridis
#    cmap = mpl.cm.inferno
if name.startswith('Ne'):
    mi = 1e0
    ma = 1e3
    cbtext = 'log(DM [$pc \cdot cm^{-3}$])'
    cmap = mpl.cm.inferno

# Automatically get the array size from the loaded numpy array
struct = np.load(fn)
print('Max = %g' % np.max(struct))
print('Mean = %g' % np.mean(struct))
print('Median = %g' % np.median(struct))
n_theta = struct.shape[0]
n_phi = struct.shape[1]
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
ttheta, pphi = np.meshgrid(theta, phi, indexing = 'ij')
struct.clip(mi,ma, out=struct)
fig = plt.figure()

# Plot cartesian projection
#plt.imshow(struct.T, cmap=cmap, norm='log')
#plt.xticks([])
#plt.yticks([])
#plt.colorbar()

# HealPy transformation
nside = 100
pixel_indices = hp.ang2pix(nside, ttheta, pphi)
m = np.ones(hp.nside2npix(nside))
m[pixel_indices] = struct
m.clip(mi, ma, out=m)

# Plot as Mollweide projection
#hp.mollview(m, fig=fig, norm='log', title='',  rot=(180, 0, 0), flip='geo', cmap=cmap)
hp.mollview(m, fig=fig, min=mi, max=ma, norm='log', title='',  rot=(180, 0, 0), flip='geo', cmap=cmap)

# Make colorbar better
cb = fig.get_axes()[1]
ticks = np.logspace(np.log10(m.min()), np.log10(m.max()), 4, endpoint=True)
cb.set_xticks(ticks)
logticks = ['%2.1f' % tick for tick in np.log10(ticks)]
cb.set_xticklabels(logticks)
cb.text( 0.5, -4.0, cbtext, transform=cb.transAxes, ha="center", va="bottom")
plt.tight_layout()
plt.savefig('%s.png' % name)
