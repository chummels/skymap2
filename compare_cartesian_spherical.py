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

if len(sys.argv) != 3:
    sys.exit('Usage: python %s <numpy_filename1> <numpy_filename2>' % sys.argv[0])

fn1 = sys.argv[1]
fn2 = sys.argv[2]
name = os.path.splitext(os.path.basename(fn1))[0]

# Key to hydrogen column density or dispersion measure based on filename
if name.startswith('NH'):
    mi = 1e13
    #ma = np.inf
    ma = 1e22
    cbtext = 'log($N_{H}$ [$cm^{-2}$])'
    cmap = mpl.cm.viridis
    cmap = mpl.cm.inferno
if name.startswith('Ne'):
    mi = 1e0
    ma = 1e3
    cbtext = 'log(DM [$pc \cdot cm^{-3}$])'
    cmap = mpl.cm.inferno

# Automatically get the array size from the loaded numpy array
struct1 = np.load(fn1)
struct2 = np.load(fn2)
print('Cartesian:')
print('Max = %g' % np.max(struct1))
print('Mean = %g' % np.mean(struct1))
print('Median = %g' % np.median(struct1))
print('Spherical:')
print('Max = %g' % np.max(struct2))
print('Mean = %g' % np.mean(struct2))
print('Median = %g' % np.median(struct2))
struct2 = struct2[:, 1024:2048].T # center to only look at the inner 1/2 in phi
print('Spherical Center pi in Phi:')
print('Max = %g' % np.max(struct2))
print('Mean = %g' % np.mean(struct2))
print('Median = %g' % np.median(struct2))
plt.imshow(struct2.T, cmap=cmap, norm='log')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
plt.savefig('%s_spherical_center.png' % name)
struct3 = struct1/struct2
struct3 = np.nan_to_num(struct3, nan=1.0)
print('Cartesian / Spherical:')
print('Max = %g' % np.max(struct3))
print('Mean = %g' % np.mean(struct3))
print('Median = %g' % np.median(struct3))

# Central disk region for both projections
struct1 = struct1[448:576, 448:576]
struct2 = struct2[448:576, 448:576]
plt.clf()
plt.imshow(struct1.T, cmap=cmap, norm='log')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
plt.savefig('%s_cartesian_center_zoom.png' % name)
plt.clf()
plt.imshow(struct2.T, cmap=cmap, norm='log')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
plt.savefig('%s_spherical_center_zoom.png' % name)
plt.clf()
struct4 = struct1/struct2
struct4 = np.nan_to_num(struct4, nan=1.0)
print('Zoom Cartesian / Zoom Spherical:')
print('Max = %g' % np.max(struct4))
print('Mean = %g' % np.mean(struct4))
print('Median = %g' % np.median(struct4))
plt.imshow(struct4.T, cmap=cmap, norm='log')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
plt.savefig('%s_residual_center_zoom.png' % name)
plt.clf()

fig = plt.figure()

# Plot cartesian projection
plt.imshow(struct3.T, cmap=cmap, norm='log')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
plt.savefig('%s_residual.png' % name)
