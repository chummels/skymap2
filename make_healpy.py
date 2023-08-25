"""
Code for converting Sam's theta-phi numpy arrays to a healpy image
"""
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import cmocean
import sys
import os

def get_plot_presets(type_of_plot):
    if type_of_plot == 'NH':
        mi = 1e13
        ma = 1e21
        cmap = mpl.cm.viridis
        norm = 'log'
        unit='N$_H$ [cm$^{-2}$]'
        cbar_ticks=[1e13, 1e15, 1e17, 1e19, 1e21]
    elif type_of_plot == 'DM':
        mi = 1e0
        ma = 1e3
        cmap = mpl.cm.inferno
        norm = 'log'
        unit='DM [pc cm$^{-3}$]'
        cbar_ticks=[1, 10, 100, 1000]
    elif type_of_plot == 'RM':
        mi = -1e3
        ma = 1e3
        cmap = cmocean.cm.balance_r
        norm = 'symlog'
        unit='RM [rad m$^{-2}$]'
        cbar_ticks=[-1000, -100, -10, 0, 10, 100, 1000]
    else:
        sys.exit('%s is not a recognized type of plot' % type_of_plot)

    d = dict(mi=mi, ma=ma, cmap=cmap, norm=norm, unit=unit, cbar_ticks=cbar_ticks)
    return d

def plot_healpy(data, data_type, radius=None, rho=None, num=None, angle=0):
    """
    Create healpy plot for dataset data
    """
    d = get_plot_presets(data_type)
    n_theta = data.shape[0]
    n_phi = data.shape[1]
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    ttheta, pphi = np.meshgrid(theta, phi, indexing = 'ij')
    fig = plt.figure()

    # HealPy transformation
    nside = 100
    pixel_indices = hp.ang2pix(nside, ttheta, pphi)
    m = np.ones(hp.nside2npix(nside))
    m[pixel_indices] = data
    m.clip(d['mi'], d['ma'], out=m)

    llabel = rlabel = None
    if radius is not None: llabel='%s kpc' % radius
    if rho is not None: rlabel=r'$\rho$ = %g' % rho

    projview(m, fig=fig, min=d['mi'], max=d['ma'], coord=["G"], flip="geo",
            projection_type="mollweide", cmap=d['cmap'], rot=(180+angle, 0, 0),
            norm=d['norm'], unit=d['unit'], cbar_ticks=d['cbar_ticks'],
            title=None, llabel=llabel, rlabel=rlabel)
    plt.savefig('%s_%d_%d.png' % (data_type, radius, num))
    plt.close('all')

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit('Usage: python %s <numpy_filename>' % sys.argv[0])

    fn = sys.argv[1]
    name = os.path.splitext(os.path.basename(fn))[0]

    title=None
    rlabel=None
    llabel=None
    fontname="serif"

    # Automatically get the array size from the loaded numpy array
    struct = np.load(fn)
    print('Min = %g' % np.min(struct))
    print('Max = %g' % np.max(struct))
    print('Mean = %g' % np.mean(struct))
    print('Median = %g' % np.median(struct))
    n_theta = struct.shape[0]
    n_phi = struct.shape[1]
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    ttheta, pphi = np.meshgrid(theta, phi, indexing = 'ij')
    fig = plt.figure()

    # HealPy transformation
    nside = 100
    pixel_indices = hp.ang2pix(nside, ttheta, pphi)
    m = np.ones(hp.nside2npix(nside))
    m[pixel_indices] = struct
    m.clip(mi, ma, out=m)

    projview(m, fig=fig, min=mi, max=ma, coord=["G"], flip="geo", projection_type="mollweide", cmap=cmap, rot=(180, 0, 0), norm=norm, title=title, rlabel=rlabel, llabel=llabel, unit=unit, cbar_ticks=cbar_ticks)
    #hp.mollview(m, fig=fig, min=mi, max=ma, norm=norm, title='',  rot=(0, 0, 0), flip='geo', cmap=cmap)
    # y=8
    #hp.mollview(m, fig=fig, min=mi, max=ma, norm=norm, title='',  rot=(270, 0, 0), flip='geo', cmap=cmap)
    # y=-8
    #hp.mollview(m, fig=fig, min=mi, max=ma, norm=norm, title='',  rot=(90, 0, 0), flip='geo', cmap=cmap)

    # Make colorbar better
    #if norm == 'log':
    #    cb = fig.get_axes()[1]
    #    #ticks = np.logspace(np.log10(m.min()), np.log10(m.max()), 4, endpoint=True)
    #    ticks = np.logspace(np.log10(mi), np.log10(ma), 4, endpoint=True)
    #    cb.set_xticks(ticks)
    #    logticks = ['%2.1f' % tick for tick in np.log10(ticks)]
    #    cb.set_xticklabels(logticks)
    #    cb.text( 0.5, -4.0, cbtext, transform=cb.transAxes, ha="center", va="bottom")

    # Add text
    fig = plt.gcf()
    #text = '10 kpc'
    #fig.text(0.05, 0.94, text, horizontalalignment='left', size=14, weight='heavy', color='k', transform=fig.transFigure)
    #text2 = 'x=8'
    #fig.text(0.05, 0.88, text2, horizontalalignment='left', size=14, weight='heavy', color='k', transform=fig.transFigure)
    plt.savefig('%s_healpy.png' % name)

    plt.clf()
    ax = plt.gca()
    log_struct = np.log10(struct)
    bins = np.linspace(0,3,30)
    counts, bins = np.histogram(log_struct, bins=bins)
    norm = counts / struct.size # PDF
    ax.stairs(norm, bins, color='k', fill=True)
    ax.set_yscale('log')
    ax.set_ylabel('PDF')
    ax.set_ylim(1e-6,1e-1)
    ax.set_xlabel('log(DM (pc cm$^{-3}$))')
    #ax.text(0.05, 0.94, text, horizontalalignment='left', size=14, weight='heavy', color='k', transform=ax.transAxes)
    #ax.text(0.05, 0.88, text2, horizontalalignment='left', size=14, weight='heavy', color='k', transform=ax.transAxes)
    plt.savefig('%s_pdf.png' % name)
