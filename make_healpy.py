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
    """
    By specifying whether this is a DM, RM, or NH map, it sets some presets for
    the Healpy projections
    """
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

def plot_healpy(data, data_type, radius=None, rho=None, num=None, angle=0, multiplot=False):
    """
    Create healpy plot for dataset data
    """
    d = get_plot_presets(data_type)
    n_theta = data.shape[0]
    n_phi = data.shape[1]
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    ttheta, pphi = np.meshgrid(theta, phi, indexing = 'ij')
    if multiplot:
        fig = plt.figure(figsize=(6.4, 9))
    else:
        fig = plt.figure()

    # HealPy transformation
    nside = 100
    pixel_indices = hp.ang2pix(nside, ttheta, pphi)
    m = np.ones(hp.nside2npix(nside))
    m[pixel_indices] = data
    m.clip(d['mi'], d['ma'], out=m)

    llabel = rlabel = sub = None
    if radius is not None: llabel='%s kpc' % radius
    if rho is not None: rlabel=r'$\rho$ = %2.1g' % rho
    if multiplot: sub=211

    projview(m, fig=fig, min=d['mi'], max=d['ma'], coord=["G"], flip="geo",
            projection_type="mollweide", cmap=d['cmap'], rot=(180+angle, 0, 0),
            norm=d['norm'], unit=d['unit'], cbar_ticks=d['cbar_ticks'],
            title=None, llabel=llabel, rlabel=rlabel, sub=sub)
    if multiplot:
        plot_PDF(data, data_type, radius=radius, rho=rho, num=num, multiplot=True)
        plt.tight_layout()
    plt.savefig('%s_%d_%d.png' % (data_type, radius, num))
    plt.close('all')

def plot_PDF(data, data_type, radius=None, rho=None, num=None, multiplot=False):
    """
    Create probability distribution function plots
    """
    d = get_plot_presets(data_type)
    if multiplot:
        plt.subplot(212)
    else:
        fig = plt.Figure()
    ax = plt.gca()
    if data_type == 'DM':
        log_data = np.log10(data)
        bins = np.linspace(np.log10(d['mi']),np.log10(d['ma']),30)
        counts, bins = np.histogram(log_data, bins=bins)
        norm = counts / data.size # PDF
        ax.stairs(norm, bins, color='k', fill=True)
        ax.set_yscale('log')
        ax.set_ylabel('PDF')
        ax.set_ylim(1e-6,1e-0)
        ax.set_xlabel('log(DM (pc cm$^{-3}$))')
        if not multiplot:
            if radius is not None:
                text='%s kpc' % radius
                ax.text(0.05, 0.94, text, horizontalalignment='left', size=14,
                        weight='heavy', color='k', transform=ax.transAxes)
            if rho is not None:
                text2=r'$\rho$ = %2.1g' % rho
                ax.text(0.95, 0.94, text2, horizontalalignment='right', size=14,
                        weight='heavy', color='k', transform=ax.transAxes)
        if not multiplot:
            plt.savefig('%s_%d_%d_PDF.png' % (data_type, radius, num))
    return
