import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from gizmopy.load_from_snapshot import *
from gizmopy.load_fire_snap import *
from gizmopy.quicklook import *
from weighted_2D_map import *
import astropy.units as u
from astropy.coordinates import cartesian_to_spherical

def center_and_clip_mod(xyz,center_xyz,r_cut):
    '''
    trim vector, re-center it, and clip keeping particles of interest
    '''    
    xyz = xyz-center_xyz
    d2 = np.sum(xyz*xyz, axis=1)
    ok = np.where(d2 < r_cut*r_cut)[0]
    xyz = xyz.take(ok, axis=0)
    return xyz, ok

def return_perp_vectors(a):
    ''' procedure which will return, for a given input vector A_in, 
    the perpendicular unit vectors B_out and C_out which form perpendicular axes to A '''
    eps = 1.0e-10
    a = np.array(a,dtype='f');
    a /= np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
    for i in range(len(a)):
        if (a[i]==0.): a[i]=eps;
        if (a[i]>=1.): a[i]=1.-eps;
        if (a[i]<=-1.): a[i]=-1.+eps;
    a /= np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
    ax=a[0]; ay=a[1]; az=a[2];
    ## use a fixed rotation of the a-vector by 90 degrees:
    ## (this anchors the solution so it changes *continously* as a changes)
    t0=np.double(np.pi/2.e0);
    bx=0.*ax; by=np.cos(t0)*ay-np.sin(t0)*az; bz=np.sin(t0)*ay+np.cos(t0)*az;
    bmag = np.sqrt(bx**2 + by**2 + bz**2); bx/=bmag; by/=bmag; bz/=bmag;
    ## c-sign is degenerate even for 'well-chosen' a and b: gaurantee right-hand 
    ##  rule is obeyed by defining c as the cross product: a x b = c
    cx=(ay*bz-az*by); cy=-(ax*bz-az*bx); cz=(ax*by-ay*bx); 
    cmag = np.sqrt(cx**2 + cy**2 + cz**2); cx/=cmag; cy/=cmag; cz/=cmag;
    B_out=np.zeros(3); C_out=np.zeros(3);
    B_out[:]=[bx,by,bz]; C_out[:]=[cx,cy,cz];
    return B_out, C_out
    
def xyz_after_center_and_rotate(sdir, snum, findcen=True, field_of_view=100., edgeon=False): #see quicklook and image_maker
    afac=load_fire_snap('Time',5,sdir,snum);
    max_search = 1000.*afac/2.;
    if(max_search<200.): max_search=200.;
    if(max_search>1000.): max_search=1000.;
    cen,n0=estimate_zoom_center(sdir,snum,ptype=0,min_searchsize=0.005,quiet=True,
        search_deviation_tolerance=0.0005,search_stepsize=1.25,max_searchsize=max_search);
    xyz,ok = center_and_clip_mod(load_fire_snap('Coordinates',0,sdir,snum),cen,20.) # load positions
    mass = load_fire_snap('Masses',0,sdir,snum).take(ok,axis=0)
    vxyz = load_fire_snap('Velocities',0,sdir,snum).take(ok,axis=0) # load velocities
    vxyz = vxyz - np.median(vxyz,axis=0) # reset to median local velocity about r=0
    x_h  = load_fire_snap('NeutralHydrogenAbundance',0,sdir,snum).take(ok,axis=0) 
    I_touse = x_h * mass # will normalize to column density for m in 1e10 Msun and r in kpc, then correct for H versus He fractions
    jvec=np.cross(vxyz,xyz); 
    m_jvec=(I_touse*(jvec.transpose())).transpose(); # compute j vector
    # m_jvec=((jvec.transpose())).transpose(); # compute j vector
    xyz_t=xyz.transpose();
    r=np.sqrt(xyz_t[0]**2+xyz_t[1]**2+xyz_t[2]**2)
    ok_tmp = np.where((r>0)&(r<20))[0]
    m_jvec = m_jvec.take(ok_tmp,axis=0)
    
    j_tot=np.sum(m_jvec,axis=0);  z_vec=j_tot/np.sqrt(np.sum(j_tot*j_tot)); # this is the z-vector
    x_vec,y_vec = return_perp_vectors(z_vec) 
    # x_vec,y_vec,z_vec = im_maker.get_angmom_projection_vectors(sdir,snum,rcut=field_of_view/2.,center=cen);
    if(edgeon): return cen, [x_vec,z_vec,y_vec]
    return cen, [x_vec,y_vec,z_vec]

def get_vals(xyz, ok, proj_matrix, sdir, snum, spectrum = False):
    '''Generic function to select relevant data from FIRE sims for a given cut'''
    data_dict = {}
    xyz  = xyz.take(ok,axis=1); data_dict['xyz'] = xyz;
    hsml = load_fire_snap('Hsml',0,sdir,snum).take(ok,axis=0); data_dict['hsml'] = hsml;
    if 'MagneticField' in load_fire_snap('Keys',0,sdir,snum):
        Bxyz = np.dot( proj_matrix , (load_fire_snap('MagneticField',0,sdir,snum).take(ok,axis=0)).transpose() )
        data_dict['Bxyz'] = Bxyz;
    mass = load_fire_snap('Masses',0,sdir,snum).take(ok,axis=0); data_dict['mass'] = mass;
    T    = load_fire_snap('Temperature',0,sdir,snum).take(ok,axis=0); data_dict['T'] = T;
    rho  = load_fire_snap('Density',0,sdir,snum).take(ok,axis=0); data_dict['rho'] = rho;
    n = load_fire_snap('NumberDensity',0,sdir,snum).take(ok,axis=0); data_dict['NumberDensity'] = n
    x_e  = load_fire_snap('ElectronAbundance',0,sdir,snum).take(ok,axis=0); data_dict['x_e'] = x_e;
    x_h  = load_fire_snap('NeutralHydrogenAbundance',0,sdir,snum).take(ok,axis=0); data_dict['x_h'] = x_h;
    vxyz = np.dot( proj_matrix , (load_fire_snap('Velocities',0,sdir,snum).take(ok,axis=0)).transpose() ); data_dict['vxyz'] = vxyz;
    vol = load_fire_snap('Volume',0,sdir,snum).take(ok,axis=0); data_dict['vol'] = vol
    if 'CosmicRayEnergy' in load_fire_snap('Keys',0,sdir,snum):
        Ecr = load_fire_snap('CosmicRayEnergy',0,sdir,snum).take(ok,axis=0)
        data_dict['Ecr'] = Ecr

    if spectrum == True:
        Scr = load_fire_snap('CosmicRayMomentumDistSlope',0,sdir,snum).take(ok,axis=0); data_dict['Scr'] = Scr
        if 'MolecularMassFraction' in load_from_snapshot('Keys',0,sdir,snum):
            x_mol = load_fire_snap('MolecularMassFraction',0,sdir,snum).take(ok,axis=0);data_dict['x_mol'] = x_mol

    return data_dict


def loadData(path_input, snum, spectrum=False, xlen=30, depth=200, edgeon=False):
    '''
    Load a simulation snapshot and return arrays of useful quantities.
    path_input = '/path/to/snapshot/', string
    snum = snapshot number, int
    spectrum = True for CRSpec runs, False for single-bin, bool
    xlen = sets the x and y range in kpc, integer
    depth = range in z is equal to in kpc, float
    snum = the number in the snapshot filename, integer
    edgeon = whether to reorganize cube when centering & rotating to be edge on (see xyz_after_center_and_rotate)
    single_bin = whether this is a single CR bin sim or full spectrum (True/False)
    Outputs arrays of:
        gas number density, particle volume (?), thermal electron density,
        magnetic field strength x, y and z, particle data (?)
        also outputs the integer def center_and_clip(xyz,center_xyz,r_cut):

    xyz = xyz-center_xyz;
    d2 = np.sum(xyz*xyz, axis=1)
    ok = np.where(d2 < r_cut*r_cut)[0]
    xyz = xyz.take(ok, axis=0)
    return xyz, okxlen
    '''
    center, proj_matrix = xyz_after_center_and_rotate(
        path_input, snum, edgeon=edgeon)

    xyz = np.dot(proj_matrix, (load_fire_snap(
        'Coordinates', 0, path_input, snum)-center).transpose())

    particle_filter = np.where((np.abs(xyz[0]) < xlen) & (
        np.abs(xyz[1]) < xlen) & (np.abs(xyz[2]) < depth))[0]

    data = get_vals(
        xyz, particle_filter, proj_matrix, path_input, snum, spectrum=spectrum)

    return data

path_input = '/panfs/ds09/hopkins/sponnada/m12i/'
# path_input = '/panfs/ds09/hopkins/sponnada/m12f/mhdcv/snapdir_600'
snum = 600
xlen = 100
depth = 100

unit_NH = 1.248e24 
unit_DM = unit_NH/3e18
unit_DM_spherical = unit_DM #for some reason need offset to match cartesian values - check with Phil
unit_NH_spherical  = unit_NH

data = loadData(path_input, snum, spectrum = False, \
                                                    xlen = xlen, depth = depth, edgeon = False)


Ne, NH, _ = construct_weighted2dmap(data['xyz'][0], data['xyz'][2], data['hsml'],
                                    data['mass']*data['x_e'], data['mass']*data['x_h'], xlen=xlen, set_aspect_ratio=1.0, pixels=512)

np.save('Ne_{}_kpc_{}_depth.npy'.format(xlen, depth), Ne*unit_DM)

np.save('NH_{}_kpc_{}_depth.npy'.format(xlen, depth), NH*unit_NH)


plt.figure()
plt.imshow(np.log10(Ne.T*unit_DM),
           extent=[-xlen, xlen, -xlen, xlen], cmap='inferno')
plt.xlabel(r'x [kpc]')
plt.ylabel(r'z [kpc]')
plt.colorbar(label=r'log $_{10}$DM [pc cm$^{-3}$]')
plt.savefig('Ne_{}_kpc_{}_depth.png'.format(xlen, depth))


plt.figure()
plt.imshow(np.log10(NH.T*unit_NH),
           extent=[-xlen, xlen, -xlen, xlen], cmap='inferno')
plt.xlabel(r'x [kpc]')
plt.ylabel(r'z [kpc]')
plt.colorbar(label=r'log $_{10}$N$_{\rm H}$ [cm$^{-2}$]')
plt.savefig('NH_{}_kpc_{}_depth.png'.format(xlen, depth))




solar_circ_vec = np.array([100,0,0])

#xyz centered on a point in Solar Circle 
xs = data['xyz'][0] - solar_circ_vec[0]
ys = data['xyz'][1] - solar_circ_vec[1]
zs = data['xyz'][2] - solar_circ_vec[2]

cartesian_radii = np.sqrt(xs**2+ys**2+zs**2)

filter_r = [200]

radial_filters = [np.where(cartesian_radii < i) for i in filter_r]

for x in range(len(radial_filters)):

    filt = radial_filters[x]

    r, lat, lon = cartesian_to_spherical(xs[filt], ys[filt], zs[filt])

    lon -= np.pi*u.rad

    print(np.nanmin(lat),np.nanmax(lat))
    print(np.nanmin(lon), np.nanmax(lon))



    Ne, NH, _ = construct_weighted2dmap(lat, lon, data['hsml'][filt]/r,
                                        data['mass'][filt]*data['x_e'][filt]/(r**2), data['mass'][filt]*data['x_h'][filt]/(r**2), xlen=np.pi/2, set_aspect_ratio=2.0, pixels=512)

    np.save('Ne_{}_kpc_{}_depth_spherical_{}.npy'.format(xlen,depth,filter_r[x]),Ne*unit_DM_spherical)
    np.save('NH_{}_kpc_{}_depth_spherical_{}.npy'.format(xlen, depth,filter_r[x]), NH*unit_NH_spherical)


    plt.figure()
    plt.imshow(np.log10(Ne*unit_DM_spherical), extent=[-np.pi,
            np.pi, -np.pi/2, np.pi/2],vmin=0,vmax=5, cmap='inferno')
    plt.ylabel(r'latitude [rad]')
    plt.xlabel(r'longitude [rad]')
    plt.colorbar(label=r'log$_{10}$ DM [pc cm$^{-3}$]')
    plt.savefig('Ne_{}_kpc_{}_depth_spherical_{}.png'.format(xlen,depth,filter_r[x]))

    plt.figure()
    plt.imshow(np.log10(NH*unit_NH_spherical), extent=[-np.pi,
            np.pi, -np.pi/2, np.pi/2], cmap='inferno')
    plt.ylabel(r'latitude [rad]')
    plt.xlabel(r'longitude [rad]')
    plt.colorbar(label=r'log$_{10}$ NH [cm$^{-2}$]')
    plt.savefig('NH_{}_kpc_{}_depth_spherical_{}.png'.format(xlen, depth,filter_r[x]))




