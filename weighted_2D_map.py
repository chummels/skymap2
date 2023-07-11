import numpy as np
import ctypes
import pfh_utils as util
import visualization.image_maker as im_maker
from gizmopy.load_from_snapshot import *
from gizmopy.load_fire_snap import *
from gizmopy.quicklook import *



# some utility functions for below
def fcor(x):
    return np.array(x,dtype='f',ndmin=1)
def vfloat(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float));
def cfloat(x):
    return ctypes.c_float(x);
def checklen(x):
    return len(np.array(x,ndmin=1));

# core routine for constructing a weighted projection-map on a grid from irregular data
def construct_weighted2dmap( x, y, hsml, weight, weight2=0, weight3=0,
        xlen = 10, pixels = 720, set_aspect_ratio = 1.0 ):

    ## set some basic initial values
    xypixels=pixels; xpixels=xypixels; ypixels = np.around(float(xpixels)*set_aspect_ratio).astype(int);
    ylen=xlen*set_aspect_ratio;
    xmin=-xlen; xmax=xlen; ymin=-ylen; ymax=ylen;

    ## load the routine we need
    exec_call=util.return_python_routines_cdir()+'/SmoothedProjPFH/allnsmooth.so'
    smooth_routine=ctypes.cdll[exec_call];
    ## make sure values to be passed are in the right format
    N=checklen(x); x=fcor(x); y=fcor(y); M1=fcor(weight); M2=fcor(weight2); M3=fcor(weight3); H=fcor(hsml)
    xpixels=np.int(xpixels); ypixels=np.int(ypixels)
    ## check for whether the optional extra weights are set
    NM=1;
    if(checklen(M2)==checklen(M1)):
        NM=2;
        if(checklen(M3)==checklen(M1)):
            NM=3;
        else:
            M3=np.copy(M1);
    else:
        M2=np.copy(M1);
        M3=np.copy(M1);
    ## initialize the output vector to recieve the results
    XYpix=xpixels*ypixels; MAP=ctypes.c_float*XYpix; MAP1=MAP(); MAP2=MAP(); MAP3=MAP();
    ## main call to the imaging routine
    smooth_routine.project_and_smooth( \
        ctypes.c_int(N), \
        vfloat(x), vfloat(y), vfloat(H), \
        ctypes.c_int(NM), \
        vfloat(M1), vfloat(M2), vfloat(M3), \
        cfloat(xmin), cfloat(xmax), cfloat(ymin), cfloat(ymax), \
        ctypes.c_int(xpixels), ctypes.c_int(ypixels), \
        ctypes.byref(MAP1), ctypes.byref(MAP2), ctypes.byref(MAP3) );
    ## now put the output arrays into a useful format
    MassMap1=np.ctypeslib.as_array(MAP1).reshape([xpixels,ypixels]);
    MassMap2=np.ctypeslib.as_array(MAP2).reshape([xpixels,ypixels]);
    MassMap3=np.ctypeslib.as_array(MAP3).reshape([xpixels,ypixels]);

    return MassMap1, MassMap2, MassMap3
