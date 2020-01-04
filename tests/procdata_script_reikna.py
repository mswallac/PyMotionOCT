# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:14:50 2019

@author: Mike
based on example provided here:
    https://www.drdobbs.com/open-source/easy-opencl-with-python/240162614?pgno=2
    
"""

from pyopencl import cltypes
from pyopencl import array
import pyopencl as cl
from reikna import cluda
from reikna.cluda import Snippet
from reikna.core import Transformation, Type, Annotation, Parameter
from reikna.algorithms import PureParallel
import reikna.transformations as transformations
import numpy as np
import time

# Load some test data!
rawdata = np.load('data.npy').squeeze()

# Load chirp matrix, containing wavelengths corresponding to spectrum bins for lambda->k interpolation
lam = np.load('lam.npy').astype(np.float16)
# Define apodization window
window = np.hanning(2048).astype(np.float16)

lam_min = np.amin(lam)
lam_max = np.amax(lam)
d_lam = np.array([lam_max-lam_min],dtype=np.float)
d_k = (1/lam_min - 1/lam_max)/2048
k = np.array([1/((1/lam_max)+d_k*i) for i in range(2048)],dtype=np.float16)

nn0 = np.zeros(2048,dtype=np.uint16)
nn1 = np.zeros(2048,dtype=np.uint16)

for i in range(2048):
    res = np.abs(lam-k[i])
    minind = np.argmin(res)
    if res[minind]>=0:
        nn0[i]=minind-1
        nn1[i]=minind
    else:
        nn0[i]=minind
        nn1[i]=minind+1
        
### OPENCL SIDE
        
platform = cl.get_platforms()
platform = platform[0]
device = platform.get_devices()
device = device[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

api = cluda.ocl_api()
thr = api.Thread.create()

program = thr.compile("""
KERNEL void interp_hann(
    GLOBAL_MEM float *dest, GLOBAL_MEM float *d_lam,
    GLOBAL_MEM float *win, GLOBAL_MEM float *lambda_spec,
    GLOBAL_MEM int *nn0, GLOBAL_MEM int *nn1,
    GLOBAL_MEM float *lam, GLOBAL_MEM float *k)
{
    const SIZE_T gid = get_local_id(0);
  	float y1 = lambda_spec[nn0[gid]];  // y-values from neighbors in spectrum
  	float y2 = lambda_spec[nn1[gid]];
  	float x1 = lam[nn0[gid]];  // corresponding initial wavelength
  	float x = k[gid];  // linear-in-wavenumber interpolation point
  
  	if (y1 == y2)
  	{
  		dest[gid] = y1*win[gid];
  	}
  	else
  	{
  		dest[gid] = (y1 + (x - x1) / (y2 - y1) * d_lam[0]) * win[gid];
  	}
}
""")

aline = np.array(rawdata[:,0],dtype=np.float16)
lspec_g = thr.to_device(aline)
r_g = thr.empty_like(lspec_g)
win_g = thr.to_device(window)
nn0_g = thr.to_device(nn0)
nn1_g = thr.to_device(nn1)
lam_g = thr.to_device(lam)
k_g = thr.to_device(k)

t=time.time()

interp_hann = program.interp_hann
interp_hann.prepare(256)
interp_hann(r_g,d_lam,win_g,lspec_g,nn0_g,nn1_g,lam_g,k_g,local_size=256, global_size=256)

#res=r_g.get()

#print(res)
