# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:14:50 2019

@author: Mike
based on example provided here:
    https://www.drdobbs.com/open-source/easy-opencl-with-python/240162614?pgno=2
    
"""

import pyopencl as cl
from pyopencl import cltypes
from pyopencl import array
from pyopencl.elementwise import ElementwiseKernel
import numpy as np
import time
import matplotlib.pyplot as plt
from reikna import fft

if __name__ == "__main__":
    
    # Load some test data!
    rawdata = np.load('fig8_13_raw_5000nm_2p0.npy')
    posdata = np.zeros((2048,40))
    flat_rawdata = rawdata.flatten()
    #t0 = flat_rawdata[0:2048*40].astype(np.uint16)  # The first B-scan for use with the demo\
    t0=flat_rawdata.astype(np.uint16)
    n = 40  # Total A-scans in input (not all are included with B)
    x = 40  # A-scans in output
    
    B1 = posdata[2,:].astype(np.bool)
    B2 = posdata[3,:].astype(np.bool)
    b = np.logical_or(B1,B2)

    b = np.zeros(40).astype(bool)
    b[0:40] = 1
    
    # Load chirp matrix, containing wavelengths corresponding to spectrum bins for lambda->k interpolation
    lam = np.load('lam.npy')
    # Define apodization window
    window = np.hanning(2048).astype(np.float16)
    
    lam_min = np.amin(lam)
    lam_max = np.amax(lam)
    d_lam = lam_max-lam_min
    d_k = (1/lam_min - 1/lam_max)/2048
    k = np.array([1/((1/lam_max)+d_k*i) for i in range(2048)])
    
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
    
    ## Step #1. Obtain an OpenCL platform.
    platform = cl.get_platforms()
    platform = platform[0]
    
    print('\nAcquired platform: \n\t'+platform.name)
     
    ## Step #2. Obtain a device id for at least one device (accelerator).
    device = platform.get_devices()
    device = device[0]
    
    print('Acquired device: \n\t'+device.name)
    extensions = ['\t'+x for x in device.extensions.split()]
    print('Device extensions: ')
    for i in range(len(extensions)):
        print(extensions[i])
     
    ## Step #3. Create a context for the selected device.
    context = cl.Context([device])
    print('Created context.')     
    
    ## Step #6. Create one or more kernels from the program functions.
    interp_hann = ElementwiseKernel(context,
                               "float *y1, float *y2, float *x, float *x1, float d_lam, float *win, float *res",
                               "res = (y1[i] + (x[i] - x1[i]) / (y2[i] - y1[i]) * d_lam) * win[i]",
                               "interp")
        
    ## Step #7. Create a command queue for the target device.
    queue = cl.CommandQueue(context)
    
    ## Step #8. Allocate device memory and move input data from the host to the device memory.   
    result = np.zeros(t0.shape)
    
    ## INCOMPLETE need to put all the inputs as pocl arrays with cl.array.to_device(np_arr)
    
    ## Step #9. Associate the arguments to the kernel with kernel object.
    
    ## Step #10. Deploy the kernel for device execution.
    
    # check that array sizes are all correct
    
    # run kernel and get event to wait for
    result=interp_hann(queue, result.shape, (1,), raw_buf, win_buf, k_buf,
                        n0_buf, n1_buf, lam_buf, d_lam, dest_buf)
     
    ## Step #11. Move the kernel’s output data to host memory.
    
    
    print(result)
    ## Step #12. Release context, program, kernels and memory.
    ## PyOpenCL performs this step for you, and therefore,
    ## you don't need to worry about cleanup code
