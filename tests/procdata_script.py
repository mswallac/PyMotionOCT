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
    
    #
    # Some of the same stuff done in motionOCT worker setup
    #
    
    # Load some test data!
    rawdata = np.load('data.npy')
    t0 = rawdata.flatten().astype(np.uint16)
    
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
    program = cl.Program(context, """
        __kernel void interp_hann(__global const double *lambda_spec,
        __global const double *win, __global const double *k,__global int *nn0,
        __global int *nn1,__global const double *lam, double d_lam, __global double *result)
        {
            int gid = get_global_id(0);
            int gid1 = gid % 2048;
    		double y1 = lambda_spec[nn0[gid1]];  // y-values from neighbors in spectrum
    		double y2 = lambda_spec[nn1[gid1]];
    		double x1 = lam[nn0[gid1]];  // corresponding initial wavelength
    		double x = k[gid1];  // linear-in-wavenumber interpolation point
            
    		if (y1 == y2)
    		{
    			result[gid] = y1*win[gid1];
    		}
    		else
    		{
    			result[gid] = (y1 + (x - x1) / (y2 - y1) * d_lam) * win[gid1];
    		}

        }
        """).build()
    
    ## Step #7. Create a command queue for the target device.
    queue = cl.CommandQueue(context)
    
    ## Step #8. Allocate device memory and move input data from the host to the device memory.   
    result = np.zeros(t0.shape)
    mem_flags = cl.mem_flags
    n0_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=nn0)
    n1_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=nn1)
    win_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=window)
    k_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=k)
    raw_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=t0)
    lam_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=lam)
    dest_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, result.nbytes)
    ## Step #9. Associate the arguments to the kernel with kernel object.
    
    ## Step #10. Deploy the kernel for device execution.
    
    # check that array sizes are all correct
    print(nn0.shape,nn1.shape,window.shape,k.shape,t0.shape,lam.shape,result.shape)
    
    # run kernel and get event to wait for
    program.interp_hann(queue, (len(result),), None, raw_buf, win_buf, k_buf,
                        n0_buf, n1_buf, lam_buf, d_lam, dest_buf)
     
    ## Step #11. Move the kernel’s output data to host memory.
    
    cl.enqueue_copy(queue,result,dest_buf)
    
    print(result)
    ## Step #12. Release context, program, kernels and memory.
    ## PyOpenCL performs this step for you, and therefore,
    ## you don't need to worry about cleanup code
