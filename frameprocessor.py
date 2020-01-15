# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:17:16 2020

@author: Mike
"""

import numpy as np
import pyopencl as cl
from pyopencl import cltypes
from pyopencl import array
from reikna.fft import FFT
from reikna import cluda
import time
import matplotlib.pyplot as plt

class FrameProcessor():
    
    def npcast(self,inp,dt):
        return np.asarray(inp).astype(dt)
    
    def rshp(self,inp,shape):
        return np.reshape(inp,shape,'C')
    
    def __init__(self,nlines):
        
        # Define data formatting
        n = nlines # number of A-lines per frame
        alen = 2048 # length of A-line / # of spec. bins
        self.dshape = (alen*n,)
        self.dt_prefft = np.float32
        self.dt_fft = np.complex64
        self.data_prefft = self.npcast(np.zeros(self.dshape),self.dt_prefft)
        self.data_fft = self.npcast(np.zeros(self.dshape),self.dt_fft)
        
        # Load spectrometer bins and prepare for interpolation / hanning operation
        hanning_win = self.npcast(np.hanning(2048),self.dt_prefft)
        lam = self.npcast(np.load('lam.npy'),self.dt_prefft)
        lmax = np.max(lam)
        lmin = np.min(lam)
        kmax = 1/lmin
        kmin = 1/lmax
        self.d_l = (lmax - lmin)/alen
        self.d_k = (kmax - kmin)/alen
        self.k_raw = self.npcast([1/x for x in (lam)],self.dt_prefft)
        self.k_lin = self.npcast([kmax-(i*self.d_k) for i in range(alen)],self.dt_prefft)
        
        # Find nearest neighbors for interpolation prep.
        nn0 = np.zeros((2048,),np.int32)
        nn1 = np.zeros((2048,),np.int32)
        for i in range(0,2048):
            res = np.abs(self.k_raw-self.k_lin[i])
            minind = np.argmin(res)
            if i==0:
                nn0[i]=0
                nn1[i]=1
            if res[minind]>=0:
                nn0[i]=minind-1
                nn1[i]=minind
            else:
                nn0[i]=minind
                nn1[i]=minind+1
            
        self.nn0=nn0
        self.nn1=nn1
        
        # Initialize PyOpenCL platform, device, context, queue
        self.platform = cl.get_platforms()
        self.platform = self.platform[0]
        self.device = self.platform.get_devices()
        self.device = self.device[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        # POCL input buffers
        mflags = cl.mem_flags
        self.win_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=hanning_win)
        self.nn0_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=self.nn0)
        self.nn1_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=self.nn1)
        self.k_lin_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=self.k_lin)
        self.k_raw_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=self.k_raw)
        self.d_k_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=self.d_k)
        
        # POCL output buffers
        self.npres_interp = self.npcast(np.zeros(self.dshape),self.dt_prefft)
        self.npres_hann = self.npcast(np.zeros(self.dshape),self.dt_prefft)
        self.result_interp = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.npres_interp)
        self.result_hann = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.npres_hann)
        
        # Define POCL global / local work group sizes
        self.global_wgsize = (2048,n)
        self.local_wgsize = (512,1)
        
        # Initialize Reikna API, thread, FFT plan, output memory
        self.api = cluda.ocl_api()
        self.thr = self.api.Thread.create()
        self.outp = self.thr.array((2048,n),self.dt_fft)
        self.result = self.npcast(np.zeros((2048,n)),self.dt_fft)
        self.fft = FFT(self.result,axes=(0,)).compile(self.thr)
        
        # kernels for hanning window, and interpolation
        self.program = cl.Program(self.context, """
        __kernel void hann(__global float *inp, __global const float *win, __global float *res)
        {
            int i = get_global_id(0)+(get_global_size(0)*get_global_id(1));
            int j = get_local_id(0)+(get_group_id(0)*get_local_size(0));
            res[i] = inp[i]*win[j];
        }
        
        __kernel void interp(__global float *y,__global const int *nn0,__global const int *nn1,
                             __global const float *k_raw,__global const float *k_lin,__global float *res)
        {
            int i_shift = (get_global_size(0)*get_global_id(1));
            int i = get_global_id(0)+i_shift;
            int j = get_local_id(0)+(get_group_id(0)*get_local_size(0));
            float x1 = k_raw[nn0[j]];
            float x2 = k_raw[nn1[j]];
            float y1 = y[i_shift+nn0[j]];
            float y2 = y[i_shift+nn1[j]];
            float x = k_lin[j];
            res[i]=y1+((x-x1)*((y2-y1)/(x2-x1))); 
        }
        """).build()
        
        self.hann = self.program.hann
        self.interp = self.program.interp
    
        # wraps reikna fft
    def FFT(self,data):
        inp = self.thr.to_device(self.npcast(data,self.dt_fft))
        self.fft(self.outp,inp,inverse=0)
        self.result = self.outp.get()
        return
    
    def interp_hann(self,data):
        self.data_pfg = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        self.hann.set_args(self.data_pfg,self.win_g,self.result_hann)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        self.interp.set_args(self.result_hann,self.nn0_g,self.nn1_g,self.k_raw_g,self.k_lin_g,self.result_interp)
        cl.enqueue_nd_range_kernel(self.queue,self.interp,self.global_wgsize,self.local_wgsize)
        cl.enqueue_copy(self.queue,self.npres_interp,self.result_interp)
        return
    
    def proc_frame(self,data):
        self.interp_hann(data)
        self.FFT(self.rshp(self.npres_interp,(2048,n)))
        res2=np.fft.fft(self.rshp(self.npres_interp,(2048,n)),axis=0)
        return self.result,res2
        
if __name__ == '__main__':
    
    n=200
    fp = FrameProcessor(n)
    data = np.load('data.npy').flatten()
    data = fp.npcast(data[0:2048*n],fp.dt_prefft)
    plt.figure()
    plt.plot(data)
    t=time.time()
    res,res2 = fp.proc_frame(data)
    res = np.reshape(res,(2048,n),'C')
    plt.subplot(121)
    plt.imshow(20*np.log10(np.abs(res[0:2048,:])))
    plt.subplot(122)
    plt.imshow(20*np.log10(np.abs(res2[0:2048,:])))