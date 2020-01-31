# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:17:16 2020

@author: Mike
"""

from reikna.core import Annotation, Type, Transformation, Parameter
from reikna.cluda import dtypes, any_api
from pyopencl import cltypes
from pyopencl import array
import pyopencl as cl
from reikna.fft import FFT
from reikna import cluda
import numpy as np
import time
import os
import matplotlib.pyplot as plt

class FrameProcessor():

    def npcast(self,inp,dt):
        return np.asarray(inp).astype(dt)

    def rshp(self,inp,shape):
        return np.reshape(inp,shape,'C')
    
    # thanks @fjarri https://github.com/fjarri/reikna/blob/develop/examples/demo_real_to_complex_fft.py
    def get_complex_trf(self,arr):
        complex_dtype = dtypes.complex_for(arr.dtype)
        return Transformation(
            [Parameter('output', Annotation(Type(complex_dtype, arr.shape), 'o')),
            Parameter('input', Annotation(arr, 'i'))],
            """
            ${output.store_same}(
                COMPLEX_CTR(${output.ctype})(
                    ${input.load_same},
                    0));
            """)
    
    def set_nlines(self,nlines):
        n = nlines # number of A-lines per frame
        alen = 2048 # length of A-line / # of spec. bins
        self.n = n
        self.dshape = (alen,n)
        self.data_prefft = self.npcast(np.zeros(self.dshape),self.dt_prefft)
        self.data_fft = self.npcast(np.zeros(self.dshape),self.dt_fft)
        # POCL output buffers
        self.npres_interp = self.npcast(np.zeros(self.dshape),self.dt_prefft)
        self.npres_hann = self.npcast(np.zeros(self.dshape),self.dt_prefft)
        self.result_interp = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_interp)
        self.result_hann = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_hann)
        
        # Define POCL global / local work group sizes
        self.global_wgsize = (2048,n)
        self.local_wgsize = (256,1)

        self.trf = self.get_complex_trf(self.data_prefft)
        self.fft = FFT(self.trf.output,axes=(0,))
        self.cfft = self.fft.parameter.input.connect(self.trf, self.trf.output, new_input=self.trf.input).compile(self.thr)
        self.fft_buffer = self.thr.empty_like(self.cfft.parameter.output)
        return

    def set_apod_win(self,win):
        self.apod_win=win
        self.win_g = cl.Buffer(self.context, self.mflags.READ_ONLY | 
                               self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.apod_win)
        return
    
    def set_chirp_arr(self,arr):
        alen = 2048
        self.chirp_arr = arr
        lam = self.chirp_arr
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
        mflags=self.mflags
        self.nn0_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.ALLOC_HOST_PTR | mflags.COPY_HOST_PTR, hostbuf=self.nn0)
        self.nn1_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.ALLOC_HOST_PTR | mflags.COPY_HOST_PTR, hostbuf=self.nn1)
        self.k_lin_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.ALLOC_HOST_PTR | mflags.COPY_HOST_PTR, hostbuf=self.k_lin)
        self.k_raw_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.ALLOC_HOST_PTR | mflags.COPY_HOST_PTR, hostbuf=self.k_raw)
        self.d_k_g = cl.Buffer(self.context, mflags.READ_ONLY | mflags.ALLOC_HOST_PTR | mflags.COPY_HOST_PTR, hostbuf=self.d_k)
        return
    
    def __init__(self,nlines):
        
        # Initialize PyOpenCL platform, device, context, queue
        self.platform = cl.get_platforms()
        self.platform = self.platform[0]
        self.device = self.platform.get_devices()
        self.device = self.device[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        # POCL memflags
        self.mflags = cl.mem_flags
        mflags=self.mflags
        
        # Initialize Reikna API, thread, FFT plan, output memor
        self.api = cluda.ocl_api()
        self.thr = self.api.Thread(self.queue)
        
        # Define data formatting
        alen = 2048 # length of A-line / # of spec. bins
        self.dt_prefft = np.float32
        self.dt_fft = np.complex64
        
        # Load spectrometer bins and prepare for interpolation / hanning operation
        hanning_win = self.npcast(np.hanning(2048),self.dt_prefft)
        self.set_chirp_arr(self.npcast(np.load('lam.npy'),self.dt_prefft))
        
        
        # Set apodization window and framesize (# of a-lines)
        self.set_apod_win(hanning_win)
        self.set_nlines(nlines)
        
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
            int i_glob = get_global_id(0)+i_shift;
            int i_loc = get_local_id(0)+(get_group_id(0)*get_local_size(0));
            float x1 = k_raw[nn0[i_loc]];
            float x2 = k_raw[nn1[i_loc]];
            float y1 = y[i_shift+nn0[i_loc]];
            float y2 = y[i_shift+nn1[i_loc]];
            float x = k_lin[i_loc];
            res[i_glob]=y1+((x-x1)*((y2-y1)/(x2-x1))); 
        }
        """).build()

        self.hann = self.program.hann
        self.interp = self.program.interp
    
    
    
    # Wraps FFT kernel
    def FFT(self,data):
        self.cfft(self.fft_buffer, data)
        return

    # Wraps interpolation and hanning window kernels
    def interp_hann(self,data):
        self.data_pfg = cl.Buffer(self.context, self.mflags.COPY_HOST_PTR | self.mflags.ALLOC_HOST_PTR, hostbuf=data)
        self.hann.set_args(self.data_pfg,self.win_g,self.result_hann)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        self.interp.set_args(self.result_hann,self.nn0_g,self.nn1_g,self.k_raw_g,self.k_lin_g,self.result_interp)
        cl.enqueue_nd_range_kernel(self.queue,self.interp,self.global_wgsize,self.local_wgsize)
        return
    
    def hann_wrap(self,data):
        self.data_pfg = cl.Buffer(self.context, self.mflags.COPY_HOST_PTR | self.mflags.ALLOC_HOST_PTR, hostbuf=data)
        self.hann.set_args(self.data_pfg,self.win_g,self.result_hann)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        return

    def proc_frame(self,data):
        self.interp_hann(data)
        self.FFT(self.result_interp)
        res_gpu = self.fft_buffer.get()
        return res_gpu
    
    def proc_frame_no_interp(self,data):
        self.hann_wrap(data)
        self.FFT(self.result_hann)
        res_gpu = self.fft_buffer.get()
        return res_gpu

if __name__ == '__main__':
    # Number of frames to benchmark with and empty lists for framerate / aline rate
    nf=10000
    ns=[]
    fs=[]
    afs=[]
    fp = FrameProcessor(2)
    
    for i,n in enumerate(range(2,66,2)):
        
        fp.set_nlines(n)
        # Initialize frameprocessor object
        if i==0:
            print('Using Device: %s'%(str(fp.device.name)))
        
        # Load / reshape / cast data
        data = np.load('data.npy').flatten()[0:2048*n].astype(np.float32).reshape(2048,n)  
        times=[]
        
        # Process many frames to get accurate bench
        for i in range(nf):
            t = time.time()
            res = fp.proc_frame(data)
            times.append(time.time()-t)
        
        # Calculate benchmark stats and add to lists
        avginterval = np.mean(times)
        frate=(1/avginterval)
        afrate=frate*n
        fs.append(frate)
        afs.append(afrate)
        ns.append(n)
        
        print('With n = %d '%n)
        print('Average framerate over %d frames: %.0fHz'%(nf,frate))
        print('Effective A-line rate over %d frames: %.0fHz'%(nf,afrate))
        
    np.save('ns.npy',ns)
    np.save('fs.npy',fs)
    np.save('afs.npy',afs)