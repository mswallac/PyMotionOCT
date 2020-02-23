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
import matplotlib.pyplot as plt
from reikna.cluda import Snippet
from reikna.algorithms import PureParallel
from reikna.linalg import MatrixMul
import reikna.transformations as transformations
from reikna.transformations import combine_complex
import time

class DisplacemenctCalc():

    def npcast(self,inp,dt):
        return np.asarray(inp).astype(dt)

    def rshp(self,inp,shape):
        return np.reshape(inp,shape,'C')
    
    def set_nlines(self,nlines):
        # Define data formatting
        self.dt = np.complex64
        n = nlines # number of A-lines per frame
        alen = 2048 # length of A-line / # of spec. bins
        self.nlines = n
        self.dshape = (alen,n)
        
        # Define POCL global / local work group sizes
        self.global_wgsize = (2048,n)
        self.local_wgsize = (256,1)

        self.fft_in = Type(self.dt, shape=self.dshape)
        self.fft = FFT(self.fft_in,axes=(0,))
        self.cfft = self.fft.compile(self.thr)
        return

    def set_win(self,win):
        self.win=win
        self.win_g = cl.Buffer(self.context, self.mflags.READ_ONLY | 
                               self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.win)
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
        
        # Set framesize (# of a-lines)
        self.set_nlines(nlines)
        
        # Prepare / hanning operation
        hanning_win = self.npcast([np.hanning(2048)for x in range(nlines)],self.dt)
        self.set_win(hanning_win)
        
        self.npres_hann_a = self.npcast(np.zeros(self.dshape),self.dt)
        self.result_hann_a = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_hann_a)
        self.npres_hann_b = self.npcast(np.zeros(self.dshape),self.dt)
        self.result_hann_b = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_hann_b)
        self.fft_buffer_a = self.thr.empty_like(self.fft.parameter.output)
        self.fft_buffer_b = self.thr.empty_like(self.fft.parameter.output)
        
        # Set apodization window and framesize (# of a-lines)
        self.ga_fft = self.thr.array((self.dshape), dtype=np.complex64)
        self.gb_fft = self.thr.array((self.dshape), dtype=np.complex64)
        self.r = self.thr.array((self.dshape), dtype=np.complex64)
        self.r_ifft = self.thr.array((self.dshape), dtype=np.complex64)
        self.max_r = self.thr.array((self.dshape), dtype=np.complex64)
        
        # kernel for hanning window
        self.program = cl.Program(self.context, """
        __kernel void hann(__global float *inp, __global const float *win, __global float *res)
        {
            int i = get_global_id(0)+(get_global_size(0)*get_global_id(1));
            int j = get_local_id(0)+(get_group_id(0)*get_local_size(0));
            res[i] = inp[i]*win[j];
        }
        """).build()
        self.hann = self.program.hann

    def example(self,ga,gb):
        self.hann.set_args(ga,self.win_g,self.result_hann_a)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        self.hann.set_args(gb,self.win_g,self.result_hann_b)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        self.FFT(self.fft_buffer_a,self.result_hann_a)
        self.FFT(self.fft_buffer_b,self.result_hann_b)
        return self.fft_buffer_a,self.fft_buffer_b

    # Wraps FFT kernel
    def FFT(self,out,data):
        self.cfft(out, data)
        return
        
if __name__ == '__main__':
    # Number of frames to benchmark with and empty lists for framerate / aline rate
    n=60
    dc=DisplacemenctCalc(n)
    ga = np.random.random((2048,60)) + np.random.random((2048,60))*1j
    ga = ga.astype(dc.dt)
    gb = np.random.random((2048,60)) + np.random.random((2048,60))*1j
    gb = gb.astype(dc.dt)
    ga_g = cl.Buffer(dc.context, dc.mflags.ALLOC_HOST_PTR | dc.mflags.COPY_HOST_PTR, hostbuf=ga)
    gb_g = cl.Buffer(dc.context, dc.mflags.ALLOC_HOST_PTR | dc.mflags.COPY_HOST_PTR, hostbuf=gb)
    times=[]
    n_frames=10000
    for x in range(n_frames):
        t=time.time()
        a,b=dc.example(ga_g,gb_g)
        times.append(time.time()-t)
        
    # Calculate benchmark stats and add to lists
    avginterval = np.mean(times)
    frate=(1/avginterval)
    afrate=frate*n
    print('Average framerate of %.1fHz over %d frames'%(frate,n_frames))    
    
