# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:29:32 2019

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

class AlineProcessor:
    
    def __init__(self,data):
        
        # load wavelength array, hanning window, linear in wavenumber interpolation prereqs.
        lam = np.load('lam.npy').astype(np.float32)
        win = np.hanning(2048).astype(np.float32)
        lam_min = np.amin(lam)
        lam_max = np.amax(lam)
        d_lam = np.asarray([lam_max-lam_min]).astype(np.float32)
        d_k = (1/lam_min - 1/lam_max)/2048
        k = np.array([1/((1/lam_max)+d_k*i) for i in range(2048)]).astype(np.float32)
        nn0 = np.zeros((2048,),np.int32)
        nn1 = np.zeros((2048,),np.int32)
        # find nearest neighbors for each point in the lambda spec for interp prep.
        for i in range(1,2047):
            res = np.abs(lam-k[i])
            minind = np.argmin(res)
            if res[minind]>=0:
                nn0[i]=minind-1
                nn1[i]=minind
            else:
                nn0[i]=minind
                nn1[i]=minind+1
        
        self.nn0_np=nn0
        self.nn1_np=nn1
        # do preliminary setup for opencl side of things (platform,device,context,queue)
        self.dtype = np.float32
        self.platform = cl.get_platforms()
        self.platform = self.platform[0]
        self.device = self.platform.get_devices()
        self.device = self.device[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        mflags = cl.mem_flags
        self.nn0 = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=nn0)
        self.nn1 = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=nn1)
        self.k = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=k)
        self.win = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=win)
        self.lam = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=lam)
        self.d_lam = cl.Buffer(self.context, mflags.READ_ONLY | mflags.COPY_HOST_PTR, hostbuf=d_lam)
        
        self.npdata = np.zeros((2048,)).astype(np.float32)
        self.npres = np.zeros((2048,)).astype(np.float32)
        self.npres1 = np.zeros((2048,)).astype(np.float32)
        self.result_ih = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.npres)
        self.result_nan = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.npres1)
        self.global_wgsize = (2048,)
        self.local_wgsize = (512,)
        self.data = cl.Buffer(self.context, mflags.COPY_HOST_PTR, hostbuf=self.npdata)
        self.result = cl.Buffer(self.context, mflags.COPY_HOST_PTR, hostbuf=self.npres)
        
        # do reikna setup for FFT
        self.api = cluda.ocl_api()
        self.thr = self.api.Thread.create()
        self.dshape = (2048)
        arbdata = self.thr.to_device(np.zeros(self.dshape).astype(np.complex64))
        self.outp = self.thr.array(self.dshape,np.complex64)
        self.fft = FFT(arbdata,axes=(0,)).compile(self.thr)
        
        self.program = cl.Program(self.context, """
        __kernel void hann(__global float *lam_s, __global float *win, __global float *res)
        {
            int i = get_global_id(0);
            res[i] = lam_s[i]*win[i];
        }
        
        __kernel void interp(__global float *lam_s, __global float *lam, __global float *k, __global int *nn0,
                                  __global int *nn1, __global float *d_lam, __global float *win, __global float *res)
        {
            int i = get_global_id(0);
            
            float y1 = lam_s[nn0[i]];
            float y2 = lam_s[nn1[i]];
            float x1 = k[i];
            float dl = d_lam[0];
            float x2 = x1+dl;
            
    		if (y1==y2)
            {
    			res[i] = y1;
    		}
    		else
    		{
    			res[i] = (y1 + (x1 - x2)/(y1 - y2))*dl;
    		}

        }
        """).build()
        
        self.interp = self.program.interp
        self.hann = self.program.hann
        
    def reikna_FFT(self,data):
        inp = self.thr.to_device(data)
        self.fft(self.outp,inp,inverse=0)
        return self.outp.get()
        
    def interp_hann_wrapper(self,data):
        #apply hanning window
        self.data = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        self.hann.set_args(self.data,self.win,self.result_ih)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        
        #do interpolation
        self.interp.set_args(self.result_ih,self.lam,self.k,self.nn0,self.nn1,self.d_lam,self.win,self.result_nan)
        cl.enqueue_nd_range_kernel(self.queue,self.interp,self.global_wgsize,self.local_wgsize)
        cl.enqueue_copy(self.queue,self.npres1,self.result_nan)
        
        plt.plot(self.npres1)
        
        return self.npres1
    
    def process_aline(self,data):
        self.interp_hann_wrapper(data)
        ft = self.reikna_FFT(self.npres)
        return ft

if __name__ == '__main__':
    
    data = np.load('data.npy')
    data = data.flatten()
    data = data[0:2048].astype(np.float32)
    aproc = AlineProcessor(data)
    times = []
    result = (aproc.process_aline((data)))
        
    