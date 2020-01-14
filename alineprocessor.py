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
    
    def __init__(self,localsize):
        
        # load wavelength array, hanning window, linear in wavenumber interpolation prereqs.
        lam = np.load('lam.npy')
        win = np.hanning(2048).astype(np.float16)
        lam_min = np.amin(lam)
        lam_max = np.amax(lam)
        d_lam = np.asarray([lam_max-lam_min])
        d_k = (1/lam_min - 1/lam_max)/2048
        k = np.array([1/((1/lam_max)+d_k*i) for i in range(2048)])
        nn0 = np.zeros((2048,),int)
        nn1 = np.zeros((2048,),int)
        
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
        self.dtype = np.complex64
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
        
        self.npdata = np.zeros((2048))
        self.npres = np.zeros((2048))
        self.result_ih = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.npres)
        self.result_nan = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.npres)
        self.global_wgsize = (2048,)
        self.local_wgsize = (localsize,)
        self.data = cl.Buffer(self.context, mflags.COPY_HOST_PTR, hostbuf=self.npdata)
        self.result = cl.Buffer(self.context, mflags.COPY_HOST_PTR, hostbuf=self.npres)
        
        # do reikna setup for FFT
        self.api = cluda.ocl_api()
        self.thr = self.api.Thread.create()
        self.dshape = (2048)
        arbdata = self.thr.to_device(np.zeros(self.dshape).astype(self.dtype))
        self.outp = self.thr.array(self.dshape,self.dtype)
        self.fft = FFT(arbdata,axes=(0,)).compile(self.thr)
        self.program = cl.Program(self.context, """
        __kernel void interp_hann(global float *lam_s, global float *lam, global float *k, global int *nn0,
                                  global int *nn1, global float *d_lam, global float *win, global float *res)
        {
            int i = get_global_id(0);
            
    		if ((lam_s[nn0[i]] - lam_s[nn1[i]]) < 1E-12)
            {
    			res[i] = lam_s[nn0[i]]*win[i];
    		}
    		else
    		{
    			res[i] = (lam_s[nn0[i]] + (k[i] - lam[nn1[i]])/(lam_s[nn0[i]] - lam_s[nn1[i]]))*d_lam[0]*win[i];
    		}

        }
        __kernel void replace_nan(global float *input, global float *res)
        {
            int i = get_global_id(0);
            float val = input[i];
            
            if (isnan(val))
            {
                res[i]=1;
            }
            else
            {
                res[i]=val;
            }
        }
        """).build()
        
    def reikna_FFT(self,data):
        inp = self.thr.to_device(data)
        self.fft(self.outp,inp,inverse=0)
        return self.outp.get()
        
    def interp_hann_wrapper(self,data):
        self.data = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        self.program.interp_hann(self.queue,self.global_wgsize,self.local_wgsize,self.data,self.lam,self.k,self.nn0,self.nn1,
                                     self.d_lam,self.win,self.result_ih)
        self.program.replace_nan(self.queue,self.global_wgsize,self.local_wgsize,self.result_ih,self.result_nan)
        cl.enqueue_copy(self.queue,self.npres,self.result_nan)
        return self.npres
        return
    
    def process_aline(self,data):
        self.interp_hann_wrapper(data)
        ft = self.reikna_FFT(self.npres)
        return ft

if __name__ == '__main__':
    local_wgsize=32
    #initialize frameprocessor
    print('With local group size %d'%local_wgsize)
    t=time.time()
    aproc = AlineProcessor(local_wgsize)
    print('Initialized in %.0fms: '%(1000*(time.time()-t)))
    
    nt=20000
    intervals = []
    results=[]
    data = (np.load('data.npy')[:,:,0])
    
    for i in range(nt):
        t = time.time()
        results.append(aproc.process_aline(np.ascontiguousarray(data[:,i])))
        intervals.append(time.time()-t)
        
    print('Over %d alines:'%nt)
    print('Average A-line rate: %.0fHz'%(1/(np.mean(intervals))))
    print('Average A-line proc. time: %.2fms'%(1000*(np.mean(intervals))))
        
        
    