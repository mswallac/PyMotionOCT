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

class FrameProcessor:
    
    def __init__(self):
        
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
                
        # do preliminary setup for opencl side of things (platform,device,context,queue)
        self.dtype = np.complex64
        self.platform = cl.get_platforms()
        self.platform = self.platform[0]
        self.device = self.platform.get_devices()
        self.device = self.device[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        self.nn0 = self.cast_to_dev(nn0)
        self.nn1 = self.cast_to_dev(nn1)
        self.k = self.cast_to_dev(k)
        self.win = self.cast_to_dev(win)
        self.lam = self.cast_to_dev(lam)
        self.d_lam = self.cast_to_dev(d_lam)
        
        # do reikna setup for FFT
        self.api = cluda.ocl_api()
        self.thr = self.api.Thread.create()
        self.dshape = (2048,40)
        arbdata = self.thr.to_device(np.zeros(self.dshape).astype(self.dtype))
        self.fft = FFT(arbdata,axes=(0,)).compile(self.thr)
        self.interp_hann = cl.elementwise.ElementwiseKernel(self.context,
                               "float *lam_s, float *lam, float *k, int *nn0, int *nn1, float *d_lam, float *win, float *res",
                               "res[i] = lam_s[nn0[i]] + (k[i] - lam[nn1[i]])/(lam_s[nn0[i]] - lam_s[nn1[i]])*d_lam[0]*win[i]",
                               "interp")
        # 
        # replace nan with y1 using scan kernel
    
    def reikna_FFT(self,data):
        inp = self.thr.to_device(data)
        outp = self.thr.array(inp.shape,self.dtype)
        self.fft(outp,inp,inverse=0)
        return outp.get()
    
    def interp_hann_wrapper(self,data):
        temp = self.cast_to_dev(data[:,0])
        result = cl.array.empty_like(temp)
        outp = []
        for i in range(data.shape[1]):
            self.set_data_dev(temp,data[:,i])
            evt = self.interp_hann(temp,self.lam,self.k,self.nn0,self.nn1,self.d_lam,self.win,result)
            outp.append(result.get())
        return np.array(outp)
    
    def cast_to_dev(self,data):
        return cl.array.to_device(self.queue,np.ascontiguousarray(data.astype(data.dtype)))
    
    def set_data_dev(self,temp,data):
        temp.set(np.ascontiguousarray(data))
        return
    
    def process_frame(self,data):
        ih = self.interp_hann_wrapper(data)
        ft = self.reikna_FFT(ih)
        print(ft)
        return ft

if __name__ == '__main__':
    
    #initialize frameprocessor
    t=time.time()
    frameproc = FrameProcessor()
    print('Initialization time (s): ',time.time()-t)
    
    #lets take a look at how fast reikna FFT is
    shape = (2048,40)
    for i in range(10):
        data = np.ascontiguousarray(np.random.normal(size=shape))
        t=time.time()
        result = frameproc.process_frame(data)
        print(result)
        print('Frame Proc. Time #'+str(i+1)+' time (s): ',"%.20f"%(time.time()-t))
        
        
    