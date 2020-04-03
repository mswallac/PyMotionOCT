# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:17:16 2020

@author: Mike
"""

import matplotlib.pyplot as plt
import time

from reikna.core import Annotation, Type, Transformation, Parameter
from reikna.cluda import dtypes, any_api, Snippet
from reikna.algorithms import Reduce, predicate_sum
from reikna.fft import FFT
from reikna import cluda
from pyopencl import cltypes
from pyopencl import array
from numba import jit
import pyopencl as cl
import numpy as np

def arg_max(arr):
    return np.argmax(arr)

def arr_sum(arr):
    return np.sum(arr,axis=1)

def arg_max_ij(arr):
    return np.unravel_index(arg_max(arr),arr.shape)

class DisplacemenctCalc():
    
    def npcast(self,inp,dt):
        return np.asarray(inp).astype(dt)

    def rshp(self,inp,shape):
        return np.reshape(inp,shape,'C')
    
    def set_nlines(self,nlines):
        # Define data formatting
        self.dt = np.complex64
        n = nlines # number of A-lines per frame
        alen = 242 # length of A-line / # of spec. bins
        self.nlines = n
        self.dshape = (alen,n)
        
        # Define POCL global / local work group sizes
        self.global_wgsize = (242,n)
        self.local_wgsize = (11,1)

        self.fft_in = Type(self.dt, shape=self.dshape)
        self.fft_in_sum = Type(self.dt, shape=(self.dshape[1],))
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
        
        # Initialize Reikna API, thread, FFT plan, output memor
        self.api = cluda.ocl_api()
        self.thr = self.api.Thread(self.queue)
        
        # Set framesize (# of a-lines)
        self.set_nlines(nlines)
        
        # Prepare / hanning operation
        hanning_win = self.npcast([np.hanning(242) for x in range(nlines)],self.dt)
        self.set_win(hanning_win)
        
        self.npres_hann_a = self.npcast(np.zeros(self.dshape),self.dt)
        self.result_hann_a = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_hann_a)
        self.npres_hann_b = self.npcast(np.zeros(self.dshape),self.dt)
        self.result_hann_b = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_hann_b)
        self.fft_buffer_a = self.thr.empty_like(self.fft.parameter.output)
        self.fft_buffer_b = self.thr.empty_like(self.fft.parameter.output)
        
        # BUFFERS / GPU ARRRAYS FOR PHASE CORRELATION
        self.ga_fft = self.thr.array((self.dshape), dtype=np.complex64)
        self.gb_fft = self.thr.array((self.dshape), dtype=np.complex64)
        self.result_r= self.thr.array((self.dshape), dtype=np.complex64)
        self.r_ifft = self.thr.array((self.dshape), dtype=np.complex64)
        self.npres_r_ifft_abs = self.npcast(np.zeros(self.dshape),np.float32)
        self.r_ifft_abs = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_r_ifft_abs)
        self.npres_max_r = self.npcast(np.zeros(self.dshape),np.float32)
        self.max_r = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_max_r)
        
        # BUFFERS / GPU ARRAYS FOR AGPF
        self.npres_agpf = self.npcast(np.zeros(self.dshape),np.float32)
        self.res_agpf = cl.Buffer(self.context, self.mflags.ALLOC_HOST_PTR | self.mflags.COPY_HOST_PTR, hostbuf=self.npres_agpf)
        
        # kernel for hanning window
        self.program = cl.Program(self.context, """
        #include <pyopencl-complex.h>
        __kernel void hann(__global cfloat_t *inp, __global const float *win, __global cfloat_t *res)
        {
            int i = get_global_id(0)+(get_global_size(0)*get_global_id(1));
            int j = get_local_id(0)+(get_group_id(0)*get_local_size(0));
            res[i] = cfloat_mulr(inp[i],win[j]);
        }
        __kernel void cpspec(__global cfloat_t *in1, __global const cfloat_t *in2, __global cfloat_t *res)
        {
            int i = get_global_id(0)+(get_global_size(0)*get_global_id(1));
            cfloat_t gagb = cfloat_mul(in1[i],cfloat_conj(in2[i]));
            double gagb_abs = cfloat_abs(gagb)+0.0001;
            res[i] = cfloat_divider(gagb,gagb_abs);
        }
        __kernel void agpf_mult(__global cfloat_t *in1, __global const cfloat_t *in2, __global cfloat_t *res)
        {
            int i = get_global_id(0)+(get_global_size(0)*get_global_id(1));
            res[i] = (cfloat_mul(in1[i],cfloat_conj(in2[i])));
        }
        __kernel void agpf_arg(__global cfloat_t *in1, __global float *res)
        {
            int i = get_global_id(0)+(get_global_size(0)*get_global_id(1));
            res[i] = cfloat_argument(in1[i]);
        }
        __kernel void f_abs(__global cfloat_t *in1, __global float *res)
        {
            int i = get_global_id(0)+(get_global_size(0)*get_global_id(1));
            res[i] = cfloat_abs(in1[i]);
        }
        """).build()

        self.hann = self.program.hann
        self.cpspec = self.program.cpspec
        self.f_abs = self.program.f_abs
        self.agpf_mult = self.program.agpf_mult
        self.agpf_arg = self.program.agpf_arg

    def phase_corr(self,ga,gb):
        self.hann.set_args(ga,self.win_g,self.result_hann_a)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        self.hann.set_args(gb,self.win_g,self.result_hann_b)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        self.FFT(self.fft_buffer_a,self.result_hann_a,0)
        self.FFT(self.fft_buffer_b,self.result_hann_b,0)
        self.cpspec.set_args(self.fft_buffer_a.data,self.fft_buffer_b.data,self.result_r.data)
        cl.enqueue_nd_range_kernel(self.queue,self.cpspec,self.global_wgsize,self.local_wgsize)
        self.FFT(self.r_ifft,self.result_r,1)
        self.f_abs.set_args(self.r_ifft.data,self.r_ifft_abs)
        cl.enqueue_nd_range_kernel(self.queue,self.f_abs,self.global_wgsize,self.local_wgsize)
        cl.enqueue_copy(self.queue, self.npres_r_ifft_abs, self.r_ifft_abs)
        idxs = arg_max_ij(self.npres_r_ifft_abs)
        return idxs
    
    def phase_corr_nj(self,ga,gb):
        self.hann.set_args(ga,self.win_g,self.result_hann_a)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        self.hann.set_args(gb,self.win_g,self.result_hann_b)
        cl.enqueue_nd_range_kernel(self.queue,self.hann,self.global_wgsize,self.local_wgsize)
        self.FFT(self.fft_buffer_a,self.result_hann_a,0)
        self.FFT(self.fft_buffer_b,self.result_hann_b,0)
        self.cpspec.set_args(self.fft_buffer_a.data,self.fft_buffer_b.data,self.result_r.data)
        cl.enqueue_nd_range_kernel(self.queue,self.cpspec,self.global_wgsize,self.local_wgsize)
        self.FFT(self.r_ifft,self.result_r,1)
        self.f_abs.set_args(self.r_ifft.data,self.r_ifft_abs)
        cl.enqueue_nd_range_kernel(self.queue,self.f_abs,self.global_wgsize,self.local_wgsize)
        cl.enqueue_copy(self.queue, self.npres_r_ifft_abs, self.r_ifft_abs)
        idxs = arg_max_ij(self.npres_r_ifft_abs)
        return idxs
    
    def agpf(self,ga,gb):
        self.agpf_mult.set_args(ga,gb,self.res_agpf)
        cl.enqueue_nd_range_kernel(self.queue,self.agpf_mult,self.global_wgsize,self.local_wgsize)
        cl.enqueue_copy(self.queue, self.npres_agpf, self.res_agpf)
        return arr_sum(self.npres_agpf)
        
    
    # Wraps FFT kernel
    def FFT(self,out,data,inv):
        self.cfft(out, data,inverse=inv)
        return
        
if __name__ == '__main__':
    widths=np.arange(2,200,20)
    rates = []
    for idx,n in enumerate(widths):
        # Number of frames to benchmark with and empty lists for framerate / aline rate
        dc=DisplacemenctCalc(np.int16(n))
        
        # Relative path to data in the form of .npy file of format [Z,X,B,T]
        file = 'fig8_1.0x-1.npy'
        data = np.load('C:\\Users\\black\\Google Drive\\PC Workspace\\Senior Design\\axial motion\\2-18-20-oct-motion\\'+file)
        ga = dc.npcast(data[:,:,1,749],dc.dt)
        plot = False
    
        
        gb = dc.npcast(data[:,:,1,740],dc.dt)
        ga_g = cl.Buffer(dc.context, dc.mflags.ALLOC_HOST_PTR | dc.mflags.COPY_HOST_PTR, hostbuf=ga)
        gb_g = cl.Buffer(dc.context, dc.mflags.ALLOC_HOST_PTR | dc.mflags.COPY_HOST_PTR, hostbuf=gb)
        times=[]
        n_frames=10000
        
        for x in range(n_frames):
            t=time.time()
            rc=dc.phase_corr(ga_g,gb_g)
            rc=dc.phase_corr(ga_g,gb_g)
            a=dc.agpf(ga_g,gb_g)
            times.append(time.time()-t)
            
        # Calculate benchmark stats and add to lists
        avginterval = np.mean(times)
        frate=(1/avginterval)*n*2
        rates.append(frate)
            
        print('%d/%d'%(idx+1,len(widths)))
        print(frate)
    plt.plot(widths,np.array(rates)/1000)
    plt.title('A-line rate of GPU Motion Quantification Algorithm')
    plt.xlabel('A-lines per frame')
    plt.ylabel('A-line rate (kHz)')
    plt.legend([''])
    arb = np.array([0,185])
    ln1 = np.array([38,38])
    ln2 = np.array([76,76])
    ln3 = np.array([146,146])
    plt.plot(arb,ln1,'k--',linewidth=1.5,alpha=0.8)
    plt.plot(arb,ln2,'k--',linewidth=1.5,alpha=0.8)
    plt.plot(arb,ln3,'k--',linewidth=1.5,alpha=0.8)
    
