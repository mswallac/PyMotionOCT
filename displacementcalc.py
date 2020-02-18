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

class DispacementCalc():

    def npcast(self,inp,dt):
        return np.asarray(inp).astype(dt)

    def rshp(self,inp,shape):
        return np.reshape(inp,shape,'C')
    
    def set_nlines(self,nlines):
        n = nlines # number of A-lines per frame
        alen = 2048 # length of A-line / # of spec. bins
        self.nlines = n
        self.dshape = (alen,n)
        # Define POCL global / local work group sizes
        self.global_wgsize = (2048,n)
        self.local_wgsize = (256,1)
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
        self.dt = np.float32
        
        # Set apodization window and framesize (# of a-lines)
        self.set_nlines(nlines)
        
        # kernels for hanning window, and interpolation
        self.program = cl.Program(self.context, """
        __kernel void func(__global float *inp, __global float *res)
        {
            return inp;
        }
        """).build()

    

