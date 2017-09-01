#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 15:11:39 2014

@author: worker
"""
import numpy as np
import myutils.image as image
import mygpu.image as gimage
from   mygpu.shared_gpu import sg
#from   myutils.mymath import minpeaks
from   myutils.utils import tprint #,part_idxs #,params2dict #array2str,batch_idxs
from   scipy.fftpack import fftshift # ifftshift
from   numpy.fft import fft2,ifft2
#import mygpu.utils as gutils
from   pycuda import gpuarray
from   pycuda.compiler import SourceModule
import os
import myutils.mrc as mrc
from   myutils.filenames import SaveAble
from   scipy.misc import imread 
import matplotlib.gridspec as gridspec
from   matplotlib import pyplot as plt
#from   micro import Movie
from   drift import Drift
import setrun as sr
from   state import st
#import simplejson
#import params
from   myutils.myplotlib import plot
#import pyprind
#from   scipy import interpolate
import scikits.cuda.fft as cufft
import pycuda.driver as drv
import copy
from   params import pobj as p
#import myutils.scratch as scratch
#import myutils.filenames as fn
#import myutils.cluster as cluster
#import engine_funs as ef
import myutils.utils as utils
import drift

def adjust_ctfs(ctfs,ddefs,rots):
    nctfs = len(ctfs)
    newc  = []
    for k in range(nctfs):
        c = copy.copy(ctfs[k])
        c.rot += rots[k]
        c.du  += ddefs[k]
        c.dv  += ddefs[k]
        newc.append(c)
    return np.array(newc)        

def bcurve_piecewise(sz,sstarts,bstarts,psize,**kwargs):
    # use last bcurve at high frequencies that have not been defined yet
    extend_hf = kwargs.pop('extend_hf',False)
    s1d       = image.psize2s1d(sz,psize)  
    s2_1d     = s1d**2    
    # Construct 1d curve of previous scales    
    nsteps   = sstarts.size
    bcurve1d = np.ones(len(s1d),dtype = 'float32')
    vprev    = 1.0 
    sidxs    = np.int32([0])
    sprev    = sstarts[0]
    for k in range(1,nsteps):
        if sstarts[k-1] > s1d[-1]:
            break
        sidxs   = np.where(np.logical_and(s1d >= sprev, s1d < sstarts[k]))[0]
        if sidxs != []:  
            sprev = sstarts[k]
            bcurve1d[sidxs] = np.exp(-bstarts[k-1]*s2_1d[sidxs]/4.0)*vprev/\
                              np.exp(-bstarts[k-1]*s2_1d[max(sidxs[0]-1,0)]/4)
            vprev = bcurve1d[sidxs[-1]]   
            bcurve1d[s1d >= sstarts[k]] = vprev if extend_hf else 1.0 
    return bcurve1d,s1d,vprev 

def apply_bfact3D(V,bfact,psize,av_fact=1.0):
    N,M,P     = V.shape
    P2        = P//2 + 1
    sg.allocate_blocked()
    bf_ker    = apply_bfact3D_fft_kernel() 
    fftplan   = cufft.Plan((N,M,P), np.float32, np.complex64, batch=1)
    ifftplan  = cufft.Plan((N,M,P),np.complex64,np.float32, batch=1)  
    V_gpu     = gpuarray.to_gpu(V)    
    VF_gpu    = gpuarray.zeros((N,M,P2),np.complex64)
    cufft.fft(V_gpu,VF_gpu,fftplan)  
    apply_bfact3D_fft_ext(bf_ker,VF_gpu,bfact,psize/av_fact,av_fact)   
    cufft.ifft(VF_gpu,V_gpu,ifftplan) 
    VO = V_gpu.get()/np.float32((N*M*P))
    del bf_ker,fftplan,ifftplan
    del V_gpu,VF_gpu
    sg.release()
    return VO

def lp_ang2bfact(lp_ang,psize):
    ''' which bfactor to use to produce filter that cuts frequencies above resolution 
        lp_ang (in angstrom)'''
    return 2*((np.pi*lp_ang/psize)**2)   
    
def apply_ctf_kernel(): 
    path = os.environ['PYTHONPATH'].split(os.pathsep)                           
    return SourceModule('''
        #include "ctf.cu" 
        __device__ float ctfabs(float ctf) {
           return fabs(ctf);
        } // ctfabs
        __device__ float ctffull(float ctf) {
           return ctf;
        } // ctfabs
        __device__ float ctfsign(float ctf) {
           return (ctf>=0)?1.0:-1.0;  
        } // ctfabs
        // define pointer to function
        typedef float (*func)(float x);        
        __device__ func pctffun;        
         __global__ void _apply_ctf(const float2* imins,
                                    float2*       imouts,
                                    const float* dus,
                                    const float* dvs,
                                    const float* rots,
                                    const float* As,
                                    const float psize,
                                    const float cs,
                                    const float lam,                              
                                    const int ylen,
                                    const int ctf_type,
                                    const int lowone,
                                    const int imstep){
            // 0 - abs, 1 - regular, 2 - sign
            int x       = blockIdx.x;
            int d       = blockIdx.y;  // image number 
            int xlen    = gridDim.x;
            int ylen2   = ylen/2 + 1;
            float du    = dus[d];
            float dv    = dvs[d];
            float rot   = rots[d];
            float A     = As[d];
            const float2* imin = &imins[d*imstep];  
            float2* imout      = &imouts[d*xlen*ylen2];
            switch (ctf_type){
                case 0: 
                    pctffun = ctfabs;
                    break;
                case 1:
                    pctffun = ctffull;
                    break;
                case 2:
                    pctffun = ctfsign;
                    break;                    
            } // switch
            for (int y      = threadIdx.x; y < ylen2; y += blockDim.x){  
                int idx     = x*ylen2+y;
                float ctf   = _CTF(x,y,psize,rot,cs,lam,du,dv,A,xlen,ylen,lowone);
                ctf         = pctffun(ctf);                                                   
                // multiply complex by scalar
                float2 a    = imin[idx]; 
                float2 c; c.x = a.x*ctf; c.y = a.y*ctf;                
                imout[idx]  = c; 
            } // for x                                           
        } // main ''', include_dirs=path).get_function('_apply_ctf')       
                

def apply_ctf_ext(kernel,imsfin_gpu,imsfout_gpu,ylen,dus_gpu,dvs_gpu,
                  rots_gpu,As_gpu,psize,cs,lam,**kwargs):  
    ''' Multiply many/one complex fft matrix by many ctf functions'''
    # apply regular ctf by default, 0 - absctf, 1 - regctf, 2 - phaseflip
    N_THREADS    = 32
    ctf_type     = kwargs.pop('ctf_type','regular');
    lowone       = kwargs.pop('lowone',0);
    t = {'abs':0,'regular':1,'flip':2}[ctf_type]    
    xlen,ylen2   = imsfin_gpu.shape[-2:]
    D            = dus_gpu.size
    nims         = int(np.prod(imsfin_gpu.shape[:-2]))
    imstep       = int(0) if nims==1 else xlen*ylen2 
    kernel(imsfin_gpu,imsfout_gpu,dus_gpu,dvs_gpu,rots_gpu,As_gpu,np.float32(psize),np.float32(cs),
           np.float32(lam),np.int32(ylen),np.int32(t),np.int32(lowone),np.int32(imstep),
           block=(N_THREADS,1,1),grid=(int(xlen),int(D),1)) 
           
def gen_ctfs_gpu(ker,ctfs,sz,psize,**kwargs):
    ''' use GPU to generate CTFS by applying CTFS on one valued image '''
    N,M  = sz
    M2   = M//2+1   
    nims = len(ctfs)
    dus_gpu,dvs_gpu,rots_gpu,As_gpu = ctfs2gpu(ctfs) 
    onef_gpu = gpuarray.empty((N,M2),np.complex64)
    onef_gpu.fill(np.complex64(1.0))
    ctfs_gpu = gpuarray.empty((nims,N,M2),np.complex64)
    apply_ctf_ext(ker,onef_gpu,ctfs_gpu,M,dus_gpu,dvs_gpu,rots_gpu,As_gpu,
                  psize,ctfs[0].cs,ctfs[0].lam,**kwargs) 
    return np.float32(np.real(ctfs_gpu.get()))                            
        
def ctfs2gpu(ctfs): 
    dus_gpu  = gpuarray.to_gpu(np.float32([cc.du for cc in ctfs]))
    dvs_gpu  = gpuarray.to_gpu(np.float32([cc.dv for cc in ctfs]))
    rots_gpu = gpuarray.to_gpu(np.float32([cc.rot for cc in ctfs]))
    As_gpu   = gpuarray.to_gpu(np.float32([cc.A for cc in ctfs]))  
    return dus_gpu,dvs_gpu,rots_gpu,As_gpu  

def load_micro(out_dir,mname):
    ctf_dir = sr.ctf_path(out_dir)
    return mrc.load(os.path.join(ctf_dir,mname+'.mrc'))[0]

def center_spectrum(p, win_sz):
    p  = image.fft_half2full2D(p,win_sz)  
    p  = fftshift(p,axes=[-2,-1])
    # This line contains fixed pattern noise, so remove it!
    #hz = win_sz[0]//2
    #p[...,hz,:] = (p[...,hz+1,:] + p[...,hz-1,:])/2
    return p  
    
def frame2ps(im,win_sz,ovlp):
    sz           = im.shape
    xposv, yposv = image.tile2D(sz, win_sz, ovlp)                              
    [xpos, ypos] = image.ndgrid(xposv, yposv)
    xpos,ypos    = xpos.flatten(), ypos.flatten() 
    n_win        = xpos.size
    p = np.zeros(win_sz, 'float32')
    for w in range(n_win):
        imf = fft2(im[xpos[w]:(xpos[w]+win_sz[0]),ypos[w]:(ypos[w]+win_sz[1])])  
        p += np.abs(imf)**2   
    return np.float32(p/np.prod(win_sz))   

def psdiff(im,imctf,c,win_sz,ovlp):
    sz           = im.shape
    xposv, yposv = image.tile2D(sz, win_sz, ovlp)                              
    [xpos, ypos] = image.ndgrid(xposv, yposv)
    xpos,ypos    = xpos.flatten(), ypos.flatten() 
    n_win        = xpos.size
    p = np.zeros(win_sz, 'float32')
    for w in range(n_win):
        # apply ctf on subimage
        lim = im[xpos[w]:(xpos[w]+win_sz[0]), 
                   ypos[w]:(ypos[w]+win_sz[1])]
        cim = np.float32(np.real(ifft2(fft2(lim)*c)))
        # get ctf corrected subimage
        sim = imctf[xpos[w]:(xpos[w]+win_sz[0]), 
                   ypos[w]:(ypos[w]+win_sz[1])]
        # power spectrum of difference
        imf = fft2(sim-cim)  
        p += np.abs(imf)**2   
    return np.float32(p/np.prod(win_sz))  
    
#from profilehooks import profile
#@profile(immediate=True)     
def power_spectrum(mv,drft,win,ovlp,fbatch): 
    ram_mem    = p['ctf']['mem_GB']*1e9    
    M2         = win//2+1
    nframes,sz = mv.nframes(),mv.shape()
    xp,yp      = image.tile2D(sz,(win,win),ovlp)
    xg,yg      = image.ndgrid(xp,yp, )
    xg,yg      = xg.flatten(),yg.flatten() 
    xpos,ypos  = drft.xpos,drft.ypos
    nwin       = xg.size
    dxy        = np.float32(drft.dxy)
    # frame batch
    fs         = np.arange(0,nframes-fbatch+1,max(np.floor((0.5*fbatch)),1),dtype='int32')
    nfbatches  = len(fs)
    # spatial batch
    fft_mem    = 4*np.dtype('float32').itemsize*win*win
    batch      = np.floor(ram_mem/fft_mem) 
    groups     = utils.part_idxs(range(nwin),batch=batch)    
    sg.allocate_blocked()
    #uf_ker     = drift.undrift_sum_frame_kernel()
    ss_ker     = gimage.squared_sum_c64_kernel()
    sn_gpu     = gpuarray.empty((win,M2),np.float32)
    pw         = np.zeros((win,M2),np.float32)
    for k in range(nfbatches):     
        idxs   = np.arange(fs[k],fs[k]+fbatch,dtype=np.int32)
        # keep noise variance constant in averaging equal to that of a single frame      
        mim    = drift.calc_undrifted_sum(mv.frames[idxs[0]:idxs[-1]+1],mv.gain,
                                          xpos,ypos,dxy[idxs])/np.float32(fbatch**0.5)  #np.float32(np.random.normal(size=mv.gain.shape))                                            
        for g in groups:
            glen   = len(g) 
            stack  = image.sub_image2D(mim,xpos=xg[g],ypos=yg[g],win=win)
            s_gpu  = gpuarray.to_gpu(stack)
            sf_gpu = gpuarray.empty((glen,)+(win,M2),np.complex64)    
            fft_p  = cufft.Plan((win,win),np.float32,np.complex64,batch=glen)                     
            cufft.fft(s_gpu,sf_gpu,fft_p) 
            gimage.squared_sum_c64_ext(ss_ker,sf_gpu,sn_gpu)
            pw    += sn_gpu.get()
            del fft_p,sf_gpu,s_gpu
    sg.release()  
    # accumulate variance of all frames
    return nframes*pw/np.float32(win*win*nwin*nfbatches)     
    
def combined_ps(ps,crop_factor,du,dv,rot,A,psize,lam,cs):    
    sz      = np.int32(ps.shape[-2:])   
    crop_sz = np.int32(utils.nextmult(np.ceil(sz*crop_factor),2))
    ps      = image.crop2D(ps,crop_sz)  
    ps      = np.log(ps)
    model   = fftshift(ctf_2d(sz,du,dv,rot,A,psize,lam,cs))
    model   = image.crop2D(model,crop_sz)
    cent    = crop_sz//2
    ps[cent[0]-5:cent[0]+5,cent[1]-5:cent[1]+5] = ps.min()   
    ps      = image.background_remove2D(np.squeeze(ps),sz[-1]/20.0) 
    ps      -= ps.min()
    ps      /= ps.max()
    ps[:,crop_sz[1]//2:] = np.abs(model[:,crop_sz[1]//2:])
    return ps

def gamma2s(gamma,d,lam,cs):
    ''' returns s only for gamma before abberation inversion '''
    a   = 0.5*np.pi*(lam**3)*cs
    b   = np.pi*lam*d  
    return np.sqrt((-b-np.sqrt(b**2+4.0*a*gamma))/(2.0*a))

def s2gamma(s,d,lam,cs): 
    s2      = s**2
    return np.pi*lam*s2*(0.5*cs*(lam**2)*s2 + d)

def calc_zeros(du,dv,A,lam,cs,smax):
    # find ctf zeros until smax
    d   = (du+dv)/2.0
    z,k = [],0 
    a   = 0.5*np.pi*(lam**3)*cs
    b   = np.pi*lam*d    
    k0  = np.floor(2.0*(b**2)/(4.0*a*np.pi))
    while True: 
        if 2*k < k0: 
            gamma = -k*np.pi
            s = np.sqrt((-b - np.sqrt(b**2+4.0*a*gamma))/(2.0*a)) 
        else:
            gamma = -(k-k0)*np.pi
            s = np.sqrt((-b + np.sqrt(b**2-4.0*a*gamma))/(2.0*a))   
        if s > smax or np.isnan(s):
            break;
        z.append(s)
        k += 1
    return np.float32(z)        
    
def estim_def(ps,dus,dvs,rots,psize,cs,lam,A,gPeriod,sRing,mxres):
    gStep  = np.pi/gPeriod        
    sg.allocate_blocked()  
    e_ker  = equiphase_kernel()
    rb_ker = remove_baseline_kernel()
    cr_ker = corr_ripples_kernel()        
    rmeans_gpu = equiphase_ext(e_ker,ps,psize,cs,lam,dus,dvs,rots,A,gStep,mxres)
    fgraph = rmeans_gpu.get()
    remove_baseline_ext(rb_ker,rmeans_gpu,gPeriod)        
    nGamma = rmeans_gpu.shape[-1]
    ws     = np.ones(nGamma,np.float32)
    # remove first rings from the analysis
    ws[:gPeriod*sRing] = 0.0        
    corrs  = corr_ripples_ext(cr_ker,rmeans_gpu,ws,gPeriod,gStep).get()
    rmeans = rmeans_gpu.get()
    del rmeans_gpu
    sg.release()
    [psi,defi,roti] = np.unravel_index(corrs.argmax(),corrs.shape)        
    return dus[defi],dvs[defi],rots[roti], \
            rmeans[psi,defi,roti,:],fgraph[psi,defi,roti,:]
    
def astig_defoc(du,dv,rot):
    du*np.cos()
    
def calc_rad(sz):
    xn,yn = image.cart_coords2D(sz)
    #xn,yn = xn/sz[0],yn/sz[1]
    return fftshift(np.sqrt(xn**2 + yn**2))    

def s_1D(win,psize):
    xn = np.arange(win//2,dtype='float32')
    return np.float32(xn)/(win*psize)

def bfact_1d(sz,bfact,psize):
    x  = np.arange(sz//2,dtype=np.float32)/sz    
    s2 = (x**2.0)/(psize**2)    
    return np.exp(-bfact*s2/4)
    
def bfact_3d(sz,bfact,psize):    
    xn,yn,zn = image.cart_coords3D(sz)
    xn,yn,zn = xn/sz[0],yn/sz[1],zn/sz[2]
    s2 = (xn**2.0+yn**2.0+zn**2.0)/(psize**2)
    return np.fft.ifftshift(np.exp(-bfact*s2/4))
    
def apply_bfact3D_fft_kernel(): 
    return SourceModule(""" 
    __device__ int ifftshift(int x, int N){
        int hdimx = N/2;
        return (x < hdimx) ? x + hdimx : x - hdimx;      
    }                       
    __global__ void __apply_bfact3D(float2* V, 
                                   const float bfact,
                                   const float psize,
                                   const float r2max,
                                   const int zlen){
        int x    = blockIdx.x;
        int y    = blockIdx.y;
        int xlen = gridDim.x;
        int ylen = gridDim.y; 
        int zlenfull = 2*(zlen-1);
        float shx  = (float)ifftshift(x,xlen)/xlen-0.5;
        float shy  = (float)ifftshift(y,ylen)/ylen-0.5;
        float rxy  = shx*shx + shy*shy;
        float ps2  = psize*psize;
        for (int z = threadIdx.x; z < zlen; z += blockDim.x){  
            int pidx = x*ylen*zlen + y*zlen + z;
            float2 val = V[pidx];            
            float shz  = (float)ifftshift(z,zlenfull)/zlenfull - 0.5;        
            float r2   = (rxy + shz*shz);                                                              
            float s2   = r2/ps2;
            float bf   = (r2>r2max)?1.0f:expf(-bfact*s2/4);            
            val.x  *= bf;
            val.y  *= bf; 
            V[pidx] = val;
        } // for y
    } // main """).get_function('__apply_bfact3D') 
    
def apply_bfact3D_fft_ext(kernel,VF_gpu,bfact,psize,lp_fact=1.0):
    ''' lp_fact - factor by which the maximum frequency is larger than cut-off frequency '''
    N_THREADS = 32
    sz = VF_gpu.shape
    r2max = (sz[-1]/(2*lp_fact))**2
    kernel(VF_gpu,np.float32(bfact),np.float32(psize),np.float32(r2max),np.int32(sz[-1]),
           block=(N_THREADS,1,1),grid=(int(sz[0]),int(sz[1]),1))            
    
def apply_bcurve_fft_kernel(): 
    return SourceModule(""" 
    __device__ int ifftshift(int x, int N){
        int hdimx = N/2;
        return (x < hdimx) ? x + hdimx : x - hdimx;      
    }                       
    __global__ void __apply_bfact3D(float2* V, 
                                   const float* bcurve,
                                   const int zlen,
                                   const int blen){
        int x    = blockIdx.x;
        int y    = blockIdx.y;
        int xlen = gridDim.x;
        int ylen = gridDim.y; 
        int zlenfull = 2*(zlen-1);
        float shx  = (float)ifftshift(x,xlen)/xlen-0.5;
        float shy  = (float)ifftshift(y,ylen)/ylen-0.5;
        float rxy  = shx*shx + shy*shy;
        for (int z = threadIdx.x; z < zlen; z += blockDim.x){  
            int pidx = x*ylen*zlen + y*zlen + z;
            float2 val = V[pidx];            
            float shz  = (float)ifftshift(z,zlenfull)/zlenfull-0.5;        
            float r2   = rxy + shz*shz;            
            int   r1   = roundf(sqrtf(r2)*xlen);  
            r1         = min(r1,blen-1);
            float bf   = bcurve[r1];                                                          
            val.x  *= bf;
            val.y  *= bf; 
            V[pidx] = val;
        } // for y
    } // main """).get_function('__apply_bfact3D')    

def apply_bcurve_fft_ext(kernel,VF_gpu,b_gpu):
    ''' lp_fact - factor by which the maximum frequency is larger than cut-off frequency '''
    N_THREADS = 32
    sz    = VF_gpu.shape
    blen  = b_gpu.shape[0]
    kernel(VF_gpu,b_gpu,np.int32(sz[-1]),np.int32(blen),
           block=(N_THREADS,1,1),grid=(int(sz[0]),int(sz[1]),1))            
           
def apply_bfact3D_gpu(V,bfact,psize,av_fact=1.0):
    sg.allocate_blocked()
    bf_ker    = apply_bfact3D_fft_kernel() 
    N,M,P     = V.shape
    fftplan   = cufft.Plan((N,M,P), np.float32, np.complex64, batch=1)
    ifftplan  = cufft.Plan((N,M,P), np.complex64, np.float32, batch=1)        
    P2        = P//2 + 1
    V_gpu     = gpuarray.to_gpu(V)
    VF_gpu    = gpuarray.zeros((N,M,P2),np.complex64)
    # transform volume
    cufft.fft(V_gpu,VF_gpu,fftplan) 
    # apply bfactor
    apply_bfact3D_fft_ext(bf_ker,VF_gpu,bfact,psize/av_fact,av_fact)
    # transform back
    cufft.ifft(VF_gpu,V_gpu,ifftplan)  
    Vb = V_gpu.get()
    del bf_ker,fftplan,ifftplan
    del V_gpu,VF_gpu    
    sg.release()
    return Vb 
    
def ctf_1d(s,d,A,lam,cs):
    gamma   = s2gamma(s,d,lam,cs)
    #fall    = np.exp(-bfact*s2/4)
    return (np.sqrt(1.0-A**2)*np.sin(gamma) + A*np.cos(gamma))     

def ctf_2d(sz,du,dv,rot,A,psize,lam,cs): 
    x,y   = image.cart_coords2D(sz)
    x,y   = x/sz[0],y/sz[1]
    r     = np.sqrt(x**2+y**2)
    s     = r/psize        
    ang   = -rot
    r[r==0] = 1.0 # avoid zero divided by zero
    xr    = (x*np.cos(ang) + y*np.sin(ang))/r
    yr    = (x*(-np.sin(ang)) + y*np.cos(ang))/r
    d     = du*(xr**2) + dv*(yr**2)    
    c     = ctf_1d(s,d,A,lam,cs)
    return np.float32(np.fft.ifftshift(c))    
    
def bfact_2d(sz,bfact,psize):    
    xn,yn = image.cart_coords2D(sz)
    xn,yn = xn/sz[0],yn/sz[1]
    s2 = (xn**2+yn**2)/(psize**2)
    return np.fft.ifftshift(np.exp(-bfact*s2/4))    
    
def apply_bfact2d(im,bfact,psize):   
    b = bfact_2d(im.shape,bfact,psize)
    return np.float32(np.real(np.fft.ifft2(np.fft.fft2(im)/b)))
    
def apply_bfact3d(V,bfact,psize):   
    b = bfact_3d(V.shape,bfact,psize)
    return np.float32(np.real(np.fft.ifftn(np.fft.fftn(V)/b)))                  

def plot_t0t1(sz,psize,t0,t1):
    s = s_1D(sz,psize)
    plot(s,s2t0t1(s,t0,t1))

def s2t0t1(s,t0,t1):
    return (np.exp(t0/(s+t1)**2 - t0/(t1**2)))    
    
def apply_t0t1(im,t0,t1,psize):   
    sz    = im.shape
    xn,yn = image.cart_coords2D(sz)
    xn,yn = xn/sz[0],yn/sz[1]
    s = np.sqrt(xn**2+yn**2)/psize
    b = np.fft.ifftshift(s2t0t1(s,t0,t1))
    return np.float32(np.real(np.fft.ifft2(np.fft.fft2(im)/b)))        
    
def res2rad(res,win,psize):
    return int(np.round(psize*win/res))               

def kev2lam(vkev):                                                  
    m0 = 9.109e-31 # [kg]                  (Electron mass      )
    e  = 1.602e-19 # [Cb]                  (Electron charge    )
    c  = 2.998e+8  # [m/sec]               (Speed of light     )
    h  = 6.626e-34 # [kg*m^2/sec]          (Planck's constant  )
    v  = 1e3*vkev; #[Volt]          (Acceleraton voltage)
    lam=h/np.sqrt(2.0*m0*e*v*(1+e*v/(2.0*m0*c**2.0))); # [m]
    lam_ang = lam*1e10
    return lam_ang       
    
def aber_s(d,A,lam,cs):
    ''' spatial frequency that precedes phase inversion due to spherical aberrations '''
    #d   = (du+dv)/2.0
    a   = 0.5*np.pi*(lam**3)*cs
    b   = np.pi*lam*d    
    ka  = np.floor((b**2)/(4.0*a*np.pi))-1
    return np.sqrt((-b - np.sqrt(b**2-4.0*a*ka*np.pi))/(2.0*a)) 
                                 
def corr_ripples_kernel():
    ''' Correlate phase responses with sine function '''
    return SourceModule('''
            #define max(x,y) ((x>y)?x:y)
            __global__ void __corrrip(const float* rmeans,
                                      float* corrs,
                                      const float* ws,
                                      const int nGamma,
                                      const int period,
                                      const float gStep) {
            // rmeans  - nPs x ndef x nrots x nGamma
            // corrs   - nPs x ndef x nrots 
            //////////////////////////////        
            int nRots    = blockDim.x;
            int nDef     = gridDim.x;
            //int nPs    = gridDim.y;
            int rotIdx   = threadIdx.x;   
            int defIdx   = blockIdx.x;
            int psIdx    = blockIdx.y; 
            int myidx    = psIdx*nDef*nRots+defIdx*nRots+rotIdx;
            const float* rmean = &rmeans[myidx*nGamma];
            // find mean of one cycle
            float mn = 0;
            for (int g=0; g<period; g++)
                mn += fabs(sinf(g*gStep));
            mn /= period;
            float corr   = 0.0;
            //float rnorm  = 0.0;
            //float rmn    = 0.0;
            float lcorr  = 0.0;
            for(int g=0; g<nGamma; g++){    
                int gp      = g-period;
                float gamma = g*gStep;
                float w     = ws[g];
                float sg    = fabs(sinf(gamma))-mn;
                float rg    = rmean[g];
                //rmn        += rg;
                //rnorm      += rg*rg; // fabs(rg);
                //float arg   = fabs(rg);
                //if (arg > rmax) rmax=arg;
                // accumulate local ripple correlation
                lcorr      += w*rg*sg*sg;                
                if ((g % period == 0) && (gp >= 0)){
                    // reset ripple normalization
                    //corr += lcorr/rnorm;
                    //rnorm = rnorm - rmn*rmn/period;
                    //corr += (lcorr/sqrtf(rnorm)); 
                    //corr += lcorr/cbrtf(rnorm);
                    // sqrtf to make it robust to outliers
                    corr += sqrtf(sqrtf(max(lcorr,0.0f)));
                    //corr += cqrtf(max(lcorr,0.0f));
                    //corr += (lcorr); 
                    //rnorm = 0.0;
                    lcorr = 0.0;
                    //rmn   = 0.0;
                } // if g                    
            }//for g            
            corrs[myidx] = corr;
        } // main ''').get_function('__corrrip')  
        
def corr_ripples_ext(kernel,rmeans_gpu,ws,period,gStep):
    nPs,nDef,nRots,nGamma = rmeans_gpu.shape
    corrs_gpu = gpuarray.empty((nPs,nDef,nRots),np.float32)    
    ws_gpu    = gpuarray.to_gpu(ws)
    kernel(rmeans_gpu,corrs_gpu,ws_gpu,np.int32(nGamma),np.int32(period),
           np.float32(gStep),block=(int(nRots),1,1),grid=(int(nDef),int(nPs),1))        
    del ws_gpu
    return corrs_gpu
                                     
def remove_baseline_kernel():
    return SourceModule('''
            __device__ float calc_gain(float yp,float yn,int rp,int rn) {
                return (yn-yp)/(float)(rn-rp);                      
            }//calc_gain
            __device__ float calc_offs(float yp,float yn,int rp,int rn) {
                return (yp*(float)rn-yn*(float)rp)/(float)(rn-rp);                      
            }//calc_gain            
            __global__ void __normrip(float* rmeans,
                                      const int nGamma,
                                      const int period) {
            // rmeans  - nPs x ndef x nrots x nGamma
            //////////////////////////////        
            int nRots    = blockDim.x;
            int nDef     = gridDim.x;
            //int nPs    = gridDim.y;
            int rotIdx   = threadIdx.x;   
            int defIdx   = blockIdx.x;
            int psIdx    = blockIdx.y; 
            float* rmean = &rmeans[psIdx*nDef*nRots*nGamma+defIdx*nRots*nGamma+rotIdx*nGamma];
            // remove baseline
            float gain   = 0.0f;
            float offs   = 0.0f;                
            for(int g=0; g<nGamma; g++){  
                int gn = g+period;
                if ((g % period == 0) && (gn<nGamma)){
                    float yp = rmean[g];
                    float yn = rmean[gn];                        
                    gain = calc_gain(yp,yn,g,gn);
                    offs = calc_offs(yp,yn,g,gn);
                } // if g    
                rmean[g] = rmean[g]-gain*g-offs;
            } // for g                                    
        } // main ''').get_function('__normrip')    

def remove_baseline_ext(kernel,rmeans_gpu,period):
    ''' Removes baseline from equiphase ripples '''
    nPs,nDef,nRots,nGamma = rmeans_gpu.shape
    kernel(rmeans_gpu,np.int32(nGamma),np.int32(period),
           block=(int(nRots),1,1), grid=(int(nDef),int(nPs),1))
                                                                
def equiphase_kernel(): 
    path = os.environ['PYTHONPATH'].split(os.pathsep)                           
    return SourceModule('''
            #include "ctf.cu"
            #include "math.h"
            #define N_THREADS 32
            #define MAX_GAM 2048
            __global__ void __equiphase(const float* pss,
                                     const float  psize,
                                     const float  cs,
                                     const float  lam,
                                     const float* dus,
                                     const float* dvs,
                                     const float* rots,
                                     const float A,
                                     const int xlen,
                                     const int nGamma,
                                     const float smax,
                                     const float gStep,
                                     float* rmeans) {
            // rmeans  - nPs x ndef x nrots x nGamma
            // ps      - nPs x xlen x ylen2  
            //////////////////////////////        
            int nRots  = gridDim.x;
            int nDef   = gridDim.y;
            //int nPs    = gridDim.z;
            int rotIdx = blockIdx.x;   
            int defIdx = blockIdx.y;
            int psIdx  = blockIdx.z; 
            int ylen2  = xlen/2+1;             
            float du   = dus[defIdx];
            float dv   = dvs[defIdx];   
            float rot  = rots[rotIdx];                                 
            const float* ps = &pss[psIdx*xlen*ylen2];  
            float* rmean = &rmeans[psIdx*nDef*nRots*nGamma+defIdx*nRots*nGamma+rotIdx*nGamma];
            float gStepi = 1.0f/gStep;             

            __shared__ float sh_sumgam[MAX_GAM];            
            __shared__ int sh_count[MAX_GAM];            

            // init shared mem
            for(int k=threadIdx.x; k<nGamma; k+=blockDim.x){
                sh_sumgam[k] = 0.0f; 
                sh_count[k]  = 0;                
            }
            __syncthreads();

            for(int x=0; x<xlen; x++){
                for(int y=threadIdx.x; y<ylen2; y+=blockDim.x){
                    float s     = _S(x,y,psize,xlen,xlen);
                    if (s<smax){
                        float lps   = ps[x*ylen2+y]; // /(2*M_PI*s);
                        float gamma = -_GAMMA(x,y,psize,rot,cs,lam,du,dv,A,xlen,xlen);   
                        int gidx    = roundf(gamma*gStepi);
                        //gidx = (gidx<nGamma)?gidx:(nGamma-1);
                        if (gidx < nGamma){
                            // accumulate power spectrum at gammas
                            atomicAdd(&sh_sumgam[gidx],lps);
                            atomicAdd(&sh_count[gidx],1);
                        }
                    } // if s
                } // for y
            } // for x           
            __syncthreads();
            // copy the result to global memory
            for(int k=threadIdx.x; k<nGamma; k+=blockDim.x)
                rmean[k] = sh_sumgam[k]/sh_count[k];
        } // main ''',include_dirs=path).get_function('__equiphase')             
         
def equiphase_ext(kernel,ps,psize,cs,lam,dus,dvs,rots,A,gStep,mxres): 
    ''' Accumulates power spectrum energy for each gamma/phase shift. '''
    assert(ps.dtype=='float32')
    N_THREADS  = 32
    MAX_GAM    = 2048 # can be increased even more 
    xlen,ylen2 = ps.shape[-2:]
    nps        = int(np.prod(ps.shape[:-2]))
    nrots,ndef = rots.size,dus.size
    dmax       = max(dus.max(),dvs.max())
    smax       = min(aber_s(dmax,A,lam,cs),(1.0/mxres))
    # calc maximum phase angle
    gmax       = s2gamma(smax,dmax,lam,cs) #np.pi*lam*s2*(0.5*cs*(lam**2)*s2 + dmax)
    nGamma     = int(np.ceil(-gmax/gStep))    
    assert(nGamma < MAX_GAM)
    rmeans_gpu = gpuarray.zeros((nps,ndef,nrots,nGamma),np.float32)
    ps_gpu     = gpuarray.to_gpu(ps)        
    dus_gpu    = gpuarray.to_gpu(dus) 
    dvs_gpu    = gpuarray.to_gpu(dvs) 
    rots_gpu   = gpuarray.to_gpu(rots)    
    kernel(ps_gpu,np.float32(psize),np.float32(cs),np.float32(lam),
           dus_gpu,dvs_gpu,rots_gpu,np.float32(A),np.int32(xlen),
           np.int32(nGamma),np.float32(smax),np.float32(gStep),rmeans_gpu,
           grid=(int(nrots),int(ndef),int(nps)),block=(N_THREADS,1,1))
    del rots_gpu,ps_gpu,dus_gpu,dvs_gpu     
    return rmeans_gpu
    
def estim_noise_var(rmean,du,dv,cs,lam,gPeriod):   #!!  
    gStep   = np.pi/gPeriod        
    #sg.allocate_blocked()  
    #e_ker  = equiphase_kernel()
    #rmean  = equiphase_ext(e_ker,ps,psize,cs,lam,np.float32([du]),
    #                       np.float32([dv]),np.float32([rot]),
    #                       A,gStep,mxres)[0,0,0].get() 
    #sg.release()
    # extract zeros
    zr     = rmean[::gPeriod]  
    # corresponding spatial frequency
    s      = gamma2s(-np.arange(0,rmean.size,gPeriod)*gStep,0.5*(du+dv),lam,cs)  
    # noise variance (assumed to be white)
    nv     = np.sum(zr[1:]/s[1:])/np.sum(1.0/s[1:])
    return nv   
#%%       
                                             
class CTF(SaveAble):    
    
    def __init__(self, mname,cs=None,vol=None,*args, **kwargs):
        super(CTF, self).__init__(*args,base_name=mname,**kwargs)
        self.initialized = False
        self.init(cs,vol)
    
    def init(self, cs, vol):
        self.cs         = cs
        self.lam        = kev2lam(vol) if vol is not None else None
        self.du         = None # defocus in Angstroms not set yet
        self.dv         = None # 
        self.rot        = None # astigmatism orientation angle not set yet
        self.A          = None # phase contrast (shift)
        self.win_sz     = None
        self.graph      = None 
        self.fgraph     = None 
        
    def __ps_filename(self, basedir): 
        return self.fullname(basedir) + '_ps.mrc'
    def __binned_filename(self,basedir):
        return self.fullname(basedir) + '_binned.png'
    def flipped_filename(self, basedir):
        return self.fullname(basedir) # + '.mrc'        
    def __report_filename(self, basedir):
        return self.fullname(basedir) + '_report.png'                
                    
    def __repr__(self):
        if self.initialized:
            return "CTF object, du,dv=[%.1f,%.1f], astig rot %.1f, A %.2f" % (self.du,self.dv,self.rot,self.A)      
        else:
            return "CTF object not initialized, either run init(...) followed by crunch(...), or load(...) from file"        

    def load_ps(self, base_dir):
        return mrc.load(self.__ps_filename(base_dir))
                                
    def calc_zeros(self,smax):         
        return calc_zeros(self.du,self.dv,self.A,self.lam,self.cs,smax)        
        
    def phase_flip(self,im,psize): 
        c  = self.ctf_2d(im.shape,psize)
        ch = image.fft_full2half2D(np.sign(c))
        sg.allocate_blocked()
        im = gimage.apply_mask_fft2D(im,ch)
        sg.release()
        return im         
        
    def phase_flip_fft(self,imf,psizef): 
        c  = self.ctf_2d((imf.shape[0],)*2,psizef)
        ch = image.fft_full2half2D(np.sign(c))
        return imf*ch            

    def apply_abs_fft(self,imf,psizef): 
        c  = self.ctf_2d((imf.shape[0],)*2,psizef)
        ch = image.fft_full2half2D(np.abs(c))
        return imf*ch            
                
    def save_imctf(self,imctf,base_dir):
        ff_name = self.flipped_filename(base_dir) + '.mrc'
        #tprint('Saving phase flipped micrograph %s ...' % ff_name)
        mrc.save(imctf,ff_name,ifExists='overwrite')

    def save_imctf_fft(self,imf,psize,nvar,base_dir):
        ff_name = self.flipped_filename(base_dir) + '.npy'
        hdr = {'psize':psize,'noise variance':nvar}
        np.save(ff_name,(imf,hdr))
        #tprint('Saving phase flipped micrograph %s ...' % ff_name)
        #mrc.save(imctf,ff_name,ifExists='overwrite')
        
    def load_imctf_fft(self,base_dir):
        return np.load(self.flipped_filename(base_dir)+'.npy')        

    def load_im_ctf2(self, base_dir):
        return mrc.load(self.flipped_filename(base_dir)+'.mrc')[0]        
        
    def load_frames_mean(self, base_dir):
        return imread(self.__binned_filename(base_dir))             
                    
    def normalized_graph(self,psize):
        fbfact = p['ctf']['fbfact']    
        win    = p['ctf']['win']        
        s2r    = psize*win;
        graph  = np.float32(self.graph)
        lg     = len(graph)
        ug     = np.zeros(lg,np.float32)
        s      = s_1D(win,psize)
        z      = self.calc_zeros(s[-1])
        z      = z[z<s[lg-1]]
        for k in range(1,len(z)):
            rp = int(np.round(z[k]*s2r))
            if rp == lg-1:
                break
            if k == len(z)-1:
                rn = lg-1
            else:
                rn = min(int(np.round(z[k+1]*s2r)),lg-1)
            yp = graph[rp]
            yn = graph[rn]
            g  = (yn-yp)/(rn-rp)
            o  = (yp*rn-yn*rp)/(rn-rp)    
            ug[rp:rn+1] = graph[rp:rn+1]-(g*np.arange(rp,rn+1)+o)
        s2 = s[:lg]**2
        ug = ug*np.exp(fbfact*s2/4) 
        ug = ug/max(ug)  
        return ug          
        
    def report_result(self,base_dir,imf,pw,dxy,psize,crop_res):
        #%%
        #from matplotlib.ticker import MaxNLocator,FormatStrFormatter      
        # prepare the example micrograph
        gPeriod = p['ctf']['period']
        cs      = p['ctf']['cs']
        lam     = kev2lam(p['ctf']['volkev'])
        d       = 0.5*(self.du+self.dv)
        
        gStep  = np.pi/gPeriod                
        win    = self.win_sz[0]        
        tprint('Generating CTF figure ...')
        # bin micro 
        bnsz   = int(1024)
        imfb   = image.crop2D_fft_half(imf,(bnsz,bnsz))
        imfb   = image.fft_half2full2D(imfb,(bnsz,bnsz))
        im     = np.real(np.fft.ifft2(imfb))
        #im     = image.bin2D_fft(im,dst_sz=[1024,1024])
        gs     = gridspec.GridSpec(7,8)
        plt.ioff() 
        ax1_1  = plt.subplot(gs[:4,:4])
        ax1_2  = plt.subplot(gs[:4,4:8])
        ax4    = plt.subplot(gs[4,:])
        ax2    = plt.subplot(gs[5:7,:]) 
        cps    = self.combined_ps(pw,psize,2.0*psize/crop_res)
        ax1_1.imshow(cps, cmap=plt.cm.gray, interpolation='bilinear') 
        ax1_1.get_xaxis().set_visible(False)
        ax1_1.get_yaxis().set_visible(False)      
        ax1_1.set_aspect('equal')
        for res in [5.6,4.7,3.8,3.3,3.0,2.75,2.5,2.3]:
            rad = cps.shape[0]//2-res2rad(res,win,psize) 
            ax1_1.annotate('%.1f'%res, xy=(cps.shape[0]//1.75,rad), 
                         xytext=(cps.shape[0]//1.75,rad), 
                         color='blue',weight='bold',fontsize=12)
        ax1_1.set_title("d=%d,%d ASTIG=%d\nwin=%d, A=%.2f, psize=%.3f" % \
                        (self.du,self.dv,np.abs(self.dv-self.du),win,self.A,psize))   
        ax1_2.imshow(im,cmap=plt.cm.gray, interpolation='bilinear')      
        ax1_2.get_xaxis().set_visible(False)        
        ax1_2.get_yaxis().set_visible(False)      
        ax1_2.set_title("Flipped, noise var %.2f" % self.nv)              
        graph   = np.float32(self.graph)
        nGamma  = len(self.graph)
        gammas  = -gStep*np.arange(nGamma)
        res     = 1.0/gamma2s(gammas,d,lam,cs)  
        ylims   = [graph[3*gPeriod:].min(),graph[3*gPeriod:].max()]
        # plot period ticks
        for g in np.arange(0,nGamma,gPeriod):  
            ax4.plot([g,g],np.array(ylims),color='red',linewidth=0.5)
        
        ax4.plot(graph, color='green')
        ticks    = np.linspace(0,nGamma-1,20,dtype='int32')
        ax4.set_xticks(ticks)  
        ax4.set_xlim(ticks[[0,-1]])
        ax4.set_ylim(ylims)
        tlabels = ['%.2f' % r for r in res[ticks]] #res in image.rad2res(np.int32(rall)[ticks],win,psize)]        
        ax4.set_xticklabels(tlabels)
        ax4.set_xlabel('resolution,A')
        ax4.xaxis.set_label_coords(0.5,0.2)                
        
        #ax4.plot(res[[residx,residx]],np.array([css.min(), css.max()]),color='red')
        #ax4.invert_xaxis()
        ax4.get_yaxis().set_visible(False)
        #rdxy = drift.regularize_drifts(dxy,1.0)
        ax2.hold(True)        
        ax2.plot(dxy[:,:,0].mean(axis=1),label='x drift')        
        ax2.plot(dxy[:,:,1].mean(axis=1),label='y drift')        
        #ax2.plot(rdxy[:,:,0].mean(axis=1),label='reg x drift')        
        #ax2.plot(rdxy[:,:,1].mean(axis=1),label='reg y drift')        
        ax2.set_xlim([0,dxy.shape[0]-1])
        #ax2.get_yaxis().set_visible(False)
        #ax2.set_title("Mean drift in A") 
        ax2.set_xlabel('# frame')
        ax2.set_ylabel('drift, A')
        ax2.xaxis.set_label_coords(0.5,0.10)        
        ax2.yaxis.set_label_coords(0.025,0.5) 
        ax2.yaxis.grid(True)        
        ax2.legend()
        #%%
        #ax4.set_title("Model fit (zoom in)")
        plt.gcf().set_figheight(12)
        plt.gcf().set_figwidth(14.0)
        fig = plt.gcf()
        fig.subplots_adjust(wspace=.1,hspace=0.2,left=0.03,right=0.98,bottom=0.05,top=0.93)
        fig.savefig(self.__report_filename(base_dir))
        
    def get_mean_def(self):
        return 0.5*(self.du+self.dv)
                                                            
    def ctf_2d(self,sz,psize):
        return ctf_2d(sz,self.du,self.dv,self.rot,self.A,psize,self.lam,self.cs)      

    def apply_ctf(self,im,psize,**kwargs): 
        ctype = kwargs.pop('ctf_type','regular')
        t = {'abs':0,'regular':1,'flip':2}[ctype]                
        ct = self.ctf_2d(im.shape,psize)    
        if t == 0: ct = np.abs(ct)
        if t == 2: ct = np.sign(ct)
        return np.float32(np.real(ifft2(fft2(im)*ct)))                 
                      
    def ctf_1d(self,sz,psize):
        s   = np.arange(sz[0]//2 + 1)/(psize*sz[0])
        return ctf_1d(s, self.d, self.A, self.lam, self.cs)        
                             
    def combined_ps(self,pw,psize,crop_factor=0.5):
        return combined_ps(pw,crop_factor,self.du,self.dv,self.rot,
                           self.A,psize,self.lam,self.cs)  #self.load_ps(base_dir)
                           
    def refine_defoc(self,partmn,msks,prots,drange,ndefs,psize):
        # look for better defocuses
        nstacks = partmn.shape[0]
        # new defocus shift
        newdd   = np.zeros((nstacks,),np.float32)
        sz      = partmn.shape[-2:]
        N,M     = sz    
        M2      = M//2+1    
        ctf_ker = apply_ctf_kernel()
        # prerp ctf params
        parts_gpu    = gpuarray.to_gpu(partmn)
        fparts_gpu   = gpuarray.zeros((ndefs,)+(N,M2),np.complex64)
        fcparts_gpu  = gpuarray.zeros((ndefs,)+(N,M2),np.complex64)
        cparts_gpu   = gpuarray.zeros((ndefs,)+sz,np.float32)
        As_gpu       = gpuarray.to_gpu(self.A*np.ones((ndefs,1),np.float32))
        stream       = drv.Stream()   
        fft2_plan_s  = cufft.Plan(sz,np.float32,np.complex64,stream=stream,batch=nstacks)    
        ifft2_plan_d = cufft.Plan(sz,np.complex64,np.float32,stream=stream,batch=ndefs)
        cufft.fft(parts_gpu,fparts_gpu,fft2_plan_s) 
        for s in range(nstacks):
            ddefs    = np.float32(np.linspace(-drange,drange,ndefs))        
            dus_gpu  = gpuarray.to_gpu(ddefs+self.du)
            dvs_gpu  = gpuarray.to_gpu(ddefs+self.dv)
            rots_gpu = gpuarray.to_gpu((self.rot+prots)*np.ones((ndefs,1),np.float32))    
            # 0 - absctf, 1 - regctf, 2 - phaseflip
            apply_ctf_ext(ctf_ker,fparts_gpu[s],fcparts_gpu,sz[1],dus_gpu,dvs_gpu,
                          rots_gpu,As_gpu,psize,self.cs,self.lam,ctf_type='abs')                             
            cufft.ifft(fcparts_gpu,cparts_gpu,ifft2_plan_d) 
            cparts   = cparts_gpu.get()/(M*N)            
            # normalize stacks to unit norm
            cparts   = cparts/np.sqrt(np.sum(cparts**2,axis=(-2,-1)))[:,None,None]
            # collect data from masked regions
            cparts   = image.stack2vecs(cparts,msks[s])            
            # l2 based cost function
            #print np.sum(cparts**2,axis=1).max(),np.sum(cparts**2,axis=1).min()
            midx     = np.sum(cparts**2,axis=1).argmax() 
            newdd[s] = ddefs[midx]
        return newdd
        
    def crunch(self,ps,nframes,psize): 
        win     = p['ctf']['win']
        ndefs   = p['ctf']['ndefs']
        franges = np.float32(p['ctf']['fine_range'])
        nsteps  = np.int32(p['ctf']['nsteps'])
        drange  = np.float32(p['ctf']['drange'])
        psize   = st.get_psize() 
        nrots   = p['ctf']['nrots']
        mxres   = p['ctf']['mxres']
        sRing   = p['ctf']['startRing']
        gPeriod = p['ctf']['period']
        cs      = p['ctf']['cs']
        vkev    = p['ctf']['volkev'] 
        A       = p['ctf']['A'] 
        coarseres = p['ctf']['coarseres']        
        lam     = kev2lam(vkev)
        rots    = np.float32(np.arange(nrots)*np.pi/(2*nrots))
        gStep   = np.pi/gPeriod                
        #%%
        
        tprint("Estimating CTF params ...")
        #defs              = np.float32(np.linspace(drange[0],drange[1],ndefs)) 
        #du,dv,rot,graph,_ = estim_def(ps,defs,defs,np.float32([0]),psize,cs,lam,A,gPeriod,sRing,mxres)
        # use first rings to coarsely find defocus 
        defs      = np.float32(np.linspace(drange[0],drange[1],ndefs)) 
        du,dv,rot,graph,_ = estim_def(ps,defs,defs,np.float32([0]),psize,cs,lam,A,gPeriod,sRing,coarseres)
        tprint("Coarse fitting I, range [%.1f,%.1f]Å, nsteps %d, found def %.1f, rot %.2f, mxres %.2fÅ" % \
                (drange[0],drange[1],ndefs,du,rot,coarseres))                                                                                                                                                                                                         
        # find a finer defocus still with no astigmatism search 
        defs      = np.float32(np.linspace(1.1*du,0.9*du,ndefs)) 
        du,dv,rot,graph,_ = estim_def(ps,defs,defs,np.float32([0]),psize,cs,lam,A,gPeriod,sRing,mxres)
        tprint("Coarse fitting II, range [%.1f,%.1f]Å, nsteps %d, found def %.1f, rot %.2f, mxres %.2fÅ" % \
                (defs[0],defs[-1],ndefs,du,rot,mxres))                                                                                                                                                                                                                 
        # look for astigmatism
        for k in range(len(nsteps)):
            defsu       = np.float32(np.linspace(du-franges[k],du+franges[k],nsteps[k])) 
            defsv       = np.float32(np.linspace(dv-franges[k],dv+franges[k],nsteps[k])) 
            dus,dvs     = image.ndgrid(defsu,defsv)
            dus,dvs     = dus.flatten(),dvs.flatten()
            du,dv,rot,graph,fgraph = estim_def(ps,dus,dvs,rots,psize,cs,lam,A,gPeriod,sRing,mxres)
            mxs         = gamma2s(-graph.size*gStep,0.5*(du+dv),lam,cs)            
            tprint("Fine fitting, range %.1fÅ, step %.1fÅ, def[%.1f,%.1f], rot %.2f, mxres %.2fÅ" % \
                    (franges[k],2.0*franges[k]/nsteps[k],du,dv,rot,1.0/mxs)) 
                                                                                                                                                                                        
        self.nv          = estim_noise_var(fgraph,du,dv,cs,lam,gPeriod) 
        self.nframes     = nframes                                                      
        self.du          = float(du)  
        self.dv          = float(dv)  
        self.rot         = float(rot) 
        self.A           = float(A)  
        self.initialized = True    
        self.win_sz      = (float(win),)*2
        self.graph       = graph.tolist() 
        self.fgraph      = fgraph.tolist()
                          
def load_ctf(in_dir,micro):
    c = CTF(micro,0,0)
    c.load_json(in_dir)
    return c

def load_ctfs(ctf_path,micros):
    cs        = p['ctf']['cs']
    vol       = p['ctf']['volkev']
    ctfs      = list()
    umicros,iidxs = np.unique(micros,return_inverse=True)
    for micro in umicros: 
        c     = CTF(micro,cs,vol)
        c.load_json(ctf_path)
        ctfs.append(c)
    return [copy.copy(ctfs[idx]) for idx in iidxs]   
            
def unctf_micro(mv): 
    #%%
    ctf_dir     = p.get_path('ctf')
    drift_dir   = p.get_path('drift')
    cs          = p['ctf']['cs']
    vkev        = p['ctf']['volkev']
    win         = p['ctf']['win']
    crop_res    = p['ctf']['crop_res']
    ovlp        = p['ctf']['ovlp']
    fbatch      = p['ctf']['batch']  
    
    ctf_dir     = p.get_path('ctf')
    psize       = mv.psize
    m           = mv.micro
    drft        = Drift(m)      
    drft.load_json(drift_dir) 

    ps = power_spectrum(mv,drft,win,ovlp,fbatch)   
    # remove extreme power spectrum values
    md = np.median(ps)*2.0
    ps[ps>md]=md      
    
    pw = center_spectrum(ps,(win,win))    
        
    c = CTF(m)
    c.init(cs,vkev) 
    c.crunch(ps,mv.nframes(),psize)     
    #c.save_pw(pw,ctf_dir)

    # save phase flipped
    imd     = drft.load_undrift(drift_dir)
    szradix = st.get_micro_sz_radix()
    imf,bn  = gimage.fft2sz(imd,szradix)
    psizef  = bn*psize
    # save tranformed micrograph
    
    imf     = c.phase_flip_fft(imf,psizef)
    c.save_imctf_fft(imf,psizef,c.nv,ctf_dir)
    
    #imctf   = c.phase_flip(imd,psize)
    #c.save_imctf(imctf, ctf_dir)    
    # save resulting figure
    c.report_result(ctf_dir,imf,pw,np.float32(drft.dxy)*psize,psize,crop_res)  
    # save found CTF params
    c.save_json(ctf_dir)
    
############ GARBAGE #################################### 
#    from profilehooks import profile
#    @profile(immediate=True)     
#def phase_flip_stacks(stacks,ctfs,psize): 
#    # apply ctf on stacks np.dtype(np.float32).itemsize
#    # generate same ctf for all frames
#    nstacks,nframes = stacks.shape[:2]
#    gpumem  = p['gpu']['mem_GB']*(2**30) 
#    batch   = gpumem/(6*stacks[0].nbytes)
#    sidxs   = part_idxs(range(nstacks),batch=batch)  
#    cstacks = np.zeros(stacks.shape,np.float32)
#    for idxs in sidxs:
#        cstacks[idxs] = apply_ctfs_gpu(stacks[idxs],ctfs[idxs],psize,ctf_type='flip')
#    return cstacks   
    #%%
    #imf_gpu = gpuarray.to_gpu(imf)
    #im = gimage.ifft2_batch2D(imf_gpu,(scalesz,)*2).get()

    ##################################################
    #lam = kev2lam(vkev)
    #from scipy.interpolate import interp1d
    
    #c.load_json(ctf_dir)
#    ps = power_spectrum(mv,drft,win,ovlp,mv.nframes())
    #ps = power_spectrum(mv,drft,win,ovlp,fbatch)               
    
    #zr[0]  = zr[1] # zero frequency is always problem                       
    # calc spatial frequency at zeros                          
    # extract monotonic subsequence    
    #plot(rmean)
    #%%
    
    # sample zeros
    #z      = calc_zeros(c.du,c.dv,c.A,c.lam,c.cs,s[-1])  
    # resample at zeros
    #zr     = interp1d(s,rmean)(z)      
    # obtain noise graph
    #n      = interp1d(z,zr)(s)
    ###################################################

#def apply_ctfs_gpu(refs,ctfs,psize,**kwargs):  
#    ''' This function can apply many ctfs to many sources, or to one source '''
#    # apply regular ctf by default, 0 - absctf, 1 - regctf, 2 - phaseflip
#    sh    = ctfs.shape
#    ctfs  = ctfs.flatten()
#    npart = np.prod(sh)
#    N,M   = refs.shape[-2:]
#    nr    = int(np.prod(refs.shape[:-2]))
#    assert(npart==nr or nr==1)
#    M2    = M//2 + 1
#    # init kernels    
#    actf_kernel   = apply_ctf_kernel()    
#    stream        = drv.Stream()   
#    fft2_plan_d   = cufft.Plan((N,M),np.float32,np.complex64,
#                               stream=stream,batch=npart)    
#    ifft2_plan_d  = cufft.Plan((N,M),np.complex64,np.float32,
#                               stream=stream,batch=npart)
#    fft2_plan_nr  = cufft.Plan((N,M),np.float32,np.complex64,
#                               stream=stream,batch=1) if nr==1 else fft2_plan_d                                     
#    dus_gpu,dvs_gpu,rots_gpu,As_gpu = ctfs2gpu(ctfs)      
#    ref_gpu   = gpuarray.to_gpu(refs)
#    reff_gpu  = gpuarray.empty((nr,N,M2),dtype='complex64')
#    rfc_gpu   = gpuarray.empty((npart,N,M2),dtype='complex64')
#    refs_gpu  = gpuarray.empty((npart,N,M),dtype='float32')
#    cufft.fft(ref_gpu,reff_gpu,fft2_plan_nr) 
#    apply_ctf_ext(actf_kernel,reff_gpu,rfc_gpu,
#                  M,dus_gpu,dvs_gpu,rots_gpu,As_gpu,
#                  psize,ctfs[0].cs,ctfs[0].lam,**kwargs)   
#    #print np.sqrt((rfc_gpu[0].get()**2).sum())                          
#    cufft.ifft(rfc_gpu,refs_gpu,ifft2_plan_d)    
#    refsout = refs_gpu.get()/np.float32(M*N)
#    del actf_kernel,fft2_plan_nr,fft2_plan_d,ifft2_plan_d
#    del dus_gpu,dvs_gpu,rots_gpu,As_gpu
#    del ref_gpu,reff_gpu,rfc_gpu,refs_gpu
#    return np.reshape(refsout,sh+(N,M))              
#        ngraph   = self.normalized_graph(psize)   
#        c1d      = np.abs(self.ctf_1d(self.win_sz,psize))
#        c1d      = c1d[:glen]  
#        call     = []
#        nall     = []
#        rall     = [] 
#        for wr in wrgs:  
#            r0   = res2rad(wr[0],win,psize)
#            r1   = res2rad(wr[1],win,psize)
#            r1   = min(r1,glen-1)
#            if r0 > (glen-1):
#                break;
#            rall = rall + range(r0,r1+1)
#            css  = c1d[r0:r1+1]
#            css -= css.mean()
#            ngg  = ngraph[r0:r1+1]
#            call = call + css.tolist()       
#            nall = nall + (ngg-ngg.mean()).tolist() 
#            ax4.plot([len(rall),len(rall)],np.array([css.min(),css.max()]),color='red',linewidth=2.0)
#        ax4.plot(np.float32(nall))
#        ax4.plot(np.float32(call), color='green')        
#    def save_pw(self,pw,base_dir):   
#        psname = self.__ps_filename(base_dir)
#        tprint('Saving power spectrum to %s' % psname)
#        mrc.save(pw,psname)    
    
#    def check_admissible(self,psize,worst_res,astig_deviaton,max_A): 
#        res = list()
#        if self.estim_res(psize) > worst_res:
#            res.append("resolution")
#        if abs(self.rat-1.0) > astig_deviaton:
#            res.append("astig ratio")
#        if self.A > max_A:
#            res.append("Af")
#        return res
        
#    def save_sel_stat(self,out_path,psize):
#        ''' save selection statistics to a json file '''
#        d     = {'estim_res':float(self.estim_res(psize)),
#                'astig_rat':float(self.rat),'A':float(self.A)}
#        with open(self.jsonname(out_path), "w") as outfile:
#            simplejson.dump(d, outfile, indent=4)
    
#                
#        #%%
#        ndefs   = p['ctf']['ndefs']
#        ndifs   = p['ctf']['ndifs']
#        arange  = p['ctf']['arange']
#        #nrats   = p['ctf']['nrats']
#        nrots   = p['ctf']['nrots']
#        maxA    = p['ctf']['maxA']
#        nA      = p['ctf']['nA']
#        #minrat  = p['ctf']['min_rat']
#        #maxrat  = p['ctf']['max_rat']  
#        dmin    = p['ctf']['dmin']
#        dmax    = p['ctf']['dmax']
#        wrgs    = np.float32(p['ctf']['w_rings'])
#        fbfact  = p['ctf']['fbfact']
#        cs      = p['ctf']['cs']
#        vkev    = p['ctf']['volkev'] 
#        ccut    = p['ctf']['coarse_cut']             
#        lam     = kev2lam(vkev)
#        win     = pw.shape[-1]
#        As      = np.float32(np.linspace(0.0,maxA,nA))    
#        rots    = np.float32(np.arange(nrots)*np.pi/(2*nrots))
#        #rats    = np.float32(np.linspace(minrat,maxrat,nrats)) 
        #rats[np.argmin(np.abs(rats-1.0))] = 1.0  # make sure that unit ratio is there      
#
#        # defocus range        
#        defs    = np.float32(np.linspace(dmin,dmax,ndefs)) 
#        # astigmatism range
#        difs    = np.float32(np.linspace(-arange/2.0,arange/2,ndifs)) 
#        #%%
#                                            
#        # Estimate ctf in 2 iterations     
#        # In the first iteration we don't look for astigmatism
#        tprint('Coarse CTF fitting, CUTOFF %.1fA...' % ccut)    
#        du,dv,ro,A = estim_def(pw,As,np.float32([0.0]),defs,np.float32([0.0]),
#                                  wrgs,ccut,fbfact,psize,lam,cs)[:4]        
#
#        # perform a finer defocus search and astigmatism
#        ab_s      = aber_s(du,dv,A,lam,cs) # cutoff due to spherical aberration
#        # don't go beyond sherical aberration rings
#        ncut      = np.maximum(1.0/ab_s,wrgs[-1,-1]) 
#        tprint('Fine CTF fitting, CUTOFF %.1fA ...' % ncut)    
#        d         = (du*dv)/2.0                                                                     
#        defs      = np.float32(np.linspace(1.1*d,0.9*d,ndefs)) 
#        # use all frequencies for the estimation
#        du,dv,ra,A,graph = estim_def(pw,As,rots,defs,difs,wrgs,ncut,fbfact,psize,lam,cs)[:5]
    
    
#def wrings2gammas(wrgs,d,A,lam,cs):     
#    s = 1.0/wrgs
#    return s2gamma(s,d,A,lam,cs)

#def estim_def(pw,As,rots,defs,difs,wrgs,mxres,fbfact,psize,lam,cs): 
#    win         = pw.shape[-1]    
#    s           = s_1D(win,psize)
#    nps         = int(np.prod(pw.shape[:-2]))
#    strtr       = res2rad(wrgs[0,0],win,psize)     
#    mxr         = min(res2rad(mxres,win,psize)+1,win//2)
#    
#    nrots,ndifs,nA,ndefs = len(rots),len(difs),len(As),len(defs)    
#    sg.allocate_blocked()  
#    cp_ker      = calc_polsum_kernel()
#    ac_ker      = adapt_corr_kernel()
#    rmeans_gpu  = gpuarray.empty((nps,ndifs,nrots,mxr),np.float32)
#    calc_polsum_ext(cp_ker,rmeans_gpu,pw,mxr,difs,rots) 
#    corrs_gpu   = gpuarray.zeros((nps,difs,nrots,nA,ndefs),np.float32)
#    one_loc     = np.zeros(mxr,np.float32)
#    for wr in wrgs:
#        r0      = res2rad(wr[0],win,psize)       
#        r1      = res2rad(wr[1],win,psize)      
#        one_loc[r0:r1] = 1.0        
#    # zero out weights outside water rings        
#    rw          = np.float32(np.exp(fbfact*(s[:mxr]**2)/4.0))*one_loc
#    adapt_corr_ext(ac_ker,rmeans_gpu,corrs_gpu,rw,As,defs,strtr,mxr,psize,lam,cs,win)
#    corrs       = corrs_gpu.get()
#    rmeans      = rmeans_gpu.get()
#    del cp_ker,ac_ker,rmeans_gpu,corrs_gpu
#    sg.release()
#    [psi,rati,roti,Ai,defi] = np.unravel_index(corrs.argmax(),corrs.shape)    
#    mcorr = corrs.max()
#    tprint('def = %.2f, rat=%.4f, rot=%.2f, A=%.3f, corr=%.4f' % 
#            (defs[defi],rats[rati],rots[roti],As[Ai],mcorr)) 
#    return defs[defi],rats[rati],rots[roti],As[Ai],\
#           rmeans[:,rati,roti,:],mcorr            
    
#    def estim_ps_multi_frame(self,base_dir,mv,drft,win_sz,ovlp,n_chunk_frames,max_mem_GB):
#        psname = self.__ps_filename(base_dir)
#        if not os.path.isfile(psname):   
#            tprint('Power spectrum file %s not found, calculating it using movie...' % psname)                 
#            ps,xpos,ypos = power_spectrum(mv,drft,win_sz,ovlp,n_chunk_frames,periodograms=False,
#                                          max_mem_GB=max_mem_GB)  
#            tprint('Saving power spectrum to %s' % psname)
#            mrc.save(ps, psname)     
#        else:
#            tprint('Loading power spectrum %s' % psname)
#            ps = mrc.load(psname)             
#        return ps    
    
#def adapt_corr_kernel():
#    return SourceModule('''
#            #include "math.h"
#            #define sign(x) ((x>0)?1:0)
#            __device__ float get_zero_s(float a,float b,float k) {
#                int k0 = (int)(2.0f*b*b)/(4.0f*a*M_PI);
#                if (2*k < k0) 
#                    return sqrtf((-b-sqrt(b*b-4.0f*a*k*M_PI))/(2.0f*a)); 
#                else
#                    return sqrtf((-b+sqrt(b*b+4.0f*a*(k-k0)*M_PI))/(2.0f*a));                     
#            } // get_zero_s
#            __device__ float calc_gain(float yp,float yn,int rp,int rn) {
#                return (yn-yp)/(float)(rn-rp);                      
#            }//calc_gain
#            __device__ float calc_offs(float yp,float yn,int rp,int rn) {
#                return (yp*(float)rn-yn*(float)rp)/(float)(rn-rp);                      
#            }//calc_gain            
#            __global__ void __corr(const float*  rmeans,
#                                   const float*  rweights,
#                                   const float*  defs,
#                                   const float*  As,
#                                   const int     nA,
#                                   const int     startr,
#                                   const float   psize,   
#                                   const float   lam,
#                                   const float   cs, 
#                                   const int     psSize,
#                                   const int     maxR,
#                                   const int     nRots,                                                                  
#                                   float* corrs){                                 
#            // const int    nConsZeros,
#            // rmeans   - nPs x nRats x nRots x maxR
#            // rweights - 1 x maxR correlation weight per frequency
#            // corrs    - nPs x nRats x nRots x nA x nDefs
#            ////////////////////////////// 
#            //int ratIdx = threadIdx.x;
#            int psIdx  = blockIdx.x/nA;                    
#            int aidx   = blockIdx.x%nA;
#            int nDefs  = gridDim.y;
#            int defIdx = blockIdx.y;   
#            int ratIdx = blockIdx.z;            
#            int nRats  = gridDim.z;
#            // calucate my defocus
#            float def  = defs[defIdx];
#            float A    = As[aidx];
#            float s2r  = psize*(float)psSize;
#            float r2s  = 1.0f/s2r;
#            for (int rotIdx = threadIdx.x; rotIdx < nRots; rotIdx += blockDim.x){                        
#                const float* rmean = &rmeans[psIdx*nRats*nRots*maxR+
#                                       ratIdx*nRots*maxR+
#                                       rotIdx*maxR];
#                float* corr = &corrs[psIdx*nRats*nRots*nDefs*nA +
#                                      ratIdx*nRots*nDefs*nA +
#                                      rotIdx*nDefs*nA+aidx*nDefs + defIdx];
#                float sqA1 = sqrtf(1.0f-A*A); 
#                // correlation of one ctf ripple
#                // find first zero s
#                float ss   = (float)startr*r2s;
#                float sp   = 0.0f;
#                float sn   = 0.0f;
#                float a    = 0.5f*M_PI*lam*lam*lam*cs;
#                float b    = M_PI*lam*def; 
#                int zidx;
#                for(zidx=1; sn<ss; zidx++){                    
#                    sp = sn;                    
#                    sn = get_zero_s(a,b,zidx);
#                } //  for zidx
#                // here sp,sn - prev,next zeros
#                int rp      = roundf(sp*s2r);
#                int rn      = roundf(sn*s2r);
#                float crr   = 0.0f;
#                float sumw  = 0.0f; // sum of all weights
#                // init linear detrending
#                float yp = rmean[rp];
#                float yn = rmean[rn];                        
#                float gn = calc_gain(yp,yn,rp,rn);
#                float o  = calc_offs(yp,yn,rp,rn);                                
#                for (int r = startr; r < maxR; r++){
#                    float s   = (float)r*r2s;
#                    float s2  = s*s;
#                    float g   = M_PI*lam*s2*(0.5f*cs*lam*lam*s2+def);
#                    float sn  = sinf(g);
#                    float cs  = cosf(g);
#                    // abs ctf of current r
#                    float actf = abs(sqA1*sn + A*cs);                          
#                    if (r > rn){
#                        rp   = rn;
#                        rn   = roundf(get_zero_s(a,b,zidx++)*s2r);
#                        rn   = min(rn,maxR-1);
#                        // update linear detrending
#                        yp   = rmean[rp];
#                        yn   = rmean[rn];                        
#                        gn   = calc_gain(yp,yn,rp,rn);
#                        o    = calc_offs(yp,yn,rp,rn);
#                    } // if r                  
#                    float m = rmean[r]-gn*r-o; 
#                    float w = rweights[r];
#                    crr += m*actf*w;   
#                    sumw += w;                                     
#                }// for r
#                *corr = crr/sumw; 
#            } // for rot                                 
#        } //main''').get_function('__corr')
#
#def adapt_corr_ext(ker,rmeans_gpu,corrs_gpu,rw,As,defs,startr,maxr,psize,lam,cs,win):
#    N_THREADS   = 32
#    nps,nrats,nrots,nA,ndef  = corrs_gpu.shape 
#    defs_gpu    = gpuarray.to_gpu(defs)                                  
#    As_gpu      = gpuarray.to_gpu(As)
#    rw_gpu      = gpuarray.to_gpu(rw)
#    ker(rmeans_gpu,rw_gpu,defs_gpu,As_gpu,np.int32(nA),np.int32(startr),
#        np.float32(psize),np.float32(lam),np.float32(cs),np.int32(win),
#        np.int32(maxr),np.int32(nrots),corrs_gpu,
#        grid=(int(nps*nA),int(ndef),int(nrats)),block=(N_THREADS,1,1)) #int(nrats),N_THREADS,1))
#    del defs_gpu,As_gpu,rw_gpu          
    
def calc_polsum_kernel():    
    return SourceModule('''
            __global__ void __polSum(const float* ps,
                const float* rats,
                const float* rots,
                const int psSize,
                const int maxR,
                float* rmeans) {
            // rmeans  - nPs x nRats x nRots x maxR
            // ps      - nPs x psSize x psSize  
            //////////////////////////////    
            int nRots  = blockDim.x;
            int rotIdx = threadIdx.x;           
            int nRats  = gridDim.x;
            int ratIdx = blockIdx.x;
            // Periodogram index
            int psIdx  = blockIdx.y;        
            const float* pPs = &ps[psIdx*psSize*psSize];    
            float  irat= 1/rats[ratIdx];    
            float  rot = rots[rotIdx]; 
            float sr   = sinf(rot);
            float cr   = cosf(rot);
            // always floor value here 
            int center = floorf(psSize/2); 
            // start with dc value
            rmeans[psIdx*nRats*nRots*maxR+
                   ratIdx*nRots*maxR+rotIdx*maxR] = pPs[center*psSize + center];
            for (int r = 1; r < maxR; r++){
                float ang     = 0.0;
                float rs      = 0.0;
                float nrAng   = r*M_PI;   // scan only half circle (pi) due to reflection symmetry
                float angStep = 1.0/r;  // this step covers one pixel at radius lzpos            
                float* rmean  = &rmeans[psIdx*nRats*nRots*maxR+
                                        ratIdx*nRots*maxR+rotIdx*maxR+r];
                for (int a = 0; a < nrAng; a++){
                    float sa  = sinf(ang);
                    float ca  = cosf(ang);            
                    float rx  = sa*r*irat;
                    float ry  = ca*r;            
                    float rxr = rx*cr + ry*sr;
                    float ryr = -rx*sr + ry*cr;            
                    int irxr  = (int) roundf(rxr + center);
                    int iryr  = (int) roundf(ryr + center);            
                    rs += pPs[irxr*psSize + iryr];
                    // increment angle
                    ang += angStep; 
                } // for a
                *rmean = rs/nrAng;  
            } // for r  
        } // main ''').get_function('__polSum')    
    
    
def calc_polsum_ext(kernel,rmeans_gpu,ps,maxr,rots,rats):
    ''' Calculates polar sums after applying astigmatic distortions. '''
    assert(ps.dtype=='float32')
    sz        = ps.shape[-2:]
    nps       = int(np.prod(ps.shape[:-2]))
    nrats     = np.int32(rats.size)
    nrots     = np.int32(rots.size)
    max_astig = np.maximum(1.0/rats[0], rats[-1])
    maxr      = np.minimum((np.float32(sz[0])//2-1)/max_astig,maxr)    
    crop_sz   = np.array([np.ceil(maxr*max_astig)*2]*2,dtype='int32')
    ps        = image.crop2D(ps,crop_sz)    
    #assert(nrats*nrots <= gutils.n_threads())
    rats_gpu   = gpuarray.to_gpu(rats)
    rots_gpu   = gpuarray.to_gpu(rots)    
    #rmeans_gpu = gpuarray.empty((nps,nrats,nrots,np.int32(maxr)), dtype='float32')
    ps_gpu     = gpuarray.to_gpu(ps)    
    kernel(ps_gpu,rats_gpu,rots_gpu,np.int32(crop_sz[0]),np.int32(maxr),
            rmeans_gpu,grid=(int(nrats),int(nps),1),block=(int(nrots),1,1))
    del rats_gpu,rots_gpu,ps_gpu                     
    
#    def estim_res(self,psize):
#        return estim_res(self.graph,self.win_sz,psize,self.start_res,self.skip_zeros,
#                         self.d,self.A,self.lam,self.cs)        
    
    
#def mem_calc_costs(ps_size, n_rats, n_rots, n_A, tot_defs):
#    ''' calculates how much gpu memory is needed to calclate 
#        costs of one periodogram '''
#    # powers spectrum, costs, cdef, for one power spectrum    
#    return (ps_size*ps_size + n_rats*n_rots*n_A*tot_defs + 1)*4    
    
    
#s   = s_2D(sz,psize)
# rotated coords
# x = sin(th), y = cos(th), d = du*cos^2(th-rot)+dv*sin^2(th-rot)
# cos(th-rot) = cos(th)*cos(rot)+sin(th)*sin(rot)
# sin(th-rot) = sin(th)*cos(rot)-cos(th)*sin(rot)
#sinth,costh = x,y    
#d = du*((costh*np.cos(rot)+sinth*np.sin(rot))**2) + \
#    dv*((sinth*np.cos(rot)-costh*np.sin(rot))**2)    

#def astig_coords(x,y,rat,rot):
#    # rotate
#    ang = -rot
#    xr  = x*np.cos(ang) + y*np.sin(ang)
#    yr  = x*(-np.sin(ang)) + y*np.cos(ang)
#    # scale
#    xr *= rat
#    return xr,yr
#
#def astig_coords_inv(x,y,rat,rot):
#    # scale
#    x /= rat
#    # rotate
#    xr = x*np.cos(rot) + y*np.sin(rot)  
#    yr = x*(-np.sin(rot)) + y*np.cos(rot)  
#    return xr,yr           

#def s_2D(sz,psize):
#    xn,yn = image.cart_coords2D(sz)
#    xn,yn = xn/sz[0],yn/sz[1]
#    #xn,yn = astig_coords(xn,yn,rat,rot)
#    return np.sqrt(xn**2 + yn**2)/psize      
    
#def s_2D(sz,rat,rot,psize):
#    xn,yn = image.cart_coords2D(sz)
#    xn,yn = xn/sz[0],yn/sz[1]
#    xn,yn = astig_coords(xn,yn,rat,rot)
#    return np.sqrt(xn**2 + yn**2)/psize      
    
#def apply_ctfs_m2m_gpu(refs,ctfs,psize,**kwarg): 
#    npart = len(ctfs)
#    N,M   = refs.shape[-2:]
#    M2    = M//2 + 1
#    # init kernels    
#    actfmm_ker  = apply_ctf_m2m_kernel()    
#    stream      = drv.Stream()   
#    fft2_plan_d = cufft.Plan((N,M),np.float32,np.complex64,
#                              stream=stream,batch=npart)    
#    ifft2_plan_d = cufft.Plan((N,M),np.complex64,np.float32,
#                              stream=stream,batch=npart)
#    defs_gpu,rats_gpu,rots_gpu,As_gpu = ctfs2gpu(ctfs)      
#    refs_gpu  = gpuarray.to_gpu(refs)
#    refsf_gpu = gpuarray.empty((npart,N,M2),dtype='complex64')
#    cufft.fft(refs_gpu,refsf_gpu,fft2_plan_d) 
#    apply_ctf_m2m_ext(actfmm_ker,refsf_gpu,M,defs_gpu,rats_gpu,rots_gpu,As_gpu,
#                      psize,ctfs[0].cs,ctfs[0].lam,0.0,**kwarg)                                 
#    cufft.ifft(refsf_gpu,refs_gpu,ifft2_plan_d)    
#    refs = refs_gpu.get()/np.float(M*N)
#    del actfmm_ker,fft2_plan_d,ifft2_plan_d
#    del defs_gpu,rats_gpu,rots_gpu,As_gpu
#    del refs_gpu,refsf_gpu
#    return refs            
    
#def apply_ctf_one2m_kernel():  
#    path = os.environ['PYTHONPATH'].split(os.pathsep)                           
#    return SourceModule('''
#         #include "ctf.cu"
#        __device__ float ctfabs(float ctf) {
#           return fabs(ctf);
#        } // ctfabs
#        __device__ float ctffull(float ctf) {
#           return ctf;
#        } // ctfabs
#        __device__ float ctfsign(float ctf) {
#           return (ctf>=0)?1.0:-1.0;  
#        } // ctfabs
#        // define pointer to function
#        typedef float (*func)(float x);        
#        __device__ func pctffun;            
#         __global__ void _apply_ctf(const float2* imin,
#                                  float2* imsout, 
#                                  const float* defs,
#                                  const float* rats,
#                                  const float* rots,
#                                  const float* As,
#                                  const float psize,
#                                  const float cs,
#                                  const float lam,                              
#                                  const float bfact,
#                                  const int y_len,
#                                  const int ctf_type){
#            int x       = blockIdx.x;
#            // image number 
#            int d       = blockIdx.y;  
#            int x_len   = gridDim.x;
#            int y_len2  = y_len/2 + 1;
#            float def   = defs[d];
#            float rat   = rats[d];
#            float rot   = rots[d];
#            float A     = As[d];
#            int imsidx  = d*x_len*y_len2;
#            float2* imout   = &imsout[imsidx];
#            switch (ctf_type){
#                case 0: 
#                    pctffun = ctfabs;
#                    break;
#                case 1:
#                    pctffun = ctffull;
#                    break;
#                case 2:
#                    pctffun = ctfsign;
#                    break;                    
#            } // switch            
#            for (int y      = threadIdx.x; y < y_len2; y += blockDim.x){  
#                int idx     = x*y_len2+y;
#                float2 a    = imin[idx]; 
#                float ctf   = _CTF(x,y,psize,rot,rat,cs,lam,def,A,
#                                   bfact,x_len,y_len);
#                ctf = pctffun(ctf);
#                // multiply complex by scalar
#                float2 c; c.x = a.x*ctf; c.y = a.y*ctf;                
#                imout[idx]  = c; 
#            } // for x                                           
#        } // main ''',include_dirs=path).get_function('_apply_ctf') 
#
#def apply_ctf_one2m_ext(kernel,imf_gpu,imout_gpu,y_len,defs_gpu,rats_gpu,
#                         rots_gpu,As_gpu,psize,cs,lam,bfact=0.0,**kwargs):  
#    ''' Multiply one complex fft matrix by many ctf functions'''
#    ctf_type = kwargs.pop('ctf_type',1);     
#    x_len,y_len2 = imf_gpu.shape[-2:]
#    D            = defs_gpu.size
#    kernel(imf_gpu,imout_gpu,defs_gpu,rats_gpu,rots_gpu,
#           As_gpu,np.float32(psize),np.float32(cs),
#           np.float32(lam),np.float32(bfact),np.int32(y_len),np.int32(ctf_type),
#           block=(min(int(y_len2),gutils.n_threads()),1,1),
#           grid=(int(x_len),int(D),1),**kwargs)             
    
    
#    def plot_distortions(self): 
#        sz   = [8192,8192]
#        ovlp = 0.5
#        ax = plt.figure().gca()        
#        res,dist = self.simulate_ps_distortion(sz,[128,128],ovlp,6,1)
#        ax.plot(res, dist,color='blue', label='win128')        
#        res,dist = self.simulate_ps_distortion(sz,[256,256],ovlp,6,1)
#        ax.plot(res, dist,color='green', label='win256')        
#        res,dist = self.simulate_ps_distortion(sz,[512,512],ovlp,6,1)
#        ax.plot(res, dist,color='red', label='win512')        
#        res,dist = self.simulate_ps_distortion(sz,[1024,1024],ovlp,6,1)
#        ax.plot(res, dist,color='magenta', label='win1024')        
#        res,dist = self.simulate_ps_distortion(sz,[2048,2048],ovlp,6,1)
#        ax.plot(res, dist,color='cyan', label='win2048')        
#        ax.invert_xaxis()        
#        plt.xlabel('resolution in A')
#        plt.ylabel('dif_ps/ps')        
#        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                   ncol=5, mode="expand", borderaxespad=0.)
#        plt.draw()
    
    
#def to_polar(im, rat=1.0, rot=0.0):
#    ''' Converts cartesic image to polar image given astigmatism ratio and angle'''
#    sz = np.int32(im.shape)
#    R  = sz[0]//2
#    r,th   = np.mgrid[0:R,0:2*np.pi*R]
#    th     = th/R    
#    x,y    = r*np.cos(th),r*np.sin(th)
#    x,y    = astig_coords_inv(x,y,rat,rot)    
#    x,y    = np.int32(np.round(x+sz[0]//2)),np.int32(np.round(y+sz[1]//2))
#    x[x<0] = 0
#    x[x>sz[0]-1]=sz[0]-1
#    y[y<0] = 0
#    y[y>sz[1]-1]=sz[1]-1
#    return im[x,y]        

#def ctf_zeros(sz, d, A, psize, lam, cs):
#    s   = np.arange((sz[0]//2))/(psize*sz[0])
#    ctf = ctf_1d(s, d, A, lam, cs)
#    mn  = minpeaks(np.abs(ctf))
#    mx  = minpeaks(-np.abs(ctf))
#    return np.int32(mn),np.int32(mx)    
    
#def corr_peak_normalize(rmeans):
#    ''' normalize peaks to have amplitude 1 '''
#    sg.allocate_blocked()            
#    mod = SourceModule('''
#            #include <float.h>    
#            #define max(x,y) ((x>y)?x:y)
#            #define min(x,y) ((x<y)?x:y)     
#            __global__ void normalize(float* rmeans,const int maxR){
#                // rmeans  - nPs x nRats x nRots x maxR
#                ////////////////////////////// 
#                int nRots  = blockDim.x;
#                int rotIdx = threadIdx.x;  
#                int ratIdx = blockIdx.x;
#                int nRats  = gridDim.x;
#                int psIdx  = blockIdx.y;         
#                float* rmean = &rmeans[psIdx*nRats*nRots*maxR+
#                                       ratIdx*nRots*maxR+
#                                       rotIdx*maxR];  
#                float mn = rmean[maxR-1];                                       
#                float mx = mn;                                       
#                for (int r = maxR-2; r > 0; r--){
#                    float rloc   = rmean[r];
#                    //float rloc_1 = rmean[r-1];
#                    //float rloc1  = rmean[r+1];
#                    //if ((rloc > rloc_1) && (rloc > rloc1) && (rloc >= (mx+mn)/2))
#                        // this is local maximum
#                        mx = max(rloc,mx);
#                    //if ((rloc < rloc_1) && (rloc < rloc1) && (rloc <= (mx+mn)/2))
#                        // this is local minimum
#                        mn = min(rloc,mn);
#                    if (mx > mn)
#                        // normalize rloc
#                        rloc = (rloc - mn)/(mx-mn);                        
#                    rloc     = min(rloc,1.0f);                      
#                    rmean[r] = max(rloc,-0.0f);                      
#                } // for r                                       
#            }''') 
#    nps,nrats,nrots,maxr = rmeans.shape            
#    normalize  = mod.get_function('normalize')
#    grid       = (int(nrats),int(nps),1)
#    block      = (int(nrots),1,1)
#    rmeans_gpu = gpuarray.to_gpu(rmeans)    
#    normalize(rmeans_gpu,np.int32(maxr),grid=grid,block=block)
#    rmeans = rmeans_gpu.get()
#    sg.release() 
#    return rmeans  

#def model_corr(rmeans,defs,As,startr,psize,lam,cs,ps_size):
#    ''' Regular correlation based fitting'''
#    nps,nrats,nrots,maxr = rmeans.shape     
#    ndef      = np.int32(defs.size)
#    nA        = np.int32(As.size)
#    sg.allocate_blocked()            
#    assert(nrots <= gutils.n_threads())        
#    mod = SourceModule('''
#            #include "math.h"
#            #define sign(x) ((x>0)?1:0)
#            __global__ void corr(const float*  rmeans,
#                                 const float*  defs,
#                                 const float*  As,
#                                 const int     nA,
#                                 const int     startr,
#                                 const float   psize,   
#                                 const float   lam,
#                                 const float   cs, 
#                                 const int     psSize,
#                                 const int     maxR,
#                                 const int     nRots,                                                                  
#                                 float* corrs){
#            // const int    nConsZeros,
#            // rmeans  - nPs x nRats x nRots x maxR
#            // corrs   - nPs x nRats x nRots x nA x nDefs
#            ////////////////////////////// 
#            int nRats  = blockDim.x;
#            int ratIdx = threadIdx.x;
#            int psIdx  = blockIdx.x/nA;                    
#            int aidx   = blockIdx.x%nA;
#            int nDefs  = gridDim.y;
#            int defIdx = blockIdx.y;   
#            // calucate my defocus
#            float def    = defs[defIdx];
#            float A      = As[aidx];
#            float invs   = 1.0f/(psize*(float)psSize);
#            float mm     = 0.0f; // data mean           
#            float cc     = 0.0f; // model mean           
#            //float mmm    = 0.0f; // data norm                  
#            for (int rotIdx = threadIdx.y; rotIdx < nRots; rotIdx += blockDim.y){                        
#                const float* rmean = &rmeans[psIdx*nRats*nRots*maxR+
#                                       ratIdx*nRots*maxR+
#                                       rotIdx*maxR];
#                float* corr = &corrs[ psIdx*nRats*nRots*nDefs*nA +
#                                      ratIdx*nRots*nDefs*nA +
#                                      rotIdx*nDefs*nA+aidx*nDefs + defIdx];
#                float sqA1 = sqrtf(1.0f-A*A); 
#                // correlation of one ctf ripple
#                float onecorr    = 0.0f;   
#                int  sampcount   = 0;     // counts samples inside ripples        
#                for (int r = startr; r < maxR; r++){
#                    float s   = (float)r*invs;
#                    float s2  = s*s;
#                    float g   = M_PI*lam*s2*(0.5f*cs*lam*lam*s2+def);
#                    float sn  = sinf(g);
#                    float cs  = cosf(g);
#                    float ctf = sqA1*sn + A*cs;                
#                    // give more weight to higher freq
#                    float actf = abs(ctf)*r; 
#                    float m = rmean[r]*r; 
#                    mm  += m;
#                    cc  += actf;
#                    //mmm += m*m;
#                    onecorr += m*actf;
#                    sampcount++;
#                } // for r
#                // normalize by number of ripples
#                onecorr -= cc*mm/(float)sampcount;            
#                *corr = onecorr; 
#            } // for rotIdx
#        }''')
#    corr       = mod.get_function('corr')
#    grid       = (int(nps*nA),int(ndef),1)
#    block      = (int(nrats),int(min(nrots,gutils.n_threads()/nrats)),1)    
#    defs_gpu   = gpuarray.to_gpu(defs);
#    As_gpu     = gpuarray.to_gpu(As);
#    corrs_gpu  = gpuarray.empty((np.int32(nps),np.int32(nrats),
#                                 np.int32(nrots),np.int32(nA),
#                                 np.int32(ndef)),dtype='float32')
#    rmeans_gpu = gpuarray.to_gpu(rmeans)    
#    corr(rmeans_gpu,defs_gpu,As_gpu,nA,np.int32(startr),
#         np.float32(psize),np.float32(lam),np.float32(cs),np.int32(ps_size),
#         np.int32(maxr),np.int32(nrots),corrs_gpu,grid=grid,block=block)
#    corrs = corrs_gpu.get()
#    sg.release() 
#    return corrs  

        #graph    = np.float32(self.graph)   
        #ngraph  -= ngraph.mean()         
        #startidx = res2rad(res_estim*1.3,self.win_sz,psize)   
        #stopidx  = res2rad(res_estim*0.7,self.win_sz,psize)  
        #stopidx  = np.minimum(stopidx,ngraph.size-1)
        #residx   = res2rad(res_estim,self.win_sz,psize)-startidx                 
        #res      = rad2res(np.arange(startidx,stopidx+1),self.win_sz,psize)   
        #c1d     -= c1d.mean()         
        #print res.shape, graph[startidx:stopidx].shape
        #gss      = graph[startidx:stopidx+1]
        #ax2.plot(res,gss)
        #ax2.plot(res[[residx,residx]],np.array([gss.min(), gss.max()]),color='red')
        #ax2.invert_xaxis()
#        ax2.plot(dxy[:,:,0].mean(axis=1),label='x drift')        
#        ax2.plot(dxy[:,:,1].mean(axis=1),label='y drift')        
#        #ax2.get_yaxis().set_visible(False)
#        ax2.set_title("Mean drift in A") 
        #handles, labels = ax.get_legend_handles_labels()        
        #ax2.legend() #([xhand,yhand],['x drift', 'y drift'])
        
        #startidx = res2rad(res_estim*1.3,self.win_sz,psize)   
        #stopidx  = res2rad(res_estim*0.7,self.win_sz,psize) 
        #stopidx  = np.minimum(stopidx,ngraph.size-1)
        #residx   = res2rad(res_estim,self.win_sz,psize)-startidx                 
        #res      = rad2res(np.arange(startidx,stopidx+1),self.win_sz,psize)   
        #css      = c1d[startidx:stopidx+1]
        #css     -= css.mean()
        # construct graph for water ring regions        
#        ax4.plot(res,ngraph[startidx:stopidx+1])

#def estim_res(ngraph,win_sz,psize,mid_res,skip_zeros,d,A,lam,cs):
#    x       = np.float32(ngraph)
#    glen    = x.size     
#    all_zer,all_one = ctf_zeros(win_sz,d,A,psize,lam,cs)  
#    all_zer = all_zer[all_zer<=glen]
#    s       = np.arange((win_sz[0]//2))/(psize*win_sz[0])
#    ctf     = abs(ctf_1d(s, d, A, lam, cs))
#    startr  = res2rad(mid_res,win_sz,psize)
#    all_zer = all_zer[all_zer>=startr]
#    badcount = 0
#    for k in range(all_zer.size-1):
#        arg = np.arange(all_zer[k]+1,all_zer[k+1],dtype='int32')
#        # correlate ripple
#        p   = np.dot(ctf[arg]-np.mean(ctf[arg]),x[arg]-np.mean(x[arg]))  
#        if p < 0:
#            badcount += 1
#        if badcount >= skip_zeros-1:
#            break              
#    return rad2res(all_zer[k],win_sz,psize)        
    
    #%%
#    ext         = p['input']['ext']
#    psize       = p['microscope']['psize']
#    dmin        = p['ctf']['dmin']
#    dmax        = p['ctf']['dmax']
#    ndefs       = p['ctf']['ndefs']
#    sres        = p['ctf']['start_res']
#    eres        = p['ctf']['end_res']
#    ovlp        = p['ctf']['ovlp']
#    chunk       = p['ctf']['chunk']
#    mem_GB      = p['ctf']['mem_GB']
#    maxA        = p['ctf']['maxA']
#    nA          = p['ctf']['nA']
#    min_rat     = p['ctf']['min_rat']
#    max_rat     = p['ctf']['max_rat']
#    nrats       = p['ctf']['nrats']
#    nrots       = p['ctf']['nrots']
    #lam         = kev2lam(vkev)    
#    scr_path    = scratch.sch.proc_path    
#    #mv          = Movie(in_dirs=in_dirs,out_dir=scr_path,micro=m,ext=ext)
#    mv          = Movie(in_dirs=in_dirs,out_dir=scr_path,micro=m)
#    mv.load2ram()
    #%%    
#    mv.extract()
    #ps = c.estim_ps_multi_frame(ctf_dir,mv,drft,[win,win],ovlp,chunk,mem_GB)    
    
    #c.crunch(ps,psize,dmin,dmax,ndefs,eres,sres,
    #         skip_zeros,maxA,nA,min_rat,max_rat,nrats,nrots)             
    #imctf = c.apply_ctf(drft.load_undrift(drift_dir),psize)
    
#def ctf_sin_1d(s, d, lam, cs):
#    s2      = s**2
#    gamma   = np.pi*lam*s2*(0.5*cs*(lam**2)*s2 + d)
#    return np.sin(gamma)      
#
#def ctf_cos_1d(s, d, lam, cs):
#    s2      = s**2
#    gamma   = np.pi*lam*s2*(0.5*cs*(lam**2)*s2 + d)
#    return np.cos(gamma)      
#        d,ra,ro,A = estim_def_zero_tilt_new(ps,psize,self.cs,self.lam,start_res,2.0*end_res,defs, 
#                                             np.float32([1.0]),np.float32([0.0]),
#                                             As,skip_zeros)[0:5] 
#        da,raa,roa,Aa,ga,nga,ca,resa = [],[],[],[],[],[],[],[]
#        era    = np.linspace(end_res*1.5,end_res,10)
#        # sweep end resolutions er
#        for er in era:                                          
#            d,ra,ro,A,graph,ngraph,cost = estim_def_zero_tilt_new(ps,psize,self.cs,self.lam,start_res,er,
#                                                                  defs,rats,rots,As,skip_zeros) 
#            da.append(d); raa.append(ra); roa.append(ro); Aa.append(A); 
#            ga.append(graph); nga.append(ngraph); ca.append(cost); 
#            res = estim_res(np.squeeze(ngraph).tolist(),list(win_sz),psize,
#                            start_res,skip_zeros,d,A,self.lam,self.cs)
#            tprint('Estimated resolution %.2f, end res %.2f' % (res,er))
#            resa.append(res)  
#        # find maximum endres
#        mnidx = np.argmin(resa)      
#def calc_polsum(*argv):   
#    sg.allocate_blocked()            
#    res = calc_polsum_ext(calc_polsum_kernel(), *argv)
#    res = res.get()    
#    sg.release()
#    return res       
#def batch_ctf(out_dir): 
#    #in_dirs   = p['input']['paths']  
#    drift_dir = p.get_path('drift')
#    ctf_dir   = p.get_path('ctf')
#    fn.mkdir_assure(ctf_dir)        
#    micros    = fn.dif_dirs([drift_dir],'.json',ctf_dir,'.json')
#    nmicros   = len(micros)
#    cmd_seqs  = list()    
#    tprint('Estimating ctfs of %d micros' % (nmicros,))
#    for midx in range(nmicros):      
#        cmd_seqs.append([(ef.ctf_one_micro,(out_dir,micros[midx]))])        
#    cluster.execute_sequences(cmd_seqs) 

#    def simulate_ps_distortion(self,sz,win_sz,ovlp,startres,niters): 
#        ''' calcualte distortion in power spectrum due to window size '''
#        ppsr1d = np.zeros(win_sz[0]//2, dtype='float32')
#        psize  = 0.61 #params.get_ctf_dict().psize
#        for it in range(niters):
#            # white noise of the large size
#            im    = np.random.normal(0,1,sz)
#            # ctf applied to the whole image
#            imctf = self.apply_ctf(im,psize)
#            # power spectrum of white noise using win_sz         
#            ps  = frame2ps(im,win_sz,ovlp)
#            # ctf model of size win_sz
#            ct  = self.ctf_2d(win_sz,psize)    
#            # fftshift, etc
#            ps  = center_spectrum(ps,win_sz)
#            # power sepctrum of difference between windows of imctf
#            # and windows of im after application of ct
#            psd = psdiff(im,imctf,ct,win_sz,ovlp)
#            psd = center_spectrum(psd,win_sz)
#            # ratio between the power sepctrum of difference and
#            # noise spectrum (which should be constant)
#            psr = psd/ps
#            # circular mean
#            ppsr   = to_polar(psr,1,0)
#            ppsr1d += np.mean(ppsr,axis=1)
#        # convert pixel indexes to resolution in angstroms
#        res    = rad2res(np.arange(ppsr1d.size),win_sz,psize)
#        # cut indexes to insteresting resolution region
#        idx    = np.where(res<startres)[0][0]        
#        return res[idx:],ppsr1d[idx:]/niters
#        

#def zeros_matrix(sz,psize,end_res,max_astig,defs,As,lam,cs,init_zero_idx):
#    # index of maximum zero relevant to a given end_res
#    max_idx = np.round(sz[0]*psize/end_res)
#    # determine max number of zeros that don't overflow
#    nd      = defs.size
#    na      = As.size
#    msz     = np.array(sz)/max_astig
#    n_zer   = np.zeros([na,nd], dtype='float32')
#    max_idx = min(max_idx, msz[0]//2)
#    zer     = list()    
#    one     = list()    
#    for a in range(na):
#        zera = list()
#        onea = list()
#        for d in range(nd):
#            all_zer,all_one = ctf_zeros(sz,defs[d],As[a],psize,lam,cs)
#            zera.append(all_zer)
#            onea.append(all_one)
#            all_zer    = all_zer[all_zer < max_idx]
#            n_zer[a,d] = all_zer.size #min(all_zer.size,all_ones.size)
#        zer.append(zera)    
#        one.append(onea)            
#    # work with the minimum of non-spilling number of zeros      
#    min_zeros = n_zer.min()
#    o_mat     = np.zeros([na,nd,min_zeros-init_zero_idx], dtype='float32')
#    z_mat     = np.zeros([na,nd,min_zeros-init_zero_idx], dtype='float32')
#    # fill the zeros matrix with zeros
#    for a in range(na):
#        for d in range(nd):
#            #all_ones,all_zer = ctf_extremes(sz,defs[d],As[a],psize,lam,cs,blend)
#            all_zer      = zer[a][d]
#            all_one      = one[a][d]            
#            z_mat[a,d,:] = all_zer[init_zero_idx:min_zeros]
#            o_mat[a,d,:] = all_one[init_zero_idx:min_zeros]            
#            #(all_zer[init_zero_idx:min_zeros] + all_zer[init_zero_idx+1:min_zeros+1])/2.0
#    return o_mat,z_mat      



#def estim_def_zero_tilt_new(ps,psize,cs,lam,start_res,end_res,defs, 
#                            rats,rots,As,skip_zeros):           
#                                
####%%###########################################                               
##    micro = 'A2aGs_20160508_0544'
##    out_dir = '/jasper/result/A2aGs_20160508/'
##    maxA=0.0
##    nA = 1
##    nrots = 64
##    min_rat=1.0
##    max_rat=1.0
##    nrats = 1
##    dmin=-1000
##    dmax = -30000
##    ndefs = 4096
##    start_res=10.0
##    end_res=2.0    
##    
##    c = CTF(micro)
##    c.load_json(out_dir + 'ctf/')
##    
##    lam = c.lam
##    cs  = c.cs
##    
##    As   = np.float32(np.linspace(0.0,maxA,nA))    
##    rots = np.float32(np.arange(nrots)*np.pi/(2*nrots))
##    rats = np.float32(np.linspace(min_rat,max_rat,nrats)) 
##    rats[np.argmin(np.abs(rats-1.0))] = 1.0  # make sure that unit ratio is there          
##    defs = np.float32(np.linspace(dmin,dmax,ndefs))    
##    
##    psize = params.pixel_size_A(out_dir)
##    
##    ps = mrc.load(out_dir+'/ctf/' + micro + '_ps.mrc')
###%%############################################                                
#                                
#    pssz    = np.int32(ps.shape[-2:])
#    cent    = pssz//2    
#    # average powerspectrums if more than one given
#    ps      = np.mean(ps,axis=-3)[None,:,:]
#    # starting frequency for correlation
#    # crop power spectrum to meet the maximum resolution
#    mxr     = res2rad(end_res,pssz,psize)+1
#    # use gpu to create polar sums subject to astig distortions
#    #zeros   = ctf_zeros(pssz,(defs[0]+defs[-1])/2.0,0,psize,lam,cs)[0] 
#    # spatial low-pass res for background removal based on first zeros spacing
#    # zspace  = np.minimum(zeros[2]-zeros[1], pssz[0]/20)     
#    zspace  = zeros[2]-zeros[1]  
#    lp_res  = 2*zspace    
#    
#    ps      = image.background_remove2D(np.squeeze(ps),lp_res)[None,:,:] 
#    ps[:,cent[0]-5:cent[0]+5,cent[1]-5:cent[1]+5] = 0.0    
#
#    rmeans  = calc_polsum(ps,mxr,rats,rots)    
#    rmeans  = image.background_remove1D(rmeans,lp_res,shape='gaussian')
#    rmeans  = image.background_remove1D(rmeans,lp_res,shape='gaussian')
#    rmeans  = image.background_remove1D(rmeans,lp_res,shape='gaussian')
#    nrmeans = corr_peak_normalize(rmeans)
#    
#    # get start radius
#    strtr   = np.floor(res2rad(start_res,pssz,psize)) 
#    # correlate normalized polar sums with the ctf model    
#    corrs   = model_corr(nrmeans,defs,As,strtr,psize,lam,cs,pssz[0])
#    [psi,rati,roti,Ai,defi] = np.unravel_index(corrs.argmax(), corrs.shape)    
#                             
#    mcorr = corrs.max()
#    tprint('def = %.2f, rat=%.2f, rot=%.2f, A=%.3f, corr=%.4f' % 
#            (defs[defi],rats[rati],rots[roti],As[Ai],mcorr)) 
#            
#    return defs[defi],rats[rati],rots[roti],As[Ai],\
#           rmeans[:,rati,roti,:],nrmeans[:,rati,roti,:],mcorr   


#def correct_low_freq(im,ctfs,ws,psize,dc_thresh=0.1,**kwargs): 
#    ''' corrects the effect of ctf on low-res frequencies in the average shape '''
#    first_one_only=kwargs.pop('first_one_only',False)
#    all_but_first =kwargs.pop('all_but_first',False)
#    sz    = im.shape
#    # calc average 1d ctf
#    if (ws.ndim == 2):
#        # weigh each frequency components individually
#        c     = np.zeros(ws.shape[1],dtype='float32')
#        for k in range(len(ctfs)):
#            c += ws[k,:]*np.abs(ctfs[k].ctf_1d(sz,psize))
#    else:
#        # use one weight for all freq components
#        c = ws[0]*np.abs(ctfs[0].ctf_1d(sz,psize))
#        for k in range(1,len(ctfs)):
#            c += ws[k]*np.abs(ctfs[k].ctf_1d(sz,psize))
#                    
#    #c /= ws.sum()
#    #plot(c)
#    if first_one_only:
#        zidxs,oidxs = ctf_zeros_ones(c)
#        if oidxs != []:
#            c[oidxs[0]:]=1.0 
#                        
#    if all_but_first:
#        zidxs,oidxs = ctf_zeros_ones(c)
#        if oidxs != []:
#            c[:oidxs[0]]=1.0 
#        else:
#            c[:]=1.0
#                         
#    # apply low-res threshold
#    c = np.maximum(c,dc_thresh)
#    # apply correction in the frequency domain
#    xn,yn = image.cart_coords2D(sz)
#    r     = np.sqrt((xn/sz[0])**2+(yn/sz[1])**2)*sz[0]
#    r     = np.round(r)
#    r     = np.int32(np.minimum(r,len(c)-1))
#    c2d   = np.fft.ifftshift(c[r])
#    imf   = np.fft.fft2(im)*(1.0/c2d)
#    # remove dc
#    imf[0,0] = 0;
#    return np.float32(np.real(np.fft.ifft2(imf))) 


#    def estim_ps_one_frame(self,base_dir,im,win_sz,ovlp):
#        psname = self.__ps_filename(base_dir)        
#        if not os.path.isfile(psname):   
#            tprint('Power spectrum file %s not found, calculating it using a single frame...' % psname)      
#            ps = frame2ps(im,win_sz,ovlp)   
#            #ps = center_spectrum(ps,win_sz)
#            ps = np.fft.fftshift(ps)            
#            ps[win_sz[0]//2,:] = (ps[win_sz[0]//2-1,:] + ps[win_sz[0]//2+1,:])/2
#            ps[:,win_sz[1]//2] = (ps[:,win_sz[1]//2-1] + ps[:,win_sz[1]//2+1])/2
#            ps = ps[None,:,:]
#            tprint('Saving power spectrum to %s' % psname)
#            mrc.save(ps, psname)     
#        else:
#            tprint('Loading power spectrum %s' % psname)
#            ps = mrc.load(psname)             
#        return ps        

#def ctf_zeros_ones(c):
#    # find ones 
#    pc    = c[0]
#    up    = True
#    ones  = []
#    zeros = [0]
#    for idx in range(1,len(c)):
#        if up and c[idx] < pc:
#            ones.append(idx-1)
#            up = False
#        if not up and c[idx] > pc:
#            zeros.append(idx-1)
#            up = True
#        pc = c[idx]
#    return zeros,ones      

#def  main(**kwargs): 
#    ''' Main function that performs ctf estimation ''' 
#    micro     = kwargs['micro']
#    out_dir   = kwargs['out_dir']
#    scratch   = kwargs['scratch']
#    win_sz    = kwargs.pop('window', 4096)
#    ovlp      = kwargs.pop('overlap', 0.5)  # window overlap for power spectrum
#    psize     = kwargs.pop('psize', 0.31)   # pixel size in angstroms
#    crop_res  = kwargs.pop('crop_res', 2.5)   # pixel size in angstroms
#    end_res   = kwargs.pop('end_res', 2.0)  # best resolution in angstroms
#    start_res = kwargs.pop('start_res', 10.0)  # lowest resolution allowed for fit used for A estimation
#    skip_zeros= kwargs.pop('skip_zeros', 5) # maximum number of zeros that are allowed to be skipped in fitting
#    cs        = kwargs.pop('cs', 2.7e7)     # spherical aberation
#    vol       = kwargs.pop('volkev', 300)   # voltage in kev
#    dmin      = kwargs.pop('dmin', -1e4)    # minumum defocus to look for 
#    dmax      = kwargs.pop('dmax', -3e4)    # maximum defocus to look for 
#    ndefs     = kwargs.pop('ndefs', 4096)   # number of defocus values to look for
#    maxA      = kwargs.pop('maxA', 1.0)     # maximum phase contrast
#    nA        = kwargs.pop('nA', 100)       # number of A values to test
#    nrots     = kwargs.pop('nrot', 32)     # number of astigmatism angles to search in [0,pi/2]
#    min_rat   = kwargs.pop('min_rat', 0.95) # minimum astigmatism ratio
#    max_rat   = kwargs.pop('max_rat', 1.05) # maximum astigmatism ratio
#    nrats     = kwargs.pop('nrats', 32)     # number of ratio samples nrots*nrats < 1024
#    n_chunk_frames = kwargs.pop('n_chunk_frames', 6) # number of frames to average for power spectrum    
#    max_mem_GB = kwargs.pop('max_mem_GB', 4)  # memory used to store power spectrum windows
#
#    micro_scr = os.path.join(scratch, micro)
#    ctf_dir   = sr.ctf_path(out_dir)
#    drift_dir = sr.drift_path(out_dir) 
#    # initialize shared gpu resource management
#    sg.init(2.0) 
#    # movie should be there already
#    mv        = Movie(dst_dir=micro_scr)  
#    # drift also should exist already
#    drft      = Drift(micro)   
#    drft.load_json(drift_dir)
#    psize     = params.get_ctf_dict(out_dir)["psize"]
#    c = CTF(micro)
#    c.init(cs,vol)
#    # estimate power spectrum
#    ps = c.estim_ps_multi_frame(ctf_dir,mv,drft,[win_sz,win_sz],ovlp,n_chunk_frames,max_mem_GB)    
#    c.crunch(ps,psize,dmin,dmax,ndefs,end_res,start_res,skip_zeros,maxA,nA,min_rat,max_rat,nrats,nrots)             
#    # save resulting figure
#    imctf = c.phase_flip(drft.load_undrift(drift_dir),psize)
#    #imctf = c.apply_ctf(drft.load_undrift(drift_dir),psize)
#    c.save_imctf(imctf, ctf_dir)    
#    # save phase flipped
#    c.report_result(ctf_dir,imctf,drft.mean_drift()*psize,psize,crop_res)  
#    # save found CTF params
#    c.save_json(ctf_dir)
#    sg.disconnect_myproc()
#
#def get_parser():
#    import argparse    
#    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
#                                     description='CTF Estimation.',
#                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#                                     epilog='Example: ctf.py -m GDH_Glutamate_20140817_0235 -t /scratch/GDH_Glutamate_20140817/ -o /fhgfs/test/GDH_Glutamate_20140817/ -sr 5.0 -er 2.4 -p 0.61')
#    parser.add_argument('-m','--micro', help='micrograph name', 
#                        default=argparse.SUPPRESS, required=True)
#    parser.add_argument('-t','--scratch', help='Scratch directory', 
#                        default = argparse.SUPPRESS, required=True)                    
#    parser.add_argument('-o','--out_dir', help='Project output directory', 
#                        default = argparse.SUPPRESS, required=True)                    
#    parser.add_argument('-p','--psize', help='Pixel size in Angstroms', type=float, default=0.31, required=False)
#    parser.add_argument('-cr','--crop_res', help='Cropping resolution of report', type=float, default=2.5, required=False)
#    parser.add_argument('-w','--window', help='Processing window size', type=int, default=1024, required=False)
#    parser.add_argument('-v','--overlap', help='Processing window overlap', type=float, default=0.8, required=False)
#    #parser.add_argument('-mr','--mid_res', help='Medium for CTF fit and phase blending', type=float, default=4.0, required=False)
#    parser.add_argument('-er','--end_res', help='End resolution for CTF fit', type=float, default=1.5, required=False)
#    parser.add_argument('-sr','--start_res', help='Lowest resolution used for CTF fit', type=float, default=10.0, required=False)
#    parser.add_argument('-sz','--skip_zeros', help='maximum number of zeros that are allowed to be skipped in fitting', type=int, default=2, required=False)
#    parser.add_argument('-cs','--cs', help='Spherical aberration in Angstroms', type=float, default=2.7e7, required=False)
#    parser.add_argument('-vk','--volkev', help='Voltage in kev', type=float, default=300, required=False)
#    parser.add_argument('-dn','--dmin', help='Minimum defocus value in Angstroms', type=float, default=-0.3e4, required=False)
#    parser.add_argument('-dx','--dmax', help='Maximum defocus value in Angstroms', type=float, default=-3e4, required=False)
#    parser.add_argument('-nd','--ndefs', help='Number of defocus values to test at coarse stage', type=int, default=4096, required=False)
#    parser.add_argument('-ma','--maxA', help='Maximum value of phase contrast', type=float, default=0.0, required=False)
#    parser.add_argument('-na','--nA', help='Number of phase contrast values to try ', type=int, default=1, required=False)
#    parser.add_argument('-nr','--nrot', help='Number of astigmatism angles to search in [0,pi/2], nrots < 1024', type=int, default=128, required=False)
#    parser.add_argument('-mnr','--min_rat', help='Minimum astigmatism ratio', type=float, default=0.95, required=False)
#    parser.add_argument('-mxr','--max_rat', help='Maximum astigmatism ratio', type=float, default=1.05, required=False)
#    parser.add_argument('-ns','--nrats', help='Number of ratio samples', type=int, default=32, required=False)
#    parser.add_argument('-ch','--n_chunk_frames', help='Size of chunk of frames for average', type=int, default=6, required=False)
#    parser.add_argument('-mm','--max_mem_GB', help='Memory used to store power spectrum windows', type=int, default=4, required=False)    
#    return parser
#    
#def get_default_args():
#    return vars(get_parser().parse_args(['--micro','skip','--scratch','skip','--out_dir','skip'])) # ,'--psize','0.415'
#    
##def get_params_file(param_path):
##    return sr.get_ctf_params_file(param_path)
#    
######### MAIN HERE ##################################################
#if __name__ == "__main__":    
#    kwargs = vars(get_parser().parse_args())
#    # call main function
#    main(**kwargs)
#    
############## TESTS ##########################################    
#def test_absctf_gpu(sz,rot,rat,df,A,pSize,cs,lam,blendRes): 
#    sg.allocate_blocked()    
#    path = os.environ['PYTHONPATH'].split(os.pathsep)    
#    mod = SourceModule('''    
#        #include "ctf.cu"
#        __global__ void _test(float2* imout,
#                              const int y_len,
#                              const float rot,
#                              const float rat,
#                              const float def,
#                              const float A,
#                              const float pSize,
#                              const float cs,
#                              const float lam,
#                              const float blendRes){
#            int x       = blockIdx.x;
#            int x_len   = gridDim.x;
#            for (int y = threadIdx.x; y<y_len; y+=blockDim.x){   
#                 int idx    = x*y_len+y;
#                 float ctf  = _CTF(x,y,pSize,rot,rat,cs,lam,def,A,
#                                       blendRes,x_len,y_len);
#                 float2 c; c.x = fabs(ctf); c.y = 0;                
#                 imout[idx] = c;
#                 } // for y
#        } // main
#        ''', include_dirs=path)
#    x_len,y_len = tuple(sz)
#    _test = mod.get_function('_test')   
#    imout_gpu   = gpuarray.empty(sz,dtype='complex64')    
#    _test(imout_gpu,np.int32(y_len),np.float32(rot),np.float32(rat),
#          np.float32(df),np.float32(A),np.float32(pSize),np.float32(cs),
#          np.float32(lam),np.float32(blendRes),
#          block = (int(y_len),1,1), grid = (int(x_len),1,1))  
#    imout = imout_gpu.get()
#    sg.release()
#    return imout       
#
#from myutils.myplotlib import imshow,clf
#def run_test_CTF_gpu():
#    cs,vol = 2.7e7,300
#    psize  = 0.61
#    micro = 'BetaGal_PETG_20141217_0303'
#    c = CTF(micro)
#    c.init(cs,vol)
#    c.load_json('/fhgfs/git/projects/SP/tests/ctf/')
#        
#    ctfsz = [512,512]
#    ctf   = c.ctf_2d(ctfsz,psize)
#    ctft  = test_absctf_gpu(ctfsz,c.rot,c.rat,c.d,c.A,psize,c.cs,c.lam,c.blend)
#    ctfdif = np.abs(ctf) - np.abs(ctft)
#    
#    clf()
#    imshow(np.abs(ctf))
#    imshow(np.abs(ctft))
#    imshow(ctfdif)     
#    
#def run_test_CTF_estim():
#    cs,vol = 2.7e7,300
#    psize  = 0.61
#    micro  = 'BetaGal_PETG_20141217_0303'
#    c = CTF(micro)
#    c.init(cs,vol)
#    c.load_json('/fhgfs/git/projects/SP/tests/ctf/')    
#    
#    sz = [2048,2048]
#    cc = c.ctf_2d(sz,psize)
#    cc = np.fft.fftshift(np.abs(cc))
#    nrots = 64
#    start_res = 10.0
#    end_res   = 1.7
#    As   = np.float32([0])
#    defs = np.float32(np.linspace(1.1*c.d,0.9*c.d,1024))    
#    rats = np.float32(np.linspace(0.9,1.1,16))
#    rats[np.argmin(np.abs(rats-1.0))] = 1.0
#    rots = np.float32(np.arange(nrots)*np.pi/(2*nrots))        
#    A = 0.0; skip_zeros = 2;
#    d,ra,ro,A,graph,ngraph,cost = estim_def_zero_tilt_new(cc[None,:,:],psize,c.cs,c.lam,start_res,end_res,
#                                                          defs,rats,rots,As,skip_zeros)                                                                                                                                 
#    tprint("orig def %f, ra %f ro %f, estim def %f, ra %f, ro %f" % (c.d,c.rat,c.rot,d,ra,ro)) 
#

#def power_spectrum(mv,drft,win_sz,ovlp,n_chunk_frames, **kwargs): 
#    to_correct   = kwargs.pop('drift_correct', True)    
#    periodograms = kwargs.pop('periodograms', False)    
#    avail_memGB  = kwargs['max_mem_GB']
#    #avail_memGB  = psutil.phymem_usage()[1]*1e-9/sg.ndevices()
#    # maximum allowed memory for spectrum calculation
#    max_mem      = 0.6*avail_memGB
#    
#    xposv, yposv = image.tile2D(mv.shape()[-2:], win_sz, ovlp, 
#                                marg=max(drft.win_sz[0]-win_sz[0], drft.max_drift))                              
#    [xpos, ypos] = image.ndgrid(xposv, yposv)
#    xpos,ypos    = xpos.flatten(), ypos.flatten() 
#    n_win        = xpos.size
#    # memory needed for one periodogram window
#    mem_win      = np.prod(win_sz)*4
#    batch        = np.floor(max_mem*1e9/mem_win)
#    if periodograms:                                                       
#        p = np.zeros([n_win] + win_sz, dtype='float32')
#    else:
#        p = np.zeros([1] + win_sz, dtype='float32')
#
#    flen  = len(drft.frames)   
#    fovlp = 0.5
#    step  = int(np.ceil(n_chunk_frames*(1-fovlp)))
#    frame_starts = np.arange(0,flen,step)   
#    frame_starts = frame_starts[frame_starts<=flen-n_chunk_frames]
#    win_chunks   = part_idxs(np.arange(n_win,dtype='int32'), batch=batch)   
#    
#    pr = pyprind.ProgBar(len(frame_starts),track_time=False,width=50)                    
#    frames = np.uint32(drft.frames)
#    for sidx in frame_starts:  
#        s = np.minimum(flen-n_chunk_frames,sidx)
#        didxs = np.uint32(s) + np.arange(n_chunk_frames,dtype='uint32')
#        fidxs = frames[didxs]
#        for widxs in win_chunks:
#            xl,yl = xpos[widxs],ypos[widxs]
#            # read drift-corrected windows for periodograms     
#            if to_correct: 
#                stacks = mv.mean_stacks(xl,yl,win_sz[0],drft,frames=fidxs,didxs=didxs)                                           
#            else:
#                stacks = mv.mean_stacks(xl,yl,win_sz[0],frames=fidxs)           
#            # Use GPU to calculate power spectrum
#            if periodograms:                                                   
#                sg.allocate_blocked() 
#                pl  = gimage.fft2_batch_mem_protected(stacks)
#                sg.release()
#                pl  = np.abs(pl)**2
#                p[widxs] = center_spectrum(pl,win_sz)              
#            else:
#                sg.allocate_blocked()             
#                pl  = gimage.sum_fft_norm_mem_protected(stacks) 
#                sg.release()
#                # aggregate power spectrum
#                p   = p + center_spectrum(pl,win_sz)[None,:,:]              
#            del stacks
#        pr.update()
#    return p/p.max(),xposv+win_sz[0]//2,yposv+win_sz[1]//2


#def apply_sqrctf_one2m_kernel():  
#    path = os.environ['PYTHONPATH'].split(os.pathsep)                           
#    return SourceModule('''
#         #include "ctf.cu"
#         __global__ void _apply_sqrctf(const float2* imin,
#                              float2* imsout, 
#                              const float* defs,
#                              const float* rats,
#                              const float* rots,
#                              const float* As,
#                              const float psize,
#                              const float cs,
#                              const float lam,                              
#                              const float bfact,
#                              const int y_len){
#            int x       = blockIdx.x;
#            // image number 
#            int d       = blockIdx.y;  
#            int x_len   = gridDim.x;
#            int y_len2  = y_len/2 + 1;
#            float def   = defs[d];
#            float rat   = rats[d];
#            float rot   = rots[d];
#            float A     = As[d];
#            int imsidx  = d*x_len*y_len2;
#            float2* imout   = &imsout[imsidx];
#            for (int y      = threadIdx.x; y < y_len2; y += blockDim.x){  
#                int idx     = x*y_len2+y;
#                float2 a    = imin[idx]; 
#                float ctf   = _CTF(x,y,psize,rot,rat,cs,lam,def,A,
#                                   bfact,x_len,y_len);
#                //ctf = fabs(ctf);
#                ctf = ctf*ctf;
#                // multiply complex by scalar
#                float2 c; c.x = a.x*ctf; c.y = a.y*ctf;                
#                imout[idx]  = c; 
#            } // for x                                           
#        } // main ''',include_dirs=path).get_function('_apply_sqrctf') #           

#def apply_sqrctfs_one2m_gpu(ref,ctfs,psize): 
#    npart = len(ctfs)
#    N,M   = ref.shape[-2:]
#    M2    = M//2 + 1
#    # init kernels    
#    sctfom_kernel = apply_sqrctf_one2m_kernel()    
#    stream        = drv.Stream()   
#    fft2_plan_1   = cufft.Plan((N,M),np.float32,np.complex64,
#                              stream=stream,batch=1)      
#    fft2_plan_d  = cufft.Plan((N,M),np.float32,np.complex64,
#                              stream=stream,batch=npart)    
#    ifft2_plan_d = cufft.Plan((N,M),np.complex64,np.float32,
#                              stream=stream,batch=npart)
#    defs_gpu,rats_gpu,rots_gpu,As_gpu = ctfs2gpu(ctfs)      
#    ref_gpu   = gpuarray.to_gpu(ref)
#    reff_gpu  = gpuarray.empty((N,M2),dtype='complex64')
#    rfc_gpu   = gpuarray.empty((npart,N,M2),dtype='complex64')
#    refs_gpu  = gpuarray.empty((npart,N,M),dtype='float32')
#    cufft.fft(ref_gpu,reff_gpu,fft2_plan_1) 
#    apply_ctf_one2m_ext(sctfom_kernel,reff_gpu,rfc_gpu,
#                           M,defs_gpu,rats_gpu,rots_gpu,As_gpu,
#                           psize,ctfs[0].cs,ctfs[0].lam,0.0)                                 
#    cufft.ifft(rfc_gpu,refs_gpu,ifft2_plan_d)    
#    refs = refs_gpu.get()/np.float32(M*N)
#    del sctfom_kernel,fft2_plan_1,fft2_plan_d,ifft2_plan_d
#    del defs_gpu,rats_gpu,rots_gpu,As_gpu
#    del ref_gpu,reff_gpu,rfc_gpu,refs_gpu
#    return refs

#    def apply_absctf(self,im,psize,bfact=0.0): 
#        ct = self.ctf_2d(im.shape,psize,bfact)    
#        return np.float32(np.real(ifft2(fft2(im)*np.abs(ct))))                 

#def apply_absctf_one2m_ext(kernel,imf_gpu,imout_gpu,y_len,defs_gpu,rats_gpu,
#                         rots_gpu,As_gpu,psize,cs,lam,bfact=0.0,**kwargs):  
#    ''' Multiply one complex fft matrix by many ctf functions'''
#    x_len,y_len2 = imf_gpu.shape[-2:]
#    D            = defs_gpu.size
#    kernel(imf_gpu,imout_gpu,defs_gpu,rats_gpu,rots_gpu,
#           As_gpu,np.float32(psize),np.float32(cs),
#           np.float32(lam),np.float32(bfact),np.int32(y_len),
#           block=(min(int(y_len2),gutils.n_threads()),1,1),
#           grid=(int(x_len),int(D),1),**kwargs)   


#    def save_im4disp(self,mv):  
#        imfile = self.__binned_filename()
##        tprint("Calculating frame mean and saving it to %s" % (imfile,))
##        im = mv.mean_frame()
##        im = image.bin2D(im, im.shape[0]/1024.0)
##        sz = np.float32(im.shape)
##        im = image.background_remove2D(im, sz[0]/20.0)
#        #im = image.histeq(im)
#        imsave(imfile, mv.im4disp())  

#def sum_sqrctf_gpu_kernel():  
#    path = os.environ['PYTHONPATH'].split(os.pathsep)                           
#    return SourceModule('''
#         #include "ctf.cu" 
#         __global__ void _sum_sqrctf(float* ctfsum,
#                              const float* ws,
#                              const float* defs,
#                              const float* rats,
#                              const float* rots,
#                              const float* As,
#                              const float psize,
#                              const float cs,
#                              const float lam,                              
#                              const float bfact,
#                              const int D,
#                              const int y_len){
#            int x       = blockIdx.x;
#            int x_len   = gridDim.x;
#            int y_len2  = y_len/2 + 1;
#            for (int y  = threadIdx.x; y < y_len2; y += blockDim.x){ 
#                float sum = 0;
#                int idx   = x*y_len2+y;
#                for(int d = 0; d < D; d++){
#                    float w     = ws[d];
#                    float def   = defs[d];
#                    float rat   = rats[d];
#                    float rot   = rots[d];
#                    float A     = As[d];
#                    float ctf   = _CTF(x,y,psize,rot,rat,cs,lam,def,A,
#                                       bfact,x_len,y_len);
#                    sum      += w*ctf*ctf; 
#                } // for d                                                  
#                ctfsum[idx] = sum; 
#            } // for y                                           
#        } // main ''', include_dirs=path).get_function('_sum_sqrctf')   
#        
#def sum_sqrctf_gpu_ext(kernel,ctfsum_gpu,w_gpu,y_len,defs_gpu,rats_gpu,
#                       rots_gpu,As_gpu,psize,cs,lam,bfact=0.0):  
#    ''' '''
#    x_len,y_len2 = ctfsum_gpu.shape[-2:]
#    D            = defs_gpu.size    
#    kernel(ctfsum_gpu,w_gpu,defs_gpu,rats_gpu,rots_gpu,
#           As_gpu,np.float32(psize),np.float32(cs),
#           np.float32(lam),np.float32(bfact),np.int32(D),np.int32(y_len),
#           block=(min(int(y_len2),gutils.n_threads()),1,1),grid=(int(x_len),1,1))                                 

#b  = exp(-bfact*(s.^2)/4);  
        
#def apply_ctfs(im,ctfs,ws,psize):
#    sz = im.shape   
#    c  = np.zeros(sz[0]//2+1,dtype='float32')    
#    for k in range(len(ctfs)):
#        c += ws[k]*np.abs(ctfs[k].ctf_1d(sz,psize))
#    xn,yn = image.cart_coords2D(sz)
#    r     = np.sqrt((xn/sz[0])**2+(yn/sz[1])**2)*sz[0]
#    r     = np.round(r)
#    r     = np.int32(np.minimum(r,len(c)-1))
#    c2d   = np.fft.ifftshift(c[r])
#    imf   = np.fft.fft2(im)*c2d
#    return np.float32(np.real(np.fft.ifft2(imf)))        


#def apply_bcurve(V,sstarts,bstarts,psize):
#    N,M,P     = V.shape
#    P2        = P//2 + 1
#    bcurve,bs1d,_ = bcurve_piecewise(N,sstarts,bstarts,psize)
#    sg.allocate_blocked()
#    bf_ker    = apply_bcurve_fft_kernel() 
#    fftplan   = cufft.Plan((N,M,P), np.float32, np.complex64, batch=1)
#    ifftplan  = cufft.Plan((N,M,P),np.complex64,np.float32, batch=1)  
#    V_gpu     = gpuarray.to_gpu(V)    
#    VF_gpu    = gpuarray.zeros((N,M,P2),np.complex64)
#    b_gpu     = gpuarray.to_gpu(bcurve) 
#    cufft.fft(V_gpu,VF_gpu,fftplan)  
#    apply_bcurve_fft_ext(bf_ker,VF_gpu,b_gpu)
#    cufft.ifft(VF_gpu,V_gpu,ifftplan) 
#    VO = V_gpu.get()
#    del bf_ker,fftplan,ifftplan
#    del V_gpu,VF_gpu,b_gpu
#    sg.release()
#    return VO

#def model_corr_peaks(rmeans,defs,As,startr,psize,lam,cs,ps_size):
#    nps,nrats,nrots,maxr = rmeans.shape 
#    ndef      = np.int32(defs.size)
#    nA        = np.int32(As.size)
#    #nBlends   = np.int32(blends.size)
#    sg.allocate_blocked()            
#    assert(nrots <= gutils.n_threads())        
#    mod = SourceModule('''
#            #include "math.h"            
#            #define sign(x) ((x>0)?1:0)
#            __global__ void corr(const float*  rmeans,
#                                 const float*  defs,
#                                 const float*  As,
#                                 const int     nA,
#                                 const int     startr,
#                                 const float   psize,   
#                                 const float   lam,
#                                 const float   cs, 
#                                 const int     psSize,
#                                 const int     maxR,
#                                 const int     nRots,                                 
#                                 float* corrs){
#            // const int    nConsZeros,
#            // rmeans  - nPs x nRats x nRots x maxR
#            // corrs   - nPs x nRats x nRots x nA x nDefs
#            ////////////////////////////// 
#            int nRats  = blockDim.x;
#            //int nRots  = blockDim.y;
#            int ratIdx = threadIdx.x;
#            //int rotIdx = threadIdx.y;  
#            //int nPs    = gridDim.x/nA;
#            int psIdx  = blockIdx.x/nA;                    
#            int aidx   = blockIdx.x % nA;
#            int nDefs  = gridDim.y;
#            int defIdx = blockIdx.y;   
#            //int blendIdx = blockIdx.y%nBlend;   
#            // calucate my defocus
#            float def    = defs[defIdx];
#            float A      = As[aidx];
#            //float blend  = blends[blendIdx];
#            float invs   = 1.0f/(psize*(float)psSize);
#            float mm     = 0.0f; // data mean           
#            float cc     = 0.0f; // model mean           
#            float mmm    = 0.0f; // data norm           
#            //float ccc    = 0.0f; // model norm                
#            for (int rotIdx = threadIdx.y; rotIdx < nRots; rotIdx += blockDim.y){            
#                const float* rmean = &rmeans[psIdx*nRats*nRots*maxR+
#                                       ratIdx*nRots*maxR+
#                                       rotIdx*maxR];
#                float* corr = &corrs[ psIdx*nRats*nRots*nDefs*nA +
#                                      ratIdx*nRots*nDefs*nA +
#                                      rotIdx*nDefs*nA+aidx*nDefs + defIdx];
#                float sqA1 = sqrtf(1.0f-A*A); 
#                float pctf = 0.0f;
#                // correlation of one ctf ripple
#                float onecorr    = 0.0f;   
#                float cost       = 0.0f;   
#                int  sampcount   = 0;     // counts samples inside ripples        
#                int  ripcount    = 0;     // counts ripples      
#                for (int r = startr; r < maxR; r++){
#                    float s   = (float)r*invs;
#                    //float w   = min(blend*s,1.0f);
#                    float s2  = s*s;
#                    float g   = M_PI*lam*s2*(0.5f*cs*lam*lam*s2+def);
#                    float sn  = sinf(g);
#                    float cs  = cosf(g);
#                    //float ctf = (1.0f-w)*sn + w*(sqA1*sn + A*cs);                
#                    float ctf = sqA1*sn + A*cs;                
#                    // detect zero crossing
#                    if ( (sign(pctf)!=sign(ctf)) && (pctf!=0) && (sampcount>1) ){ 
#                        mmm = mmm - mm*mm/(float)sampcount;
#                        onecorr -= cc*mm/(float)sampcount;
#                        //onecorr  = onecorr/sqrtf(mmm*sampcount);
#                        if (onecorr > 0.0)
#                            //accumulate cost
#                            cost+=onecorr;
#                        onecorr   = 0.0f;   
#                        mm        = 0.0f;  
#                        mmm       = 0.0f;  
#                        sampcount = 0;
#                        cc        = 0.0;
#                        ripcount++;
#                    } // if sign
#                    float actf = fabs(ctf); 
#                    // aggregate ripple correllation
#                    float m = rmean[r]; 
#                    mm  += m;
#                    cc  += actf;
#                    mmm += m*m;
#                    onecorr += m*actf;
#                    sampcount++;
#                    pctf = ctf;
#                } // for r
#                // normalize by number of ripples
#                *corr = cost/ripcount; 
#            } // for rotIdx
#        }''')
#    corr       = mod.get_function('corr')
#    grid       = (int(nps*nA),int(ndef),1)
#    block      = (int(nrats),int(min(nrots,gutils.n_threads()/nrats)),1)
#    defs_gpu   = gpuarray.to_gpu(defs);
#    As_gpu     = gpuarray.to_gpu(As);
#    corrs_gpu  = gpuarray.empty((np.int32(nps), np.int32(nrats),
#                                 np.int32(nrots), np.int32(nA),
#                                 np.int32(ndef)), dtype='float32')
#    rmeans_gpu = gpuarray.to_gpu(rmeans)    
#    corr(rmeans_gpu,defs_gpu,As_gpu,nA,np.int32(startr),
#         np.float32(psize), np.float32(lam),np.float32(cs),np.int32(ps_size),
#         np.int32(maxr),np.int32(nrots),corrs_gpu,grid=grid,block=block)
#    corrs = corrs_gpu.get()
#    sg.release() 
#    return corrs  

#def find_bcurve(V,start_res,ns,bstep,bmax,psize,arat): 
#    #%%
#    N,M,P     = V.shape
#    sz        = np.float32([N,M,P])
#    P2        = P//2 + 1
#    sg.allocate_blocked()
#    bf_ker    = apply_bcurve_fft_kernel() 
#    mul_ker   = gimage.mult_c64_conj_c64_self_kernel()
#    fftplan   = cufft.Plan((N,M,P), np.float32, np.complex64, batch=1)
#    V_gpu     = gpuarray.to_gpu(V)    
#    VF_gpu    = gpuarray.zeros((N,M,P2),np.complex64)
#    # transform volume
#    cufft.fft(V_gpu,VF_gpu,fftplan)  
#    VF        = VF_gpu.get()
#    
#    #%% initialize bcurve params
#    sstarts   = np.float32([0.0])
#    bstarts   = np.float32([])
#    bn        = float(start_res)/(2.0*psize)
#    sarr      = np.linspace(1.0/start_res,1.0/(2*psize),ns)
#    grow      = True
#    #%%
#    for s in sarr:
#        bn        = 1.0/(2.0*psize*s)
#        cropsz    = np.int32(np.round(sz/bn))
#        cpsize    = psize*bn
#        CN,CM,CP  = tuple(cropsz)
#        bopt      = -1
#        if grow:
#            sstarts   = np.append(sstarts,s)
#            bstarts   = np.append(bstarts,bopt)
#        else:
#            sstarts[-1] = s
#            bstarts[-1] = bopt
#        VC_gpu    = gpuarray.zeros(tuple(cropsz),np.float32)
#        VFC       = image.crop3D_fft_half(VF,cropsz)[0]
#        VFC_gpu   = gpuarray.to_gpu(VFC)    
#        VFA_gpu   = gpuarray.zeros(VFC.shape,np.complex64) 
#        cpsz      = np.prod(VFC.shape)*np.dtype(np.complex64).itemsize   
#        b_gpu     = gpuarray.empty(cropsz[0]//2,np.float32) 
#        ifftplan  = cufft.Plan(cropsz,np.complex64,np.float32, batch=1)  
#        grow = False           
#        while bopt > -bmax:   
#            bcurve,bs1d,_ = bcurve_piecewise(cropsz[0],sstarts,bstarts,cpsize,extend_hf=True)
#            drv.memcpy_htod(b_gpu.gpudata,bcurve[:cropsz[0]//2])     
#            # copy V transform
#            drv.memcpy_dtod(VFA_gpu.gpudata,VFC_gpu.gpudata,cpsz)     
#            # apply bfactor
#            apply_bcurve_fft_ext(bf_ker,VFA_gpu,b_gpu)
#            # autocorrelate
#            gimage.mult_c64_conj_c64_self_ext(mul_ker,VFA_gpu)     
#            # transform back
#            cufft.ifft(VFA_gpu,VC_gpu,ifftplan) 
#            # autocorrelation
#            VA   = np.fft.fftshift(VC_gpu.get()) #/(N*M*P)
#            cn   = VA[CN//2,CM//2,CP//2]
#            lb   = (VA[CN//2-1,CM//2,CP//2] + \
#                    VA[CN//2+1,CM//2,CP//2] + \
#                    VA[CN//2,CM//2-1,CP//2] + \
#                    VA[CN//2,CM//2+1,CP//2] + \
#                    VA[CN//2,CM//2,CP//2-1] + \
#                    VA[CN//2,CM//2,CP//2+1])/6.0
#            rat  = cn/lb
#            if rat < arat/bstep:
#                bopt = bopt*bstep
#                bstarts[-1] = bopt
#                grow = True
#            else:
#                break  
#        tprint('res  %.2fA,bfact %.2f' % (1.0/s,-bopt))
#        del ifftplan
#    #%%        
#    del bf_ker,mul_ker,fftplan
#    del V_gpu,VC_gpu,VF_gpu,VFA_gpu    
#    sg.release()
#    return sstarts,-bstarts    

#class Bfact(SaveAble): 
#    ''' Takes care of piecewise bfactor corrections '''
#    def __init__(self,mname,*args, **kwargs):
#        super(Bfact, self).__init__(*args,base_name=mname,**kwargs)
#        #self.bstarts = [0.0]
#        #self.sstarts = [0.0]
#        self.refs1d  = []
#        self.bfact   = []
#        
#    @staticmethod
#    def estim_trend(pa):
#        s1ds     = np.arange(len(pa))
#        trnd     = pa
#        mnLoc    = np.logical_and(trnd[:-2] > trnd[1:-1], trnd[2:] > trnd[1:-1])
#        mnLoc    = np.concatenate(([True],mnLoc))
#        mnLoc[-1]=True
#        trndspln = interpolate.pchip(s1ds[mnLoc],pa[mnLoc],extrapolate=True)
#        trnd     = trndspln(s1ds)  
#        mxLoc    = np.logical_and(trnd[:-2] < trnd[1:-1], trnd[2:] < trnd[1:-1])
#        mxLoc    = np.concatenate(([True],mxLoc))
#        mxLoc[-1]=True
#        trndspln = interpolate.pchip(s1ds[mxLoc],pa[mxLoc],extrapolate=True)
#        trnd     = trndspln(s1ds)  
#        return trnd
#
#    def estim_bcurve(self,V,psize,start_res):   
#        [pa,s1d] = image.pa1Dfrom3D(V,psize)            
#        if s1d[-1] < 1.0/start_res:
#            # do nothing
#            return
#        sLoc     = s1d[:len(pa)]>1.0/start_res
#        spa      = pa[sLoc]
#        s1ds     = s1d[sLoc]
#        ideal    = IdealBfact.load()
#        ref      = ideal.interp(s1ds)
#        
#        testtrnd  = Bfact.estim_trend(spa)
#        reftrnd   = Bfact.estim_trend(ref)
#        #plot(np.concatenate([ref[None,:],reftrnd[None,:]],axis=0).transpose())    
#        bfact     = reftrnd/testtrnd
#        bfact    /= bfact[0]
#        bfact     = np.concatenate((np.ones((sLoc==False).sum(),dtype='float32'),bfact))
#        self.s1d   = s1d.tolist()
#        self.bfact = bfact.tolist() 
#        
#    def interp(self,s1d):
#        # match bfact to the new volume size
#        
#        if self.bfact == []:
#            return np.ones(len(s1d),dtype='float32')
#            
#        paspln  = interpolate.splrep(np.float32(self.s1d),
#                                     np.float32(self.bfact))
#        bfact   =  interpolate.splev(s1d,paspln) 
#        return bfact        
##    
#    def calc_sharpen_mask2d(self,sz,psize,strength):
#        s1d     = image.psize2s1d(sz[0],psize)[:(sz[0]//2)]       
#        bfact   = self.interp(s1d)
#        bfact   = np.power(bfact,strength)
#        X,Y     = image.cart_coords2D(sz)
#        r       = np.sqrt(X*X + Y*Y)
#        return bfact[np.minimum(np.int32(np.round(r)),len(bfact)-1)]

#    def apply_3d(self,V,psize,bfact,strength):
#        sz      = V.shape
#        bf      = bfact_3d(sz,bfact,psize)
#        s1d     = image.psize2s1d(sz[0],psize)[:(sz[0]//2)]       
#        curve   = self.interp(s1d)
#        #curve   = np.power(curve,strength)
#        VF      = np.fft.fftshift(np.fft.fftn(V))/np.prod(sz)
#        # radially average
#        X,Y,Z   = image.cart_coords3D(sz)
#        r       = np.sqrt(X*X + Y*Y + Z*Z)
#        B       = curve[np.minimum(np.int32(np.round(r)),len(curve)-1)]
#        VFB     = VF*np.power(B*bf,strength)
#        VB      = np.fft.ifftshift(VFB)
#        VB      = np.fft.ifftn(VB).real
#        return np.float32(VB)

#class IdealBfact(SaveAble):
#    ''' Calculates and stores ideal local differences for each spatial frequency '''    
#    def __init__(self,mname,*args, **kwargs):
#        super(IdealBfact, self).__init__(*args,base_name=mname,**kwargs)
#        self.s1d = []
#        self.pa  = []
#        #self.diff = []
#    
#    @staticmethod    
#    def save_ref_pa():
#        ''' calcualtes and returns a spline that approximates reference power amplutude'''
#        sample_file = '/jasper/models/BetaGal/betagal1.5.mrc'
#        V,psize   = mrc.load_psize(sample_file)        
#        # make model cubic with odd size
#        V         = image.make_cubic(V)
#        ideal     = IdealBfact('ideal_bfact')
#        [pa,s1d]  = image.pa1Dfrom3D(V,psize) 
#        ideal.pa  = pa.tolist()
#        ideal.s1d = s1d[:len(pa)].tolist()
#        ideal.save_json('/jasper/git/projects/SP/data/')
#        
#    def interp(self,s1ds):
#        paspln = interpolate.splrep(np.float32(self.s1d),
#                                    np.float32(self.pa))
#        return interpolate.splev(s1ds, paspln) 
#        
#    @staticmethod
#    def load():
#        ideal = IdealBfact('ideal_bfact')
#        ideal.load_json('/jasper/git/projects/SP/data/')
#        return ideal 

#def calc_costs(ps,psize,cs,lam,end_res,defs,n_defs, 
#               cdefs,rats,rots,As,init_ring):
#    assert(ps.ndim==3)    
#    n_A       = np.int32(As.size)
#    n_rots    = np.int32(rots.size)
#    n_rats    = np.int32(rats.size)
#    sz        = np.int32(ps.shape[-2:])
#    #ps_size   = np.copy(sz[0])
#    n_ps      = int(np.prod(ps.shape[:-2]))
#    max_astig = np.maximum(1.0/rats[0], rats[-1])
#    dmin      = np.float32(defs[0])
#    dmax      = np.float32(defs[-1])  
#    tot_defs  = np.int32(defs.size)
#    ddef      = (dmax-dmin)/(tot_defs+1)
#    assert(n_defs<=tot_defs)      
#    
#    # Ensure that defocus periodogram centers are far enough from defocus range ends
#    #print cdefs.min(),defs[n_defs//2]-ddef,ddef
#    assert(cdefs.min() >= defs[n_defs//2]-ddef)
#    assert(cdefs.max() <= defs[tot_defs-n_defs//2]+ddef)   
#    assert(cdefs.dtype == 'float32')
#    assert(defs.dtype  == 'float32')
#    assert(As.dtype    == 'float32')    
#    assert(rats.dtype  == 'float32')    
#    assert(rots.dtype  == 'float32')    
#    # z_mat ~ nA x tot_defs x n_ctf_zeros
#    o_mat,z_mat = zeros_matrix(sz,psize,end_res,max_astig,defs,As,
#                               lam,cs,init_ring)                                 
#    n_ctf_zeros = np.int32(z_mat.shape[2]) 
#    # pre-allocate costs
#    costs = np.zeros([n_ps,n_rats,n_rots,n_A,tot_defs], dtype='float32')        
#    sg.allocate_blocked()        
#    assert(n_defs % gutils.warp_size() == 0)
#    assert(n_defs <= gutils.n_threads())        
#    fitctf    = gutils.get_pycudafun("fitCTF.cu")
#    # treads work on small defocus neighborhoods, n_defs <= tot_defs
#    o_mat_gpu = gpuarray.to_gpu(o_mat)
#    z_mat_gpu = gpuarray.to_gpu(z_mat)
#    rots_gpu  = gpuarray.to_gpu(rots)
#    rats_gpu  = gpuarray.to_gpu(rats)
#    # choose only one center defocus, since we have only one power spectrum
#    gpu_sz    = mem_calc_costs(sz[0], n_rats, n_rots, n_A, tot_defs)
#    batch     = gutils.mem_elements(gpu_sz)
#    n_iters   = int(np.ceil(n_ps/batch))
#    for iter in range(n_iters):
#        idxs      = np.array(batch_idxs(n_ps, batch, iter)) 
#        lps       = idxs.size
#        block     = (int(n_defs), 1, 1)
#        grid      = (int(n_A*n_rats*n_rots), int(lps), 1)
#        ps_gpu    = gpuarray.to_gpu(ps[idxs])
#        costs_gpu = gpuarray.empty((np.int32(lps),n_rats,n_rots,n_A,tot_defs), dtype='float32')
#        cdefs_gpu = gpuarray.to_gpu(cdefs[idxs])            
#        fitctf(z_mat_gpu, o_mat_gpu, ps_gpu, rots_gpu, rats_gpu, cdefs_gpu, 
#               dmin, dmax, tot_defs, n_rots, n_A, 
#               costs_gpu, n_ctf_zeros, sz[0], block=block, grid=grid)           
#        costs[idxs] = costs_gpu.get()  
#        del ps_gpu
#        del costs_gpu
#    sg.release()    
#    return costs     

#def estim_tilt_def_fixed(ps,xpos,ypos,init_def,init_tilt,max_tilt,tilt_inc, 
#               psize,cs,lam,end_res,n_defs,rats,rots,As,init_ring,blend):
#    n_ps  = ps.shape[0]
#    # micrograph center for tilt estimation
#    xcent = (xpos[0]+xpos[-1])//2 
#    ycent = (ypos[0]+ypos[-1])//2
#    # calculate defocus range based on periodogram locations
#    xposp = np.float32(psize*(xpos - xcent))
#    yposp = np.float32(psize*(ypos - ycent))
#    [xposp, yposp] = image.ndgrid(xposp, yposp)
#    xposp,yposp = xposp.flatten(), yposp.flatten()
#    # obtain sphere segment pointing tao init_tilt
#    #points = image.sample_sphere(tilt_inc, max_tilt, init_tilt)
#    #ntilts = points.shape[0]
#    angs   = np.float32(np.linspace(0, max_tilt, np.ceil(max_tilt/tilt_inc)))[:,None]
#    ntilts = angs.size;
#    points = np.concatenate((np.zeros((ntilts,1),dtype='float32'), 
#                            np.sin(angs), np.cos(angs)),axis=1)
#    #plot3D(points)
#    # obtain defocus center for each periodogram given initial tilt
#    dcents = init_tilt[0]*xposp + init_tilt[1]*yposp                       
#    #defocus offset for each tilt [npos x ntilts]
#    doffs  = points[:,0]*xposp[:,None] + points[:,1]*yposp[:,None]
#    # determine defocus range for ctf sampling
#    mnoffs = doffs.min()
#    mxoffs = doffs.max()
#    # fine defocus spacing - there should be space of > n_defs//2 
#    # before dcents.min() and after dcents.max()
#    ddef   = (mxoffs-dcents.max())/(n_defs-n_defs//2+1)
#    ddef   = np.minimum(ddef, (dcents.min()-mnoffs)/(n_defs//2+1))    
#    tot_defs = np.ceil((mxoffs-mnoffs)/ddef)
#    assert(tot_defs>=n_defs)    
#    cdefs  = init_def + dcents
#    defs   = np.float32(np.linspace(init_def+mnoffs,init_def+mxoffs,tot_defs))
#    costs  = calc_costs(ps,psize,cs,lam,end_res,defs,n_defs,
#                        cdefs,rats,rots,As,init_ring,blend)
#    #costs ~ n_ps x n_rat x n_rot x n_A x tot_defs
#    #select best tilt
#    t_costs = np.zeros([ntilts]+list(costs.shape[1:-1]), dtype='float32')
#    # t_costs ~ ntilts x n_rat x n_rot x n_A
#    for t in range(ntilts):
#        # convert offset to dspace index
#        tdidx = np.round((tot_defs-1)*(doffs[:,t]-mnoffs)/(mxoffs-mnoffs))    
#        # mean cost for this tilt
#        tc    = 0
#        for k in range(n_ps):
#            tc += costs[k,:,:,:,tdidx[k]]
#        t_costs[t] = tc        
#    # get optimal params
#    [ti,rati,roti,Ai] = np.unravel_index(t_costs.argmax(), t_costs.shape)
#    tprint('tilt=[%s], rat=%.2f, rot=%.2f, A=%.3f,cost=%.2f' % 
#           (array2str(points[ti,:]),rats[rati],rots[roti],As[Ai]))
#    return points[ti,:],rats[rati],rots[roti],As[Ai],costs.max()    
    
#def wins2tilts_gpu(costs,doffs,sdefs): 
#    sg.allocate_blocked()
#    mod = SourceModule(""" 
#    __global__ void convert_costs(const float* costs, 
#                                  float* tcosts, 
#                                  const float* doffs,
#                                  const float mnoffs,
#                                  const float mxoffs,
#                                  const float dmin,
#                                  const float dmax,
#                                  const int tot_defs,
#                                  const int nA,
#                                  const int nrots, 
#                                  const int nrats,
#                                  const int nps){
#        // costs ~ n_ps x n_rat x n_rot x n_A x tot_defs   
#        // t_costs ~ ndefs x ntilts x n_rat x n_rot x n_A    
#        // doffs ~ nps x ntilts                                 
#        const int ntilts  = gridDim.x;
#        const int ratrotA = gridDim.y;
#        const int ndefs   = blockDim.x;
#        // defocus index
#        const int didx    = threadIdx.x;
#        const int tiltidx = blockIdx.x;
#        const int Aidx    = blockIdx.y % nA;
#        int ratrot        = blockIdx.y/nA;
#        const int rotidx  = ratrot % nrots;
#        const int ratidx  = ratrot/nrots;        
#        float dinc        = (dmax-dmin)/((float)ndefs-1.0f);        
#        float sdefmnoffs  = didx*dinc - mnoffs;
#        float mxmndd      = ((float)tot_defs-1.0f)/(mxoffs-mnoffs+dmax-dmin);
#        int rotA          = nrats*nA;
#        int precidx       = ratidx*rotA*tot_defs+rotidx*nA*tot_defs+Aidx*tot_defs;
#        int tcidx         = didx*ntilts*ratrotA+tiltidx*ratrotA+ratidx*rotA+rotidx*nA+Aidx;
#        for(int k=0; k<nps; k++){
#            //tdidx = np.round((tot_defs-1)*(sdefs[d]-drange[0]+doffs[:,t]-mnoffs)/(mxoffs-mnoffs+drange[1]-drange[0]))                 
#            int tidx  = roundf((sdefmnoffs+doffs[k*ntilts+tiltidx])*mxmndd); 
#            int cidx  = k*ratrotA*tot_defs+precidx+tidx;
#            tcosts[tcidx] += costs[cidx];
#        } // for        
#    } // convert_costs
#    """)
#    convert = mod.get_function('convert_costs')
#    mnoffs  = np.float32(doffs.min())
#    mxoffs  = np.float32(doffs.max())
#    ndefs   = np.int32(sdefs.size)
#    nps,ntilts  = doffs.shape
#    nps,nrats,nrots,nA,tot_defs = costs.shape
#    costs_gpu   = gpuarray.to_gpu(costs)
#    doffs_gpu   = gpuarray.to_gpu(doffs)
#    tcosts_gpu  = gpuarray.empty((ndefs,ntilts,nrats,nrots,nA), np.float32)
#    grid        = (int(ntilts),int(nrats*nrots*nA),1)
#    block       = (int(ndefs),1,1)
#    convert(costs_gpu,tcosts_gpu,doffs_gpu,mnoffs,mxoffs,
#            np.float32(sdefs[0]),np.float32(sdefs[-1]),
#            np.int32(tot_defs), np.int32(nA),
#            np.int32(nrots),np.int32(nrats),np.int32(nps),
#            grid=grid,block=block)
#    tcosts = tcosts_gpu.get()
#    sg.release()            
#    return tcosts     

#    def apply_3d_old(self,V,W,psize,bstrength=1.0,**kwargs):
#        bcurve1d,s1d = self.calc_bcurve(V.shape[0],psize,**kwargs)   
#        # modify curve by bfact strength        
#        bcurve1d = bcurve1d**bstrength
#        A        = gimage.apodize_fun(W)        
#        Vb       = apply_bcurves(V*A,bcurve1d[None,:])[0]
#        return Vb
#    def estim_bcurve_old(self,V,W,psize,ibf,scale,min_scale,
#                     scale_step,bmax,nsamples):
#        sz = np.float32(V.shape)
#        # initialize piecewise measurements
#        bstarts     = np.float32([])
#        sstarts     = np.float32([0.0])
#        #mndiff      = ibf.get_min_diff()
#        bm          = bmax
#        # loop through all bin factors
#        for sc in np.arange(min_scale,scale+scale_step,scale_step):
#            bn     = float(scale)/sc            
#            cropsz = int(sz[0]/bn) #(int(sc),)*3
#            cropsz -= np.mod(cropsz,2)
#            cropsz = (cropsz,)*3
#            cpsize = psize*bn
#            s1d    = image.psize2s1d(cropsz[0],cpsize)    
#            diff   = ibf.get_diff(s1d[cropsz[0]//2])               
#            #bm     = bmax*(mndiff/diff);
#            bf_range      = np.linspace(0,bm,nsamples)
#            # obtain all possible curves for this scale
#            bcurves1d,s1d = Bfact.bcurves_piecewise(cropsz[0],sstarts,bstarts,
#                                                    bf_range,cpsize)
#
#            tprint('Estimating bfactor curve, target sharpness %.3f ...' % diff)
#                        
#            Vc     = gimage.scale3D(V,cropsz)[0]
#            Wc     = np.float32(gimage.resample3D(W,cropsz) > 0.5)  
#            #with utils.Timer():
#            sg.allocate_blocked(prio_level=1000.0)
#            Vb     = apply_bcurves_gpu(Vc,bcurves1d)            
#            shrps  = vols2shrps(Vb,Wc)          
#            sg.release()
#             
#            # find parameter that brings us towards the desired sharpness
#            nidx   = int(nsamples-1)
#            #shrps  = np.zeros(nsamples,dtype='float32')
#            #sg.allocate_blocked()
#            for k in range(int(nsamples)):
#                #Vb       = apply_bcurves_gpu(Vc,bcurves1d[k][None,:])
#                #assert(not np.any(np.isnan(Vb.flatten())))
#                #shrps[k] = vols2shrps(Vb,Wc)[0] 
#                if shrps[k] > diff:
#                    nidx = max(k-1,0)                   
#                    break
#            #sg.release()
#            
#            #nidx    = utils.find_nearest_idx(shrps,sharpness)
#            # update curve params
#            bstarts = np.append(bstarts,bf_range[nidx])
#            # update upper limit for the bfactor
#            bm      = max(bf_range[nidx]*2.0,bmax/5.0)
#            sstarts = np.append(sstarts,s1d[cropsz[0]//2])
#            assert(not np.isnan(shrps[nidx]))
#            tprint('bfactor %.1f, shrp %.3f, res [%.1f,%.1f]' % \
#                    (bf_range[nidx],shrps[nidx],
#                     np.inf if sstarts[-2]==0 else 1.0/sstarts[-2],
#                     1.0/sstarts[-1])  )                        
#
#        # store result
#        self.bstarts = bstarts.tolist()
#        self.sstarts = sstarts.tolist()
#    def calc_bcurve(self,sz,psize,**kwargs):
#        bstarts = np.float32(self.bstarts)  
#        sstarts = np.float32(self.sstarts)
#        bcurve,s1d,_ = Bfact.bcurve_piecewise(sz,sstarts,bstarts,psize,**kwargs)
#        return bcurve,s1d   
#        
#    def calc_sharpen_mask2d(self,sz,psize,bstrength,**kwargs):
#        ''' Multiplicative sharpening mask that enhances resolution 
#            bstrength = [0,1] 1 for maximum strength '''
#        xn,yn    = image.cart_coords2D(sz)
#        r        = np.int32(np.round(np.sqrt(xn**2+yn**2)))  
#        bcurve,_ = self.calc_bcurve(sz[0],psize,**kwargs)
#        # apply strength
#        bcurve   = bcurve**bstrength
#        binv     = np.float32(1.0/bcurve)
#        lb       = len(binv)        
#        r[r>(lb-1)] = lb-1
#        return binv[r]     

#    @staticmethod        
#    def learn_from_sample(self,sample_file): 
#        ''' Uses sample file to obtain difference stats '''
#        V,psize = mrc.load_psize(sample_file)        
#        # make model cubic with odd size
#        V       = image.make_cubic(V)
#        sz      = V.shape
#        tprint("Calculating differences of volume size %d, pixel size %.2f" % (sz[0],psize))
#        W       = gimage.calc_mask(V,np.float32(sz)/2)        
#        s1d     = image.psize2s1d(sz[0],psize)[:(sz[0]//2+1)]          
#        pdiffs  = list() 
#        parg    = list() 
#        for scale in range(16,sz[0],16):
#            Vc    = gimage.scale3D(V,(scale,)*3)
#            Wc    = gimage.resample3D(W,(scale,)*3)            
#            # save local diffs fro this scale    
#            pdiffs.append(vols2shrps(Vc,Wc)[0])
#            parg.append(s1d[scale//2])
#            tprint("Spatial frequency %.3f, diff %.3f" % (parg[-1],pdiffs[-1]))                
#        tck   = interpolate.splrep(parg,pdiffs)
#        self.diff = interpolate.splev(s1d, tck).tolist()  
#        self.s1d  = s1d.tolist()
#        plot(self.s1d,self.diff)          
#    def get_diff(self,s): 
#        # look the value up
#        idx = utils.find_nearest_idx(self.s1d,s)
#        return self.diff[idx]
#    def get_min_diff(self):
#        return np.float32(self.diff).min()
#
#    @staticmethod    
#    def bcurves_piecewise(sz,sstarts,bstarts,bf_range,psize):   
#        ''' Generates bfactor piecewise curves
#            sstarts - s values that start each resolution scale
#            bstarts - bfactors of each previous resolution scale 
#            bf_range - bfactors values for last res scale for fitting '''     
#        s1d      = image.psize2s1d(sz,psize)    
#        s2_1d    = s1d**2      
#        if len(sstarts) > 1:
#            bcurve1d,_,vprev = Bfact.bcurve_piecewise(sz,sstarts,bstarts,psize)
#        else:
#            bcurve1d = np.zeros((len(s1d)), dtype='float32')
#            vprev    = 1.0             
#        # construct versions of last scale
#        nver = bf_range.size 
#        bcurves1d = np.zeros((nver,len(s1d)), dtype='float32') 
#        for k in range(nver):   
#            sidxs = np.where(s1d >= sstarts[-1])[0]
#            # roll over previous result
#            bcurves1d[k,:sidxs[0]] = bcurve1d[:sidxs[0]]
#            
#            bcurves1d[k,sidxs] = np.exp(-bf_range[k]*s2_1d[sidxs]/4)*vprev/\
#                                 np.exp(-bf_range[k]*s2_1d[max(sidxs[0]-1,0)]/4)
#            
#        return bcurves1d,s1d 
#
#def apply_bcurves(V,bcurves1d):   
#    sz       = V.shape
#    xn,yn,zn = image.cart_coords3D(sz)
#    r        = np.int32(np.round(np.sqrt(xn**2+yn**2+zn**2)))
#    # construct versions of last scale
#    nver = bcurves1d.shape[-2] 
#    Vout = np.zeros((nver,)+sz,dtype='float32')    
#    Vf   = np.fft.fftn(V)
#    for k in range(nver):   
#        binv    = np.float32(1.0/bcurves1d[k])
#        b       = np.fft.ifftshift(binv[r])
#        Vout[k] = np.float32(np.real(np.fft.ifftn(Vf*b)))                               
#    return Vout 
#    
#def apply_bcurves_gpu(V,bcurves1d):   
#    sz       = V.shape
#    xn,yn,zn = image.cart_coords3D(sz)
#    r        = np.int32(np.round(np.sqrt(xn**2+yn**2+zn**2)))
#    # construct versions of last scale
#    nver = bcurves1d.shape[-2] 
#    Vout = np.zeros((nver,)+sz,dtype='float32')    
#    Vf   = np.fft.fftn(V)
#    Vf   = image.fft_full2half3D(np.complex64(Vf))
#    Vof  = np.zeros((nver,)+Vf.shape,dtype='complex64')
#    for k in range(nver):   
#        binv    = np.float32(1.0/bcurves1d[k])
#        binv    = np.minimum(binv,1000.0)
#        b       = image.fft_full2half3D(np.fft.ifftshift(binv[r]))
#        Vof[k]  = np.complex64(Vf*b)
#    Vout = gimage.ifft3_batch_mem_protected(Vof,sz)                               
#    return Vout     
#    
#def vols2shrps(Vb,Wc):
#    ''' Calculates sharpness measure in a volume '''
#    nvol  = Vb.shape[-4] 
#    sz    = Vb.shape[-3:]
#    for v in range(nvol):        
#        Vb[v] = image.pos(Vb[v]-np.median(Vb[v]))
#        Vb[v] = Vb[v]/(Vb[v]*Wc).sum()        
#    assert(not np.any(np.isnan(Vb.flatten())))        
#    batch  = gutils.mem_elements(sz[0]*sz[1]*sz[2]*4*2, 0)   
#    groups = utils.part_idxs(range(nvol),batch=batch)
#    D = np.zeros(Vb.shape,dtype='float32')
#    for g in range(len(groups)):
#        idxs    = groups[g]
#        D[idxs] = np.abs(Vb[idxs])*gimage.loc_pixel_diff(Vb[idxs])
#    assert(not np.any(np.isnan(D.flatten())))
#    return [(d*Wc).sum() for d in D]    


#def tilted_ps4test(sz,xpos,ypos,tilt,d,rat,rot,A,psize,lam,cs,blend):
#    xcent = (xpos[0]+xpos[-1])//2 
#    ycent = (ypos[0]+ypos[-1])//2
#    # calculate defocus range based on periodogram locations
#    xposp = np.float32(psize*(xpos - xcent))
#    yposp = np.float32(psize*(ypos - ycent))
#    [xposp, yposp] = image.ndgrid(xposp, yposp)
#    xposp,yposp = xposp.flatten(), yposp.flatten()    
#    dcents = tilt[0]*xposp + tilt[1]*yposp                       
#    nps  = xposp.size    
#    ps   = np.zeros([nps]+list(sz), dtype='float32')    
#    for p in range(nps):
#        tprint('Generating power spectrum %d out pf %d' %(p,nps))
#        ps[p] = fftshift(ctf_2d(sz, d+dcents[p], rot, rat, A, psize, lam, cs, blend))     
#    return np.abs(ps)        

#def estim_tilt_def(ps,xpos,ypos,drange,max_tilt,tilt_inc, 
#                   psize,cs,lam,end_res,n_defs,rats,rots,As,init_ring,blend,**kwargs):
#    # type of tilt axis selection 'x','y','unknown'
#    # Note: 'unknown' option is much slower               
#    tilt_axis = kwargs.pop('tilt_axis','unknown')    
#    init_tilt = np.float32([0,0,1])            
#    xcent = (xpos[0]+xpos[-1])//2 
#    ycent = (ypos[0]+ypos[-1])//2
#    # calculate defocus range based on periodogram locations
#    xposp = np.float32(psize*(xpos - xcent))
#    yposp = np.float32(psize*(ypos - ycent))
#    [xposp, yposp] = image.ndgrid(xposp, yposp)
#    xposp,yposp = xposp.flatten(), yposp.flatten()
#    # obtain sphere segment pointing tao init_tilt    
#    if tilt_axis == 'unknown':
#        points = image.sample_sphere(tilt_inc, max_tilt, init_tilt)
#    else: 
#        angs   = np.float32(np.linspace(-max_tilt, max_tilt, np.ceil(2*max_tilt/tilt_inc)))[:,None]
#        if tilt_axis == 'x': 
#            points = np.concatenate((np.sin(angs), np.zeros((angs.size,1),dtype='float32'), np.cos(angs)),axis=1)
#        else:        
#            points = np.concatenate((np.zeros((angs.size,1),dtype='float32'),
#                                    np.sin(angs), np.cos(angs)),axis=1)
#    #plot3D(points)
#    # obtain defocus center for each periodogram given initial tilt
#    dcents = init_tilt[0]*xposp + init_tilt[1]*yposp                       
#    #defocus offset for each tilt [npos x ntilts]
#    doffs  = np.float32(points[:,0]*xposp[:,None] + points[:,1]*yposp[:,None])
#    # determine defocus range for ctf sampling
#    mnoffs = doffs.min()
#    mxoffs = doffs.max()
#    # half defocus range
#    hdrange = (drange[1]-drange[0])/2
#    # half defocus value
#    hd      = (drange[1]+drange[0])/2    
#    # fine defocus spacing - there should be space of > n_defs//2 
#    # before dcents.min() and after dcents.max()
#    ddef   = (hdrange+mxoffs-dcents.max())/(n_defs-n_defs//2+1)
#    ddef   = np.minimum(ddef, (hdrange+dcents.min()-mnoffs)/(n_defs//2+1))    
#    tot_defs = np.ceil((mxoffs-mnoffs+drange[1]-drange[0])/ddef)
#    assert(tot_defs>=n_defs)    
#    cdefs  = hd + dcents
#    defs   = np.float32(np.linspace(drange[0]+mnoffs,drange[1]+mxoffs,tot_defs))
#    costs  = calc_costs(ps,psize,cs,lam,end_res,defs,n_defs,
#                        cdefs,rats,rots,As,init_ring,blend)
#    #costs ~ n_ps x n_rat x n_rot x n_A x tot_defs
#    #select best tilt and defocus
#    sdefs   = np.linspace(drange[0],drange[1],np.ceil((drange[1]-drange[0])/ddef))
#    sdefs   = np.float32(sdefs)
#    # t_costs ~ ntilts x n_rat x n_rot x n_A
#    # doffs ~ nps x ntilts
#    tprint('Search defocuses %d' % sdefs.size)
#    #sg.allocate_blocked()
#    t_costs = wins2tilts_gpu(costs,doffs,sdefs)    
#    #sg.release()
##    t_costs = np.zeros([sdefs.size, ntilts]+list(costs.shape[1:-1]), dtype='float32')    
##    for d in range(sdefs.size): 
##        for t in range(ntilts):
##            # convert offset to dspace index
##            tdidx = np.round((tot_defs-1)*(sdefs[d]-drange[0]+doffs[:,t]-mnoffs)/(mxoffs-mnoffs+drange[1]-drange[0]))         
##            # mean cost for this tilt
##            tc    = 0
##            for k in range(n_ps):
##                tc += costs[k,:,:,:,tdidx[k]]
##            t_costs[d,t] = tc        
#    # get optimal params
#    mcost = t_costs.max()           
#    [di,ti,rati,roti,Ai] = np.unravel_index(t_costs.argmax(), t_costs.shape)
#    tprint('def=%.2f, tilt=[%s], rat=%.3f, rot=%.2f, A=%.3f, cost=%.2f' % 
#           (sdefs[di],array2str(points[ti,:]),rats[rati],rots[roti],As[Ai],mcost))
#    return sdefs[di],points[ti,:],rats[rati],rots[roti],As[Ai],mcost        
#
#def estim_def_tilt_fixed(ps,xpos,ypos,init_def,init_tilt,drange,
#                         psize,cs,lam,end_res,n_defs,rats,rots,As,init_ring,blend):
#    n_ps  = ps.shape[0]
#    # micrograph center for tilt estimation
#    xcent = (xpos[0]+xpos[-1])/2 
#    ycent = (ypos[0]+ypos[-1])/2
#    # calculate defocus range based on periodogram locations
#    xposp = np.float32(psize*(xpos - xcent))
#    yposp = np.float32(psize*(ypos - ycent))
#    [xposp, yposp] = image.ndgrid(xposp, yposp)
#    xposp,yposp = xposp.flatten(), yposp.flatten()   
#    dcents = init_tilt[0]*xposp + init_tilt[1]*yposp
#    hrange = (drange[1]-drange[0])/2
#    ddef   = hrange/(n_defs-n_defs//2)    
#    tot_defs = np.ceil((drange[1]-drange[0]+dcents.max()-dcents.min())/ddef)
#    assert(tot_defs>=n_defs)
#    # defocus centers for local periodogram processing
#    cdefs   = (drange[0]+drange[1])/2 + dcents
#    defs    = np.float32(np.linspace(cdefs.min()-hrange,cdefs.max()+hrange,tot_defs))  
#    costs  = calc_costs(ps,psize,cs,lam,end_res,defs,n_defs,
#                        cdefs,rats,rots,As,init_ring,blend)
#    #costs ~ n_ps x n_rat x n_rot x n_A x tot_defs
#    #select best defocus
#    # number of defocus values to test
#    ndefval = np.floor((drange[1]-drange[0])/ddef)
#    t_costs = np.zeros([ndefval+1]+list(costs.shape[1:-1]), dtype='float32')
#    # t_costs ~ ntilts x n_rat x n_rot x n_A
#    for t in range(int(ndefval+1)):
#        # current defocus being tested
#        curdef = drange[0] + t*ddef
#        # convert offset to dspace indexes
#        tdidxs = np.round((tot_defs-1)*(curdef+dcents-defs[0])/(defs[-1]-defs[0]))    
#        # mean cost for this tilt
#        tc    = 0
#        for k in range(n_ps):
#            tc += costs[k,:,:,:,tdidxs[k]]
#        t_costs[t] = tc        
#    # get optimal params
#    [di,rati,roti,Ai] = np.unravel_index(t_costs.argmax(), t_costs.shape)
#    dres = drange[0] + di*ddef    
#    tprint('def=%.2f, rat=%.2f, rot=%.2f, A=%.3f' % (dres,rats[rati],rots[roti],As[Ai]))                      
#    return dres,rats[rati],rots[roti],As[Ai]       

#def estim_def_zero_tilt(ps, psize,cs,lam,end_res,dmin,dmax,n_defs, 
#                         rats,rots,As,init_ring,blend):
#    ps       = np.mean(ps, axis=0)[None,:,:]
#    tot_defs = n_defs # here we have only one power spectrum frame 
#    defs     = np.float32(np.linspace(dmin, dmax, n_defs))
#    cdefs    = np.array([defs[tot_defs//2]])            
#    costs    = calc_costs(ps,psize,cs,lam,end_res,defs,n_defs,
#                          cdefs,rats,rots,As,init_ring,blend)
#    # get optimal params
#    [psi,rati,roti,Ai,defi] = np.unravel_index(costs.argmax(), costs.shape)
#    mcost = costs.max()
#    tprint('def = %.2f, rat=%.2f, rot=%.2f, A=%.3f, cost=%.2f' % 
#            (defs[defi],rats[rati],rots[roti],As[Ai],mcost))    
#    return defs[defi],rats[rati],rots[roti],As[Ai],mcost    

#def run_test_apply_ctf():  
#    cs,vol = 2.7e7,300
#    psize  = 0.61
#    micro  = 'BetaGal_PETG_20141217_0303'
#    c = CTF(micro)
#    c.init(cs,vol)
#    c.load_json('/fhgfs/git/projects/SP/tests/ctf/')                                                                 
#    sz = [1024,1024]
#    sg.allocate_blocked()        
#    D = 10;
#    test = np.ones((D,sz[0],sz[1]//2+1), dtype='complex64')
#    test_gpu = gpuarray.to_gpu(test)    
#    ctfs = [c for i in range(D)]    
#    defs_gpu = gpuarray.to_gpu(np.float32([cc.d for cc in ctfs]))
#    rats_gpu = gpuarray.to_gpu(np.float32([cc.rat for cc in ctfs]))
#    rots_gpu = gpuarray.to_gpu(np.float32([cc.rot for cc in ctfs]))
#    As_gpu   = gpuarray.to_gpu(np.float32([cc.A for cc in ctfs]))    
#    out_gpu  = apply_absctf_gpu(test_gpu,sz[1],defs_gpu,rats_gpu,rots_gpu,As_gpu,psize,c.cs,c.lam,c.blend)                                                              
#    out = out_gpu.get()
#    sg.release()    
#    clf()
#    test = image.fft_half2full(np.real(out[D-1]),sz)  
#    imshow(test) 
#    cc = np.abs(c.ctf_2d(sz,psize))
#    imshow(cc)
#    imshow(test-cc)
    
#def model_corr(rmeans,defs,As,startr,psize,lam,cs,blend,ps_size):
#    nps,nrats,nrots,maxr = rmeans.shape 
#    ndef      = np.int32(defs.size)
#    nA        = np.int32(As.size)
#    sg.allocate_blocked()            
#    #ws        = gutils.warp_size()
#    #assert(nrats*nrots % ws == 0)
#    assert(nrots <= gutils.n_threads())        
#    mod = SourceModule('''
#            #define pi (3.141592654f)
#            __global__ void corr(const float* rmeans,
#                                 const float* defs,
#                                 const float* As,
#                                 const int   nA,
#                                 const int   startr,
#                                 const float psize,   
#                                 const float lam,
#                                 const float cs, 
#                                 const float blend,                                  
#                                 const int psSize,
#                                 const int maxR,
#                                 float* corrs){
#            // rmeans  - nPs x nRats x nRots x maxR
#            // corrs   - nPs x nRats x nRots x nA x nDefs
#            ////////////////////////////// 
#            int nRats  = blockDim.x;
#            int nRots  = blockDim.y;
#            int ratIdx = threadIdx.x;
#            int rotIdx = threadIdx.y;  
#            //int nPs    = gridDim.x/nA;
#            int psIdx  = blockIdx.x/nA;                    
#            int aidx   = blockIdx.x % nA;
#            int nDefs  = gridDim.y;
#            int defIdx = blockIdx.y;   
#            // calucate my defocus
#            //float def  = dmin+(float)defIdx*(dmax-dmin)/((float)nDefs);
#            float def  = defs[defIdx];
#            float A    = As[aidx];
#            float invs = 1.0f/(psize*(float)psSize);
#            float mc   = 0.0f; // cos norm            
#            float mm   = 0.0f; // sin norm            
#            float cc   = 0.0f; // sin norm            
#            const float* rmean = &rmeans[psIdx*nRats*nRots*maxR+
#                                   ratIdx*nRots*maxR+
#                                   rotIdx*maxR];
#            float* corr = &corrs[psIdx*nRats*nRots*nDefs*nA +
#                                  ratIdx*nRots*nDefs*nA +
#                                  rotIdx*nDefs*nA+aidx*nDefs + defIdx];
#            float sqA1 = sqrtf(1.0f-A*A);                                  
#            for (int r = startr; r < maxR; r++){
#                float s  = (float)r*invs;
#                float w  = min(blend*s,1.0f);
#                float s2 = s*s;
#                float g  = pi*lam*s2*(0.5f*cs*lam*lam*s2+def);
#                float sn = sinf(g);
#                float cs = cosf(g);
#                float ctf = abs((1.0f-w)*sn + w*(sqA1*sn + A*cs));
#                float m  = rmean[r];
#                mc += m*ctf;
#                mm += m*m;
#                cc += ctf*ctf;
#            } // for r
#            *corr = mc/sqrtf(mm*cc);
#        }''')
#    corr       = mod.get_function('corr')
#    grid       = (int(nps*nA),int(ndef),1)
#    block      = (int(nrats),int(nrots),1)
#    defs_gpu   = gpuarray.to_gpu(defs);
#    As_gpu     = gpuarray.to_gpu(As);
#    corrs_gpu  = gpuarray.zeros((np.int32(nps),np.int32(nrats),
#                                 np.int32(nrots),np.int32(nA),
#                                 np.int32(ndef)), dtype='float32')
#    rmeans_gpu = gpuarray.to_gpu(rmeans)    
#    corr(rmeans_gpu,defs_gpu,As_gpu,nA,np.int32(startr),
#         np.float32(psize), np.float32(lam),np.float32(cs),
#         np.float32(blend), np.int32(ps_size),
#         np.int32(maxr),corrs_gpu,grid=grid,block=block)
#    corrs = corrs_gpu.get()
#    sg.release() 
#    return corrs                   

#def test_tilted_ctf():
#
#
#    #%% Testing tilted situation ############################################
#    #sz   = mv.shape()[-2:]
#    #xpos = np.arange(0,sz[0]-win_sz[0]//2,sz[0]/10)
#    #ypos = np.arange(0,sz[1]-win_sz[1]//2,sz[1]/10)
#    #tilt = np.float32(unit_vector([0,0.5,1]))
#    #ps = ctf.tilted_ps4test(win_sz,xpos,ypos,tilt,-2e4,1.0,0.9,0,psize,lam,cs,blend)
#    
#    ps,xpos,ypos = ctf.power_spectrum(mv,drft,win_sz,ovlp,
#                                      periodograms=True,max_mem_GB=4)
#    
#    #%%
#    tilt_axis = 'unknown'
#    
#    #rats   = np.float32(np.linspace(min_rat,max_rat,nrats))
#    #rots   = np.arange(nrots,dtype='float32')*np.pi/(2*nrots)
#    rats   = np.float32(1.0)[None] #np.float32(np.linspace(min_rat,max_rat,nrats))
#    rots   = np.float32(0.0)[None] #np.arange(nrots,dtype='float32')*np.pi/(2*nrots)
#    As     = np.float32(np.linspace(0,maxA,nA))
#    #defs   = np.float32(np.linspace(dmin,dmax,n_defs))
#    #As     = np.float32(0)[None] #np.float32(np.linspace(0,maxA,nA))
#    
#    ######################
#    init_tilt = np.float32([0,0,1])            
#    xcent = (xpos[0]+xpos[-1])//2 
#    ycent = (ypos[0]+ypos[-1])//2
#    # calculate defocus range based on periodogram locations
#    xposp = np.float32(psize*(xpos - xcent))
#    yposp = np.float32(psize*(ypos - ycent))
#    [xposp, yposp] = image.ndgrid(xposp, yposp)
#    xposp,yposp = xposp.flatten(), yposp.flatten()
#    # obtain sphere segment pointing tao init_tilt    
#    points = image.sample_sphere(tilt_inc, max_tilt, init_tilt)
#    #points = np.concatenate((points,tilt[None,:]), axis=0)
#    
#    dcents   = init_tilt[0]*xposp + init_tilt[1]*yposp                       
#    #defocus offset for each tilt [npos x ntilts]
#    doffs    = np.float32(points[:,0]*xposp[:,None] + points[:,1]*yposp[:,None])
#    # determine defocus range for ctf sampling
#    mnoffs   = doffs.min()
#    mxoffs   = doffs.max()
#    # half defocus range
#    hdrange  = (dmax-dmin)/2
#    # half defocus value
#    hd       = (dmax+dmin)/2    
#    # fine defocus spacing - there should be space of > n_defs//2 
#    # before dcents.min() and after dcents.max()
#    ddef     = (hdrange+mxoffs-dcents.max())/(n_defs-n_defs//2+1)
#    ddef     = np.minimum(ddef, (hdrange+dcents.min()-mnoffs)/(n_defs//2+1))    
#    tot_defs = np.ceil((mxoffs-mnoffs+dmax-dmin)/ddef)
#    assert(tot_defs>=n_defs)    
#    cdefs    = hd + dcents
#    defs     = np.float32(np.linspace(dmin+mnoffs,dmax+mxoffs,tot_defs))
#    
#    pssz    = win_sz
#    nps     = np.prod(ps.shape[:-2])
#    mxr     = ctf.res2rad(end_res,pssz,psize)
#    # starting frequency for correlation
#    strtr   = np.ceil(ctf.res2rad(mid_res,pssz,psize))
#    # crop power spectrum to meet the maximum resolution
#    crop_sz = np.array([np.floor(mxr)*2]*2,dtype='int32')
#    psc     = image.crop2D(ps,crop_sz)
#    
#    zeros   = ctf.ctf_zeros(win_sz,(dmin+dmax)/2,0,psize,lam,cs,blend)[0]
#    # spatial low-pass res for background removal based on first zeros spacing
#    zspace  = zeros[2]-zeros[1] 
#    
#    # use gpu to create polar sums subject to astig distortions
#    rmeans  = calc_polsum((psc),rats,rots)
#    rmeans  = image.background_remove1D(rmeans,zspace)
#    rmeans  = (ctf.corr_peak_normalize(rmeans) + 1.0)/2.0
#    
#    corrs   = ctf.model_corr(rmeans,defs,As,strtr,psize,lam,cs,blend,pssz[0])
#    
#    #costs ~ n_ps x n_rat x n_rot x n_A x tot_defs
#    #select best tilt and defocus
#    sdefs   = np.linspace(dmin,dmax,np.ceil((dmax-dmin)/ddef))
#    #sdefs   = np.float32(sdefs)
#    
#    t_corrs = ctf.wins2tilts_gpu(corrs,doffs,np.float32(sdefs))/nps    
#    
#    mcorr   = t_corrs.max()           
#    [di,ti,rati,roti,Ai] = np.unravel_index(t_corrs.argmax(), t_corrs.shape)
#    tprint('def=%.2f, tilt=[%s], rat=%.2f, rot=%.2f, A=%.3f, corr=%.4f' % 
#           (sdefs[di],array2str(points[ti,:],'%.4f'),rats[rati],rots[roti],As[Ai],mcorr))
#    tilt = points[ti,:]    
#    d    = sdefs[di]
#    
#    clf()
#    im = ctf.combined_ps(ps,0.5,d,0,1,0,psize,lam,cs,blend)
#    imshow(im)
#    
#def test_zero_tilt():
#    #%% Testing zero tilt situation
#    #ps = ctf.ctf_2d(win_sz, -1.2e4, np.pi/4, 1.02, 0, psize, lam, cs, blend)
#    #ps = np.fft.fftshift(np.abs(ps))[None,:,:]
#    
#    ps,xpos,ypos = ctf.power_spectrum(mv,drft,win_sz,ovlp,
#                                      periodograms=False,max_mem_GB=4)
#    
#    rats   = np.float32(np.linspace(min_rat,max_rat,nrats))
#    rots   = np.arange(nrots,dtype='float32')*np.pi/(2*nrots)
#    
#    defs   = np.float32(np.linspace(dmin,dmax,n_defs))
#    As     = np.float32(np.linspace(0,maxA,nA))
#    
#    d,ra,ro,A,corr = ctf.estim_def_zero_tilt_new(ps,psize,cs,lam,end_res,defs, 
#                                rats,rots,As,mid_res,blend)
#                                
#    clf()
#    im = ctf.combined_ps(ps,0.5,d,0,1,0,psize,lam,cs,blend)
#    imshow(im)
#    
#    #ctf.estim_def_zero_tilt(ps,psize,cs,lam,end_res,dmin,dmax, 
#    #                        n_defs,rats,rots,As,init_ring,blend);              
    
        
#            function lambda_Angs=kev2Lam(AccelVolt_keV)            
#            m0=9.109e-31 ; %[kg]                  (Electron mass      )
#            e =1.602e-19 ; %[Cb]                  (Electron charge    )
#            c =2.998e+8  ; %[m/sec]               (Speed of light     )
#            h =6.626e-34 ; %[kg*m^2/sec]          (Planck's constant  )
#            
#            V=1e3*AccelVolt_keV; %[Volt]          (Acceleraton voltage)
#            
#            lambda=h/sqrt(2*m0*e*V*(1+e*V/(2*m0*c^2))); % [m]
#            
#            lambda_Angs=lambda*1e10;
#        end % kev2Lam
    
#        function I = invPhase2DExt(I, ctfFFT)
#            I = real(ifftn(fftn(I).*sign(ctfFFT)));
#        end % invPhase2DExt     