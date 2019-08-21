# streakanalysis.utils A newer set of streak utilities in the PK Lab
"""
Created 12/09/2018
Streak utilities for use with the new streak GUI

Mostly functions for streak data analysis.
This module will contain functions used in post processing and data analysis process, including beam profile analysis.

@author: Harry Baker
Supported Files for reading: .txt and .npy
Supported Files for writing: .txt
"""
#All imports

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import lmfit
from lmfit.models import GaussianModel, VoigtModel
from collections import OrderedDict
from scipy.constants import h, c, N_A, eV, nano
from scipy import ndimage
from scipy.ndimage import median_filter

## Set of functions for beam profiles ##

def load_check(path,im_bk,im_beam):
    """
    Small function to quickly load tiff images of beam profile and to check they're loaded correctly
    Parameters:
        Path of images
        im_bk: background image file name
        im_beam: beam image file name
    Returns:
        bk: (N,M) 2D Numpy array of background
        beam: (N,M) 2D Numpy array of beam
    """
    bk = plt.imread(path+im_bk)
    beam = plt.imread(path+im_beam)
    plt.figure()
    plt.pcolormesh(bk[:,:,0])
    plt.show()
    plt.figure()
    plt.pcolormesh(beam[:,:,0])
    plt.show()
    return bk,beam

def bk_sub(beam,bk):
    """
    Background subtraction function. Also attaches pixel values to axes and checks through plotting.
    Returns:
        bk_sub: (N,M) sized 2D Numpy array of background free beam image
        im: (N,M) sized 2D Numpy array of background free beam image with axes attached
        x: M sized 1D Numpy array of X pixel values
        y: N sized 1D Numpy array of Y pixel values
    """
    bk_sub= beam[:,:,0]-bk[:,:,0]
    x = np.linspace(0,1280,len(bk_sub[0,:]))
    y = np.linspace(0,1024,len(bk_sub[:,0]))
    plt.figure()
    plt.pcolormesh(x,y,bk_sub)
    plt.show()
    im = np.zeros((y.size+1,x.size+1))*np.nan
    print(im.shape,bk_sub.shape)
    im[1:,1:]=bk_sub
    im[1:,0]=y
    im[0,1:]=x
    plt.figure()
    plt.pcolormesh(im)
    plt.show()
    return bk_sub,im,x,y

def crop_im(x,y,im,ux,lx,uy,ly):
    """
   Function to crop useless space from im
   Parameters:
        x: x axis to be cropped
        y: y axis to be cropped
        im: beam image to be cropped
        ux: Upper x value to crop below
        lx: Lower x value to crop above
        uy,ly: Same as above but for y axis.
    Returns:
        x_crop: Cropped x axis
        y_crop: Cropped y axis
        im_crop: Cropped beam image
    """
    x_mask = (x>lx)&(x<ux)
    y_mask = (y>ly)&(y<uy)
    x_crop = x.compress(x_mask)
    y_crop = y.compress(y_mask)
    im_crop = im.compress(x_mask,axis=1).compress(y_mask,axis=0)
    plt.figure()
    plt.pcolormesh(x_crop,y_crop,im_crop)
    plt.show()
    return x_crop,y_crop,im_crop

def fit_fwhm(im_crop,x_crop,y_crop,x,y):
    """
    Function that integrates the cropped image along x and y directions.
    It then fits gaussians to retrieve a beam size in x/y directions. This is then converted from pixels to micrometers.

    Parameters:
        im_crop: Cropped beam image
        x_crop: Cropped x axis
        y_crop: Cropped y axis
        x = guess for x center
        y: guess for y center

    Returns:
        size_x: Size of beam in x direction (micrometers)
        size_y: Size of beam in y direction (micrometers)

    """
    int_x = np.sum(im_crop,axis=0)-np.min(np.sum(im_crop,axis=0))
    int_y = np.sum(im_crop,axis=1)-np.min(np.sum(im_crop,axis=1))

    plt.figure()
    plt.plot(x_crop,int_x)
    plt.plot(y_crop,int_y)
    plt.show()

    gauss_x = GaussianModel(prefix='x_')
    gauss_y=GaussianModel(prefix='y_')
    pars_x = gauss_x.make_params()
    pars_y = gauss_y.make_params()


    pars_x.update(gauss_x.guess(int_x,x=x_crop))
    pars_y.update(gauss_y.guess(int_y,x=y_crop))
    pars_x['x_center'].set(x,min=x-100,max=x+100)
    pars_y['y_center'].set(y,min=y-100,max=y+100)

    out_x=gauss_x.fit(int_x,pars_x,x=x_crop)
    out_y = gauss_y.fit(int_y,pars_y,x=y_crop)
    plt.figure()
    plt.plot(x_crop,int_x,color='r',label='Data')
    plt.plot(x_crop,out_x.best_fit,color='b',label='Fit')
    plt.xlim(out_x.values['x_center']-50,out_x.values['x_center']+50)
    plt.legend()
    print(out_x.fit_report())
    plt.show()
    plt.figure()
    plt.plot(y_crop,int_y,color='r',label='Data')
    plt.plot(y_crop,out_y.best_fit,color='b',label='Fit')
    plt.xlim(out_y.values['y_center']-50,out_y.values['y_center']+50)
    plt.legend()
    print(out_y.fit_report())
    plt.show()

    pix = 5.3  # Units of micrometers/pixel
    fwhm_x = out_x.best_values['x_sigma']*2.3548200
    fwhm_y = out_y.best_values['y_sigma']*2.3548200
    size_x = fwhm_x * pix
    size_y = fwhm_y *pix
    return size_x,size_y

def beam_params(size_x,size_y,powers,ramp,wl):
    """
    Function that calculates the beam waist, spot size in cm^2, energy required (in J) for the desired fluences and the number of photons at each fluence.
    Params:
    size_x: Size of beam in x direction (micrometers)
    size_y: Size of beam in y direction (micrometers)
    powers: List of experimental powers (micro W /cm^2)
    ramp: Time of an experiment (10ns,50ns,100ns, etc...) this is used to convert W to J
    wl: Wavelength of pump beam

    Returns:
        fluences: Array of fluences in J cm^-2
        n_photon: Number of photons per cm^2
        radius: Beam waist in m
    """
    beam_diam = np.sqrt(size_x*size_y)*10**-6 # diameter in m
    radius = beam_diam/2 #radius in m
    area_m2 = np.pi*(radius*radius)  #area in m^2
    area_c2 = area_m2*1e4 #area in cm^2

    powers = np.asarray(powers)/1000  # Converts micro W to nJ/pulse
    fluences = ramp*(powers *10**-9)/area_c2 #Converts from nJ/pulse  to J cm^-2


    e_pump = h*(c/wl*nano) ## Pump energy in J
    n_photon = fluences/e_pump # Converts fluence to number of photons per cm^2
    return fluences,n_photon,radius

def poisson(j,b,sigma):
    """Poisson distribution for fluorescence saturation
    Params:
    j: Photon flux, obtained from beam_params function (n_photons)
    b: Fluorescence saturation magnitude, guessed at the value where intensity "flattens off" in plots of I vs n_photons on a log scale
    sigma: Absorption cross section of the sample, this must be obtained through fitting to the model (following function)
    """
    poisson = b*(1-(np.exp(-j*sigma)))
    return poisson

## General Functions for manipulating Streak traces ##

def sort_trace(lof,powers):
    """Function that sorts traces organised with random pump powers
    Params:
    lof: List of files
    powers: List of powers in randomised order

    Returns:
        lof_s: Dictionary of sorted list of files, in order of lowest to highest power"""
    inds = sorted(zip(powers,range(len(powers))))
    s_inds = [s[1] for s in inds]

    lof_s ={}
    powers = sorted(powers)
    for i, k in zip(s_inds,powers):
        lof_s[k] = lof[i]
    return lof_s

def unpack_trace(data,tr =False):
    """
    This function unpacks an (N,M) array into wl,t and the intensity trace

    Parameters:
        data: (N,M) 2D array in .txt format
        tr: Boolean, if True return only trace, default is False
    Returns:
        t: 1D array of length M giving time values
        wl: 1D array of length N giving wavelength values
        trace: (N,M) 2D array of intensity data
    """
    t = data[0,1:]
    wl = data[1:,0]
    trace = data[1:,1:]
    if tr == True:
        return trace
    else:
        return t,wl,trace

def pack_trace(t,wl,trace):
    """
    Function that repacks processed trace as a (N,M) 2D array
    TO DO: Add a way to save the trace as well

    Parameters:
    t: 1D array of length M containing time values
    wl: 1D array of length N containing wavelength values
    trace: (N,M) 2D array of intensity values

    Returns:
        packed: Packed trace in the form of a (N,M) 2D numpy array
        saved: Packed trace in the form of a .txt file
    """
    packed = np.empty([i+1 for i in trace.shape])
    packed.fill(np.nan)
    packed[1:,1:]=trace
    packed[0,1:]=t
    packed[1:,0]=wl
    return packed

def jacobian(wl,trace):
    """
    This function converts units of wavelength to energy. It also performs the jacobian transform on the trace to account for non-linearities

    Parameters:
         wl: Wavelength values to be transformed to energy values
         trace: Trace to be corrected with jacobian
    Returns:
        e: Energies obtained from unit conversion
        js: Jacobian corrected spectra
    """
    e = (c*h/(wl*nano))/eV
    js= trace.T*(h*c/(e**2))
    return e,js

## Functions used in post processing of data ##

def downsample_rebin(packed,t_bin,wl_bin):
    """
    Downsample a packed trace by rebinning. Axes are reshaped (t/wl)

    Parameters:
        packed: Packed trace to be rebinned
        t_bin: Rebinning factor for time axis
        wl_bin: rebinning factor for wavelength axis

    Returns:
        rebinned: Reshaped (N,M) 2D array containing new time and wavelength values as well as the rebinned trace
    """
    t,wl,trace = unpack_trace(packed)
    n_t=t.size
    n_wl=wl.size
    reshaped = trace.reshape((n_wl//wl_bin,wl_bin,n_t//t_bin,t_bin))
    rebin = reshaped.sum(axis=(1,3))
    wl = wl.reshape((n_wl//wl_bin,wl_bin)).mean(axis=1)
    t = t.reshape((n_t//t_bin,t_bin)).mean(axis=1)
    rebinned = pack_trace(t,wl,rebin)
    return rebinned

def check_data(packed,t_bin,wl_bin):
    """
    This function quickly loads, rebins and plots the raw data to check for post processing.
    Parameters:
        packed: raw streak data to be post processed
        t_bin: time rebining factor
        wl_bin: Wavelength rebining factor
    Returns:
        A plot of the rebinned raw data and the rebinned trace
    """
    t,wl,trace = unpack_trace(downsample_rebin(packed,t_bin,wl_bin))
    print(np.diff(t))
    plt.figure()
    plt.pcolormesh(t,wl,trace,vmin=np.percentile(trace,50),vmax=np.percentile(trace,99))
    return trace


def micro_mask(name,t):
    """
    Function that adds an extra mask to t for the 1us time range!
    Parameters:
        name: filename of trace being processed
        t: time values
    """
    time_mask = slice(50,-490)
    if any([s in name for s in ['1us','1000ns']]):
        start,stop,step = time_mask.start,time_mask.stop,time_mask.step
        stop = t.size+stop if stop<0 else stop
        stop = min([stop,1690])
        time_mask = slice(start,stop,step)
        return time_mask

def shear_shift(trace,t,wl,tu,tl,wlu,wll):
    """
    This function uses the affine transform to correct for shear on the CCD in the streak camera.
    It then reasserts bin widths and corrects for t0

    Parameters:
        trace: Trace to be corrected
        t: Time values
        wl: Wavelength values
        tu: Upper time value to search for IRF within
        tl: Lower time value to search for IRF within
        wlu/wll: Same as above but for wavelength bounds (note, name is to avoid confusion later)
    Returns:
        trace: Shear corrected trace
        irf: Instrument response function
        t0: Actual value of t0 to correct for
        t: Corrected time values such that trace starts at/around t=0
    """
    #This part corrects for shear
    shear_m = np.array([[1,0],[-1E-2,1]])
    c_out = 0.5*np.array(trace.shape)
    c_int = 0.5*np.array(trace.shape)
    offset =c_int-c_out.dot(shear_m)
    trace = ndimage.affine_transform(trace,shear_m.T,offset=offset)

    #This part then corrects for bin widths and t0
    dt = t[1:]-t[:-1]
    assert all (dt>0)
    new_dt = np.zeros_like(t)
    new_dt[:-1]=dt
    new_dt[-1]=dt[-1]
    dwl=wl[1:]-wl[:-1]
    new_dwl=np.zeros_like(wl)
    new_dwl[:-1]=-dwl
    new_dwl[-1]=-dwl[-1]
    trace/=new_dt[np.newaxis,:]
    trace/=new_dwl[:,np.newaxis]
    irf = trace[:,(t>tl)&(t<tu)][(wl>wll)&(wl<wlu),:].sum(axis=0)
    max_pix=np.argmax(irf)
    t0 = t[max_pix]
    t-= t0
    print('[IRF]',irf.shape)
    print('[t0]',t0)
    return trace, irf, t0, t

def shifts(t_bin,irf,t,t_mask,wl,wl_mask,trace):
    """
    Shifts the peak value as close to t0 as possible
    Parameters:
    t_bin: Rebining factor for t
    irf: Instrument response function as an array
    t: Time values to shift around
    t_mask: Range of times to search for irf
    wl: Wavelength values to shift around
    wl_mask: Range of wavelengths to search for irf
    trace: trace to apply shifts to

    Returns:
        t: Shifted time values
        wl: Shifted wavelength values
        trace: Shifted intensity values
    """
    n = t_bin
    shifts =[]
    for i in range(n):
        print(i)
        irf_s = np.sum(irf[i:(-n+i)].reshape((-1,n)),axis=0)
        loc = np.argmax(irf_s)
        t_s= np.mean(t[i:(-n+i)].reshape((-1,n)),axis=0)
        shifts.append(t_s[loc])
    shift = np.argmin(np.abs(shifts))
    t_mask=slice(t_mask.start+shift,t_mask.stop-n+shift,t_mask.step)
    t = t[t_mask]
    wl=wl[wl_mask]
    trace= trace[wl_mask,t_mask]
    return t,wl,trace

## These functions pertain purely to data analysis##

def plot_trace(t,e,trace,el,eu,tl,tu,linthresh,linscale,vmin,vmax,cmap):
    """
    This function quickly plots a Pcolormesh of the streak trace

    Parameters:
         t: Time values to plot (ns)
         e: Energy values to plot (eV)
         el: Lower e value in range to search for max (for normalisation purposes)
         eu: Upper e value in range to search for max (for normalisation)
         tl/tu: Same as above two but in time
         linthresh: Linear threshold for pcolormesh z scale
         linscale: Linear scale for pcolormesh z scale
         vmin: Minimum value shown on z scale of pcolormesh plot
         vmax: Max value shown on z scale of pcolormesh plot
         cmap: Colourmap to be used
    Returns:
        Plot of data

    """
    plt.figure()
    lim=np.max(trace.T[(e>el)&(e<eu),:][:,(t>tl)&(t<tu)])
    plt.pcolormesh(t,e,trace.T/lim,norm=colors.SymLogNorm(linthresh=linthresh,linscale=linscale,vmin=vmin,vmax=vmax),cmap=str(cmap))


def spec_slices(w,ts,tf,t,trace):
    """
    This function slices streak traces in time to give spectral slices so that spectral dynamics can be analysed.
   Parameters:
        w: Width of slice in energy or wavelength
        ts: Starting time to slice at
        tf: final time to slice to, note this should be w/2 greater than the time you wish to reach
        t: Time values to slice
        trace: Trace to be slice

    Returns:
        slices: OrderedDictionary of spectral slices of whatever name specified
    """
    trace -= np.percentile(trace,50)
    slice_w =w
    slice_c=np.arange(ts,tf,w)
    #slice_c= np.around(slice_c,decimals=2)
    slices=OrderedDict()
    dt=t[1:]-t[-1:]
    for c in slice_c:
        mask= (t-c<slice_w)&(t-c>0)
        sl=np.compress(mask,trace,axis=1)*np.compress(mask,dt)[np.newaxis,:]
        slices[c]=np.compress(mask,trace,axis=1).sum(axis=1)
    return slices

def plot_slices(x,slices,xlab,ylab,xmin,xmax,ymin,ymax,title,out_fn,scale,save):
    """
    Quick function to plot slices contained in an orderded dictionary
     Parameters
     x: X axis values to plot. Can be energy or time depending on the slices
     slices: Slices in time (spectral) or in energy (kinetics) to be plotted
     xlab/ylab: axes labels as string
     xmin,xmax : min/max x values
     ymin, ymax: as above for y axis
     title: sets legend title (energy/wavelength or time)
     scale: Used to set log scale on y axis for transient plotting

     Returns:
     Plot of the slices

    """
    plt.figure()
    colors=cm.rainbow(np.linspace(0,1,num=len(slices)))
    for i, (k,v) in enumerate(slices.items()):
        plt.plot(x,v,c=colors[i],label=k,lw=1)
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend(title=title,ncol=2,fontsize='small',loc='best')
        if scale==True:
            plt.yscale('Log')
        plt.tight_layout()
        if save==True:
            plt.savefig(out_fn)

def norm_slicemax(x,max,slices):
    """
    Function that finds the max value of each slice and then normalises each slice to this max.
    This only works when there is one clear maximum

    Parameters:
    x: x values to search for max at
    slices: slices to look for max/to normalise

    Returns:
    norm: Ordered dictionary of normalised slices
    """
    norm_idx = np.argmin(np.abs(x-max))
    norm = OrderedDict((k,v/v[norm_idx]) for k,v in slices.items())
    return norm

def norm_max(slices):
    """
    This function normalises slices to the max value of the first slice.

    Parameters:
        slices: Slices to normalise

    Returns:
        norm: Slices normalised to max of first slice
    """
    keys = np.asarray(list(slices.keys()))
    maxes = []
    for i in keys:
        max = np.max(slices[i])
        maxes.append(max)
    maxes = np.asarray(maxes)
    max = np.max(maxes)
    norm = OrderedDict()
    for i in keys:
        norm[i]=slices[i]/max
    return norm

def voigt_fit(prefix,x,slice,c,vary):
    """
    This function fits a voigt to a spectral slice. Center value can be set to constant or floated, everything else is floated.

    Parameters:
    prefix: prefix for lmfit to distinguish variables during multiple fits
    x: x values to use in fit
    slice: slice to be fit
    c: center of voigt obtained from max value of the slice
    vary: Boolean, determines whether c is floated default is True
    Returns:
         out: lmfit fit output
    """
    model = VoigtModel(prefix=prefix)
    pars = model.guess(slice,x=x)
    pars[str(prefix)+'center'].set(c,vary=vary)
    out = model.fit(slice,pars,x=x)
    return out

def kin_slices(w,es,ef,e,trace,norm):
    """
     This function slices streak traces in energy to give kinetic slices so that dynamics can be analysed.
    Parameters:
         w: Width of slice in energy
         es: Starting energy to slice at
         ef: final energy to slice to, note this should be w/2 greater than the time you wish to reach
         e: Energy values to slice
         trace: Trace to be sliced
         norm: Boolean, if true normalises all slices to max value of each slice

     Returns:
         slices: OrderedDictionary of slices of whatever name specified
     """
    trace -= np.percentile(trace, 50)
    slice_w = w
    slice_c = np.arange(es, ef, w)
    slices = OrderedDict()
    de = e[1:] - e[-1:]
    for c in slice_c:
        mask = (e - c < slice_w) & (e - c > 0)
        sl = np.compress(mask, trace, axis=0) * np.compress(mask, de)[:,np.newaxis]
        slices[c] = np.compress(mask, trace, axis=0).sum(axis=0)
        if norm==True:
            slices[c]=slices[c]/np.max(slices[c])
    return slices

def despike_median(data, size, threshold=5):
    cutoff = np.std(data) * threshold
    filtered = median_filter(data, size=size)
    reject = np.abs(filtered-data) > cutoff
    despiked = np.copy(data)
    despiked[reject] = filtered[reject]
    return despiked




