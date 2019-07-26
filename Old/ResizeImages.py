#!/usr/bin/env python

from astropy.io import fits
from scipy import stats
import numpy as np
import os,sys,glob

output_shape = (139,139)
input_shape = (512,512)


def rebin(array, dimensions=None, scale=None):

    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(array.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(array.shape)
        elif len(dimensions) != len(array.shape):
            raise RuntimeError('')
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = map(int, map(round, map(lambda x: x*scale, array.shape)))
        elif len(scale) != len(array.shape):
            raise RuntimeError('')
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\trebin(array, dimensions=(x,y))\n\trebin(array, scale=a')
    #print "Rebinning to Dimensions: %s, %s" % tuple(dimensions)
    import itertools
    dY, dX = map(divmod, map(float, array.shape), dimensions)
 
    result = np.zeros(dimensions)
    for j, i in itertools.product(*map(range, array.shape)):
        (J, dj), (I, di) = divmod(j*dimensions[0], array.shape[0]), divmod(i*dimensions[1], array.shape[1])
        (J1, dj1), (I1, di1) = divmod(j+1, array.shape[0]/float(dimensions[0])), divmod(i+1, array.shape[1]/float(dimensions[1]))
         
        # Moving to new bin
        # Is this a discrete bin?
        dx,dy=0,0
        if (I1-I == 0) | ((I1-I == 1) & (di1==0)):
            dx = 1
        else:
            dx=1-di1
        if (J1-J == 0) | ((J1-J == 1) & (dj1==0)):
            dy=1
        else:
            dy=1-dj1
        # Prevent it from allocating outide the array
        I_=min(dimensions[1]-1,I+1)
        J_=min(dimensions[0]-1,J+1)
        result[J, I] += array[j,i]*dx*dy
        result[J_, I] += array[j,i]*(1-dy)*dx
        result[J, I_] += array[j,i]*dy*(1-dx)
        result[J_, I_] += array[j,i]*(1-dx)*(1-dy)
    allowError = 0.1
    assert (array.sum() < result.sum() * (1+allowError)) & (array.sum() >result.sum() * (1-allowError))
    return result


wdir = '/home/bottrell/scratch/RealCNN/'
os.chdir(wdir)
Fire_path = '/home/bottrell/scratch/Fire/'
bands = ['g','r','i']

for label in ['Photometry','StellarMap']:

    if label is 'StellarMap':
        file_list = glob.glob(Fire_path+'StellarMap/*/*/*.npz')
        for filename in file_list:
            if 'Iso' in filename:
                output_dir = wdir+'BinaryClass/{}/Iso/'.format(label)
                outfile = output_dir+filename.split('/')[-1].replace('.npz','.fits')
            else:
                output_dir = wdir+'BinaryClass/{}/Int/'.format(label)
                outfile = output_dir+filename.split('/')[-1].replace('.npz','.fits')
            if os.access(outfile,0): continue
            f = np.load(filename,'r')
            # flux conserved but still a linear unit which can be easily scaled
            img_data = rebin(f['stellarmap'],output_shape)
            params = f['params'][()]
            f.close()
            # create fits header
            hdr = fits.Header()
            for key in params.keys():
                if len(str(params[key]))<25:
                    hdr.append((key,params[key]))
            fits.writeto(outfile,data=img_data,header=hdr)
            
    if label is 'Photometry':
        file_list = glob.glob(Fire_path+'Photometry/*/*/photo_r*.fits')
        for filename in file_list:
            if 'Iso' in filename:
                output_dir = wdir+'BinaryClass/{}/Iso/'.format(label)
                outfile = output_dir+filename.split('/')[-1]
            else:
                output_dir = wdir+'BinaryClass/{}/Int/'.format(label)
                outfile = output_dir+filename.split('/')[-1]
            for band in bands:
                filename_x = filename.replace('photo_r','photo_{}'.format(band))
                outfile_x = outfile.replace('photo_r','photo_{}'.format(band))
                if os.access(outfile_x,0): continue
                hdr = fits.getheader(filename_x)
                img_data = 10**(-0.4*(fits.getdata(filename_x)-22.5)) # nanomaggies/arcsec2
                # conserve surface brightness (correction to flux conserved rebin function)
                img_data = rebin(img_data,output_shape)*(float(output_shape[0])/input_shape[0])**2
                hdr['NAXIS1']=output_shape[1]
                hdr['NAXIS2']=output_shape[0]
                hdr['BUNIT'] = 'AB nanomaggies/arcsec2'
                fits.writeto(outfile_x,data=img_data,header=hdr)
        
