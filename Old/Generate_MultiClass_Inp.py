#!/usr/bin/env python

from astropy.io import fits
from scipy import stats
import numpy as np
import os,sys,glob
import matplotlib.pyplot as plt

wdir = '/home/bottrell/scratch/RealCNN/MultiClass/Data/'
if os.access(wdir,0): os.chdir(wdir)
else:
    print('Coult not access working directory: {}\n'.format(wdir))
    print('Exiting...')
    sys.exit(0)
    
input_shape = (139,139)
dx = input_shape[1]
dy = input_shape[0]
cat_header = '# SIMTAG SUBTAG ISNAP CAMERA\n'

for label in ['StellarMap','StellarMap_SemiReal','StellarMap_FullReal',
              'Photometry','Photometry_SemiReal','Photometry_FullReal']:
    
    for cid in ['Iso','Pair','Post']:
        
        input_dir = '/home/bottrell/scratch/RealCNN/Fire_Images_CNN/'+'{}/{}/'.format(label,cid)
        
        if 'StellarMap' in label:
            outname = '{}-{}-Inp.npy'.format(cid,label)
            # catalog of sim_tag, sub_tag, snapID, camera
            catname = '{}-{}.cat'.format(cid,label)
            if os.access(outname,0): continue
            file_list = glob.glob(input_dir+'*.fits')
            output = np.empty((len(file_list),dx*dy))
            with open(catname,'w') as c:
                c.write(cat_header)
                for i,file in enumerate(file_list):
                    data = fits.getdata(file)
                    output[i,:] = data.reshape((1,dx*dy))
                    hdr = fits.getheader(file)
                    sim_tag,sub_tag = hdr['sim_tag'],hdr['sub_tag']
                    filesplit = file.split('/')[-1].replace('_sci.fits','.fits').split('_')
                    isnap,camera = filesplit[-2],filesplit[-1].replace('.fits','')
                    c.write('{} {} {} {}\n'.format(sim_tag,sub_tag,isnap,camera))
            np.save(outname,output)
                
        if 'Photometry' in label:
            outname = '{}-{}-Inp.npy'.format(cid,label)
            # catalog of sim_tag, sub_tag, snapID, camera
            catname = '{}-{}.cat'.format(cid,label)
            if os.access(outname,0): continue
            file_list = glob.glob(input_dir+'photo_r*.fits')
            output = np.empty((len(file_list),3,dx*dy))
            with open(catname,'w') as c:
                c.write(cat_header)
                for i,file in enumerate(file_list):
                    data_g = fits.getdata(file.replace('photo_r','photo_g'))
                    data_r = fits.getdata(file)
                    data_i = fits.getdata(file.replace('photo_r','photo_i'))
                    output[i] = np.array([data_g,data_r,data_i]).reshape(3,dx*dy)
                    hdr = fits.getheader(file)
                    sim_tag,sub_tag,isnap,camera = [hdr[key] for key in ['simtag','subtag','isnap','camera']]
                    c.write('{} {} {} {}\n'.format(sim_tag,sub_tag,isnap,camera))
            np.save(outname,output)
            
