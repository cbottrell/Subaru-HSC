#!/usr/bin/env python
import numpy as np
import glob
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import scipy.ndimage
import random

cat_header = '# SIMTAG SUBTAG ISNAP CAMERA\n'
targetLen = 10000
datagen = ImageDataGenerator(   
        rotation_range=180,
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode='nearest',
        vertical_flip =True,
        zoom_range = 0.1,
        horizontal_flip =True,
        data_format= 'channels_first')

'''Each image in the input array gets an integer number of
augmentations.'''

for label in ['StellarMap','StellarMap_SemiReal','StellarMap_FullReal',
              'Photometry','Photometry_SemiReal','Photometry_FullReal']:
    
    for classID in ['Iso','Pair','Post']:
        
        nChannels = (1 if 'StellarMap' in label else 3)
        # data file name
        fileName = '/home/bottrell/scratch/RealCNN/MultiClass/Data/{}-{}-Inp.npy'.format(classID,label)
        fileOut = fileName.replace('Inp','InpAug')
        if os.access(fileOut,0): continue
        # corresponding catalog file name
        catName = '/home/bottrell/scratch/RealCNN/MultiClass/Data/{}-{}.cat'.format(classID,label)
        catOut = catName.replace('.cat','_catAug.npy')
        x = np.load(fileName)
        c = np.loadtxt(catName,dtype=str,delimiter=' ')
        # data must have shape (nRecords,nChannels,dx,dy) for augment
        dx = dy = int(np.sqrt(x.shape[-1]))
        x = x.reshape(-1,nChannels,dx,dy)
        # Number of records
        inputLen = x.shape[0]
        # number of augmentations needed to roughly acheive target
        n_augm = 0
        # iteratively increase until output meets target size
        while ((n_augm+1) * inputLen) < targetLen: n_augm+=1
        if n_augm<1: 
            print('Input already exceeds target size.')
            sys.exit(0)

        fileOut = fileName.replace('Inp','InpAug')
        catOut = catName.replace('.cat','_catAug.npy')
        # generate empty augmented data array
        x_augm = np.empty((n_augm*x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        c_augm = np.empty((n_augm*inputLen,4),dtype=object)
        
        # fill augmented data array
        for img_i in range(inputLen):
            for augm_i in range(n_augm): 
                x_augm[img_i+augm_i*inputLen] = datagen.flow(x[img_i].reshape((1,nChannels,dx,dy)), batch_size=1)[0]
                c_augm[img_i+augm_i*inputLen] = c[img_i]

        # reshape inputs
        x = x.reshape(-1,nChannels,dx*dy)
        # reshape augmented to flatten images
        x_augm = x_augm.reshape(-1,nChannels,dx*dy)
        # save to output file
        np.save(fileOut,np.concatenate([x,x_augm],axis=0))
        np.save(catOut,np.concatenate([c,c_augm],axis=0))
        print('Finished with file: {}'.format(fileOut))