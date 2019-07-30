#!/usr/bin/env python

import os,sys,time
import multiprocessing
import numpy as np
from glob import glob
from astropy.io import fits

def Generate_Files(args):
    
    fileName,catName,imgDir,classID,filterIDs = args

    if os.access(fileName,0):
        print('File already exists: {}\nDelete files and try again.'.format(fileName))
        
    elif os.access(catName,0):
        print('File already exists: {}\nDelete files and try again.'.format(catName))
    
    else:
        print('Making files:\nData: {} \nCatalogue: {}'.format(fileName,catName))

        # catalogue of input objIDs and sizes (need the size flags to know whether to keep galaxy)
        inCat = '/home/bottrell/scratch/Subaru/HyperSuprime/Catalogues/HSC-TF_all_2019-07-25.txt'
        # filters to go into catalog
        filterIDs = ['g','r','i','z','y']
        # number of corresponding channels
        nChannels = len(filterIDs)
        # input catalogue data
        inCatData = np.loadtxt(inCat,delimiter=',',dtype='str')
        # morphID (stream/shell/non)
        morphIDs = inCatData[:,5]
        # objIDs from catalog
        objIDs = inCatData[:,0].astype(int)
        # image dimensions
        dx = dy = 128

        if classID == 'Positive':
            indices = morphIDs!='non'
        if classID == 'Negative':
            indices = morphIDs=='non'
        objIDs = objIDs[indices]
        morphIDs = morphIDs[indices]

        # initialize numpy array with shape: (len(objIDs),dx,dy,nChannels)
        outData = np.empty(shape=(len(objIDs),dx,dy,nChannels),dtype=float)
        # initialize output catalogue (objID,(stream/shell/non),g_exists,r_exists,i_exists,z_exists,y_exists)
        outCat = np.empty(shape=(len(objIDs),7),dtype='<U32')

        for ii,objID in enumerate(objIDs):
            outCat[ii,0] = objID
            outCat[ii,1] = morphIDs[ii]
            for jj,filterID in enumerate(filterIDs):
                imgName = '{}{}_Cutout-Resized_{}.fits'.format(imgDir,objID,filterID)
                if not os.access(imgName,0):
                    outData[ii,:,:,jj] = np.zeros((dx,dy))
                    outCat[ii,jj+2] = 0
                else:
                    outData[ii,:,:,jj] = fits.getdata(imgName)
                    outCat[ii,jj+2] = 1

        np.save(fileName,outData)
        np.save(catName,outCat)

if __name__ == "__main__":
    
    # enviornment properties
    SLURM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    # class options
    classIDs = ['Positive','Negative']
    # filters to go into catalog
    filterIDs = ['g','r','i','z','y']
    # filter label for numpy array
    label = ''.join(filterIDs)
    # output filename holder
    _fileName_ = '/home/bottrell/scratch/Subaru/HyperSuprime/Data/Binary/{}_{}_Images.npy'
    # output catalogue name holder
    _catName_ = '/home/bottrell/scratch/Subaru/HyperSuprime/Data/Binary/{}_{}_Images_cat.npy'
    # input image directory
    imgDir = '/home/bottrell/scratch/Subaru/HyperSuprime/Data/Resized/'
    
    # pool tasks
    argList = [(_fileName_.format(classID,label),_catName_.format(classID,label),imgDir,classID,filterIDs) for classID in classIDs]
    pool = multiprocessing.Pool(SLURM_CPUS)
    pool.map(Generate_Files, argList)
    pool.close()
    pool.join()


