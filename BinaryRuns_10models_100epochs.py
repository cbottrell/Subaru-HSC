#!/usr/bin/env python

import os,sys,json
import keras
from keras import backend as K
from keras.models import Model,load_model
from keras.layers import Input,Dense,Dropout,Flatten
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from sklearn import model_selection as md
from keras import regularizers
import numpy as np
from collections import Counter

def save_model_history(model,history_filename):
    with open(history_filename,'w') as f:
        json.dump(model.history.history, f)
        
##############################################################

def balanced_class_weights(y):
   counter = Counter(y)
   majority = max(counter.values())
   return  {cls: float(majority/count) for cls, count in counter.items()}


#############################################################

######################################################################
def HSC_Subaru_CNN(args):
    
    dx,dy,nChannels = args
    
    # convolution model
    inputs = Input(shape=(dx,dy,nChannels), name='main_input')
    # first conv layer
    x = Conv2D(32, kernel_size=(5,5),activation='relu',
               padding='same',strides=(1, 1),name='Conv_1')(inputs)
    x = MaxPooling2D(pool_size=(2,2),name='MP_C1')(x)
    #x = Dropout(0.5,name='Drop_C1')(x)
    # second conv layer
    x = Conv2D(64, kernel_size=(3,3),activation='relu',
               padding='same',strides=(1, 1),name='Conv_2')(x)
    x = MaxPooling2D(pool_size=(2,2),name='MP_C2')(x)
    #x = Dropout(0.25,name='Drop_C2')(x)
    # third conv layer
    x = Conv2D(128, kernel_size=(2,2),activation='relu',
               padding='same',strides=(1, 1),name='Conv_3')(x)
    x = MaxPooling2D(pool_size=(2,2),name='MP_C3')(x)
    #x = Dropout(0.25,name='Drop_C3')(x)
    # fourth conv layer
    x = Conv2D(128, kernel_size=(3,3),activation='relu',
               padding='same',strides=(1, 1),name='Conv_4')(x)
    # x = Dropout(0.25,name='Drop_C4')(x)
    # flatten for fully connected layers
    x = Flatten(name='Flatten')(x)

    # Fully Connected Layer
    x = Dense(64,activation='relu',name='Dense_1')(x)
    x = Dropout(0.25,name='DropFCL_1')(x)
    x = Dense(16,activation='relu',name='Dense_2')(x)
    x = Dropout(0.25,name='DropFCL_2')(x)
    x = Dense(1,activation='sigmoid',name='Dense_3')(x)

    # connect and compile
    model = Model(inputs=inputs,outputs=x)
    model.compile(optimizer=keras.optimizers.Adadelta(), 
                  loss='binary_crossentropy',metrics=['accuracy'])
    return model
######################################################################

classIDs = ['Negative','Positive']
dataDir = '/home/bottrell/scratch/Subaru/HyperSuprime/Data/Binary/'
modelDir = '/home/bottrell/scratch/Subaru/HyperSuprime/Models/Binary/'
fileNames = ['{}{}_grizy_Images-AugNorm.npy'.format(dataDir,classID) for classID in classIDs]
catNames = ['{}Negative_grizy_Images_cat.npy'.format(dataDir),'{}Positive_grizy_Images-Aug_cat.npy'.format(dataDir)]

tar = np.array([])
inp = np.array([])
# Negatives
inp_neg = np.load(fileNames[0])[:,:,:,2]
inp_neg[np.isnan(inp_neg)]=0.
# inp_neg = np.zeros(shape=(20010,128,128)) # !!! for testing
cat_neg = np.load(catNames[0])
tar_neg = np.append(tar,np.ones(inp_neg.shape[0])*0)
# Positives
inp_pos = np.load(fileNames[1])[:1201,:,:,2]
#inp_pos = np.ones(shape=(20417,128,128)) # !!! for testing
# inp_pos[:,48:80,48:80] = 1. # !!! for testing
cat_pos = np.load(catNames[1])[:1201]
tar_pos = np.ones(inp_pos.shape[0])[:1201]
# Combining
inp = np.concatenate([inp_neg,inp_pos],axis=0)
cat = np.concatenate([cat_neg,cat_pos],axis=0)
tar = np.append(tar_neg,tar_pos)
# Reformatting
#tar_OH = keras.utils.to_categorical(tar)
inp = inp.reshape((*inp.shape,1))

nRecords,dx,dy,nChannels = inp.shape
nClasses=len(classIDs)
#del new_inp,new_cat

batch_size = 64
epochs=100
train_valid_split = 0.3
valid_test_split = 0.5
update = 0
version = 0
label = 'HSC-Subaru'
          
randomStates = [1,2,3,4,5,6,7,8,9,42]

for randomState in randomStates:

    modelFile = modelDir+'{}_Binary_RS-{}_v{}_{}.h5'
    historyFile = modelDir+'{}_Binary_RS-{}_v{}_{}_history.json'
    if os.access(modelFile.format(label,randomState,version,update),0): continue

    modelFile = modelFile.format(label,randomState,version,update)
    print('Model will save as: {}'.format(modelFile))
    historyFile = historyFile.format(label,randomState,version,update)
    print('History will save as: {}'.format(historyFile))

    # Split training and validation data
    inp_train, inp_valid, tar_train, tar_valid, cat_train, cat_valid = md.train_test_split(inp, tar, cat, test_size=train_valid_split, random_state=randomState)
    # Reserve some validation data as test data
    inp_valid, inp_test, tar_valid, tar_test, cat_valid, cat_test = md.train_test_split(inp_valid, tar_valid, cat_valid, test_size=valid_test_split, random_state=randomState)

    print(inp_train.shape)
    args = (dx,dy,nChannels)
    model = HSC_Subaru_CNN(args)
    
    model.summary()
    
    class_weight=balanced_class_weights(tar_train)
    
    print("Weights used:", class_weight)
    
    early_stopping = EarlyStopping(monitor='val_loss',patience=999,verbose=0,mode='auto')
    model.fit(inp_train,tar_train,shuffle=True,epochs=epochs,
              batch_size=batch_size,validation_data=[inp_valid,tar_valid],
              callbacks=[early_stopping],class_weight = class_weight) 
    
    model.save(modelFile)
    save_model_history(model=model,history_filename=historyFile)
