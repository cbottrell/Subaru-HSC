#!/usr/bin/env python

#!/usr/bin/env python

import os,sys,json
import keras
from keras import backend as K
from keras.models import Model,load_model
from keras.layers import Input,Dense,Dropout,Flatten
from keras.utils import plot_model
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from sklearn import model_selection as md
from keras import regularizers
import numpy as np

def save_model_history(model,history_filename):
    with open(history_filename,'w') as f:
        json.dump(model.history.history, f)
        
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
    x = Dropout(0.5,name='DropFCL_1')(x)
    x = Dense(16,activation='relu',name='Dense_2')(x)
    x = Dropout(0.5,name='DropFCL_2')(x)
    x = Dense(1,activation='sigmoid',name='Dense_3')(x)

    # connect and compile
    model = Model(inputs=inputs,outputs=x)
    model.compile(optimizer=keras.optimizers.Adadelta(), 
                  loss='binary_crossentropy',metrics=['accuracy'])
    return model
######################################################################

def split_data(inp,tar,cat,n_negOrig,n_neg,n_posOrig,n_pos,
               train_valid_split=0.3,valid_test_split=0.5,randomState=0):
    
    f_negAug = int(n_neg/n_negOrig)-1
    f_posAug = int(n_pos/n_posOrig)-1

    # indices of the original pre-augmentation images
    indices_neg0 = np.arange(n_negOrig).astype(int)
    indices_pos0 = np.arange(n_posOrig).astype(int)+n_neg

    # this randomstate ensures we would have the same images if we did this again
    randomState = 0
    indices_neg_train0,indices_neg_valid0 = md.train_test_split(indices_neg0,test_size=train_valid_split, random_state=randomState)
    indices_neg_valid0,indices_neg_test0  = md.train_test_split(indices_neg_valid0,test_size=valid_test_split, random_state=randomState)
    indices_pos_train0,indices_pos_valid0 = md.train_test_split(indices_pos0,test_size=train_valid_split, random_state=randomState)
    indices_pos_valid0,indices_pos_test0  = md.train_test_split(indices_pos_valid0,test_size=valid_test_split, random_state=randomState)

    indices_neg_train = np.array([]).astype(int)
    indices_neg_valid = np.array([]).astype(int)
    indices_neg_test = np.array([]).astype(int)
    indices_pos_train = np.array([]).astype(int)
    indices_pos_valid = np.array([]).astype(int)
    indices_pos_test = np.array([]).astype(int)

    # now incorporate augmented images into each dataset
    for i in range(f_negAug+1):
        indices_neg_train = np.concatenate([indices_neg_train,indices_neg_train0+i*n_negOrig])
        indices_neg_valid = np.concatenate([indices_neg_valid,indices_neg_valid0+i*n_negOrig])
        indices_neg_test = np.concatenate([indices_neg_test,indices_neg_test0+i*n_negOrig])
    for i in range(f_posAug+1):
        indices_pos_train = np.concatenate([indices_pos_train,indices_pos_train0+i*n_posOrig])
        indices_pos_valid = np.concatenate([indices_pos_valid,indices_pos_valid0+i*n_posOrig])
        indices_pos_test = np.concatenate([indices_pos_test,indices_pos_test0+i*n_posOrig])
    print('Negatives:', len(indices_neg_train),len(indices_neg_valid),len(indices_neg_test))
    print('Positives:', len(indices_pos_train),len(indices_pos_valid),len(indices_pos_test))

    # join
    indices_train = np.concatenate([indices_neg_train,indices_pos_train])
    indices_valid = np.concatenate([indices_neg_valid,indices_pos_valid])
    indices_test = np.concatenate([indices_neg_test,indices_pos_test])

    # # Now apply to full data matrix
    tar_train = tar[indices_train]
    inp_train = inp[indices_train]
    cat_train = cat[indices_train]
    tar_valid = tar[indices_valid]
    inp_valid = inp[indices_valid]
    cat_valid = cat[indices_valid]
    tar_test = tar[indices_test]
    inp_test = inp[indices_test]
    cat_test = cat[indices_test]
    return inp_train,tar_train,cat_train,inp_valid,tar_valid,cat_valid,inp_test,tar_test,cat_test



classIDs = ['Negative','Positive']
modelDir = '/home/bottrell/scratch/Subaru/HyperSuprime/Models/Binary/'
dataDir = '/home/bottrell/scratch/Subaru/HyperSuprime/Data/Binary/'
fileNames = ['{}{}_i_Images-RZSAugNorm.npy'.format(dataDir,classID) for classID in classIDs]
catNames = ['{}{}_i_Images-RZSAug_cat.npy'.format(dataDir,classID) for classID in classIDs]

np.random.seed(0)
negative_subsample = 10000

tar = np.array([])
inp = np.array([])
# Negatives
#indices_neg = np.random.choice(np.arange(negative_total),negative_samples,replace=False)
inp_neg = np.load(fileNames[0])[:negative_subsample] # [indices_neg]
print('Negatives:',inp_neg.shape)
n_neg = inp_neg.shape[0]
n_negOrig = 5000
f_negAug = int(n_neg/n_negOrig)-1
print('Number of augmentations per original image:', f_negAug)
tar_neg = np.append(tar,np.ones(n_neg)*0)
cat_neg = np.load(catNames[0])[:negative_subsample]


# Positives
inp_pos = np.load(fileNames[1])
print('Positives:',inp_pos.shape)
n_pos = inp_pos.shape[0]
n_posOrig = 1201
f_posAug = int(n_pos/n_posOrig)-1
print('Number of augmentations per original image:', f_posAug)
tar_pos = np.ones(n_pos)
cat_pos = np.load(catNames[1])
# Combining
inp = np.concatenate([inp_neg,inp_pos],axis=0).reshape(-1,128,128,1)
tar = np.append(tar_neg,tar_pos)
cat = np.concatenate([cat_neg,cat_pos],axis=0)


nRecords,dx,dy,nChannels = inp.shape
nClasses=len(classIDs)
#del new_inp,new_cat

batch_size = 32
epochs=30
train_valid_split = 0.3
valid_test_split = 0.5
update = 0
version = 3
patience = 999
label = 'HSC-Subaru'
          
randomStates = np.arange(10)

for randomState in randomStates:

    modelFile = modelDir+'{}_Binary_RS-{}_v{}_{}.h5'
    historyFile = modelDir+'{}_Binary_RS-{}_v{}_{}_history.json'
    if os.access(modelFile.format(label,randomState,version,update),0): continue

    modelFile = modelFile.format(label,randomState,version,update)
    print('Model will save as: {}'.format(modelFile))
    historyFile = historyFile.format(label,randomState,version,update)
    print('History will save as: {}'.format(historyFile))
    
    mcp_file = modelFile.replace('.h5','_wgts.hdf5')
    early_stopping = EarlyStopping(monitor='val_acc', patience=patience, verbose=1, mode='max')
    mcp_save = ModelCheckpoint(mcp_file, save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max')

    args = (dx,dy,nChannels)
    model = HSC_Subaru_CNN(args)
    
    inp_train,tar_train,cat_train,inp_valid,tar_valid,cat_valid,inp_test,tar_test,cat_test = split_data(inp,tar,cat,
                                                                                                       5000,10000,1201,9608,randomState=randomState)
    
    model.summary()

    model.fit(inp_train,tar_train,shuffle=True,epochs=epochs,
              batch_size=batch_size,validation_data=[inp_valid,tar_valid],
              callbacks=[early_stopping, mcp_save]) 
    
    save_model_history(model=model,history_filename=historyFile)
    if os.access(mcp_file,0):
        model.load_weights(mcp_file)
        os.remove(mcp_file)
    model.save(modelFile)

