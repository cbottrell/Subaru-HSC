import json
import os
import string

import keras
import numpy as np
from astropy.io import fits
from keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# TODO: hyperparam search will eventually go here
hyperparam_ranges = dict(

)


# TODO: consider dictionary of dictionaries
hyperparams = dict(
    img_width = 128,
    img_height = 128,
    img_channels = 1,
    train_epochs = 30,
    batch_size = 32, # can this be bigger

    # cnn params
    n_classes = 2,
    conv_filters = [32, 64, 128, 128],
    fully_connected_neurons = [64, 16],
    acitvation = "relu",
    use_batch_norm = False,
    dropout_rate = 0.25,

    # tt kwargs
    train_ratio = 0.8,
    desired_pos_ratio = 0.25,
    negative_sample_usuage = 1.0,

    # augmentation params
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
)

def get_positive_samples():
    raise NotImplementedError("Connor's code here")

def get_negative_samples():
    raise NotImplementedError("Connor's code here")

def summary_to_file(model_id):

    def print_fn(s):
        with open(f"{model_id}/summary.txt", "a") as f:
            f.write(s + "\n")

    return print_fn

# https://pythontips.com/2013/07/28/generating-a-random-string/
def get_random_string(length):
    all_chars = string.ascii_letters + string.digits
    return ''.join([np.random.choice(all_chars) for _ in range(length)])

def HSC_Subaru_CNN(params):

    # inputs
    in_shape = params["img_width"], params["img_height"], params["img_channels"]
    inputs = layers.Input(shape=in_shape, name="main_input")

    # convs
    x = layers.Conv2D(params["conv_filters"][0],
               kernel_size=3,
               strides=1,
               activation=params["activation"],
               name="MP_C1")(inputs)
    for i, filters in enumerate(params["conv_filters"][1:], 2):
        if params["use_batch_norm"]:
            x = layers.BatchNormalization()(x, name="BN_{}".format(i))
        x = layers.Conv2D(filters,
                          kernel_size=3,
                          strides=1,
                          activation=params["activation"],
                          name="MP_C{}".format(i))(x)
        x = layers.MaxPooling2D(pool_size=2, name="pooling_{}".format(i))(x)

    x = layers.Flatten(name="flatten")(x)

    # fully connected
    for i, neurons in enumerate(params["fully_connected_neurons"]):
        x = layers.Dense(neurons,
                         activation=params["activation"],
                         name="Dense_{}".format(i))
        x = layers.Dropout(params["dropout_rate"], name="DropFCL_{}".format(i))

    n_classes = params["n_classes"]
    x = layers.Dense(n_classes,
                     activation="softmax" if n_classes>2 else "sigmoid",
                     name="Dense_Out")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=keras.optimizers.Adadelta(),
                  loss="categorical_crossentropy" if n_classes>2 else "binary_crossentropy")

    return model

# no validation set in here!!!
def get_train_test_data(pos_samples, # no duplicates in here
                        neg_samples,
                        train_ratio=0.8,
                        desired_pos_ratio=0.25,
                        negative_sample_usuage=1.0):
    num_negative, num_positive = pos_samples.shape[0], neg_samples.shape[0]

    if negative_sample_usuage < 1:
        idxs = np.arange(num_negative)
        num_negative = int(num_negative * negative_sample_usuage)
        neg_samples = neg_samples[np.random.choice(idxs, num_negative), ...]

    # figure out how many positive examples to add to reach the desired_pos_ratio
    # P = # of positive samples
    # N = # of negative samples
    # r = desired P/(P+N)
    # n = # of additional positive samples needed to reach r
    # n = (P*(r-1) + r*N) / (1 - R)
    num_additional_pos_examples = num_positive * (desired_pos_ratio - 1)
    num_additional_pos_examples += desired_pos_ratio * num_positive
    num_additional_pos_examples /= 1 - desired_pos_ratio
    num_additional_pos_examples = int(np.ceil(num_additional_pos_examples))

    # randomly select samples from the positive set to duplicate
    idxs = np.arange(num_positive)
    selected_idxs = np.random.choice(idxs, num_additional_pos_examples)
    repeated_positive_examples = pos_samples[selected_idxs, ...]

    pos_samples = np.concatenate((pos_samples, repeated_positive_examples))
    num_positive = pos_samples.shape[0]

    pos_labels = np.ones(num_positive)
    neg_labels = np.zeros(num_negative)

    xs = np.concatenate((pos_samples, neg_samples))
    ys = np.concatenate((pos_labels, neg_labels))

    x_train, x_test, y_train, y_test = train_test_split(xs, ys,
                                                        train_size=train_ratio)

    return (x_train,
            keras.utils.to_categorical(y_train),
            x_test,
            keras.utils.to_categorical(y_test))

def get_data_generator(params):
    augementation_params =[
        "width_shift_range",
        "height_shift_range",
        "zoom_range",
        "horizontal_flip",
        "vertical_flip"
    ]

    dg_kwargs = {k:params[k] for k in augementation_params}

    return ImageDataGenerator(**dg_kwargs)

def main():
    pos_samples = get_positive_samples()
    neg_samples = get_negative_samples()

    # add updating hyperparams here
    model_id = get_random_string(8)

    tt_kwargs = [
        "train_ratio",
        "desired_pos_ratio",
        "negative_sample_usuage"
        ]
    tt_kwargs = {k:hyperparams[k] for k in tt_kwargs}

    x_train, y_train, x_test, y_test = get_train_test_data(pos_samples,
                                                           neg_samples,
                                                           **tt_kwargs)

    model = HSC_Subaru_CNN(hyperparams)
    model.summary(print_fn=summary_to_file(model_id))

    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=10,
                                   verbose=0,
                                   mode="auto")

    datagen = get_data_generator(hyperparams)

    batches = datagen.flow(x_train,
                           y_train,
                           batch_size=hyperparams["batch_size"])

    history = model.fit_generator(batches,
                                  steps_per_epoch=x_train.shape[0]//32,
                                  epochs=hyperparams["train_epochs"],
                                  callbacks=[early_stopping],
                                  validation_data=(x_test, y_test))

    model.save(f"{model_id}/model.h5")
    with open(f"{model_id}/model_hist.json", "w") as f:
        json.dump(history.history, f)

if __name__=="__main__":
    main()
