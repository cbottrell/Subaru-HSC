import json
import os
import random
import string

import comet_ml
import numpy as np
import tensorflow as tf
from astropy.io import fits
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split


from balanced_image_datagenerator import BalancedImageDataGenerator
from comet_callback import CometLogger

# Put other configurations that you want to try here
hyperparam_ranges = dict(
    conv_filters = [
        [8, 16, 32, 64],
        [64, 128, 128, 256],
        [32, 64],
        [4, 8, 16]
    ],

    fully_connected_neurons = [
        [64, 32, 16],
        [64, 64, 128],
        [256],
        [16]
    ],

    dropout_rate = [
        0.0,
        0.50,
        0.75
    ],

    batch_size = [
        16,
        64
    ],

    learning_rate = [
        0.1,
        0.01,
        0.0001,
        0.00001,
    ]
)

# TODO: consider dictionary of dictionaries
hyperparams = dict(
    img_width = 128,
    img_height = 128,
    img_channels = 1,
    train_epochs = 50,
    train_size = 0.7,
    batch_size = 32,
    pos_batch_ratio = 0.50,

    # cnn params
    n_classes = 1,
    conv_filters = [32, 64, 128, 128],
    fully_connected_neurons = [64, 16],
    activation = "relu",
    use_batch_norm = True,
    dropout_rate = 0.25,
    learning_rate = 0.001,

    # augmentation params
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 359
)

def get_positive_samples():
    return np.load("./Data/pos_samples.npy")

def get_negative_samples():
    return np.load("./Data/neg_samples.npy")

def summary_to_file(model_id):

    def print_fn(s):
        with open(f"models/{model_id}/summary.txt", "a") as f:
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
            x = layers.BatchNormalization(name="BN_{}".format(i))(x)

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
                         name="Dense_{}".format(i))(x)

        x = layers.Dropout(params["dropout_rate"], name="DropFCL_{}".format(i))(x)

    n_classes = params["n_classes"]

    x = layers.Dense(n_classes,
                     activation="softmax" if n_classes>1 else "sigmoid",
                     name="Dense_Out")(x)


    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
        tf.keras.metrics.AUC(curve="PR", name="auc_pr")

    ]

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=params["learning_rate"]),
                  loss="sparse_categorical_crossentropy" if n_classes>1 else "binary_crossentropy",
                  metrics=metrics)

    return model

def get_data_generator(params):
    augementation_params =[
        "width_shift_range",
        "height_shift_range",
        "zoom_range",
        "horizontal_flip",
        "vertical_flip"
    ]

    dg_kwargs = {k:params[k] for k in augementation_params}

    return BalancedImageDataGenerator(**dg_kwargs)

def train_model(model_id,
                params,
                train_data,
                validation_data,
                model_rating_fn,
                experiment):
    tf.keras.backend.clear_session()


    if model_id not in os.listdir("./models"):
        os.mkdir(f"./models/{model_id}")

    with open(f"models/{model_id}/params.json", "w") as f:
        json.dump(params, f)

    pos_xs, pos_ys, neg_xs, neg_ys = train_data

    model = HSC_Subaru_CNN(params)
    model.summary(print_fn=summary_to_file(model_id))
    #plot_model(model, to_file='./models/{mdoel_id}/model.png')
    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=10,
                                   verbose=1,
                                   mode="auto")

    datagen = get_data_generator(params)

    batches = datagen.flow(pos_xs,
                           pos_ys,
                           neg_xs,
                           neg_ys,
                           params["pos_batch_ratio"],
                           params["batch_size"])

    history = model.fit_generator(batches,
                                  steps_per_epoch=(pos_xs.shape[0] + neg_xs.shape[0])//params["batch_size"],
                                  epochs=params["train_epochs"],
                                  validation_data=validation_data,
                                  verbose=0,
                                  callbacks=[CometLogger(experiment)]).history

    model.save(f"models/{model_id}/model.h5")
    with open(f"models/{model_id}/model_hist.json", "w") as f:
        json_save = {}

        for k in history:
            if isinstance(history[k], list):
                json_save[k] = [float(v) for v in history[k]]
            else:
                json_save[k] = float(history[k])

        json.dump(json_save, f)

    return model_rating_fn(history)


def main():
    pos_samples = get_positive_samples()
    neg_samples = get_negative_samples()

    pos_train_x, pos_test_x = train_test_split(pos_samples, train_size=hyperparams["train_size"])
    neg_train_x, neg_test_x = train_test_split(neg_samples, train_size=hyperparams["train_size"])

    pos_train_y = np.ones(pos_train_x.shape[0])
    pos_test_y = np.ones(pos_test_x.shape[0])

    neg_train_y = np.zeros(neg_train_x.shape[0])
    neg_test_y = np.zeros(neg_test_x.shape[0])

    train_data = pos_train_x, pos_train_y, neg_train_x, neg_train_y
    test_x = np.concatenate((pos_test_x, neg_test_x))
    test_y = np.concatenate((pos_test_y, neg_test_y))
    test_data = (test_x, test_y)

    # we want some way to keep the best value over each param.
    model_rating_fn = lambda hist: min(hist["val_loss"])

    hyperparams["random_state"] = np.random.randint(100000000)

    # use the initial values before searching
    experiment = comet_ml.Experiment(api_key=os.getenv("comet_key"),
                                     project_name="cbottrell-subaru-hsc",
                                     workspace="kspa-subaru-hsc",
                                     disabled=False)

    model_id = experiment.get_key()
    experiment.log_parameters(hyperparams)

    lowest_val = train_model(model_id,
                             hyperparams,
                             train_data,
                             test_data,
                             model_rating_fn,
                             experiment)

    experiment.end()

    print(f"Initial model {model_id} scored: {lowest_val}")

    # shuffle the keys so that we don't rerun in the same order
    param_list = list(hyperparam_ranges.keys())
    random.shuffle(param_list)

    # start the search!
    for k in param_list:
        print(f"Experimenting with {k}")
        for val in hyperparam_ranges[k]:
            old_hparam = hyperparams[k]
            hyperparams[k] = val

            experiment = comet_ml.Experiment(api_key=os.getenv("comet_key"),
                                             project_name="cbottrell-subaru-hsc",
                                             workspace="kspa-subaru-hsc")
            experiment.log_parameters(hyperparams)

            model_id = experiment.get_key()

            print(f"ModelID {model_id} Trying {k}={val}")

            curr_val = train_model(model_id,
                                   hyperparams,
                                   train_data,
                                   test_data,
                                   model_rating_fn,
                                   experiment)

            experiment.end()

            if curr_val < lowest_val:
                print(f"New best! current: {curr_val} prev best {lowest_val}")
                lowest_val = curr_val
            else:
                print(f"No change. current: {curr_val} prev best {lowest_val}")
                hyperparams[k] = old_hparam

    print("Done!")

if __name__=="__main__":
    main()
