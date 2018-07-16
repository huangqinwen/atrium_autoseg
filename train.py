from __future__ import division, print_function
import os

import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import math
import SimpleITK as sitk
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
from keras import utils
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import ImageDataGenerator
import pickle
import argparse
import cv2
from augmentation import *
from unet import *
from loss import *
from generate import *



train_labels = pickle.load(open("train_labels.p","rb"))

def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your models.")

    parser.add_argument("--data", type=str, help="The data file to use for training or testing.",default='/Users/qinwenhuang/Documents/autoseg/Mini_Training')
    '''
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    '''


    parser.add_argument("--batch_size", type=int,
                        help="batch size for training", default=16)
    parser.add_argument("--optimizer", type=str,
                        help="The optimizer wanted to use.", default='adam')
    parser.add_argument("--online-learning-rate", type=float, help="The learning rate for online methods", default=0.1)
    parser.add_argument("--decay-learning-rate", type=float, help="The decay rate for learning", default=0.001)
    parser.add_argument("--validation-split", type=float, help="validation-split", default=0.2)
    parser.add_argument("--loss", type=str, help="loss function", default='binary_crossentropy')

    # TODO: Add optional command-line arguments as necessary.

    args = parser.parse_args()

    return args

#you can choose to use different optimizers
#this function takes in optimizer name and arguments for optimizer (e.g. learning rate, decay, momentum)
def use_optimizer(name, args):
    """
    7 possible optimizers - usually adam works the best
    see paper - "Adam, A Method for Stochastic Optimization" by Kingma and Ba, arXiv 1412.6980
    """
    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    if name not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(name))

    return optimizers[name](**args)
"""
certain arguments for training
"""
rotation_range = 90
width_shift_range = 0.1
height_shift_range=0.1
shear_range = 0.1
zoom_range = 0.01
fill_mode='nearest'
shuffle_train_val = True
shuffle = True
seed = None
augment_training=True
augment_validation=True
loss = 'pixel'
augment_options = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'shear_range': shear_range,
            'zoom_range': zoom_range,
            'fill_mode': fill_mode,
        }
loss_weights = 0.1
normalize = True
augmentation_args = augment_options
def check_args(args):
    mandatory_args = {'data'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception(
            'Arguments that we provided are now renamed or missing.')





# use train function to train the neural nets - if you want to train the net, just run this function with all arguments
def train(rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, fill_mode, datadir, batch_size,
          validation_split, learning_rate, decay, optimizer, loss, loss_weights):
    augmentation_args = {
        'rotation_range': rotation_range,
        'width_shift_range': width_shift_range,
        'height_shift_range': height_shift_range,
        'shear_range': shear_range,
        'zoom_range': zoom_range,
        'fill_mode': fill_mode,

    }
    train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch = create_generators(
        datadir, batch_size,
        validation_split=validation_split,
        mask = train_labels,
        shuffle_train_val=shuffle_train_val,
        shuffle=shuffle,
        seed=seed,
        normalize_images=normalize,
        augment_training=augment_training,
        augment_validation=augment_validation,
        augmentation_args=augmentation_args)
    # get image dimensions from first batch
    images, masks = next(train_generator)
    _, height, width, channels = images.shape
    _, _, _, classes = masks.shape
    # start building model
    model = unet(height=height, width=width, channels=channels, classes=classes, dropout=0.5)

    model.summary()

    optimizer_args = {
        'lr': learning_rate,
        'decay': decay
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = use_optimizer(optimizer, optimizer_args)

    if loss == 'binary_crossentropy':
        def lossfunc(y_true, y_pred):
            return weighted_categorical_crossentropy(
                y_true, y_pred, loss_weights)
    elif loss == 'dice':
        def lossfunc(y_true, y_pred):
            return sorensen_dice_loss(y_true, y_pred, loss_weights)
    elif loss == 'jaccard':
        def lossfunc(y_true, y_pred):
            return jaccard_loss(y_true, y_pred, loss_weights)
    else:
        raise Exception("Unknown loss")
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.fit_generator(generator=train_generator, validation_data=val_generator, steps_per_epoch=train_steps_per_epoch,
                        validation_steps=val_steps_per_epoch, use_multiprocessing=False, )

def main():
    args = get_args()
    #check_args(args)
    """
    certain arguments for training
    """
    rotation_range = 90
    width_shift_range = 0.1
    height_shift_range = 0.1
    shear_range = 0.1
    zoom_range = 0.01
    fill_mode = 'nearest'
    shuffle_train_val = True
    shuffle = True
    seed = None
    augment_training = True
    augment_validation = True
    loss = 'pixel'
    augment_options = {
        'rotation_range': rotation_range,
        'width_shift_range': width_shift_range,
        'height_shift_range': height_shift_range,
        'shear_range': shear_range,
        'zoom_range': zoom_range,
        'fill_mode': fill_mode,
    }
    loss_weights = 0.1
    normalize = True
    augmentation_args = augment_options
    batch_size = args.batch_size
    datadir = args.data.lower()
    validation_split = args.validation_split
    decay = args.decay_learning_rate
    learning_rate = args.online_learning_rate
    optimizer = args.optimizer.lower()
    loss = args.loss.lower()
    train(rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, fill_mode, datadir,
          batch_size, validation_split, learning_rate, decay, optimizer, loss, loss_weights)


if __name__ == "__main__":
    main()