import os
import pickle
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

from keras import utils
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import ImageDataGenerator
import cv2
from augmentation import *

def load_nrrd(full_path_filename):

	data = sitk.ReadImage(full_path_filename)
	data = sitk.Cast(sitk.RescaleIntensity(data),sitk.sitkUInt8)
	data = sitk.GetArrayFromImage(data)

	return(data)


# decode mask
def run_length_decoding(run_lengths, img):
    h, w = img.shape
    mask = np.zeros(h * w)
    if run_lengths == '[\n]':
        pass
    else:
        run_lengths_s = run_lengths[0].split()
        # print(run_lengths_s)
        for i in range(len(run_lengths_s)):
            # even number is index and odd number is # of consecutive tags
            if i % 2 == 0:
                # print(i)
                mask[(int(run_lengths_s[i]) - 1):(int(run_lengths_s[i]) + int(run_lengths_s[i + 1]) - 1)] = 1
        mask = mask.reshape((h, w)).T

    return mask


class dataLoading(object):
    """
    data directory structure
    TrainingSet/
    each patient/
    mask.nrrd, original.nrrd
    """

    def __init__(self, directory, mask_dict):
        self.directory = directory
        self.mask_dict = mask_dict
        self.mris = []
        self.mri_names = []
        self.masks = []
        #self.train_labels = train_label
        # self.load_images()
        # self.load_masks()

    def load_images(self):
        """
        load all images from training set 
        go through subdirectories and get lgemri.nrrd, which represents patients original mri images
        uses load_nrrd function 
        retun mri matrices and mri names
        """

        for root, dirs, files in os.walk(self.directory, topdown=False):
            for name in files:
                if name == 'lgemri.nrrd':
                    # print('yes')
                    patient_name = root[-20:] + '_Slice_'
                    full_name = os.path.join(root, name)
                    single_patient_image = load_nrrd(full_name)
                    num_of_slices = single_patient_image.shape[0]

                    for i in range(num_of_slices):
                        resze_single = cv2.resize(single_patient_image[i], (128,128))
                        # resze_single = resze_single.reshape(resze_single,(640,640,1))
                        # resze_single = np.expand_dims(resze_single, axis=2)
                        self.mris.append(resze_single)
                        self.mri_names.append(patient_name + str(i))
        return self.mris, self.mri_names

    def load_masks(self):
        """
        covert all masks in rle to matrix format 
        return matrix format mask list
        """
        # self.masks = []
        for idx, name in enumerate(self.mri_names):
            img = self.mris[idx]
            train_labels = pickle.load(open("train_labels.p","rb"))
            encode_cav = train_labels[name]
            # print(encode_cav)
            output_mask = run_length_decoding(encode_cav, img)
            resize_mask = cv2.resize(output_mask, (128, 128))
            self.masks.append(resize_mask)
        return self.masks

def load_images(directory, mask):
    all_data = dataLoading(directory, mask)
    image, image_name = all_data.load_images()
    masks = all_data.load_masks()
    return image, masks
#create iterator for better data iteration - this is for loading data to Neural Nets
class Iterator(object):
    def __init__(self, images, masks, batch_size,
                 shuffle=True,
                 rotation_range=90,
                 width_shift_range=0.1,
                 height_shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.01,
                 fill_mode='nearest',
                 alpha=500,
                 sigma=20):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        augment_options = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'shear_range': shear_range,
            'zoom_range': zoom_range,
            'fill_mode': fill_mode,
        }
        self.idg = ImageDataGenerator(**augment_options)
        self.alpha = alpha
        self.sigma = sigma
        self.fill_mode = fill_mode
        self.i = 0
        self.index = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()

    def next(self):
        # compute how many images to output in this batch
        start = self.i
        end = min(start + self.batch_size, len(self.images))
        augmented_images = []
        augmented_masks = []
        for n in self.index[start:end]:
            image = self.images[n]
            mask = self.masks[n]
            h,w,channels = image.shape
            #h,w = image.shape

            # stack image + mask together to simultaneously augment
            stacked = np.concatenate((image, mask), axis=2)

            # apply simple affine transforms first using Keras
            augmented = self.idg.random_transform(stacked)

            # maybe apply elastic deformation
            if self.alpha != 0 and self.sigma != 0:
                augmented = elastic_transform(
                    augmented, self.alpha, self.sigma, self.fill_mode)

            # split image and mask back apart
            augmented_image = augmented[:,:,:channels]
            augmented_images.append(augmented_image)
            augmented_mask = np.round(augmented[:,:,channels:])
            augmented_masks.append(augmented_mask)

        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return np.asarray(augmented_images), np.asarray(augmented_masks)


def normalize(x, epsilon=1e-7, axis=1):
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= np.std(x, axis=axis, keepdims=True) + epsilon


def create_generators(data_dir, batch_size, mask,validation_split=0.0,
                      shuffle_train_val=True, shuffle=True, seed=None,
                      normalize_images=True, augment_training=False,
                      augment_validation=False, augmentation_args={}):
    images, masks = load_images(data_dir, mask)

    # before: type(masks) = uint8 and type(images) = uint8
    # convert images to double-precision
    # images = images.astype('float64')
    for i, img in enumerate(images):
        images[i] = images[i].astype('float64')
    # maybe normalize image
    if normalize_images:
        for i in images:
            normalize(i, axis=1)
    images = np.dstack(images)
    images = np.rollaxis(images, -1)
    masks = np.dstack(masks)
    masks = np.rollaxis(masks, -1)
    images = np.expand_dims(images, axis=3)
    masks = np.expand_dims(masks, axis=3)
    if seed is not None:
        np.random.seed(seed)

    if shuffle_train_val:
        # shuffle images and masks in parallel
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(masks)

    # split out last %(validation_split) of images as validation set
    split_index = int((1 - validation_split) * len(images))

    if augment_training:
        train_generator = Iterator(
            images[:split_index], masks[:split_index],
            batch_size, shuffle=shuffle, **augmentation_args)
    else:
        idg = ImageDataGenerator()
        train_generator = idg.flow(images[:split_index], masks[:split_index],
                                   batch_size=batch_size, shuffle=shuffle)

    train_steps_per_epoch = np.ceil(split_index / batch_size)

    if validation_split > 0.0:
        if augment_validation:
            val_generator = Iterator(
                images[split_index:], masks[split_index:],
                batch_size, shuffle=shuffle, **augmentation_args)
        else:
            idg = ImageDataGenerator()
            val_generator = idg.flow(images[split_index:], masks[split_index:],
                                     batch_size=batch_size, shuffle=shuffle)
    else:
        val_generator = None

    val_steps_per_epoch = np.ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch,
            val_generator, val_steps_per_epoch)
