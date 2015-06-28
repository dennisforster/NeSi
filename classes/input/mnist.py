# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import numpy as np
import os
import struct
from array import array

def read_images_from_mnist(dataset = "train", path = "./data-sets/MNIST", classes=None):
    """
    Python function for importing the MNIST data set.
    """
    label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    if (classes is None):
        classes = xrange(10)

    if dataset is "train":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "test":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'test' or 'train'"

    flbl = open(fname_lbl, 'rb')
    #magic_nr, size = struct.unpack(">II", flbl.read(8))
    struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    #lbl = np.array(flbl.read(),dtype='b')
    flbl.close()

    fimg = open(fname_img, 'rb')
    #magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    size, rows, cols = struct.unpack(">IIII", fimg.read(16))[1:4]
    img = array("B", fimg.read())
    #img = np.array(fimg.read(),dtype='B')
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in classes ]
    images = np.zeros(shape=(len(ind), rows*cols))
    labels = np.zeros(shape=(len(ind)), dtype=int)
    for i in xrange(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i] = lbl[ind[i]]

    return images, labels, label_names

def add_translated_images(images, labels):
    trans_images = np.zeros(shape=(images.shape[0]*9, images.shape[1]), dtype=np.float64)
    trans_labels = np.zeros(len(labels)*9, dtype=np.int)
    for i in xrange(images.shape[0]):
        trans_images[9*i,:] = images[i,:]
        trans_images[9*i+1,:] = shift(images[i,:], [0,1])
        trans_images[9*i+2,:] = shift(images[i,:], [0,-1])
        trans_images[9*i+3,:] = shift(images[i,:], [1,0])
        trans_images[9*i+4,:] = shift(images[i,:], [-1,0])
        trans_images[9*i+5,:] = shift(images[i,:], [1,1])
        trans_images[9*i+6,:] = shift(images[i,:], [1,-1])
        trans_images[9*i+7,:] = shift(images[i,:], [-1,1])
        trans_images[9*i+8,:] = shift(images[i,:], [-1,-1])
        trans_labels[9*i:9*(i+1)] = labels[i]
    return trans_images, trans_labels

def shift(image,direction,cval=0.0,cyclic=False):
    """
    direction is a 2-dimensional array: [vertical_shift,horizontal_shift]

    vertival_shift > 0 : shift down
    vertival_shift < 0 : shift up
    horizontal_shift > 0 : shift to the right
    horizontal_shift < 0 : shift to the left

    If cyclic==True elements that are shifted beyond the last position are
    re-introduced at the first position of the same axis.
    If cyclic==False elements that are shifted beyond the last position are
    lost and the elements at the first position will be set to cval
    """
    image_2D = np.reshape(image, (np.sqrt(image.shape[0]),np.sqrt(image.shape[0])))
    if (direction[0] != 0):
        image_2D = np.roll(image_2D,direction[0],axis=0)
    if (direction[1] != 0):
        image_2D = np.roll(image_2D,direction[1],axis=1)

    if (cyclic == False):
        if (direction[0] > 0):
            image_2D[:direction[0],:] = cval
        elif (direction[0] < 0):
            image_2D[direction[0]:,:] = cval
        if (direction[1] > 0):
            image_2D[:,:direction[1]] = cval
        elif (direction[1] < 0):
            image_2D[:,direction[1]:] = cval

    return image_2D.flatten()

def add_rotated_images(images, labels, rot_angle):
    n = (len(rot_angle)+1)
    rot_images = np.zeros(shape=(images.shape[0]*n, images.shape[1]), dtype=np.float64)
    rot_labels = np.zeros(len(labels)*n, dtype=np.int)
    for i in xrange(images.shape[0]):
        rot_images[n*i,:] = images[i,:]
        for nrot in xrange(len(rot_angle)):
            rot_images[n*i+nrot+1,:] = rotate(images[i,:], rot_angle[nrot])
        rot_labels[n*i:n*(i+1)] = labels[i]
    return rot_images, rot_labels

def rotate(image, rot_angle, _cval=0.0):
    from scipy.ndimage.interpolation import rotate
    """
    rot_angel > 0 clock_wise rotation
    rot_angel < 0 anti-clock_wise rotation

    cval : Value used for points outside the boundaries of the input if mode='constant'. Default is 1.0

    """
    rot_img = rotate(
        input=np.reshape(image,
                         (np.sqrt(image.shape[0]),np.sqrt(image.shape[0]))),
        angle=(-1)*rot_angle, axes=(1, 0), reshape=False,
        order=3, mode='constant', cval=_cval, prefilter=True)
    image = rot_img.flatten()
    #Since the interpolation can cause negative values, these values are
    #set to the background value:
    image[image < _cval] = _cval
    return image