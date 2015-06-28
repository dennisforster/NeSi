# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import numpy as np
import os
import struct
from array import array
import scipy
from scipy.io import loadmat

def read_data_from_20news(dataset = "train", path = "./data-sets/20Newsgroups", classes=None):
    """
    Python function for importing the 20 Newsgroups data set.
    """
    bydatepath = path + '/20news-bydate/matlab/'

    # vocabulary_file = open(path + '/vocabulary.txt', 'r')
    # vocabulary = []
    # for line in vocabulary_file:
    #     vocabulary.append(line[0:-1]) # omits the '\n' at the end of each line

    label_names_file = open(bydatepath + '/train.map', 'r')
    label_names = []
    for line in label_names_file:
        label_names.append(line.split()[0])

    if (classes == None):
        classes = xrange(20)

    if (dataset == 'train'):
        data_file = open(bydatepath + '/train.data', 'r')
        data = np.zeros(shape=(11269, 61188), dtype=int)
        for line in data_file:
            data[int(line.split()[0])-1,int(line.split()[1])-1] = int(line.split()[2])
        label_file = open(bydatepath + '/train.label', 'r')
        labels = []
        for line in label_file:
            labels.append(int(line)-1)
        labels = np.asarray(labels, dtype=int)
    elif (dataset == 'test'):
        data_file = open(bydatepath + '/test.data', 'r')
        data = np.zeros(shape=(7505, 61188), dtype=int)
        for line in data_file:
            data[int(line.split()[0])-1,int(line.split()[1])-1] = int(line.split()[2])
        label_file = open(bydatepath + '/test.label', 'r')
        labels = []
        for line in label_file:
            labels.append(int(line)-1)
        labels = np.asarray(labels, dtype=int)
    else:
        raise ValueError, "dataset must be 'test' or 'train'"

    ind = [ k for k in xrange(data.shape[0]) if labels[k] in classes ]
    data = data[ind]
    labels = labels[ind]

    #-- tf-idf normalization
    tf = data # raw frequency: tf(t,d) = f(t,d)
    idf = np.log(data.shape[0]/(1+(data != 0).sum(0, keepdims=True)))
    data = tf*idf

    return data, labels, label_names