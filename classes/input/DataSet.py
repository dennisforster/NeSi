# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import numpy as np
import os.path
import h5py
import sys

from utils.sysfunctions import mem_usage
from utils.parallel import pprint

class DataSet:
    """
    Stores the full Data Set and allows for creation of random sub-sets
    """

    #===== Constructor ======================================
    def __init__(self):
        self._fulltraindata = None
        self._fulltrainlabel = None
        self._fulltrainlist = None
        self._fulltestdata = None
        self._fulltestlabel = None

        self._traindata = None
        self._trainlabel = None
        self._testdata = None
        self._testlabel = None

        self._name = None
        self._h5path = None

        self._labelnames = None
        self._classes = None

        self._indexlist = {
            'full_train_labeled':None,
            'full_train_unlabeled':None,
            'train_labeled': None,
            'train_unlabeled': None,
            'full_test':None,
            'test':None,
        }

    #========================================================

    #===== Data Set Properties ==============================
    def name(self):
        return self._name

    def h5path(self):
        return self._h5path
    #========================================================

    #===== Data Set Operations ==============================
    def import_dataset(self, comm, config):
        """
        Link data set to existing h5 file or create new h5 file if it
        doesn't exist
        """
        self._name = config['dataset']['name']
        path = config['dataset']['path']

        # create data set as h5 file, if such file doesn't exist
        self._h5path = os.path.join(path, '%s.h5'%self._name)
        if (comm.Get_rank() == 0):
            if not os.path.isfile(self._h5path):
                self.create_h5_dataset(self._name, path)
            # uncomment to always create new h5 file
            # else:
            #     os.remove(self._h5path)
            #     self.create_h5_dataset(self._name, path)
        h5file = h5py.File(self._h5path, 'r')
        self._labelnames = np.append(h5file['label_names'].value, 'None')
        try:
            classes = config['dataset']['classes']
        except:
            classes = self._labelnames
        self._classes = np.asarray(
            [np.where(self._labelnames == k)[0][0] for k in classes],
            dtype=int)
        comm.Barrier()
        self._indexlist['full_train_labeled'] = np.where(
            [lbl in self._classes for lbl in h5file['/train/label'].value])[0]
        self._indexlist['full_train_unlabeled'] = np.where(
            h5file['/train/label'].value == -1)[0]
        self._indexlist['full_test'] = np.arange(
            len(h5file['/test/label'].value))
        h5file.close()

        """
        # TODO:
        if not config['dataset']['USE_TEST_SET']:
            # draw from labeled data both the labeled training and test data
            sys.exit("Error (DataSet::DrawDataSets): Cannot split training \
                set as specified.\nUSE_TEST_SET = %s\ntraining_label_size = \
                %d\ntest_size = %d\navailable number of labels = %d" % \
                (config['dataset']['USE_TEST_SET'],\
                    config['dataset']['training_label_size'],\
                    config['dataset']['test_size'], max_label_size))
            # TODO: data split if no test set is available:
        """


    def pick_new_subset(self, config, dset, run=None):
        """
        1. From the labeled data points the given number of data points
           to use for training or testing are selected - these can be
           either chosen completely randomly, or evenly distributed
           between classes (and only randomly inside of each class);
           all remaining labeled data points are declared as unlabeled
        2. Then (for training) the remaining data points are chosen from
           the unlabeled data points
        """

        h5file = h5py.File(self._h5path, 'r')
        if (dset == 'train'):
            label = np.asarray(h5file['/train/label'].value, dtype=int)
            label_size = min(config['dataset']['training_label_size'],
                             len(self._indexlist['full_train_labeled']))
            data_size = min(config['dataset']['training_data_size'],
                            len(self._indexlist['full_train_labeled']) +\
                            len(self._indexlist['full_train_unlabeled']))
            if (label_size > data_size):
                label_size = data_size
                pprint("ERROR: more training labels than training data. " +
                    "These values are now set as follows:")
                pprint("training_data_size: %d", data_size)
                pprint("training_label_size: %d", label_size)
        elif (dset == 'test'):
            label = np.asarray(h5file['/test/label'].value, dtype=int)
            data_size = min(config['dataset']['test_size'],
                            len(self._indexlist['full_test']))
            label_size = data_size
        h5file.close()

        # TODO: change data_size value in saved setting accordingly
        # TODO: change label_size value in saved setting accordingly

        try:
            EVENLABELS = config['dataset']['EVENLABELS']
        except:
            EVENLABELS = False

        try:
            indexlist = config['dataset']['indexlist']
        except:
            indexlist = None

        try:
            labellist = config['dataset']['labellist']
        except:
            labellist = None

        if ((indexlist is not None) and (labellist is not None)):
            pprint('WARNING (classes.input.DataSet): when dataset.indexlist ' +
                'is given, dataset.labellist will be ignored.')

        if (indexlist is None):
            if (labellist is None):
                # draw labeled data
                labeled_index = self.draw_indexlist(label, self._classes,
                                                   label_size, EVENLABELS)
            else:
                try:
                    filename = labellist
                    h5file = h5py.File(filename, 'r')
                except:
                    filename = labellist + 'Run' + str(run+1) + 'Data.h5'
                    h5file = h5py.File(filename,'r')
                if (dset == 'train'):
                    try:
                        labeled_index = h5file['train/labeled'].value
                        pprint('WARNING: labels set according to labellist - '+
                            'label_size will be ignored.')
                        # TODO: draw labels according to label_size from labellist
                    except:
                        pprint('WARNING (classes.input.DataSet): ' +
                            'training indexlist not found in provided ' +
                            'labellist, it will be drawn randomly')
                        labeled_index = self.draw_indexlist(
                            label, self._classes, label_size, EVENLABELS)
                elif (dset == 'test'):
                    try:
                        labeled_index = h5file['test'].value
                    except:
                        pprint('WARNING (classes.input.DataSet): ' +
                            'testlist not found in provided labellist, it ' +
                            'will be drawn randomly')
                        labeled_index = self.draw_indexlist(
                            label, self._classes, label_size, EVENLABELS)
            unlabeled_index = np.asarray([], dtype=int)
            if (label_size < data_size):
                # set remaining data points as unlabeled
                remaining = [idx
                    for idx in self._indexlist['full_train_labeled']
                    if idx not in labeled_index]
                label[remaining] = int(-1)
                # draw additional unlabeled data
                unlabeled_index = self.draw_indexlist(
                    label, [-1], data_size-label_size, False)
        else:
            try:
                h5file = h5py.File(indexlist, 'r')
            except:
                h5file = h5py.File(indexlist+'Run'+str(run+1)+'Data.h5', 'r')
            if (dset == 'train'):
                labeled_index = h5file['train/labeled'].value
                unlabeled_index = h5file['train/unlabeled'].value
            elif (dset == 'test'):
                labeled_index = h5file['test'].value
            h5file.close()

        if (dset == 'train'):
            self._indexlist['train_labeled'] = labeled_index
            self._indexlist['train_unlabeled'] = unlabeled_index
        elif (dset == 'test'):
            self._indexlist['test'] = labeled_index


    def load_set(self, dset, datatype=np.float32):
        """
        Load actual data points from h5 into RAM according to the index
        list
        """
        h5file = h5py.File(self._h5path, 'r')
        if (dset == 'train'):
            data = h5file['/train/data'].value[
                    np.append(
                        self._indexlist['train_labeled'],
                        self._indexlist['train_unlabeled']
                    ).astype('int')
                ].astype(datatype)
            label = np.append(
                h5file['/train/label'].value[self._indexlist['train_labeled']],
                [-1]*len(self._indexlist['train_unlabeled']))
            # use label index corresponding to order given in
            # config[dataset][classes]
            label[label!=-1] = [np.where(self._classes == classidx)[0][0]
                for classidx in label[label != -1]]
            self.set_train_set(data,label,datatype)
        elif (dset == 'test'):
            data = h5file['/test/data'].value[self._indexlist['test']]
            label = h5file['/test/label'].value[self._indexlist['test']]
            label[label!=-1] = [np.where(self._classes == classidx)[0][0]
                for classidx in label[label != -1]]
            self.set_test_set(data,label,datatype)
        h5file.close()


    def delete_set(self, dset):
        """
        Delete dataset (but not index list) from RAM
        """
        if (dset == 'train'):
            self.set_train_set(np.asarray([], dtype=np.float32),
                             np.asarray([], dtype=np.int))
        elif (dset == 'test'):
            self.set_test_set(np.asarray([], dtype=np.float32),
                            np.asarray([], dtype=np.int))


    def create_h5_dataset(self, dataset, path):
        """
        Create h5 file for the given dataset
        """
        pprint('\nCreating %s' % dataset +
               ' h5 data set using fast, lossless compression (lzf)...',
               end='')
        # read train and test data from original dataset
        nparray_2D_train, label_train, label_names = \
            self._read_set_from(dataset, 'train', path)
        nparray_2D_test, label_test = \
            self._read_set_from(dataset, 'test', path)[0:2]
        # store in h5 file using lzf compression
        h5file = h5py.File(self._h5path, 'w-')
        h5file.create_dataset('/train/data', data=nparray_2D_train,
                              compression='lzf')
        h5file.create_dataset('/train/label', data=label_train,
                              compression='lzf')
        if (nparray_2D_test is not None):
            h5file.create_dataset('/test/data', data=nparray_2D_test,
                                  compression="lzf")
            h5file.create_dataset('/test/label', data=label_test,
                                  compression="lzf")
        h5file.create_dataset('label_names', data=label_names,
                              compression="lzf")
        h5file.close()
        pprint(' Done.')


    def total_datapoints_per_class(self, lbl):
        """
        Returns array with the number of data points for each class in
        the given label vector, with the last entry showing the number
        of unlabeled data points
        """
        dataperclass = np.asarray(
            [np.where(lbl==i)[0].shape[0]
                for i in np.append(xrange(max(lbl+1)),-1)],
            dtype=int)
        return dataperclass


    def draw_indexlist(self, lbl, classes, n_select=None, even_classes=True):
        """
        lbl: label vector
        classes: array of classes (denoted as class number, not class
        names) which will be used (if known)
        n_select: total number of data points to select

        calculate from the label vector and the selection options how
        many data points to use per class
        """
        # print "Total data points per class: \t\t\t", \
        #     self.total_datapoints_per_class(lbl)
        n_select_max_per_class = np.zeros_like(
            self.total_datapoints_per_class(lbl))
        n_select_max_per_class[classes] = \
            self.total_datapoints_per_class(lbl)[classes]
        # print "Max. usable data points per class: \t", n_select_max_per_class

        if ((n_select is None) or (n_select > np.sum(n_select_max_per_class))):
            n_select = np.sum(n_select_max_per_class)

        # select data points evenly between, or independent of classes
        if even_classes:
            """
            calculate, how many data points per class will be used
            e.g.: MNIST data points available per class:
            [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]
            using only 58000 data points yields:
            use: [5843 5842 5842 5842 5842 5421 5842 5842 5842 5842] 58000
            remaining: [ 80 900 116 289   0   0  76 423   9 107] 2000
            """
            n_select_per_class = np.zeros(len(n_select_max_per_class),
                                          dtype=int)
            n_remaining_per_class = n_select_max_per_class
            while (np.sum(n_select_per_class) < n_select):
                n_nonzero_classes = np.sum(n_remaining_per_class > 0)
                n_add_per_class = min(
                    (n_select-np.sum(n_select_per_class))/n_nonzero_classes,
                    min(n_remaining_per_class[n_remaining_per_class > 0]))
                n_select_per_class[n_remaining_per_class>0] += n_add_per_class
                n_remaining_per_class[n_remaining_per_class>0]-=n_add_per_class
                if ((n_add_per_class == 0) and
                    (np.sum(n_select_per_class) < n_select)):
                    class_index = np.where(n_remaining_per_class>0)[0]\
                        [0:(n_select-np.sum(n_select_per_class))]
                    n_remaining_per_class[class_index] -= 1
                    n_select_per_class[class_index] += 1
            # print "evenly distributed data to use per class: ", \
            #     n_select_per_class, np.sum(n_select_per_class)
            # print "remaining data per class: ", \
            #     n_remaining_per_class, np.sum(n_remaining_per_class)
            index = np.asarray([],dtype=int)
            for i in classes:
                index = np.append(
                    index,
                    np.random.permutation(np.where(lbl==i)[0])\
                        [0:n_select_per_class[i]])
        else:
            index = np.random.permutation([k for k in xrange(len(lbl))
                if lbl[k] in classes])[0:n_select]
        # print [np.sum(lbl[index] == i) for i in classes]
        return np.sort(index)


    def distribute_set(self, comm, dset='train'):
        """
        Divide and distribute the index lists for training or testing
        between all processes.
        For training, labeled and unlabeled data is divided separately
        to ensure an even distribution between processes.
        """
        self._classes = comm.bcast(self._classes, root=0)
        if (comm.Get_rank() == 0):
            if (dset == 'train'):
                for nrank in range(1,comm.Get_size()):
                    train_labeled = self._devide_list(
                        self._indexlist['train_labeled'], nthpart=nrank,
                        nparts=comm.Get_size())
                    train_unlabeled = self._devide_list(
                        self._indexlist['train_unlabeled'], nthpart=nrank,
                        nparts=comm.Get_size())
                    comm.send(train_labeled,dest=nrank,tag=1)
                    comm.send(train_unlabeled,dest=nrank,tag=2)
                self._indexlist['train_labeled'] = self._devide_list(
                    self._indexlist['train_labeled'], nthpart=0,
                    nparts=comm.Get_size())
                self._indexlist['train_unlabeled'] = self._devide_list(
                    self._indexlist['train_unlabeled'], nthpart=0,
                    nparts=comm.Get_size())
            elif (dset == 'test'):
                for nrank in range(1,comm.Get_size()):
                    test = self._devide_list(
                        self._indexlist['test'], nthpart=nrank,
                        nparts=comm.Get_size())
                    comm.send(test,dest=nrank,tag=1)
                self._indexlist['test'] = self._devide_list(
                    self._indexlist['test'], nthpart=0, nparts=comm.Get_size())
        else:
            if (dset == 'train'):
                self._indexlist['train_labeled'] = comm.recv(source=0,tag=1)
                self._indexlist['train_unlabeled'] = comm.recv(source=0,tag=2)
            elif (dset == 'test'):
                self._indexlist['test'] = comm.recv(source=0,tag=1)

    def shuffle(self, dataset='train'):
        """
        Shuffle the order of the images in the picked subset of the full
        dataset.
        """
        if (dataset == 'train'):
            index_list = range(self.get_train_data().shape[0])
        elif (dataset == 'test'):
            index_list = range(self.get_test_data().shape[0])

        np.random.shuffle(index_list)

        if (dataset == 'train'):
            self.set_train_data(self.get_train_data()[index_list])
            self.set_train_label(self.get_train_label()[index_list])
        elif (dataset == 'test'):
            self.set_test_data(self.get_test_data()[index_list])
            self.set_test_label(self.get_test_label()[index_list])
        return
    #========================================================


    #===== Full Train Set & Full Train Label ================
    #----- Setter -------------------------------------------
    def set_full_train_set(self, np_2D_training_data, training_label,
                        datatype=np.float32):
        """
        Set the full training set in the given data type (default:
        float32) with accompanying labels. If only a smaller part of
        this full training set is used for training, the training set
        will be randomly drawn from the full training set.
        """
        self.set_full_train_data(np_2D_training_data, datatype)
        self.set_full_train_label(training_label)

    def set_full_train_data(self, np_2D_training_data, datatype=np.float32):
        self._fulltraindata = np.asarray(np_2D_training_data,dtype=datatype)

    def set_full_train_label(self, training_label):
        self._fulltrainlabel = np.asarray(training_label, dtype=int)

    #----- Getter -------------------------------------------
    def get_full_train_set(self):
        return self._fulltraindata, self._fulltrainlabel

    def get_full_train_data(self):
        return self._fulltraindata

    def get_full_train_label(self):
        return self._fulltrainlabel

    #----- Properties ---------------------------------------
    def get_full_train_label_names(self):
        return self._get_label_names(self._fulltrainlabel)
    #========================================================


    #===== Train Set & Train Label ==========================
    #----- Setter -------------------------------------------
    def set_train_set(self, np_2D_training_data, training_label,
                      datatype=np.float32):
        """
        Set the training set in the given data type (default: float32)
        with accompanying labels. This set will be used during all
        training iterations of a single run.
        """
        self.set_train_data(np_2D_training_data, datatype)
        self.set_train_label(training_label)

    def set_train_data(self, np_2D_training_data, datatype=np.float32):
        self._traindata = np.asarray(np_2D_training_data, dtype=datatype)

    def set_train_label(self, training_label):
        self._trainlabel = np.asarray(training_label, dtype=int)

    #----- Getter -------------------------------------------
    def get_train_set(self):
        return self._traindata, self._trainlabel

    def get_train_data(self):
        return self._traindata

    def get_train_label(self):
        return self._trainlabel

    #----- Properties ---------------------------------------
    def number_of_train_labels(self):
        return (self._trainlabel[np.where(self._trainlabel >= 0)]).shape[0]

    def get_train_label_names(self):
        return self._get_label_names(self._trainlabel)
    #========================================================


    #===== Full Test Set & Full Test Label ==================
    #----- Setter -------------------------------------------
    def set_full_test_set(self, np_2D_test_data, test_label, datatype=np.float32):
        """
        Set the full test set in the given data type (default: float32)
        with accompanying labels. If only a smaller part of this full
        test set is used for testing, the test set will be randomly
        drawn from the full test set.
        """
        self.set_full_test_data(np_2D_test_data, datatype)
        self.set_full_test_label(test_label)

    def set_full_test_data(self, np_2D_test_data, datatype=np.float32):
        self._fulltestdata = np.asarray(np_2D_test_data, dtype=datatype)

    def set_full_test_label(self, test_label):
        self._fulltestlabel = np.asarray(test_label, dtype=int)

    #----- Getter -------------------------------------------
    def get_full_test_set(self):
        return self._fulltestdata, self._fulltestlabel

    def get_full_test_data(self):
        return self._fulltestdata

    def get_full_test_label(self):
        return self._fulltestlabel

    #----- Properties ---------------------------------------
    def get_full_test_label_names(self):
        return self._get_label_names(self._fulltestlabel)
    #========================================================


    #===== Test Set & Test Label ============================
    #----- Setter -------------------------------------------
    def set_test_set(self, np_2D_test_data, test_label, datatype=np.float32):
        """
        Set the test set in the given data type (default: float32) with
        accompanying labels. This set will be used during all training
        iterations of a single run.
        """
        self.set_test_data(np_2D_test_data, datatype)
        self.set_test_label(test_label)

    def set_test_data(self, np_2D_test_data, datatype=np.float32):
        self._testdata = np.asarray(np_2D_test_data, dtype=datatype)

    def set_test_label(self, test_label):
        self._testlabel = np.asarray(test_label, dtype=int)

    #----- Getter -------------------------------------------
    def get_test_set(self):
        return self._testdata, self._testlabel

    def get_test_data(self):
        return self._testdata

    def get_test_label(self):
        return self._testlabel

    #----- Properties ---------------------------------------
    def get_test_label_names(self):
        return self._get_label_names(self._testlabel)
    #========================================================

    #----- Utility ---------------------------------------
    def labelname(self,label):
        return self._labelnames[label]
    #========================================================


    def _devide_list(self, index_list, nthpart, nparts):
        """
        Divide index list into [nparts] Parts and return the [nthpart]th
        part. For parallelization use nthpart=rank and nparts=size to
        distribute the index list between processes.
        """
        size = len(index_list)
        first = nthpart*(size/nparts)
        if (nthpart <= size%nparts):
            first += nthpart
        else:
            first += size%nparts
        last = first + size/nparts
        if (nthpart < size%nparts):
            last += 1
        return index_list[first:last]


    def _read_set_from(self, dsetname, dset, path, classes=None):
        # TODO: label names for all data sets
        # TODO: delete classes selection, since this will only be used for
        # storing the complete set as h5 file. Also rearrange function/name to
        # make this clear.
        label_names = None
        # MNIST
        if (dsetname == 'MNIST'):
            import classes.input.mnist as mnist
            nparray_2D, label, label_names = mnist.read_images_from_mnist(
                dset, path, classes)
        elif (dsetname == 'MNIST540k'):
            # add MNIST data points translated by one pixel up, down,
            # left, right and along the diagonals
            import classes.input.mnist as mnist
            nparray_2D, label, label_names = mnist.read_images_from_mnist(
                dset, path, classes)
            if(dset == 'train'):
                nparray_2D, label = mnist.add_translated_images(
                    nparray_2D, label)
        elif (dsetname == 'MNIST+r180k'):
            # add MNIST data points rotated by a given list of angles
            angles = [-10,+10]
            import classes.input.mnist as mnist
            nparray_2D, label, label_names = mnist.read_images_from_mnist(
                dset, path, classes)
            if(dset == 'train'):
                nparray_2D, label = mnist.add_rotated_images(
                    nparray_2D, label, angels)
            # print nparray_2D.shape
        # CIFAR-10
        elif (dsetname == 'CIFAR-10'):
            import classes.input.cifar as cifar
            nparray_2D, label = cifar.read_images_from_cifar(dset, path, classes)
        # SVHN
        elif (dsetname == 'SVHN'):
            import classes.input.svhn as svhn
            nparray_2D, label = svhn.read_images_from_svhn(dset, path, classes)
        # artificial shapes
        elif (dsetname == 'artificial'):
            import classes.input.artificial as artificial
            nparray_2D, label = artificial.generate_artificial_data(
                dset, path, classes)
        # squares (artificial data from KeckEtAl2012)
        elif (dsetname == 'squares'):
            import classes.input.squares as squares
            nparray_2D, label = squares.read_images_from_squares(
                dset, path, classes)
        # IRIS
        elif (dsetname == 'IRIS'):
            import classes.input.iris as iris
            nparray_2D, label, label_names = iris.read_data_from_iris(
                dset, path, classes)
        # CalTechS (CalTech Silhouettes)
        elif (dsetname == 'CalTechS'):
            import classes.input.caltechs as caltechs
            nparray_2D, label, label_names = caltechs.read_data_from_caltechs(
                dset, path, classes)
        # 20 Newsgroups
        elif (dsetname == '20Newsgroups-tf-idf'):
            import classes.input.news as news
            nparray_2D, label, label_names = news.read_data_from_20news(
                dset, path, classes)
        return nparray_2D, label, label_names

    def _get_label_names(self, labels):
        return [self._labelnames[i] for i in labels]