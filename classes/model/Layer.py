# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

from mpi4py import MPI
import numpy as np
import classes.model.layermodels as layermodels
from layermodels import *
import h5py
#import time

class InputLayer(object):
    '''
    The Input-Layer stores the Training-Data, performs pre-processing
    operations and forwards the data to the processing layers
    '''

    def __init__(self, level):
        self._Y = None
        self._Label = None
        self._A = None
        self._nextdatapoint = 0
        self._inputsource = None
        self._level = level

    def set_input(self, inputdata, input_label, shuffle=True):
        self._Y = inputdata
        self._Label = input_label
        if shuffle:
            self.shuffle()

    def get_input_data(self):
        return self._Y

    def get_input_label(self):
        return self._Label

    def output_data(self):
        return self._Y[self._nextdatapoint]

    def output_label(self):
        return self._Label[self._nextdatapoint]

    def next_datapoint(self):
        self._nextdatapoint += 1
        if (self._nextdatapoint == self._Y.shape[0]):
            self._nextdatapoint = 0
            self.shuffle()

    def shuffle(self):
        index = np.arange(self._Y.shape[0])
        np.random.shuffle(index)
        self._Y = self._Y[index]
        self._Label = self._Label[index]

    def set_normalization(self, A):
        self._A = A

    def get_normalization(self):
        return self._A

    def set_inputsource(self, inputsource):
        if isinstance(inputsource, tuple):
            self._inputsource = inputsource
        else:
            self._inputsource = (inputsource,)

    def get_inputsource(self):
        return self._inputsource

    def output(self, inputdata):
        if (self._A is not None):
            return (self._A-inputdata.shape[0])*inputdata/np.sum(inputdata)+1.
        else:
            return inputdata

    def normalize_input(self, normalization=None, offset=1.):
        if (normalization is None):
            normalization = self._A
        if (normalization is not None):
            sumY = np.sum(self._Y,1)
            self._Y = (normalization-self._Y.shape[1])*self._Y/sumY + offset

    def normalize_inputs(self, normalization=None, offset=1.):
        if (normalization is None):
            normalization = self._A
        if (normalization is not None):
            if (len(self._Y.shape) == 2):  # Grayscale Image
                sumY = np.sum(self._Y,1,dtype=float)
                try: # faster, but quite memory intensive
                    self._Y = (normalization-self._Y.shape[1])\
                              *self._Y/sumY[:,np.newaxis] + offset
                except:
                    for i in xrange(self._Y.shape[0]):
                        self._Y[i,:] = (normalization-self._Y.shape[1])\
                              *self._Y[i,:]/sumY[i] + offset
            elif (len(self._Y.shape) == 3):  # RGB Image
                #--------------------------------------------------------------
                # Comprehensive Normalization
                # (compare Finlayson et al 1998)
                #--------------------------------------------------------------
                for _ in xrange(10):
                    delta = np.copy(self._Y)
                    #----------------------------------------------------------
                    # 1. y_d = 1./(r_d + g_d + b_d) * y_d
                    denominator1 = np.sum(self._Y,axis=2,dtype=float)
                    # catch division by zero
                    denominator1[np.where(denominator1 == 0.)] = 1.
                    self._Y = self._Y/denominator1[:,:,np.newaxis]
                    #----------------------------------------------------------
                    # 2. y_d,i = (D/3 /sum_d y_d,i) * y_d,i
                    denominator2 = np.sum(self._Y,axis=1,dtype=float)
                    self._Y = self._Y.shape[1]/3./denominator2[:,np.newaxis,:]\
                              *self._Y
                    delta = np.sum(np.abs(delta-self._Y))
                    #print self._Y[0,0,:]
                    print delta
                # catch 'true black' pixels and set them to average gray
                #self._Y[np.where(denominator1 == 0.)] = 1./3.
                self._Y *= (normalization-self._Y.shape[1])\
                           /float(self._Y.shape[1]) + offset
                print np.sum(np.sum(self._Y,axis=2),axis=1)[0:20]
                #print np.sum(self._Y,axis=1)[20:40,:]
                #print np.sum(self._Y,axis=1)[40:60,:]
                #print np.sum(self._Y,axis=1)[60:80,:]
                #print np.sum(self._Y,axis=1)[80:100,:]
                #--------------------------------------------------------------

    def imagecount(self):
        return self._Y.shape[0]

    def initialize_theano_variables(self):
        import theano
        import theano.tensor as T
        # for mb=1
        # self.s_t = T.matrix("s_%d.%d"%(self._level[0],self._level[1]), dtype='float32')
        # self.L_t = T.vector("L", dtype='int32')
        self.s_t = T.tensor3("s_%d.%d"%(self._level[0],self._level[1]), dtype='float32')
        self.L_t = T.matrix("L", dtype='int32')
        return

    def sequences(self, mode='train'):
        """
        Return the theano variables which are sequences in the learning
        iteration, i.e. which change values between the iterations by a
        given pattern which is independent from the learning iteration.
        E.g. different input data points, or decreasing learning rates.
        """
        if (mode == 'train'):
            sequences = [self.s_t, self.L_t]
        elif (mode == 'test'):
            sequences = [self.s_t]
        elif (mode == 'likelihood'):
            sequences = [self.s_t, self.L_t]
        return sequences

    def outputs_info(self, mode='train'):
        """
        Return the theano variables which are changed through the
        learning iteration. E.g. the adaptive synaptic weights.
        """
        if (mode == 'train'):
            outputs_info = []
        elif (mode == 'test'):
            outputs_info = []
        elif (mode == 'likelihood'):
            outputs_info = []
        return outputs_info

    def non_sequences(self, mode='train'):
        """
        Return the theano variables which are constant throughout the
        learning iteration. E.g. fixed learning rates.
        """
        if (mode == 'train'):
            non_sequences = []
        elif (mode == 'test'):
            non_sequences = []
        elif (mode == 'likelihood'):
            non_sequences = []
        return non_sequences


class ProcessingLayer(object):
    '''
    The Processing Layer receives its data from the Input Layer or
    previous Processing Layers and processes the data according to its
    learning rules.
    '''

    def __init__(self, level):
        self.A = None
        self.C = None
        self.D = None
        self.epsilon = None
        self.W = None
        self.mini_batch_size = None

        self._layermodel = None
        self._inputsource = None
        self._mini_batch_count = 0
        self._comm = MPI.COMM_WORLD
        self._level = level
        #self.elapsed = time.time() - time.time()
        #self.comm_time = time.time() - time.time()
        #self.ncomm = 0


    def normalize_input(self, input_data, normalization=None, offset=1.):
        if (normalization is None):
            normalization = self.A
        inputsum = np.sum(input_data)
        input_data = (normalization-input_data.shape[0])*\
            input_data/inputsum + offset
        return input_data

    def normalize_inputs(self, input_ddata, normalization=None, offset=1.):
        if (normalization is None):
            normalization = self.A
        inputsum = np.sum(input_ddata,1)
        input_ddata = (normalization-input_ddata.shape[1])*\
            input_ddata/inputsum[:,np.newaxis] + offset
        return input_ddata

    def set_learningrate(self, epsilon):
        self._layermodel.epsilon = epsilon

    def get_learningrate(self):
        return self._layermodel.epsilon

    def set_weights(self, W):
        self._layermodel.set_weights(W)

    def get_weights(self):
        return self._layermodel.get_weights()

    def set_inputsource(self, inputsource):
        if isinstance(inputsource, tuple):
            self._inputsource = inputsource
        else:
            self._inputsource = (inputsource,)

    def get_inputsource(self):
        return self._inputsource

    def set_model(self, model, model_args):
        if not model_args['Theano']:
            if (model == 'Poisson'):
                self._layermodel = layermodels.poisson.Poisson()
            elif (model == 'PoissonRecurrent'):
                self._layermodel = layermodels.poisson.Poisson_Recurrent()
            elif (model == 'MM-LabeledOnly'):
                self._layermodel = layermodels.mixturemodel.MixtureModel(
                    useunlabeled=False)
            elif (model == 'MM'):
                self._layermodel = layermodels.mixturemodel.MixtureModel(
                    useunlabeled=True)
        # Using Theano Functions (necessary for GPU computation):
        elif model_args['Theano']:
            if not model_args['Scan']:
                if (model == 'Poisson'):
                    self._layermodel = layermodels.poisson_theano.Poisson(
                        C=model_args['C'], D=model_args['D'])
                elif (model == 'PoissonRecurrent'):
                    self._layermodel = \
                        layermodels.poisson_theano.Poisson_Recurrent(
                        C=model_args['C'], D=model_args['D'])
                elif (model == 'MM-LabeledOnly'):
                    self._layermodel = \
                        layermodels.mixturemodel_theano.MixtureModel(
                            C=model_args['C'], D=model_args['D'],
                            use_unlabeled=False)
                elif (model == 'MM'):
                    self._layermodel = \
                        layermodels.mixturemodel_theano.MixtureModel(
                            C=model_args['C'], D=model_args['D'],
                            use_unlabeled=True)
            elif model_args['Scan']:
                inputsource = np.empty(0, dtype=int).reshape((0,2))
                nmultilayer = None
                nlayer = None
                for source in self._inputsource:
                    if source[0:10] == 'MultiLayer':
                        nmultilayer = int(source[10:])
                    else:
                        if nmultilayer is None:
                            nmultilayer = self._level[0]
                        if source[0:10] == 'InputLayer':
                            nlayer = 0
                        elif source[0:15] == 'ProcessingLayer':
                            nlayer = int(source[15:])
                    if ((nmultilayer is not None) and (nlayer is not None)):
                        inputsource = np.vstack((inputsource,
                                                [nmultilayer, nlayer]))
                        nmultilayer = None
                        nlayer = None
                if (model == 'Poisson'):
                    self._layermodel = layermodels.poisson_theano_scan.\
                        Poisson(
                            self._level[0], self._level[1], inputsource)
                elif (model == 'PoissonRecurrent'):
                    self._layermodel = layermodels.poisson_theano_scan.\
                        Poisson_Recurrent(
                            self._level[0], self._level[1], inputsource)
                elif (model == 'MM-LabeledOnly'):
                    self._layermodel = layermodels.mixturemodel_theano_scan.\
                        MixtureModel(
                            self._level[0], self._level[1], inputsource)
                elif (model == 'MM'):
                    print 'ERROR: MM Model not supported for theano.scan'

    def sequences(self, mode='train'):
        """
        Return the theano variables which are sequences in the learning
        iteration, i.e. which change values between the iterations by a
        given pattern which is independent from the learning iteration.
        E.g. different input data points, or decreasing learning rates.
        """
        return self._layermodel.sequences(mode)

    def outputs_info(self, mode='train'):
        """
        Return the theano variables which are changed through the
        learning iteration. E.g. the adaptive synaptic weights.
        """
        return self._layermodel.outputs_info(mode)

    def non_sequences(self, mode='train'):
        """
        Return the theano variables which are constant throughout the
        learning iteration. E.g. fixed learning rates.
        """
        return self._layermodel.non_sequences(mode)

    def initialize_weights(self, Y=None, L=None, method=None, h5path=None, h5file=None):
        W = np.zeros(shape=((self.C,)+self.D), dtype='float32')
        if (self._comm.Get_rank() == 0):
            if (method is None):
                if (Y is None):
                    method = 'random'
                elif (Y is not None):
                    method = 'input'

            if (method == 'random'):
                # random initialization with normalization
                W = np.random.random_sample((self.C,)+self.D)
                if (self.A is not None):
                    if (self.A-self.D[0] <= 0):
                        print "ERROR: Normalization Constant too low"
                    W = (self.A-self.D[0])*W/np.sum(W,1)[:,np.newaxis] + 1

            elif (method == 'even'):
                # (for classification layer) initialize all weights with
                # same value
                W = np.ones(shape=((self.C,)+self.D), dtype='float32')\
                    *1./float(self.D[0])

            elif (method == 'class'):
                # (for classification layer) initialize all weights as a
                # segregative map, e.g., for N/#classes = 2:
                # W(0) = (0.5, 0.5, 0, ..., 0),
                # W(1) = (0, 0, 0.5, 0.5, 0, ..., 0),
                # ...,
                # W(#classes-1) = (0,...,0,0.5,0.5)
                N = self.D[0]/self.C * np.ones(shape=(self.C), dtype=int)
                N[0:self.D[0]%self.C] += 1
                for i in xrange(self.C):
                    W[i,np.sum(N[0:i]):np.sum(N[0:i+1])] = 1.
                W = W/np.sum(W,1)[:,np.newaxis]

            elif ((method == 'input') & (Y is not None)):
                #--- initialize weights by using the input statistics
                try: # fast but memory intensive for big data sets
                    m_d = np.mean(Y, axis=0)
                except:
                    m_d = np.zeros_like(Y[0])
                    for i in xrange(Y.shape[0]):
                        m_d += Y[i]
                    m_d /= Y.shape[0]
                try: # fast but memory intensive for big data sets
                    v_d = np.var(Y, axis=0)
                except:
                    v_d = np.zeros_like(Y[0])
                    for i in xrange(Y.shape[0]):
                        v_d += (Y[i] - m_d) ** 2
                    v_d /= Y.shape[0]
                for i in xrange(self.C):
                    W[i] = m_d + 2.*np.sqrt(v_d)*np.random.random_sample(self.D)

            elif ((method == 'input_bylabel') & (Y is not None) & (L is not None)):
                #--- initialize weights by using the input statistics
                nlabels = np.max(L)
                m = np.zeros_like(W)
                for l in xrange(nlabels):
                    m = np.mean(Y[np.where(L==l)], axis=0)
                    std = np.std(Y[np.where(L==l)], axis=0)
                    for i in xrange(l*self.C/nlabels, (l+1)*self.C/nlabels):
                        W[i] = m + 2.*std*np.random.random_sample(self.D)

            elif ((method == 'singleinput') & (Y is not None)):
                #--- initialize weights by setting each to different input image
                ind = np.random.permutation(Y.shape[0])
                W = Y[ind[0:self.C]]

            elif (method == 'h5'):
                # initialize weights by using a h5 file, e.g., to resume a
                # previous computation
                h5file = h5py.File("%s/%s"%(h5path,h5file), 'r')
                if (len(h5file['W'].shape) == 2):
                    W = h5file['W'][:,:]
                elif (len(h5file['W'].shape) == 3):
                    W = h5file['W'][0,:,:]
                h5file.close()
        W = self._comm.bcast(W, root=0)
        self._layermodel.set_weights(W.astype('float32'))

    def learningstep(self, multilayer, input_data, input_label):
        # if A is None there will be no explicit normalization of the
        # inputs (if the data comes from an input layer this is in
        # general not necessary)
        if(self.A is not None):
            input_data = self.normalize_input(input_data)
        self._layermodel.feed(self, multilayer, input_data, input_label,
                              mode='train')
        self._mini_batch_count += 1
        if (self._mini_batch_count == self.mini_batch_size):
            self._layermodel.update()
            self._mini_batch_count = 0
        return

    def blank_step(self):
        """
        Perform a virtual training step without any input. This is only
        needed for paralellization, so that the reduce doesn't get stuck
        on processes with less data points.
        """
        self._mini_batch_count += 1
        if (self._mini_batch_count == self.mini_batch_size):
            self._layermodel.update()
            self._mini_batch_count = 0
        return

    def activation(self):
        return self._layermodel.activation()

    def output(self, multilayer, input_data):
        if(self.A is not None):
            input_data = self.normalize_input(input_data)
        self._layermodel.feed(self, multilayer, input_data, input_label=-1,
                              mode='test')
        return self._layermodel.activation()