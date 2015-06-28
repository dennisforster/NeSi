# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import numpy as np
from mpi4py import MPI
from utils import accel

from _layermodels import LayerModel
from utils.decorators import DocInherit
doc_inherit = DocInherit

#------------------------------------------------------------------------------
class Poisson(LayerModel):
    """(FF-)Mixture of Poisson layer"""

    def __init__(self):
        self.W = None
        self.epsilon = None
        self._s = None
        self._dsY = None
        self._ds = None
        self._comm = MPI.COMM_WORLD

    @doc_inherit
    def feed(self, layer, multilayer, input_data, input_label, mode=''):
        #--- TODO: add selection option to config
        #-- select activate function
        I = self._activate_function(input_data)
        #self._lin_activate_function(W, input_data)
        #-- select competition function
        self._s = self._softmax(I)
        #self._max()
        if (mode == 'train'):
            self._accumulate_weight_update(input_data)
        return

    @doc_inherit
    def update(self):
        if (self._comm.Get_size() > 1):
            # multiple-thread parallel computing
            DsY = np.zeros_like(self._dsY)
            Ds = np.zeros_like(self._ds)
            self._comm.Allreduce(self._dsY, DsY, op=MPI.SUM)
            self._comm.Allreduce(self._ds, Ds, op=MPI.SUM)
            try:
                # Grayscale Image
                self.W += self.epsilon*(DsY - Ds[:,np.newaxis]*self.W)
            except:
                # RGB Image
                self.W += self.epsilon*(
                    DsY - Ds[:,np.newaxis,np.newaxis]*self.W
                    )
            self._dsY = np.zeros_like(self._dsY)
            self._ds = np.zeros_like(self._ds)
        else:
            # single-thread computing
            try:
                # Grayscale Image
                self.W += self.epsilon*(
                    self._dsY - self._ds[:,np.newaxis]*self.W
                    )
            except:
                # RGB Image
                self.W += self.epsilon*(
                    self._dsY - self._ds[:,np.newaxis,np.newaxis]*self.W
                    )
            self._dsY = np.zeros_like(self._dsY)
            self._ds = np.zeros_like(self._ds)
        return

    @doc_inherit
    def activation(self):
        return self._s

    @doc_inherit
    def set_weights(self, W):
        self.W = W

    @doc_inherit
    def get_weights(self):
        return self.W

    def _accumulate_weight_update(self,input_data):
        try:
            self._dsY += self._s[:,np.newaxis] * input_data
            self._ds += self._s
        except:
            self._dsY = self._s[:,np.newaxis] * input_data
            self._ds = self._s
        return

    def _activate_function(self, input_data):
        try:
            #I = np.dot(np.log(W),Y)
            I = np.dot(accel.log(self.W),input_data)
            #I = np.dot(self._log(W),input_data)
        except:
            I = np.sum(np.sum(
                input_data[np.newaxis,:,:]*accel.log(self.W),
                axis=1),axis=1)
            #TODO: Check This!
        return I.astype('float64')


    def _lin_activate_function(self, W, input_data):
        return np.dot(W,input_data)

    def _softmax(self, I):
        # softmax-function with overflow fix
        # over/underflow in np.exp(x) for approximately x > 700 or x < -740
        scale = 0
        if (I[np.argmax(I)] > 700):
            scale  = I[np.argmax(I)] - 700
        if (I[np.argmin(I)] < -740 + scale):
            I[np.argmin(I)] = -740 + scale
        return np.exp(I-scale) / np.sum(np.exp(I-scale))

    def _max(self, I):
        # max-function
        s = np.zeros_like(I)
        s[np.argmax(self._I)] = 1.
        return s


#------------------------------------------------------------------------------
class Poisson_Recurrent(LayerModel):

    def __init__(self):
        self.W = None
        self.epsilon = None

        self._s = None
        self._dsY = None
        self._ds = None
        self._comm = MPI.COMM_WORLD

    @doc_inherit
    def feed(self, layer, multilayer, input_data, input_label, mode='train'):
        if (input_label == -1):
            M = np.sum(
                multilayer.Layer[
                    int(layer.get_inputsource()[1][15])
                    ].get_weights(),
                axis=0
                )
        else:
            M = multilayer.Layer[
                int(layer.get_inputsource()[1][15])
                ].get_weights()[input_label,:]
        I = self._activate_function(input_data)
        self._s = self._recurrent_softmax(I,M)
        if (mode == 'train'):
            self._accumulate_weight_update(input_data)
        return

    @doc_inherit
    def update(self):
        if (self._comm.Get_size() > 1):
            # multiple-thread parallel computing
            DsY = np.zeros_like(self._dsY)
            Ds = np.zeros_like(self._ds)
            self._comm.Allreduce(self._dsY, DsY, op=MPI.SUM)
            self._comm.Allreduce(self._ds, Ds, op=MPI.SUM)
            try:
                # Grayscale Image
                self.W += self.epsilon*(DsY - Ds[:,np.newaxis]*self.W)
            except:
                # RGB Image
                self.W += self.epsilon*(
                    DsY - Ds[:,np.newaxis,np.newaxis]*self.W
                    )
            self._dsY = np.zeros_like(self._dsY)
            self._ds = np.zeros_like(self._ds)
        else:
            # single-thread computing
            try:
                # Grayscale Image
                self.W += self.epsilon*(
                    self._dsY - self._ds[:,np.newaxis]*self.W
                    )
            except:
                # RGB Image
                self.W += self.epsilon*(
                    self._dsY - self._ds[:,np.newaxis,np.newaxis]*self.W
                    )
            self._dsY = np.zeros_like(self._dsY)
            self._ds = np.zeros_like(self._ds)
        return

    @doc_inherit
    def activation(self):
        return self._s

    @doc_inherit
    def set_weights(self, W):
        self.W = W

    @doc_inherit
    def get_weights(self):
        return self.W

    def _accumulate_weight_update(self,input_data):
        try:
            self._dsY += self._s[:,np.newaxis] * input_data
            self._ds += self._s
        except:
            self._dsY = self._s[:,np.newaxis] * input_data
            self._ds = self._s
        return

    def _activate_function(self, input_data):
        #I = np.dot(np.log(self.W),Y)
        I = np.dot(accel.log(self.W),input_data)
        #I = np.dot(self._log(self.W),input_data)
        return I.astype('float64')

    def _recurrent_softmax(self, I, M):
        # softmax-function with overflow fix
        # float64: over/underflow in np.exp(x) for ~ x>700 or x<-740
        scale = 0
        if (I[np.argmax(I)] > 700):
            scale  = I[np.argmax(I)] - 700
        if (I[np.argmin(I)] < -740 + scale):
            I[np.argmin(I)] = -740 + scale
        # for float32:
        # if (I[np.argmax(I)] > 86):
        #     scale  = I[np.argmax(I)] - 86
        return M*np.exp(I-scale) / np.sum(M*np.exp(I-scale))