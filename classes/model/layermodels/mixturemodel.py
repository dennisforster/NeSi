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
class MixtureModel(LayerModel):
    """
    """

    def __init__(self, useunlabeled):
        self.W = None
        self.epsilon = None
        self.UseUnlabeled = useunlabeled
        self._s = None
        self._comm = MPI.COMM_WORLD

    @doc_inherit
    def feed(self, layer, multilayer, input_data, input_label, mode):
        self._s = self._activate_function(input_data, input_label, mode)
        if (mode == 'train'):
            self._accumulate_weight_update(input_data)
        return

    @doc_inherit
    def update(self):
        if (self._comm.Get_size() > 1):
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

    def _accumulate_weight_update(self, input_data):
        try:
            self._dsY += self._s[:,np.newaxis] * input_data
            self._ds += self._s
        except:
            self._dsY = self._s[:,np.newaxis] * input_data
            self._ds = self._s
        return

    def _activate_function(self, input_data, input_label, mode):
        s = np.zeros(self.W.shape[0])
        if (input_label != -1):
            s[input_label] = 1.
        else:
            if ((mode == 'test') or
                ((mode == 'train') and self.UseUnlabeled)):
                s = np.sum(input_data*self.W/np.sum(self.W, axis=0), axis=1)
                s = s/np.sum(s)
        return s