# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import theano
import theano.tensor as T
import numpy as np

from _layermodels import LayerModel_Theano_Scan
from utils.decorators import DocInherit
doc_inherit = DocInherit

#------------------------------------------------------------------------------
class MixtureModel(LayerModel_Theano_Scan):
    """
    """

    def __init__(self, nmultilayer, nlayer, input_source):
        self.W_t = T.matrix("W_%d.%d"%(nmultilayer,nlayer), dtype='float32')
        self.s_t = T.matrix("s_%d.%d"%(nmultilayer,nlayer), dtype='float32')
        self.parameters_t = [
            T.scalar("epsilon_%d.%d"%(nmultilayer,nlayer), dtype='float32'),]        
        self._nmultilayer = nmultilayer
        self._nlayer = nlayer
        self._input_source = input_source
        # _input_source gives for each input variable which is not from
        # this layer the multilayer and the Layer of its source:
        # _input_source[i][0]: MultiLayer of variable i
        # _input_source[i][1]: Layer of variable i

    @doc_inherit
    def sequences(self, mode='train'):
        if (mode == 'train'):
            sequences = []
        elif (mode == 'test'):
            sequences = []
        elif (mode == 'likelihood'):
            sequences = []
        return sequences

    @doc_inherit
    def outputs_info(self, mode='train'):
        if (mode == 'train'):
            outputs_info = [self.W_t]
        elif (mode == 'test'):
            outputs_info = [self.s_t]
        elif (mode == 'likelihood'):
            outputs_info = []
        return outputs_info

    @doc_inherit
    def non_sequences(self, mode='train'):
        if (mode == 'train'):
            non_sequences = self.parameters_t
        elif (mode == 'test'):
            non_sequences = [self.W_t]
        elif (mode == 'likelihood'):
            non_sequences = [self.W_t]
        return non_sequences

    @doc_inherit
    def input_parameters(self, mode='train'):
        if (mode == 'train'):
            parameters = [
                's_%d.%d[t]'%(self._input_source[0][0], self._input_source[0][1]),
                'L[t]',
                'W_%d.%d[t-1]'%(self._nmultilayer, self._nlayer),
                'epsilon_%d.%d'%(self._nmultilayer, self._nlayer)
                ]
        elif (mode == 'test'):
            parameters = [
                's_%d.%d[t]'%(self._input_source[0][0], self._input_source[0][1]),
                'W_%d.%d'%(self._nmultilayer, self._nlayer)
                ]
        return parameters

    @doc_inherit
    def learningstep(self, Y, L, W, epsilon):
        s = self._activation(Y,L,W)
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        W_new = W + epsilon*(T.tensordot(s,Y,axes=[0,0]) -
                             T.sum(s,axis=0)[:,np.newaxis]*W)
        W_new.name = 'W_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s, W_new

    @doc_inherit
    def teststep(self, Y, W):
        # activation
        s = self._inference(Y,W)
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s

    @doc_inherit
    def set_weights(self, W):
        self.W = W

    @doc_inherit
    def get_weights(self):
        return self.W

    def _activation(self, Y, L, W):
        """Return the activation for a given input."""
        s = T.ones((L.shape[0],W.shape[0]), dtype='float32')/T.cast(W.shape[0],'float32')
        return s

    def _inference(self, Y, W):
        """Return the infered class label for a given input"""
        W_normalized = T.switch(T.eq(W,0), 0, W/T.sum(W, axis=0))
        s = T.tensordot(Y, W_normalized, axes=[1,1])
        return s