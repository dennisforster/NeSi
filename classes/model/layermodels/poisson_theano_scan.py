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
class Poisson(LayerModel_Theano_Scan):
    """(FF-)Mixture of Poisson layer for theano.scan calculation"""

    def __init__(self, nmultilayer, nlayer, input_source):
        self.W_t = T.matrix("W_%d.%d"%(nmultilayer,nlayer), dtype='float32')
        self.s_t = T.matrix("s_%d.%d"%(nmultilayer,nlayer), dtype='float32')
        self._s = None
        self.epsilon_t = T.scalar("epsilon_%d.%d"%(nmultilayer,nlayer),
                                  dtype='float32')
        self._nmultilayer = nmultilayer
        self._nlayer = nlayer
        self._input_source = input_source
        # _input_source gives for each input variable which is not from
        # this layer the multilayer and the layer of its source:
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
            sequences = []
        return outputs_info

    @doc_inherit
    def non_sequences(self, mode='train'):
        if (mode == 'train'):
            non_sequences = [self.epsilon_t]
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
    def learningstep(self, Y, W, epsilon):
        # activation
        s = self._activation(Y,W)
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        # weight update
        W_new = W + epsilon*(T.tensordot(s,Y,axes=[0,0]) -
                             T.sum(s,axis=0)[:,np.newaxis]*W)
        W_new.name = 'W_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s, W_new

    def learningstep_m1(self, Y, W, epsilon):
        """Perform a single learning step.

        This is a faster learning step for the case of
        mini-batch-size = 1.

        Keyword arguments:
        the keyword arguments must be the same as given in
        self.input_parameters(mode) for mode='train'.
        """
        # Input integration:
        I = T.dot(T.log(W),Y)
        # numeric trick to prevent overflow in the exp-function
        max_exponent = 88. - T.log(I.shape[0]).astype('float32')
        scale = theano.ifelse.ifelse(T.gt(I[T.argmax(I)], max_exponent),
            I[T.argmax(I)] - max_exponent, 0.)
        # activation: softmax with overflow protection
        s = T.exp(I-scale)/T.sum(T.exp(I-scale))
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        # weight update
        W_new = W + epsilon*(T.outer(s,Y) - s[:,np.newaxis]*W)
        W_new.name = 'W_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s, W_new

    @doc_inherit
    def teststep(self, Y, W):
        # activation
        s = self._activation(Y,W)
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s

    @doc_inherit
    def set_weights(self, W):
        self.W = W

    @doc_inherit
    def get_weights(self):
        return self.W

    def _activation(self, Y, W):
        """Returns the activation for a given input.

        Derived from the generative model formulation of Poisson
        mixtures, the formular for the activation in the network reads
        as follows:
        I_c = \sum_d \log(W_{cd})y_d
        s_c = softmax(I_c)
        """
        # input integration
        I = T.tensordot(Y,T.log(W),axes=[1,1])
        # activation
        # to prevent numerical over- or underflows in the exponential, a
        # scaling factor in the softmax function is used, utilizing the
        # identity:
        # exp(x_i)/sum_i(exp(x_i)) = exp(x_i-a)/sum_i(exp(x_i-a))
        max_exponent = 86. - T.log(I.shape[1]).astype('float32')
        scale = T.switch(
            T.gt(T.max(I, axis=1, keepdims=True), max_exponent),
            T.max(I, axis=1, keepdims=True) - max_exponent,
            0.)
        s = T.exp(I-scale)/T.sum(T.exp(I-scale), axis=1, keepdims=True)
        return s

#------------------------------------------------------------------------------
class Poisson_Recurrent(LayerModel_Theano_Scan):
    """Recurrent Mixture of Poisson layer for theano.scan calculation"""

    def __init__(self, nmultilayer, nlayer, input_source):
        self.W = None
        self.W_t = T.matrix("W_%d.%d"%(nmultilayer,nlayer), dtype='float32')
        self.s_t = T.matrix("s_%d.%d"%(nmultilayer,nlayer), dtype='float32')
        self.epsilon_t = T.scalar("epsilon_%d.%d"%(nmultilayer,nlayer),
                                  dtype='float32')
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
            non_sequences = [self.epsilon_t]
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
                'W_%d.%d[t-1]'%(self._input_source[1][0], self._input_source[1][1]),
                'W_%d.%d[t-1]'%(self._nmultilayer, self._nlayer),
                'epsilon_%d.%d'%(self._nmultilayer, self._nlayer)
                ]
        elif (mode == 'test'):
            parameters = [
                's_%d.%d[t]'%(self._input_source[0][0], self._input_source[0][1]),
                'W_%d.%d'%(self._input_source[1][0], self._input_source[1][1]),
                'W_%d.%d'%(self._nmultilayer, self._nlayer)
                ]
        return parameters

    @doc_inherit
    def learningstep(self, Y, L, M, W, epsilon):
        s = self._activation(Y,L,M,W)
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        # weight update
        W_new = W + epsilon*(T.tensordot(s,Y,axes=[0,0]) -
                             T.sum(s,axis=0)[:,np.newaxis]*W)
        W_new.name = 'W_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s, W_new

    def learningstep_m1(self, Y, L, M, W, epsilon):
        """Perform a single learning step.

        This is a faster learning step for the case of
        mini-batch-size = 1.

        Keyword arguments:
        the keyword arguments must be the same as given in
        self.input_parameters(mode) for mode='train'.
        """
        # Input integration:
        I = T.dot(T.log(W),Y)
        # recurrent term:
        vM = theano.ifelse.ifelse(
            T.eq(L,-1), # if no label is provided
            T.sum(M, axis=0),
            M[L,:]
            )
        # numeric trick to prevent overflow in the exp-function:
        max_exponent = 88. - T.log(I.shape[0]).astype('float32')
        scale = theano.ifelse.ifelse(T.gt(I[T.argmax(I)], max_exponent),
            I[T.argmax(I)] - max_exponent, 0.)
        # activation: recurrent softmax with overflow protection
        s = vM*T.exp(I-scale)/T.sum(vM*T.exp(I-scale))
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        # weight update
        W_new = W + epsilon*(T.outer(s,Y) - s[:,np.newaxis]*W)
        W_new.name = 'W_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s, W_new

    @doc_inherit
    def teststep(self, Y, M, W):
        # activation
        L = (-1)*T.ones_like(Y[:,0], dtype='int32')
        s = self._activation(Y,L,M,W)
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s

    @doc_inherit
    def set_weights(self, W):
        self.W = W

    @doc_inherit
    def get_weights(self):
        return self.W

    def _activation(self, Y, L, M, W):
        """Returns the activation for a given input.

        Derived from the generative model formulation of hierarchical
        Poisson mixtures, the formular for the activation in the network
        reads as follows:
        I_c =
         \sum_d \log(W_{cd})y_d + \log(M_{lc})        for labeled data
         \sum_d \log(W_{cd})y_d + \log(\sum_k M_{kc}) for unlabeled data
        s_c = softmax(I_c)
        """
        # Input integration:
        I = T.tensordot(Y,T.log(W),axes=[1,1])
        # recurrent term:
        vM = M[L]
        L_index = T.eq(L,-1).nonzero()
        vM = T.set_subtensor(vM[L_index], T.sum(M, axis=0))
        # numeric trick to prevent overflow in the exp-function
        max_exponent = 86. - T.ceil(T.log(I.shape[1]).astype('float32'))
        scale = T.switch(
            T.gt(T.max(I, axis=1, keepdims=True), max_exponent),
            T.max(I, axis=1, keepdims=True) - max_exponent,
            0.)
        # numeric approximation to prevent underflow in the exp-function:
        # map too low values of I to a fixed minimum value
        min_exponent = -87. + T.ceil(T.log(I.shape[1]).astype('float32'))
        I = T.switch(
            T.lt(I-scale, min_exponent),
            scale+min_exponent,
            I)
        # activation: recurrent softmax with overflow protection
        s = vM*T.exp(I-scale)/T.sum(vM*T.exp(I-scale), axis=1, keepdims=True)
        return s