# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import theano
import theano.tensor as T
import numpy as np

from _layermodels import LayerModel
from utils.decorators import DocInherit
doc_inherit = DocInherit

#------------------------------------------------------------------------------
class Poisson(LayerModel):
    """(FF-)Mixture of Poisson layer for theano calculation"""

    def __init__(self, C, D):
        self.W = theano.shared(np.ones((C,D), dtype='float32'))
        t_eps = T.scalar('epsilon', dtype='float32')
        t_Y = T.vector('Y', dtype='float32')
        t_I = T.vector('I', dtype='float32')
        t_s = T.vector('s', dtype='float32')
        self.input_integration = theano.function(
            [t_Y],
            T.dot(T.log(self.W),t_Y),
            allow_input_downcast=True
            )
        self.softmax = theano.function(
            [t_I],
            T.exp(t_I)/T.sum(T.exp(t_I)),
            allow_input_downcast=True
            )
        self.weight_update = theano.function(
            [t_Y,t_s,t_eps],
            self.W,
            updates={
                self.W:
                self.W + t_eps*(T.outer(t_s,t_Y) - t_s[:,np.newaxis]*self.W)
                },
            allow_input_downcast=True
            )
        self.epsilon = None
        self._Y = None
        self._s = None

    @doc_inherit
    def feed(self, layer, multilayer, input_data, input_label, mode=''):
        self._Y = input_data
        I = self.input_integration(self._Y)
        self._s = self.softmax(I-self._scale(I))
        return

    @doc_inherit
    def update(self):
        self.weight_update(self._Y,self._s,self.epsilon)
        return

    @doc_inherit
    def activation(self):
        return self._s

    @doc_inherit
    def set_weights(self, W):
        self.W.set_value(np.asarray(W, dtype='float32'))

    @doc_inherit
    def get_weights(self):
        return self.W.get_value()

    def _scale(self, I):
        # overflow fix for softmax-function
        # over/underflow in sum(T.exp(x)) (float32) for approximately
        # x > 88-ln(D) or x < -87
        scale = 0
        max = 88 - np.log(I.shape[0])
        min = 87
        if (I[np.argmax(I)] > max ):
            scale  = I[np.argmax(I)] - max
        if (I[np.argmin(I)] < -min + scale):
            I[np.argmin(I)] = -min + scale
        return scale

#------------------------------------------------------------------------------
class Poisson_Recurrent(LayerModel):
    """Recurrent Mixture of Poisson layer for theano calculation"""

    def __init__(self, C, D):
        self.W = theano.shared(np.ones((C,D), dtype='float32'))
        t_M = T.matrix('M', dtype='float32')
        t_vM = T.vector('M', dtype='float32')
        t_Y = T.vector('Y', dtype='float32')
        t_I = T.vector('I', dtype='float32')
        t_s = T.vector('s', dtype='float32')
        t_eps = T.scalar('epsilon', dtype='float32')
        self.input_integration = theano.function(
            [t_Y],
            T.dot(T.log(self.W),t_Y),
            allow_input_downcast=True
            )
        self.M_summation = theano.function(
            [t_M],
            T.sum(t_M, axis=0),
            allow_input_downcast=True
            )
        self.recurrent_softmax = theano.function(
            [t_I,t_vM],
            t_vM*T.exp(t_I)/T.sum(t_vM*T.exp(t_I)),
            allow_input_downcast=True
            )
        self.weight_update = theano.function(
            [t_Y,t_s,t_eps],
            self.W,
            updates={
                self.W:
                self.W + t_eps*(T.outer(t_s,t_Y) - t_s[:,np.newaxis]*self.W)
                },
            allow_input_downcast=True
            )
        self.epsilon = None
        self._Y = None
        self._s = None

    @doc_inherit
    def feed(self, layer, multilayer, input_data, input_label, mode=''):
        if (input_label == -1):
            # TODO: maybe build theano function using the shared weights
            #       of the upper layer
            vM = self.M_summation(multilayer.Layer[
                int(layer.get_inputsource()[1][15])
                ].get_weights())
            # vM = np.sum(multilayer.Layer[
            #     int(layer.get_inputsource()[1][15])
            #     ].get_weights(),axis=0)
        else:
            vM = multilayer.Layer[
                int(layer.get_inputsource()[1][15])
                ].get_weights()[input_label,:]
        self._Y = input_data
        I = self.input_integration(self._Y)
        self._s = self.recurrent_softmax(I-self._scale(I), vM)
        return

    @doc_inherit
    def update(self):
        self.weight_update(self._Y,self._s,self.epsilon)
        return

    @doc_inherit
    def activation(self):
        return self._s

    @doc_inherit
    def set_weights(self, W):
        self.W.set_value(np.asarray(W, dtype='float32'))

    @doc_inherit
    def get_weights(self, borrow_=False):
        return self.W.get_value(borrow=borrow_)

    def _scale(self, I):
        # overflow fix for softmax-function
        # over/underflow in sum(T.exp(x)) (float32) for approximately
        # x > 88-ln(D) or x < -87
        scale = 0
        max = 88 - np.log(I.shape[0])
        min = 87
        if (I[np.argmax(I)] > max ):
            scale  = I[np.argmax(I)] - max
        if (I[np.argmin(I)] < -min + scale):
            I[np.argmin(I)] = -min + scale
        return scale