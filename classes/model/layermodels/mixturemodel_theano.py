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
class MixtureModel(LayerModel):
    """
    """

    def __init__(self, C, D, use_unlabeled):
        self.W = theano.shared(np.ones((C,D), dtype='float32'))
        t_eps = T.scalar('epsilon', dtype='float32')
        t_Y = T.vector('Y', dtype='float32')
        t_s = T.vector('s', dtype='float32')
        self.activation_unlabeled = theano.function(
            [t_Y],
            T.sum(t_Y*self.W/T.sum(self.W, axis=0), axis=1),
            allow_input_downcast=True
            )
        self.activation_normalization = theano.function(
            [t_s],
            t_s/T.sum(t_s),
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
        self._delta = np.eye(C, dtype='float32')
        self._C = C
        self._use_unlabeled = use_unlabeled
        self._skipupdate = False

    @doc_inherit
    def feed(self, layer, multilayer, input_data, input_label, mode=''):
        self._Y = input_data
        if (input_label != -1):
            self._s = self._delta[input_label,:]
        else:
            if ((mode == 'test') or
                ((mode == 'train') and self._use_unlabeled)):
                self._s = self._activate_function(input_data, input_label, mode)
            else:
                self._s = np.zeros(self._C)
                # no need to update if s = (zeros)
                self._skipupdate = True
        return

    @doc_inherit
    def update(self):
        if self._skipupdate:
            self._skipupdate = False
        else:
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

    def _activate_function(self, input_data, input_label, mode):
        s = self.activation_unlabeled(input_data)
        s = self.activation_normalization(s)
        return s