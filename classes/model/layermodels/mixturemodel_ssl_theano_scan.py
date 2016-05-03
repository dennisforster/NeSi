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
            T.scalar("epsilon_%d.%d"%(nmultilayer,nlayer), dtype='float32'),
            T.scalar("threshold_%d.%d"%(nmultilayer,nlayer), dtype='float32')
            ]
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
                'epsilon_%d.%d'%(self._nmultilayer, self._nlayer),
                'threshold_%d.%d'%(self._nmultilayer, self._nlayer)
                ]
        elif (mode == 'test'):
            parameters = [
                's_%d.%d[t]'%(self._input_source[0][0], self._input_source[0][1]),
                'W_%d.%d'%(self._nmultilayer, self._nlayer)
                ]
        return parameters

    @doc_inherit
    def learningstep(self, Y, L, W, epsilon, threshold):
        s = self._activation(Y,L,W,threshold)
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        W_new = W + epsilon*(T.tensordot(s,Y,axes=[0,0]) -
                             T.sum(s,axis=0)[:,np.newaxis]*W)
        W_new.name = 'W_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        return s, W_new

    def learningstep_m1(self, Y, L, W, epsilon): # learning step for mb=1
        """Perform a single learning step.

        This is a faster learning step for the case of
        mini-batch-size = 1.

        Keyword arguments:
        the keyword arguments must be the same as given in
        self.input_parameters(mode) for mode='train'.
        """
        s = theano.ifelse.ifelse(
            T.eq(L,-1), # if no label is provided
            T.sum(W/T.sum(W, axis=0)*Y, axis=1), # Inference
            T.eye(W.shape[0], dtype='float32')[L,:] # Training
            )
        s.name = 's_%d.%d[t]'%(self._nmultilayer,self._nlayer)
        # update weights only if a label is provided
        W_new = theano.ifelse.ifelse(
            T.eq(L,-1),
            W,
            W + epsilon*(T.outer(s,Y) - s[:,np.newaxis]*W)
            )
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

    def _activation(self, Y, L, W, threshold):
        """Return the activation for a given input."""
        s = T.zeros((L.shape[0],W.shape[0]), dtype='float32')
        # use inference of all unlabeled data where p_max > threshold
        # nonzero activation only if label provided
        # L = T.switch(
        #     T.eq(L,-1), # if no label is provided
        #     T.switch(
        #         T.gt(
        #             T.max(
        #                 T.tensordot(
        #                     Y,
        #                     T.switch(T.eq(W,0), 0, W/T.sum(W, axis=0)),
        #                     axes=[1,1]),
        #                 axis=1
        #             ),
        #             0.2
        #         ),
        #         T.cast(T.argmax(
        #             T.tensordot(
        #                 Y,
        #                 T.switch(T.eq(W,0), 0, W/T.sum(W, axis=0)),
        #                 axes=[1,1]),
        #             axis=1),'int32'
        #         ),
        #         L
        #         ),
        #     L
        #     )

        # use inference of all unlabeled data
        # L = T.switch(
        #     T.eq(L,-1), # if no label is provided
        #     T.cast(T.argmax(
        #         T.tensordot(
        #             Y,
        #             T.switch(T.eq(W,0), 0, W/T.sum(W, axis=0)),
        #             axes=[1,1]),
        #         axis=1),'int32'),
        #     L
        #     )

        # use inference of all unlabeled data where p_max - p_max2 > threshold
        # nonzero activation only if label provided
        # inference = T.tensordot(
        #         Y,
        #         T.switch(T.eq(W,0), 0, W/T.sum(W, axis=0)),
        #         axes=[1,1])
        # L = T.switch(
        #     T.eq(L,-1), # if no label is provided
        #     T.switch(
        #         T.gt(T.sort(inference, axis=1)[:,-1]-T.sort(inference, axis=1)[:,-2], threshold),
        #         T.cast(T.argmax(inference,axis=1),'int32'),
        #         L
        #         ),
        #     L
        #     )
        # L_index = T.neq(L,-1).nonzero()
        # s = T.set_subtensor(s[L_index,L[L_index]], 1.)

        inference = T.tensordot(
                Y,
                T.switch(T.eq(W,0), 0, W/T.sum(W, axis=0)),
                axes=[1,1])
        BvSB = T.sort(inference,axis=1)[:,-1]-T.sort(inference,axis=1)[:,-2]
        L_inf = T.switch(
            T.eq(L,-1), # if no label is provided
            T.switch(
                T.gt(
                    BvSB,
                    threshold),
                T.cast(T.argmax(inference,axis=1),'int32'),
                -1
                ),
            -1
            )
        L_inf_index = T.neq(L_inf,-1).nonzero()
        L_index = T.neq(L,-1).nonzero()
        # set labeled
        s = T.set_subtensor(s[L_index,L[L_index]], 1.)
        # set unlabeled with inferred label
        # MAP
        s = T.set_subtensor(s[L_inf_index,L_inf[L_inf_index]], 1.)
        # weighted
        # s = T.set_subtensor(s[L_inf_index,L_inf[L_inf_index]], BvSB[L_inf_index])
        return s

    def _inference(self, Y, W):
        """Return the infered class label for a given input"""
        W_normalized = T.switch(T.eq(W,0), 0, W/T.sum(W, axis=0))
        s = T.tensordot(Y, W_normalized, axes=[1,1])
        return s