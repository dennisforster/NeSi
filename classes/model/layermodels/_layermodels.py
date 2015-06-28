# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import abc

#-----------------------------------------------------------------------
# abstract class for layermodels that use numpy (cpu) calculations or
# theano without theano.scan
class LayerModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def feed(self, layer, multilayer, input_data, input_label):
        return

    @abc.abstractmethod
    def update(self):
        return

    @abc.abstractmethod
    def activation(self):
        return

    @abc.abstractmethod
    def set_weights(self, W):
        return

    @abc.abstractmethod
    def get_weights(self):
        return


#-----------------------------------------------------------------------
# abstract class for layermodels that use theano.scan
class LayerModel_Theano_Scan(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sequences(self, mode):
        """Return sequences variables for theano.scan.

        Return the theano variables which are sequences in the theano
        scan, i.e. which change values between iterations by a given
        pattern which is independent from the scan iteration.
        E.g. different input data points, or decreasing learning rates.

        Keyword arguments:
        mode -- e.g. 'train', 'test', or 'likelihood':
            define for which mode the sequences variables should be
            returned.
        """
        return

    @abc.abstractmethod
    def outputs_info(self, mode):
        """Return outputs_info variables for theano.scan.

        Return the theano variables which are changed by and throughout
        the theano scan. E.g. the adaptive synaptic weights.

        Keyword arguments:
        mode -- e.g. 'train', 'test', or 'likelihood':
            define for which mode the sequences variables should be
            returned.
        """
        return

    @abc.abstractmethod
    def non_sequences(self, mode):
        """Return non_sequences variables for theno.scan.

        Return the theano variables which are constant throughout the
        theano scan. E.g. fixed learning rates.

        Keyword arguments:
        mode -- e.g. 'train', 'test', or 'likelihood':
            define for which mode the sequences variables should be
            returned.
        """
        return

    @abc.abstractmethod
    def input_parameters(self, mode):
        """Return names of input theano variables.

        Returns the names of the required input theano variables for the
        learning step as list of strings. These are derived from the
        option 'InputSource' in the config file.

        Nomenclature:
        's_[#].[#]': activation/output of the given MultiLayer.Layer
        'W_[#].[#]': weights of the given MultiLayer.Layer
        'L': label from first input layer

        the variables are followed by the iterationstep:
        '[t]': sequential variable at iterationstep t
        '[t-1]': outputs_info variable from previous iterationstep t-1

        e.g. s_1.1[t] denotes the output of the first processing layer
        in the first multilayer at iterationstep t

        Keyword arguments:
        mode -- e.g. 'train', 'test', or 'likelihood':
            define for which mode the sequences variables should be
            returned.
        """
        return

    @abc.abstractmethod
    def learningstep(self, *args):
        """Perform a single learning step.

        This performs a single learning step and returns the
        activation for the given input and the learned synaptic weights.

        Keyword arguments:
        the keyword arguments must be the same as given in
        self.input_parameters(mode) for mode='train'.
        """
        return

    @abc.abstractmethod
    def teststep(self, *args):
        """Perform a single test step.

        This performs a single learning step and returns the
        activation for the given input.

        Keyword arguments:
        the keyword arguments must be the same as given in
        self.input_parameters(mode) for mode='test'.
        """
        return

    @abc.abstractmethod
    def set_weights(self, W):
        """Set weights of network to given value."""
        return

    @abc.abstractmethod
    def get_weights(self):
        """Get weights of the network."""
        return