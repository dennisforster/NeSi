# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import math
import time
import numpy as np

from classes.model.Layer import InputLayer, ProcessingLayer
from utils.parallel import pprint

class MultiLayer(object):
    '''
    classdocs
    '''

    #def __init__(self, config, model):
    def __init__(self, level):
        '''
        Constructor
        '''
        #self._config = config
        #self._model = model
        self.Layer = []
        self._niterations = 0
        self._niteration = 0
        self._figure = []
        self._blank_step = False
        self._level = level
        self._nrun = 0
        self._mini_batch_size = 1
        self._scan_batch_size = None

    def run(self):
        return self._nrun

    def next_run(self):
        self._nrun += 1

    def add_inputlayer(self):
        level = [self._level, len(self.Layer)]
        newLayer = InputLayer(level)
        self.Layer.append(newLayer)
        del newLayer

    def initialize_inputlayer(self, Y, Label, A,
                             nlayer=0,theanoscan=False):
        if (self.Layer[nlayer].__class__.__name__ == 'InputLayer'):
            self.Layer[nlayer].set_normalization(A)
            self.Layer[nlayer].set_input(Y, Label)
            if (A is not None):
                self.Layer[nlayer].normalize_inputs()
            if theanoscan:
                self.Layer[nlayer].initialize_theano_variables()

    def add_processinglayer(self, n = 1):
        for _ in range(n):
            level = [self._level, len(self.Layer)]
            newLayer = ProcessingLayer(level)
            self.Layer.append(newLayer)
            del newLayer

    def initialize_processinglayer(
        self, model, parameters, inputsource,
        nlayer=1, Theano=False, Scan=False, D=None, Y=None, L=None,
        method=None, h5path=None, h5file=None):
        if (self.Layer[nlayer].__class__.__name__ == 'ProcessingLayer'):
            self.Layer[nlayer].parameters = parameters
            # inputsource must be set before Model when using theano.scan
            self.Layer[nlayer].set_inputsource(inputsource)
            self.Layer[nlayer].C = parameters['C']
            self.Layer[nlayer].A = parameters['A']
            self.Layer[nlayer].D = D
            model_args = parameters
            model_args.update({'D':D[0], 'Theano':Theano, 'Scan':Scan})
            self.Layer[nlayer].set_model(model, model_args)
            self.Layer[nlayer].initialize_weights(Y, L, method, h5path, h5file)

    def compile_theano_functions(self):
        import theano
        #--- Training ---
        sequences = [item
            for layer in self.Layer
            for item in layer.sequences(mode='train')]
        outputs_info = [item
            for layer in self.Layer
            for item in layer.outputs_info(mode='train')]
        non_sequences = [item
            for layer in self.Layer
            for item in layer.non_sequences(mode='train')]
        weights, updates = theano.scan(
            fn=self._learningstep_scan,
            sequences=sequences,
            outputs_info=outputs_info,
            non_sequences=non_sequences)
        try:
            # Only save last weight
            results = [w[-1] for w in weights]
        except:
            results = [weights[-1]]
        # Compile function
        self._learningiteration_scan = theano.function(
            inputs=sequences + outputs_info + non_sequences,
            outputs=results,
            name='learning_iteration')

        #--- Testing ---
        sequences = [item
            for layer in self.Layer
            for item in layer.sequences(mode='test')]
        outputs_info = [item
            for layer in self.Layer
            for item in layer.outputs_info(mode='test')]
        non_sequences = [item
            for layer in self.Layer
            for item in layer.non_sequences(mode='test')]
        activations, updates = theano.scan(
            fn=self._teststep_scan,
            sequences=sequences,
            outputs_info=outputs_info,
            non_sequences=non_sequences)
        # Compile function
        self._activation_scan = theano.function(
            inputs=sequences + outputs_info + non_sequences,
            outputs=activations,
            name='activation')
        return

    def name(self):
        return 'M'+str(self._level)

    def set_iterations(self,niterations):
        self._niterations = niterations

    def get_iterations(self):
        return self._niterations

    def set_iteration(self, niteration):
        self._niteration = niteration

    def get_iteration(self):
        return self._niteration

    def number_of_layers(self):
        return len(self.Layer)

    def activaton(self,ilayer):
        if (ilayer < self.number_of_layers()):
            return self.Layer[ilayer].activation()

    def learningstep(self):
        for i in xrange(1, self.number_of_layers()):
            if (self.Layer[i].get_inputsource()[0] == 'InputLayer'):
                input_data = self.Layer[0].output_data()
            elif (self.Layer[i].get_inputsource()[0][0:15] ==
                'ProcessingLayer'):
                input_data = self.Layer[
                    int(self.Layer[i].get_inputsource()[0][15:])
                    ].activation()
            input_label = self.Layer[0].output_label()
            self.Layer[i].learningstep(self, input_data, input_label)
        self.Layer[0].next_datapoint()

    def blank_step(self):
        """
        needed for MPI parallelization, otherwise the reduce gets stuck on
        processes with less data points
        """
        for i in xrange(1, self.number_of_layers()):
            self.Layer[i].blank_step()

    def learning_iteration(self, theanoscan):
        t0 = time.time()
        if not theanoscan:
            logn = int(math.log10(self.Layer[0].imagecount()))
            pprint(' -- Image %*d'%(logn+1,0), end='')
            for i in xrange(self.Layer[0].imagecount()):
                if (((i+1)%1000 == 0) or ((i+1) == self.Layer[0].imagecount())):
                    pprint('\b'*(logn+1) + '%*d'%(logn+1, i+1), end='')
                self.learningstep()
            if self._blank_step:
                self.blank_step()
            self._niteration += 1
            pprint('\b \b'*(10+logn+1), end='')
            """
            from mpi4py import MPI
            import time
            if (MPI.COMM_WORLD.Get_rank() == 0):
                if (self._nrun == 1):
                    print '%d Processes' % MPI.COMM_WORLD.Get_size()
                    print 'C\tL1 total\tL1 comm\tL1 calls\tL2 ',
                    print 'total\tL2 comm\tL2 calls'
                print '%d\t%f\t%f\t%f\t%f\t%f\t%f' % (
                    self.Layer[1].GetNumberOfNeurons(), self.Layer[1].elapsed,
                    self.Layer[1].comm_time, self.Layer[1].ncomm,
                    self.Layer[2].elapsed, self.Layer[2].comm_time,
                    self.Layer[2].ncomm
                    )

            self.Layer[1].elapsed = time.time() - time.time()
            self.Layer[2].elapsed = time.time() - time.time()
            self.Layer[1].comm_time = time.time() - time.time()
            self.Layer[2].comm_time = time.time() - time.time()
            self.Layer[1].ncomm = 0
            self.Layer[2].ncomm = 0
            """
        else:
            inputs = {}
            weights = {}
            parameters = {}
            mbs = self._mini_batch_size
            if self._scan_batch_size is None:
                nbatches = 1
                scan_batch_size = self.Layer[0].get_input_data().shape[0]
                # don't set self._scan_batch_size, because for testing
                # the data shape will be different
            else:
                nbatches = int(np.ceil(
                    self.Layer[0].get_input_data().shape[0]
                    /float(self._scan_batch_size)))
                scan_batch_size = self._scan_batch_size
            for nbatch in xrange(nbatches):
                for layer in self.Layer:
                    if layer.__class__.__name__ == 'InputLayer':
                        layer.shuffle()
                        try:
                            data = layer.get_input_data().astype('float32')[
                                nbatch*scan_batch_size:
                                (nbatch+1)*scan_batch_size]
                            label = layer.get_input_label().astype('int32')[
                                nbatch*scan_batch_size:
                                (nbatch+1)*scan_batch_size]
                        except:
                            data = layer.get_input_data().astype('float32')[
                                nbatch*scan_batch_size:]
                            label = layer.get_input_label().astype('int32')[
                                nbatch*scan_batch_size:]
                        # if mbs > 1: (...)
                        data = np.append(
                            data,
                            np.zeros(
                                ((mbs - data.shape[0]%mbs)%mbs, data.shape[1]),
                                dtype='float32'),
                            axis=0)
                        data = data.reshape(
                            (data.shape[0]/mbs, mbs, data.shape[1]))
                        label = np.append(
                            label,
                            (-1)*np.ones((mbs - label.shape[0]%mbs)%mbs,
                                dtype='int32'),
                            axis=0)
                        label = label.reshape((label.shape[0]/mbs, mbs))
                        inputs[layer.s_t.name] = data
                        inputs[layer.L_t.name] = label
                    elif layer.__class__.__name__ == 'ProcessingLayer':
                        weights[layer._layermodel.W_t.name] = \
                            layer.get_weights().astype('float32')
                        # TODO: generalize parameters in Layer
                        parameters.update(dict(zip(
                            [item.name
                                for item in layer._layermodel.parameters_t],
                            [layer.parameters[item.name.rpartition('_')[0]]
                                for item in layer._layermodel.parameters_t]
                            )))
                        # parameters[layer._layermodel.epsilon_t.name] = \
                        #     np.asarray(layer.get_learningrate(), dtype='float32')
                sequences = [inputs[item.name]
                    for layer in self.Layer
                    for item in layer.sequences(mode='train')]
                outputs_info = [weights[item.name]
                    for layer in self.Layer
                    for item in layer.outputs_info(mode='train')]
                non_sequences = [parameters[item.name]
                    for layer in self.Layer
                    for item in layer.non_sequences(mode='train')]
                args = sequences + outputs_info + non_sequences
                learnedweights = self._learningiteration_scan(*args)
                for n in xrange(len(learnedweights)):
                    self.Layer[n+1].set_weights(learnedweights[n])
            self._niteration += 1
        pprint(' (%f s)' % (time.time() - t0), end='')

    def learning_iterations(self, theanoscan):
        for _ in xrange(self._niterations):
            #print "Iteration %d" % (i)
            self.learning_iteration(theanoscan)

    def _learningstep_scan(self, *args):
        argsdict = {a.name:a for a in args}
        for layer in self.Layer:
            if layer.__class__.__name__ == 'ProcessingLayer':
                layerargs = [argsdict[param]
                    for param in layer._layermodel.input_parameters('train')]
                args += layer._layermodel.learningstep(*layerargs)
                # or .learningstep_m1 for mb=1 for some mild speed-up
                argsdict = {a.name:a for a in args}
        return [argsdict[item.name+'[t]']
            for layer in self.Layer
            for item in layer.outputs_info(mode='train')]

    def _teststep_scan(self, *args):
        argsdict = {a.name:a for a in args}
        for layer in self.Layer:
            if layer.__class__.__name__ == 'ProcessingLayer':
                layerargs = [argsdict[param]
                    for param in layer._layermodel.input_parameters('test')]
                args += (layer._layermodel.teststep(*layerargs),)
                # or .learningstep_m1 for mb=1 for some mild speed-up
                argsdict = {a.name:a for a in args}
        return [argsdict[item.name+'[t]']
            for layer in self.Layer
            for item in layer.outputs_info(mode='test')]
    # TODO: combine _LearningStep_Scan and _TestStep_Scan

    def output(self,input_data,nlayer):
        """
        returns the Output of Layer['nlayer'] if the MultiLayer is fed
        with 'input_data'
        """
        if (nlayer == 0):
            return self.Layer[0].output(input_data)
        else:
            if (self.Layer[nlayer].get_inputsource()[0] == 'InputLayer'):
                source = 0
            elif (self.Layer[nlayer].get_inputsource()[0][0:15] ==
                'ProcessingLayer'):
                source = int(self.Layer[nlayer].get_inputsource()[0][15])

            return self.Layer[nlayer].output(self,self.output(input_data,source))