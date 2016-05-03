# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import numpy as np
from mpi4py import MPI
import sys
import time
import math

from classes.model.MultiLayer import MultiLayer
from utils.mathfunctions import log_poisson_function
from utils.parallel import pprint
from utils.sysfunctions import mem_usage

class Model(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.MultiLayer = []
        self._run = 0
        self._theanoscan = False
        self._t_sum_log_likelihood = None
        self._t_sum_log_likelihood_W = None


    #=== Properties ===========================================================

    def number_of_multilayers(self):
        return len(self.MultiLayer)

    def number_of_layers(self):
        total = []
        for i in xrange(self.number_of_multilayers()):
            total.append(self.MultiLayer[i].number_of_layers())
        return total


    #=== Model Construction ===================================================

    def add_multilayer(self,niterations=0):
        level = len(self.MultiLayer)+1
        newML = MultiLayer(level)
        newML.set_iterations(niterations)
        self.MultiLayer.append(newML)

    def build(self,config):
        """
        Build Frame of the Model:
        Create the MultiLayers with according Layers based on 'config'
        This is a pure creation of the structure of MultiLayers and
        Layers. To use the model the Layers must be initialized first.
        (e.g. via Model.InitializeMultiLayer).
        """
        self.MultiLayer = []
        for multilayer in config['model'].keys():
            if (multilayer[0:10] == 'MultiLayer'):
                nmultilayer = int(multilayer[10])-1
                niterations = config['model'][multilayer]['Iterations']
                self.add_multilayer(niterations)
                for layer in config['model'][multilayer].keys():
                    if (layer == 'InputLayer'):
                        self.MultiLayer[nmultilayer].add_inputlayer()
                    elif (layer[0:15] == 'ProcessingLayer'):
                        self.MultiLayer[nmultilayer].add_processinglayer()

    def clear(self):
        for nML in xrange(self.number_of_multilayers()):
            for nL in xrange(self.MultiLayer[nML].number_of_layers()):
                self.MultiLayer[nML].Layer[nL].__init__([nML+1,nL])

    def initialize_multilayer(self, nmultilayer, config, dataset):
        """
        Initialize given MultiLayer according to config.
        The Layers get initialized with the data depending on their
        InputSource
        """
        # Set if theano and/or theano.scan should be used for learning
        # iterations
        try:
            self._theano = config['config']['Theano']
        except:
            self._theano = False

        try:
            self._theanoscan = config['config']['Theano']*config['config']['Scan']
        except:
            self._theanoscan = False

        try:
            mini_batch_size = config['config']['mini_batch_size']
        except:
            mini_batch_size = 1
        if self._theano and not self._theanoscan and (mini_batch_size > 1):
            pprint("WARNING: theano (without scan) doesn't support " +
                   "mini-batches yet. mini_batch_size set to 1.")
            mini_batch_size = 1
        self.MultiLayer[nmultilayer]._mini_batch_size = mini_batch_size

        try:
            scan_batch_size = config['config']['scan_batch_size']
        except:
            scan_batch_size = None
        self.MultiLayer[nmultilayer]._scan_batch_size = scan_batch_size

        # Set the Data for the InputLayer of the MultiLayer depending on
        # InputSource (DataSet or another MultiLayer)
        InputSource = config['model']['MultiLayer'+str(nmultilayer+1)]\
            ['InputLayer']['InputSource']
        self.MultiLayer[nmultilayer].Layer[0].set_inputsource(InputSource)
        if (InputSource == 'DataSet'):
            # If the InputSource is "DataSet" get the Training Data +
            # Label from the given DataSet
            Y = dataset.get_train_data()
            Label = dataset.get_train_label()
        elif (InputSource[0][0:10] == 'MultiLayer'):
            # If the InputSource is a "MultiLayer[#]" get as Training
            # Data the Output of that MultiLayer.
            if not self._theanoscan:
                Y = np.empty(shape=(dataset.get_train_data().shape[0],
                             config['model'][InputSource[0]][InputSource[1]]['C']),
                             dtype='float32')
                for i in xrange(dataset.get_train_data().shape[0]):
                    Y[i,:] = self.output(dataset.get_train_data()[i],
                                         int(InputSource[0][10])-1)
            else:
                # TODO: Implement Scan-Splitting for big data sets
                inputdata = []
                weights = {}
                activations = {}
                ml = self.MultiLayer[int(InputSource[0][10])-1]
                if ml._scan_batch_size is None:
                    nbatches = 1
                    scan_batch_size = dataset.get_train_data().shape[0]
                else:
                    nbatches = int(np.ceil(
                        dataset.get_train_data().shape[0]
                        /float(ml._scan_batch_size)))
                    scan_batch_size = ml._scan_batch_size
                for layer in ml.Layer:
                    if layer.__class__.__name__ == 'InputLayer':
                        data = dataset.get_train_data()
                        label = dataset.get_train_label()
                        layer.set_input(data, label, shuffle=False)
                        layer.normalize_inputs()
                        data = layer.get_input_data().astype('float32')
                        # TODO: use layer.output() !
                        N = data.shape[0]
                        D = data.shape[1]
                        data = data.reshape((1, N, D))
                        inputdata.append(data)
                    elif layer.__class__.__name__ == 'ProcessingLayer':
                        weights[layer._layermodel.W_t.name] = \
                            layer.get_weights().astype('float32')
                        activations[layer._layermodel.s_t.name] = np.zeros(
                            (N, weights[layer._layermodel.W_t.name].shape[0]),
                            dtype='float32')
                # TODO: reconstruct this from layers as in Train
                # (done for outputs_info and non_sequences):
                Y = np.empty(shape=(dataset.get_train_data().shape[0],
                             config['model'][InputSource[0]][InputSource[1]]['C']),
                             dtype='float32')
                for nbatch in xrange(nbatches):
                    sequences = [inputdata[0][:,nbatch*scan_batch_size:\
                                              (nbatch+1)*scan_batch_size,:]]
                    outputs_info = [activations[item.name]\
                            [nbatch*scan_batch_size:(nbatch+1)*scan_batch_size,:]
                        for layer in ml.Layer
                        for item in layer.outputs_info(mode='test')]
                    non_sequences = [weights[item.name]
                        for layer in ml.Layer
                        for item in layer.non_sequences(mode='test')]
                    args = sequences + outputs_info + non_sequences
                    Y[nbatch*scan_batch_size:(nbatch+1)*scan_batch_size] = \
                        ml._activation_scan(*args)[0]
            Label = dataset.get_train_label()

        self.MultiLayer[nmultilayer].set_iterations(
            config['model']['MultiLayer'+str(nmultilayer+1)]['Iterations'])

        plcount = 0
        # if the number of datapoints changes on different layers/
        # multilayers the next line must be changed
        ndatapoints = Y.shape[0]
        self.MultiLayer[nmultilayer].set_iteration(0)
        for nlayer in xrange(self.MultiLayer[nmultilayer].number_of_layers()):
            #--- Initialize Input Layer ---
            if (self.MultiLayer[nmultilayer].Layer[nlayer].__class__.__name__
                == 'InputLayer'):
                InputSource = config['model']['MultiLayer'+str(nmultilayer+1)]\
                    ['InputLayer']['InputSource']
                A = config['model']['MultiLayer'+str(nmultilayer+1)]\
                    ['InputLayer']['A']
                if (A == 'Default'):
                    if (InputSource[0][0:10] == 'MultiLayer'):
                        A = config['model'][InputSource[0]][InputSource[1]]\
                            ['C'] + 1
                        # if you change this formular remember to also
                        # change it in Output->WriteSetting (!)
                    else:
                        A = None
                try:
                    theanoscan = config['config']['Theano']*config['config']['Scan']
                except:
                    theanoscan = False
                self.MultiLayer[nmultilayer].initialize_inputlayer(
                    Y,Label,A,nlayer,theanoscan)

            #--- Processing Layer ---
            elif (self.MultiLayer[nmultilayer].Layer[nlayer].__class__.__name__
                == 'ProcessingLayer'):
                plcount += 1
                InputSource = config['model']['MultiLayer'+str(nmultilayer+1)]\
                    ['ProcessingLayer'+str(plcount)]['InputSource']
                if (type(InputSource) == str):
                    InputSource = (InputSource,)
                C = config['model']['MultiLayer'+str(nmultilayer+1)]\
                    ['ProcessingLayer'+str(plcount)]['C']
                if (C == 'Default'):
                    try:
                        C = len(config['dataset']['classes'])
                    except:
                        C = config['dataset']['nclasses']
                epsilon = config['model']['MultiLayer'+str(nmultilayer+1)]\
                    ['ProcessingLayer'+str(plcount)]['epsilon']
                try:
                    if (epsilon == 'Default'):
                        # if you change this formular remember to also change
                        # it in Output->WriteSetting (!)
                        if (config['model']['MultiLayer'+str(nmultilayer+1)]\
                            ['ProcessingLayer'+str(plcount)]['Model']
                            == 'MM-LabeledOnly'):
                            epsilon = min(C/2.* \
                                1./config['dataset']['training_label_size'],
                                1.)
                        else:
                            epsilon = C/2. * \
                                1./config['dataset']['training_data_size']
                    elif (epsilon[0] == 'factor'):
                        if (config['model']['MultiLayer'+str(nmultilayer+1)]\
                            ['ProcessingLayer'+str(plcount)]['Model']
                            == 'MM-LabeledOnly'):
                            epsilon = min(epsilon[1] * \
                                C/float(config['dataset']['training_label_size']),
                                1.)
                        else:
                            epsilon = min(epsilon[1] * \
                                C/float(config['dataset']['training_data_size']),
                                1.)
                except:
                    pass

                Model = config['model']['MultiLayer'+str(nmultilayer+1)]\
                    ['ProcessingLayer'+str(plcount)]['Model']
                L = self.MultiLayer[nmultilayer].Layer[0].get_input_label()
                if (InputSource[0] == 'InputLayer'):
                    Y = self.MultiLayer[nmultilayer].Layer[0].get_input_data()
                    D = Y.shape[1:]
                elif (InputSource[0][0:15] == 'ProcessingLayer'):
                    D = (config['model']['MultiLayer'+str(nmultilayer+1)]\
                        [InputSource[0]]['C'],)
                    Y = None
                else:
                    D = None
                    Y = None
                    print "ERROR: %s"%'model',
                    print "| MultiLayer%d"%(nmultilayer+1),
                    print "| ProcessingLayer%d:"%nlayer,
                    print "Invalid InputSource"

                """ # obsolete
                # for recurrent model: number of neurons in following layer
                try:
                    K = config['model']['MultiLayer'+str(nmultilayer+1)]\
                        ['ProcessingLayer'+str(plcount+1)]['C']]
                    if (K == 'Default'):
                        K = len(config['dataset']['classes'])
                except:
                    K = None
                """

                # optional arguments:
                try:
                    A = config['model']['MultiLayer'+str(nmultilayer+1)]\
                        ['ProcessingLayer'+str(plcount)]['A']
                    if (A == 'Default'):
                        # if you change this formular remember to also change
                        # it in Output->WriteSetting (!)
                        if (InputSource[0][0:15] == 'ProcessingLayer'):
                            A = D + 1
                        else:
                            A = None
                except:
                    A = None

                try:
                    threshold = config['model']['MultiLayer'+str(nmultilayer+1)]\
                        ['ProcessingLayer'+str(plcount)]['threshold']
                except:
                    threshold = None

                h5path = None
                h5file = None
                try:
                    InitMethod = config['model']['MultiLayer'+str(nmultilayer+1)]\
                        ['ProcessingLayer'+str(plcount)]['Initialization']
                    if isinstance(InitMethod, tuple):
                        InitMethod = InitMethod[0]
                        if (InitMethod == 'h5'):
                            h5path = config['model']\
                                ['MultiLayer'+str(nmultilayer+1)]\
                                ['ProcessingLayer'+str(plcount)]\
                                ['Initialization'][1]
                            # expected h5 file name format:
                            # "Run[i]M[j]L[k].h5"
                            # e.g. "Run3M1L2.h5" for weights of 1st MultiLayer,
                            # 2nd Layer, the 3rd Run (counting starts with 1).
                            # For each Run, their must be an individual h5 file
                            # present.
                            try:
                                h5file = config['model']\
                                    ['MultiLayer'+str(nmultilayer+1)]\
                                    ['ProcessingLayer'+str(plcount)]\
                                    ['Initialization'][2]
                            except:
                                h5file = "Run%dM%dL%d.h5"%(
                                    self._run+1,nmultilayer+1,nlayer)
                except:
                    InitMethod = None

                try:
                    Theano = config['config']['Theano']
                except:
                    Theano = False

                try:
                    Scan = config['config']['Scan']
                except:
                    Scan = False

                Parameters = {
                    'C':C,
                    'A':A,
                    'epsilon':epsilon,
                    'threshold':threshold
                    }
                self.MultiLayer[nmultilayer].initialize_processinglayer(
                    Model,Parameters,InputSource,nlayer,Theano,
                    Scan,D,Y,L,InitMethod,h5path,h5file
                    )

                if (np.ceil(float(config['dataset']['training_data_size'])/\
                            MPI.COMM_WORLD.Get_size() > ndatapoints)):
                    self.MultiLayer[nmultilayer]._blankstep = True
        if self._theanoscan:
            self.MultiLayer[nmultilayer].compile_theano_functions()


    #=== Application ==========================================================

    def train(self, nmultilayer, output, config, dataset):
        ml = self.MultiLayer[nmultilayer]
        ml.next_run()

        pprint('%d.%d.2.1 - Visualize Weights' % (
            self._run+1, nmultilayer+1), end='')
        output.visualize_all_weights(self,nmultilayer,config)
        pprint(' (%0.2f MB)' % mem_usage())

        # if (len(self.MultiLayer) == 1):
            # TODO: for greedy model: output Likelihood of up to the
            #       current MultiLayer
        pprint('%d.%d.2.2 - LogLikelihood' % (
            self._run+1, nmultilayer+1), end='')
        output.write_loglikelihood(self, nmultilayer)
        pprint(' (%0.2f MB)' % mem_usage())

        # variables for stopping criterion
        # TODO: generalize for all MultiLayers
        try:
            STOPPING_CRITERION = config.get()\
                ['model']['MultiLayer1']['StoppingCriterion']
        except:
            STOPPING_CRITERION = False
        if STOPPING_CRITERION:
            try:
                mvngwidth = int(config.get()\
                    ['model']['MultiLayer1']['MovingWidth'])
            except:
                pprint('WARNING (model.Model::Train): No width for \
                    moving average was given. It will be set to %d'%20)
                mvngwidth = 20
            loglikelihood = np.asarray([], dtype=np.float32)
            mvng_avg, mvng_std = 0., 0.
            max_mvng_avg = float('-inf')
            last_weights = []
        else:
            loglikelihood = np.asarray([None])
        STOP = False

        MPI.COMM_WORLD.Barrier()
        pprint('%d.%d.2.3 - Training Iterations' % (self._run+1,nmultilayer+1))
        for niteration in xrange(ml.get_iterations()):
            pprint('Iteration: %*d' % (
                int(math.log10(ml.get_iterations()))+1,
                ml.get_iteration()+1), end='')

            # pprint('2.2.3.1 - Convergence', end='')
            output.conv_pre(ml)
            # pprint(' - Memory usage: %s (Mb)' % mem_usage())
            MPI.COMM_WORLD.Barrier()

            # pprint('2.2.3.2 - Learning Iteration', end='')
            ml.learning_iteration(self._theanoscan)
            # pprint(' - Memory usage: %s (Mb)\n' % mem_usage())
            MPI.COMM_WORLD.Barrier()

            # experimental: variing learning rates
            # if ((niteration % 1 == 0) and
            #     (ml.Layer[1]._epsilon >= 0.0008)):
            #     ml.Layer[1]._epsilon *= 0.98
            # MPI.COMM_WORLD.Barrier()

            # to save learned weights every iteration
            # output.save_weights(self.MultiLayer[nmultilayer])

            # to save posterior distribution of training data every iteration
            # if nmultilayer == self.number_of_multilayers()-1:
            #     output.save_posterior(self, config, dataset)

            # Stopping criterion
            if (len(self.MultiLayer) == 1):
                # TODO: for greedy model: output Likelihood of up to the
                #       current MultiLayer
                if STOPPING_CRITERION:
                    t0 = time.time()
                    loglikelihood = np.append(
                        loglikelihood,
                        self.loglikelihood()
                        )
                    if (niteration >= mvngwidth):
                        # save only the last #mvngwidth values
                        loglikelihood = loglikelihood[1:]
                    pprint(' | Log-Likelihood: %f (%f s)'%(
                        loglikelihood[-1], time.time()-t0), end='')

                if STOPPING_CRITERION:
                    # save only the last #mvngwidht/2-1 weights
                    # for centered moving average
                    last_weights.append([ml.Layer[nlayer].get_weights()
                        for nlayer in xrange(1,len(ml.Layer))])
                    if (niteration > mvngwidth/2):
                        last_weights = last_weights[1:]
                    # calculate moving average over last #mvngwidth values
                    if (niteration >= mvngwidth-1):
                        mvng_avg = np.mean(loglikelihood)
                        mvng_std = np.std(loglikelihood)
                        pprint(' | Moving Average (%d): %f +- %f'%
                            ((niteration+1 - mvngwidth/2),mvng_avg,mvng_std),
                            end='')
                        if (mvng_avg > max_mvng_avg):
                            max_mvng_avg = mvng_avg
                        elif (mvng_avg < max_mvng_avg - mvng_std):
                            # if the moving average drops below the maximum
                            # moving average by more than the moving standard
                            # deviation, stop the learning iteration and revert
                            # back to the point of the centered moving average
                            for nlayer in xrange(1,len(ml.Layer)):
                                ml.Layer[nlayer].set_weights(last_weights[0][nlayer-1])
                            stopping_iteration = niteration+1 - mvngwidth/2
                            pprint('\nStopping criterion met at iteration %d.'%
                                stopping_iteration)
                            STOP = True
                            ml.set_iteration(stopping_iteration)
                            ml.set_iterations(stopping_iteration)

                # abort on numerical error (nan in any of the weights)
                if any([np.any(np.isnan(ml.Layer[nlayer].get_weights()))
                        for nlayer in xrange(1,len(ml.Layer))]):
                    pprint('\nNumerical error: try to decrease learning ' +
                        'rate and/or mini-batch size.')
                    STOP = True
                    try:
                        ml.set_iteration(
                            stopping_iteration)
                        ml.set_iterations(
                            stopping_iteration)
                    except:
                        pass

            this_loglikelihood = output.write_loglikelihood(self, nmultilayer, loglikelihood[-1])
            # pprint(' - Memory usage: %s (Mb)' % mem_usage())

            if (len(self.MultiLayer) == 1):
                # pprint('2.2.3.7 - Test Error', end='')
                output.write_online_results(self, config, dataset, this_loglikelihood)
                # pprint(' - Memory usage: %s (Mb)' % mem_usage())

            # pprint('2.2.3.4 - Visualize Weights', end='')
            output.visualize_all_weights(self,nmultilayer,config)
            # pprint(' - Memory usage: %s (Mb)' % mem_usage())

            # pprint('2.2.3.5 - Convergence 2', end='')
            output.conv_post(self.MultiLayer[nmultilayer])
            # pprint(' - Memory usage: %s (Mb)' % mem_usage())

            # pprint('2.2.3.6 - Visualize Convergence', end='')
            output.visualize_convergence(self,nmultilayer)
            # pprint(' - Memory usage: %s (Mb)' % mem_usage())
            if STOP:
                break
            pprint('')  #linebreak

    def test(self,testdata,testlabel):
        """
        Calculate test error based on the Output of the last
        Processing Layer of the last MultiLayer
        """
        if not self._theanoscan:
            ncorrect = np.asarray(0.)
            ntotal = np.asarray(testdata.shape[0])
            for i in xrange(ntotal):
                # classification = np.argmax(
                #     np.sum(self.MultiLayer[0].output(testdata[i],1)*\
                #            self.MultiLayer[0].Layer[2].get_weights(),1)
                #     )
                activation = self.output(
                    testdata[i],
                    self.number_of_multilayers()-1)
                classification = np.argmax(activation)
                if (classification == testlabel[i]):
                    ncorrect += 1.
            sumcorrect = np.zeros_like(ncorrect)
            sumtotal = np.zeros_like(ntotal)
            MPI.COMM_WORLD.Allreduce(ncorrect, sumcorrect, op=MPI.SUM)
            MPI.COMM_WORLD.Allreduce(ntotal, sumtotal, op=MPI.SUM)
            test_error = (1. - sumcorrect/sumtotal)*100.
        else:
            # TODO: Implement theano Output class in multilayer
            inputdata = {}
            weights = {}
            init_activations = {}
            # activation[ML][Layer]:
            activation = [[None]*(len(ml.Layer)-1) for ml in self.MultiLayer]
            for nml in xrange(len(self.MultiLayer)):
                ml = self.MultiLayer[nml]
                if ml._scan_batch_size is None:
                    nbatches = 1
                    scan_batch_size = testdata.shape[0]
                else:
                    nbatches = int(np.ceil(
                        testdata.shape[0]
                        /float(ml._scan_batch_size)))
                    scan_batch_size = ml._scan_batch_size
                for nbatch in xrange(nbatches):
                    for nl in xrange(len(ml.Layer)):
                        layer = ml.Layer[nl]
                        if layer.__class__.__name__ == 'InputLayer':
                            if (layer.get_inputsource()[0] == 'DataSet'):
                                try:
                                    data = testdata[nbatch*scan_batch_size:
                                        (nbatch+1)*scan_batch_size]
                                    label = testlabel[nbatch*scan_batch_size:
                                        (nbatch+1)*scan_batch_size]
                                except:
                                    data = testdata[nbatch*scan_batch_size:]
                                    label = testlabel[nbatch*scan_batch_size:]
                                # TODO: use layer.output of input layer
                                old_data = layer.get_input_data()
                                old_label = layer.get_input_label()
                                layer.set_input(data, label, shuffle=False)
                                layer.normalize_inputs()
                                data = layer.get_input_data().astype('float32')
                                layer.set_input(old_data, old_label, shuffle=False)
                            elif (layer.get_inputsource()[0][0:10] == 'MultiLayer'):
                                input_ml = int(layer.get_inputsource()[0][10:])-1
                                input_l = int(layer.get_inputsource()[1][15:])-1
                                data = activation[input_ml][input_l]
                                try:
                                    data = data[nbatch*scan_batch_size:
                                                (nbatch+1)*scan_batch_size]
                                except:
                                    data = data[nbatch*scan_batch_size:]
                            N = data.shape[0]
                            D = data.shape[1]
                            data = data.reshape((1, N, D))
                            inputdata[layer.s_t.name] = data
                        elif layer.__class__.__name__ == 'ProcessingLayer':
                            wname = layer._layermodel.W_t.name
                            weights[wname] = layer.get_weights().astype('float32')
                            C = weights[wname].shape[0]
                            init_activations[layer._layermodel.s_t.name] = \
                                np.zeros((N,C), dtype='float32')
                    sequences = [inputdata[item.name]
                        for layer in ml.Layer
                        for item in layer.sequences(mode='test')]
                    outputs_info = [init_activations[item.name]
                        for layer in ml.Layer
                        for item in layer.outputs_info(mode='test')]
                    non_sequences = [weights[item.name]
                        for layer in ml.Layer
                        for item in layer.non_sequences(mode='test')]
                    args = sequences + outputs_info + non_sequences
                    activation_scan = ml._activation_scan(*args)
                    if (activation[nml][0] is None):
                        for n in xrange(len(activation[nml])):
                            activation[nml][n] = activation_scan[n]
                    else:
                        for n in xrange(len(activation[nml])):
                            activation[nml][n] = np.append(
                                activation[nml][n],
                                activation_scan[n],
                                axis=-2)
            # TODO: figure out why the greedy model has activation shape (N,D),
            # and the ff shape (1,N,D)
            activation = activation[-1][-1]
            classification = np.argmax(activation, axis=-1).flatten()
            test_error = (1.-np.sum(classification == testlabel)\
                /float(classification.shape[0]))*100.
        return test_error, activation

    def loglikelihood(self, data=None, mode='unsupervised'):
        """
        Calculate the log-likelihood of the given data under the model
        parameters.

        Keyword arguments:
        data: nparray (data,label) or 'None' for input data
        mode=['unsupervised','supervised']: calculate supervised or
              unsupervised loglikelihood
        """
        # To avoid numerical problems the log-likelihood has to be
        # calculated in such a more costly way by using intermediate
        # logarithmic functions

        # input data
        if data is None:
            Y = self.MultiLayer[0].Layer[0].get_input_data().astype('float32')
        else:
            Y = np.asarray(
                [self.MultiLayer[0].Layer[0].output(y) for y in data[0]],
                dtype='float32')

        # labels
        if (mode == 'supervised'):
            if data is None:
                L = self.MultiLayer[0].Layer[0].get_input_label()
            else:
                L = data[1]
        elif (mode == 'unsupervised'):
            L = (-1)*np.ones(Y.shape[0])
        
        # weights & dimensions
        W = self.MultiLayer[0].Layer[1].get_weights().astype('float32')
        N = Y.shape[0]
        C = W.shape[0]
        D = W.shape[1]
        if (self.number_of_multilayers() == 2):
            try:
                M = self.MultiLayer[1].Layer[1].get_weights()
            except:
                M = None
        elif ((self.number_of_multilayers() == 1) and
              (self.MultiLayer[0].number_of_layers() == 3)):
            M = self.MultiLayer[0].Layer[2].get_weights()
        else:
            M = None
        try:
            K = M.shape[0]
        except:
            K = None

        if not self._theano:
            if M is None:
                ones = np.ones(shape=(C,D), dtype=float)
                log_likelihood = np.empty(N, dtype=float)
                for ninput in xrange(N):
                    sum_log_poisson = np.sum(
                        log_poisson_function(ones*Y[ninput,:], W), axis=1)
                    a = np.max(sum_log_poisson)
                    log_likelihood[ninput] = -np.log(C) + a + \
                        np.log(np.sum(np.exp(sum_log_poisson - a)))
            else:
                ones = np.ones(shape=(C,D), dtype=float)
                log_likelihood = np.empty(N, dtype=float)
                for ninput in xrange(N):
                    sum_log_poisson = np.sum(
                        log_poisson_function(ones*Y[ninput,:], W), axis=1)
                    a = np.max(sum_log_poisson)
                    if (L[ninput] == -1):
                        log_likelihood[ninput] = a + np.log(np.sum(
                            np.exp(sum_log_poisson-a)*np.sum(M,axis=0)/float(K)))
                    else:
                        log_likelihood[ninput] = a + np.log(np.sum(
                            np.exp(sum_log_poisson - a)*
                            M[L[ninput],:]/float(K)))
            mean_log_likelihood = np.mean(log_likelihood)
            sum_log_likelihood = np.zeros_like(mean_log_likelihood)
            MPI.COMM_WORLD.Allreduce(mean_log_likelihood, sum_log_likelihood, op=MPI.SUM)
            mean_log_likelihood = sum_log_likelihood/float(MPI.COMM_WORLD.Get_size())
        else:
            import theano
            import theano.tensor as T
            ml = self.MultiLayer[0]
            if ml._scan_batch_size is None:
                nbatches = 1
                scan_batch_size = ml.Layer[0].get_input_data().shape[0]
            else:
                nbatches = int(np.ceil(
                    ml.Layer[0].get_input_data().shape[0]
                    /float(ml._scan_batch_size)))
                scan_batch_size = ml._scan_batch_size
            batch_log_likelihood = np.zeros(nbatches, dtype='float32')
            if M is None:
                if (self._t_sum_log_likelihood_W is None):
                    Y_t = T.matrix('Y', dtype='float32')
                    L_t = T.vector('L', dtype='int32')
                    W_t = T.matrix('W', dtype='float32')
                    sum_log_poisson = T.tensordot(Y_t,T.log(W_t), axes=[1,1]) \
                        - T.sum(W_t, axis=1) \
                        - T.sum(T.gammaln(Y_t+1), axis=1, keepdims=True)
                    a = T.max(sum_log_poisson, axis=1, keepdims=True)
                    logarg = T.sum(T.exp(sum_log_poisson-a), axis=1)
                    log_likelihood = -T.log(C) + a[:,0] + T.log(logarg)
                    # Compile theano function
                    self._t_sum_log_likelihood_W = theano.function(
                        [Y_t,L_t,W_t],
                        T.sum(log_likelihood),
                        on_unused_input='ignore')
                for nbatch in xrange(nbatches):
                    batch_log_likelihood[nbatch] = self._t_sum_log_likelihood_W(
                        Y[nbatch*scan_batch_size:
                            (nbatch+1)*scan_batch_size].astype('float32'),
                        L[nbatch*scan_batch_size:
                            (nbatch+1)*scan_batch_size].astype('int32'),
                        W.astype('float32'))
            else:
                if (self._t_sum_log_likelihood is None):
                    Y_t = T.matrix('Y', dtype='float32')
                    L_t = T.vector('L', dtype='int32')
                    W_t = T.matrix('W', dtype='float32')
                    M_t = T.matrix('M', dtype='float32')
                    sum_log_poisson = T.tensordot(Y_t,T.log(W_t), axes=[1,1]) \
                        - T.sum(W_t, axis=1) \
                        - T.sum(T.gammaln(Y_t+1), axis=1, keepdims=True)
                    M_nlc = M_t[L_t]
                    L_index = T.eq(L_t,-1).nonzero()
                    M_nlc = T.set_subtensor(M_nlc[L_index], T.sum(M_t, axis=0))
                    # for numerics: only account for values, where M_nlc is
                    # not zero
                    a = T.switch(
                        T.eq(M_nlc, 0.),
                        T.cast(T.min(sum_log_poisson), dtype = 'int32'),
                        T.cast(sum_log_poisson, dtype = 'int32'))
                    a = T.max(a, axis=1, keepdims=True)
                    # logarg = T.switch(
                    #     T.eq(M_nlc, 0.),
                    #     0.,
                    #     T.exp(sum_log_poisson-a).astype('float32')*M_nlc\
                    #         /M_t.shape[0].astype('float32'))
                    logarg = T.switch(
                        T.eq(M_nlc, 0.),
                        0.,
                        T.exp(sum_log_poisson-a.astype('float32'))
                    )
                    logarg = T.sum(logarg, axis=1)
                    log_likelihood = a[:,0].astype('float32') + T.log(logarg)
                    # Compile theano function
                    self._t_sum_log_likelihood = theano.function(
                        [Y_t,L_t,W_t,M_t],
                        T.sum(log_likelihood),
                        on_unused_input='ignore')
                    """
                    # LL_scan:
                    ll_t = T.scalar('loglikelihood', dtype='float32')
                    sequences = [Y_t, L_t]
                    outputs_info = [ll_t]
                    non_sequences = [W_t, M_t]
                    likelihood, updates = theano.scan(
                        fn=self._loglikelihood_step,
                        sequences=sequences,
                        outputs_info=outputs_info,
                        non_sequences=non_sequences)
                    result = likelihood[-1]
                    # Compile function
                    self._loglikelihood_scan = theano.function(
                        inputs=sequences + outputs_info + non_sequences,
                        outputs=result,
                        name='loglikelihood')
                    """
                for nbatch in xrange(nbatches):
                    batch_log_likelihood[nbatch] = self._t_sum_log_likelihood(
                        Y[nbatch*scan_batch_size:
                            (nbatch+1)*scan_batch_size].astype('float32'),
                        L[nbatch*scan_batch_size:
                            (nbatch+1)*scan_batch_size].astype('int32'),
                        W.astype('float32'),
                        M.astype('float32'))
            mean_log_likelihood = np.sum(batch_log_likelihood)/float(N)
        return mean_log_likelihood

    def _loglikelihood_step(self, Y_t, L_t, ll_t, W_t, M_t):
        import theano
        import theano.tensor as T
        sum_log_poisson = T.tensordot(Y_t, T.log(W_t), axes=[0,1]) \
            - T.sum(W_t, axis=1) - T.sum(T.gammaln(Y_t+1))
        M_nlc = theano.ifelse.ifelse(
            T.eq(L_t, -1),
            T.sum(M_t, axis=0),
            M_t[L_t]
            )
        # for numerics: only account for values, where M_nlc is not zero
        a = T.switch(
            T.eq(M_nlc, 0.),
            T.min(sum_log_poisson),
            sum_log_poisson
            )
        a = T.max(a, keepdims=True)
        logarg = T.switch(
            T.eq(M_nlc, 0.),
            0.,
            T.exp(sum_log_poisson - a)*M_nlc/T.cast(M_t.shape[0], dtype='float32')
            )
        logarg = T.sum(logarg)
        return ll_t + a[0] + T.log(logarg)

    def output(self, Input, nmultilayer):
        """
        returns the Output of given MultiLayer when the Model is fed
        with given Input Data
        """
        if (self.MultiLayer[nmultilayer].Layer[0].get_inputsource()[0]
            == 'DataSet'):
            # if the InputSource of the MutliLayer is the DataSet, the Output
            # of the last ProcessingLayer of that MultiLayer is returned
            return self.MultiLayer[nmultilayer].output(
                Input, self.MultiLayer[nmultilayer].number_of_layers()-1)
        elif (self.MultiLayer[nmultilayer].Layer[0].get_inputsource()[0][0:10]
            == 'MultiLayer'):
            # if the InputSource is another MultiLayer the output of the last
            # Processing Layer of the given MultiLayer fed with the output of
            # its source is returned
            source = int(self.MultiLayer[nmultilayer].Layer[0]\
                .get_inputsource()[0][10])-1
            # the '-1' is because of the numbering:
            # MultiLayer1 -> nmultilayer = 0
            return self.MultiLayer[nmultilayer].output(
                self.output(Input,source),
                self.MultiLayer[nmultilayer].number_of_layers()-1)