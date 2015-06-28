# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import os
import datetime
from configobj import ConfigObj
from mpi4py import MPI
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

class Output(object):
    """
    This class handels all the file output operations.
    """

    def __init__(self):
        """
        Constructor
        """
        self._txtfoldername = ''
        self._txtfilename = ''
        self._CONV_OUTPUT = False
        self._CONV_PLOT = False
        self._CONV_ONLINE = False
        self._CONV_EVERY_N_ITERATIONS = 0
        self._PICTURE_OUTPUT = False
        self._PICTURE_EVERY_N_ITERATIONS = 0
        self._RESULTS_OUTPUT = True
        self._SETTING_OUTPUT = True
        self._SAVE_WEIGHTS = False
        self._LIKELIHOOD_OUTPUT = False
        self._LIKELIHOOD_ONLINE_OUTPUT = False
        self._LIKELIHOOD_EVERY_N_ITERATIONS = 0

    def read_config(self, config):
        """
        Read the file output options from config file.
        """

        #--- Path Name ---
        try:
            if config['output']['DATEPREFIX_PATH']:
                self._txtfoldername = str(datetime.date.today()) + ' - '
        except:
            self._txtfoldername = ''

        try:
            if (config['config']['foldername'] is None):
                self._txtfoldername = \
                    self._txtfoldername + config['config']['Model']
            else:
                self._txtfoldername = \
                    self._txtfoldername + config['config']['foldername']
        except:
            self._txtfoldername = \
                self._txtfoldername + config['config']['Model']


        #--- File Name ---
        try:
            if config['output']['DATEPREFIX_FILE']:
                self._txtfilename = \
                    self._txtfilename + str(datetime.date.today()) + ' - '
        except:
            self._txtfilename = ''
        try:
            if (config['config']['filename'] is None):
                self._txtfilename = \
                    self._txtfilename + config['config']['Model']
            else:
                self._txtfilename = \
                    self._txtfilename + config['config']['filename']
        except:
            self._txtfilename = self._txtfilename + config['config']['Model']


        #--- Picture Output Options ---
        try:
            self._PICTURE_OUTPUT = config['output']['PICTURE_OUTPUT']
        except:
            self._PICTURE_OUTPUT = False

        try:
            self._PICTURE_EVERY_N_ITERATIONS = \
                config['output']['PICTURE_EVERY_N_ITERATIONS']
        except:
            self._PICTURE_EVERY_N_ITERATIONS = 20


        #--- Convergence Output Options ---
        try:
            self._CONV_OUTPUT = config['output']['CONV_OUTPUT']
        except:
            self._CONV_OUTPUT = False

        try:
            self._CONV_PLOT = config['output']['CONV_PLOT']
        except:
            self._CONV_PLOT = False

        try:
            self._CONV_ONLINE = config['output']['CONV_ONLINE_OUTPUT']
        except:
            self._CONV_ONLINE = False

        try:
            self._CONV_EVERY_N_ITERATIONS = \
                config['output']['CONV_EVERY_N_ITERATIONS']
        except:
            self._CONV_EVERY_N_ITERATIONS = 20


        #--- Likelihood Output Options ---
        try:
            self._LIKELIHOOD_OUTPUT = config['output']['LIKELIHOOD_OUTPUT']
        except:
            self._LIKELIHOOD_OUTPUT = False

        try:
            self._LIKELIHOOD_ONLINE_OUTPUT = \
                config['output']['LIKELIHOOD_ONLINE_OUTPUT']
        except:
            self._LIKELIHOOD_ONLINE_OUTPUT = False

        try:
            self._LIKELIHOOD_EVERY_N_ITERATIONS = \
                config['output']['LIKELIHOOD_EVERY_N_ITERATIONS']
        except:
            self._LIKELIHOOD_EVERY_N_ITERATIONS = 1


        #--- Results Output Options ---
        try:
            self._RESULTS_OUTPUT = config['output']['RESULTS_OUTPUT']
        except:
            self._RESULTS_OUTPUT = True
        try:
            self._RESULTS_ONLINE_OUTPUT = \
                config['output']['RESULTS_ONLINE_OUTPUT']
        except:
            self._RESULTS_ONLINE_OUTPUT = False

        try:
            self._RESULTS_EVERY_N_ITERATIONS = \
                config['output']['RESULTS_EVERY_N_ITERATIONS']
        except:
            self._RESULTS_EVERY_N_ITERATIONS = 1



        #--- Setting Output Options ---
        try:
            self._SETTING_OUTPUT = config['output']['SETTING_OUTPUT']
        except:
            self._SETTING_OUTPUT = True


        #--- h5 Output Options ---
        try:
            self._SAVE_WEIGHTS = config['output']['SAVE_WEIGHTS']
        except:
            self._SAVE_WEIGHTS = False

        try:
            self._SAVE_DATA_INDEX = config['output']['SAVE_DATA_INDEX']
        except:
            self._SAVE_DATA_INDEX = True


    def create_output_files(self, config):
        """
        Create all necessary files and folders for output.
        """
        #--- Create Output Folder ---
        if (os.path.exists('./output/%s/' % self._txtfoldername)):
            npath = 0
            while (os.path.exists('./output/%s/' % (
                self._txtfoldername + ' (' + str(npath) + ')'))):
                npath += 1
            self._txtfoldername = self._txtfoldername + ' (' + str(npath) + ')'
        os.makedirs('./output/%s/' % self._txtfoldername)

        #--- Create Picture Folder ---
        if (self._PICTURE_OUTPUT or self._CONV_PLOT):
            os.makedirs('./output/%s/pictures/' % self._txtfoldername)
        if self._CONV_OUTPUT:
            for multilayer in config.get()['model'].keys():
                if (multilayer[0:10] == 'MultiLayer'):
                    filename = './output/%s/%s' % (
                        self._txtfoldername,
                        self._txtfilename + ' - Convergence - M' \
                            + multilayer[10] + '.txt')
                    self._out_file_conv = open(filename, 'a')
                    if (os.stat(filename).st_size == 0):
                        self._out_file_conv.write('%Iteration')
                        # TODO: implement better way to count number of layers
                        for nlayer in xrange(1,
                            len(config.get()['model'][multilayer].keys())-1):
                            self._out_file_conv.write('\tdW%dmean\tdW%dmax'%(
                                nlayer,nlayer))
                        self._out_file_conv.write('\n')
                    self._out_file_conv.close()

        #--- Create Online Likelihood File ---
        if self._LIKELIHOOD_ONLINE_OUTPUT:
            filename = './output/%s/%s' % (
                self._txtfoldername,
                self._txtfilename + ' - LogLikelihood.txt')
            self._out_file_llh = open(filename,'a')
            self._out_file_llh.write('%Iteration\tLogLikelihood\n')
            self._out_file_llh.close()

        #--- Create Results File ---
        if self._RESULTS_OUTPUT:
            filename = './output/%s/%s' % (
                self._txtfoldername,
                self._txtfilename + ' - Results.txt')
            self._results_file = open(filename,'a')
            if (os.stat(filename).st_size == 0):
                self._results_file.write(str(datetime.date.today()) + '\n')
                for params in reversed(config.changed_parameters().keys()):
                    self._results_file.write(str(params))
                    self._results_file.write('\t')
                self._results_file.write('Test-Error [%]')
                if self._LIKELIHOOD_OUTPUT:
                    self._results_file.write('\tLog-Likelihood\n')
                else:
                    self._results_file.write('\n')
            self._results_file.close()

        #--- Create Online Results File ---
        if self._RESULTS_ONLINE_OUTPUT:
            filename = './output/%s/%s' % (
                self._txtfoldername,
                self._txtfilename + ' - Online Results.txt')
            self._results_online_file = open(filename,'a')
            if (os.stat(filename).st_size == 0):
                self._results_online_file.write(str(datetime.date.today())+'\n')
                self._results_online_file.write('Iteration\t')
                for params in reversed(config.changed_parameters().keys()):
                    self._results_online_file.write(str(params) + '\t')
                self._results_online_file.write('Test-Error [%]')
                if self._LIKELIHOOD_OUTPUT:
                    self._results_online_file.write('\tLog-Likelihood\n')
                else:
                    self._results_online_file.write('\n')
            self._results_online_file.close()

        #--- Create Setting File ---
        if self._SETTING_OUTPUT:
            self._setting = ConfigObj()
            self._setting.filename= './output/%s/%s' % (self._txtfoldername,
                self._txtfilename + ' - Setting.txt')

        #--- Create h5 Folder ---
        if (self._SAVE_WEIGHTS or self._SAVE_DATA_INDEX):
            os.makedirs('./output/%s/h5/' % self._txtfoldername)

    def conv_pre(self, multilayer):
        """
        For convergence output: Save the state of the weights before the
        next training iteration.
        """
        if self._CONV_OUTPUT:
            self._Wbefore = []
            if (MPI.COMM_WORLD.Get_rank() == 0):
                for nlayer in xrange(1,multilayer.number_of_layers()):
                    self._Wbefore.append(np.copy(
                        multilayer.Layer[nlayer].get_weights()))

    def conv_post(self, multilayer):
        """
        For convergence output: Compare the state of the weights to the
        state when ConvBefore() was called and save the root mean
        squared deviation of all weights and the maximum summed total
        change of a single weight.
        """
        if self._CONV_OUTPUT:
            if (MPI.COMM_WORLD.Get_rank() == 0):
                self._out_file_conv = open('./output/%s/%s' % (
                    self._txtfoldername, self._txtfilename +
                    ' - Convergence - ' + multilayer.name() + '.txt'),'a')
                self._out_file_conv.write('%d\t' % multilayer.get_iteration())
                for nlayer in xrange(1,multilayer.number_of_layers()):
                    Wbefore = self._Wbefore[nlayer-1]
                    Wafter = multilayer.Layer[nlayer].get_weights()
                    deltaWmean = np.sqrt(np.mean((Wbefore-Wafter)**2))
                    deltaWmax = np.max(np.sum(np.abs(Wbefore-Wafter),axis=1))
                    self._out_file_conv.write(
                        '%.10f\t%.10f\t' % (deltaWmean, deltaWmax))
                self._out_file_conv.write('\n')
                self._out_file_conv.close()

    def visualize_convergence(self, model, nmultilayer):
        """
        Plot the weight convergence.
        """
        if (self._CONV_OUTPUT
            and self._CONV_PLOT
            and (MPI.COMM_WORLD.Get_rank() == 0)
            and ((self._CONV_ONLINE
                    and (model.MultiLayer[nmultilayer].get_iteration() %
                         self._CONV_EVERY_N_ITERATIONS == 0))
                or (model.MultiLayer[nmultilayer].get_iteration() ==
                      model.MultiLayer[nmultilayer].get_iterations())) ):
            filename = './output/%s/%s' % (
                self._txtfoldername,
                self._txtfilename + ' - Convergence - M' + str(nmultilayer+1)\
                + '.txt')
            with open(filename) as f:
                data = f.read()
            data = data.split('\n')
            # exclude first row (Comment) and last row ('\n'):
            NLayers = model.MultiLayer[nmultilayer].number_of_layers()
            for nlayer in xrange(1,NLayers):
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.set_title('Weight Convergence')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('dW**2')
                ax1.set_yscale('log')
                x = [row.split('\t')[0] for row in data[1:-1]]
                ymean = [row.split('\t')[(nlayer*2)-1] for row in data[1:-1]]
                ymax = [row.split('\t')[(nlayer*2)] for row in data[1:-1]]
                # because of log scale -> replace 0. with 0.0000000001
                for index, item in enumerate(ymean):
                    if (item == '0.0000000000'):
                        ymean[index] = '0.0000000001'
                for index, item in enumerate(ymax):
                    if (item == '0.0000000000'):
                        ymean[index] = '0.0000000001'
                start = 0
                for i in xrange(len(x)):
                    try:
                        if (x[i] == '1'):
                            start = i
                        if (i == len(x)-1):
                            ax1.plot(x[start:i+1],ymean[start:i+1])
                            ax1.plot(x[start:i+1],ymax[start:i+1])
                        elif (x[i+1] == '1'):
                            ax1.plot(x[start:i+1],ymean[start:i+1])
                            ax1.plot(x[start:i+1],ymax[start:i+1])
                    except:
                        pass
                filename = './output/%s/pictures/Convergence - %s - M%dL%d.png'%(
                    self._txtfoldername,
                    self._txtfilename,
                    nmultilayer+1,
                    nlayer)
                plt.savefig(filename)
                plt.close(fig)

    def save_weights(self, multilayer):
        """
        Save the current weights in h5 file.
        """
        if (self._SAVE_WEIGHTS and (MPI.COMM_WORLD.Get_rank() == 0)):
            for nlayer in xrange(1, multilayer.number_of_layers()):
                filename = './output/%s/h5/%s' % (
                    self._txtfoldername,
                    'Run%d%sL%d.h5' % (
                        multilayer.run(),
                        multilayer.name(),
                        nlayer))
                h5file = h5py.File(filename)
                h5file['W'] = multilayer.Layer[nlayer].get_weights()
                h5file.close()

    def save_data_index(self, nrun, dataset, mode=''):
        """
        Save the index numbers of the used data in h5 file.
        """
        if (self._SAVE_DATA_INDEX and (MPI.COMM_WORLD.Get_rank() == 0)):
            filename = './output/%s/h5/%s' % (
                self._txtfoldername,
                'Run%dData.h5' % (nrun))
            h5file = h5py.File(filename)
            if ((mode == 'train') or (mode == '')):
                h5file['train/unlabeled'] = dataset._indexlist['train_unlabeled']
                h5file['train/labeled'] = dataset._indexlist['train_labeled']
            if ((mode == 'test') or (mode == '')):
                h5file['test'] = dataset._indexlist['test']
            h5file.close()

    def visualize_weights(self, model, nmultilayer, nlayer, config):
        """
        Save pictures of the visualized weights of a given layer.
        """
        if (config.get()['dataset']['name'][0:12] == '20Newsgroups'):
            pass
        else:
            from visualize_as_matrix import visualize_weights
            visualize_weights(self, model, nmultilayer, nlayer, config)

    def visualize_all_weights(self, model, nmultilayer, config):
        """
        Save pictures of the visualized weights of all layers.
        """
        if (config.get()['dataset']['name'][0:12] == '20Newsgroups'):
            from visualize_20news import visualize_all_weights
        else:
            from visualize_as_matrix import visualize_all_weights
        visualize_all_weights(self, model, nmultilayer, config)

    def visualize_inputs(self, model, nmultilayer, config):
        """
        Save pictures of the visualized input data.
        """
        if (config.get()['dataset']['name'][0:12] == '20Newsgroups'):
            pass
        else:
            from visualize_as_matrix import visualize_inputs
            visualize_inputs(self, model, nmultilayer, config)

    def write_setting(self, config):
        """
        Save the used config setting.
        """
        if (self._SETTING_OUTPUT and (MPI.COMM_WORLD.Get_rank() == 0)):
            self._setting['========== Config =========='] = config['config']
            self._setting['========== DataSet =========='] = config['dataset']
            self._setting['========== Output =========='] = config['output']
            self._setting['========== Model =========='] = config['model']

            s_model = self._setting['========== Model ==========']
            for ml in s_model.keys(): # ml: MultiLayer
                for l in s_model[ml].keys(): # l: Layer
                    try:
                        if (s_model[ml][l]['A'] == 'Default'):
                            InputSource = s_model[ml][l]['InputSource']
                            if (InputSource[0:10] != 'InputLayer'):
                                if (InputSource[0:15] == 'ProcessingLayer'):
                                    D = config['model'][ml][InputSource]['C']
                                    s_model[ml][l]['A'] = D + 1
                                elif (InputSource[0][0:10] == 'MultiLayer'):
                                    D = config['model'][InputSource[0]]\
                                        [InputSource[1]]['C']
                                    s_model[ml][l]['A'] = D + 1
                                else:
                                    s_model[ml][l]['A'] = None
                            else:
                                s_model[ml][l]['A'] = None
                    except:
                        pass

                    try:
                        C = config['model'][ml][l]['C']
                        if (C == 'Default'):
                            try:
                                C = len(config['dataset']['classes'])
                            except:
                                C = config['dataset']['nclasses']
                        s_model[ml][l]['C'] = C
                    except:
                        pass

                    try:
                        if (s_model[ml][l]['Model'] == 'MM-LabeledOnly'):
                            N = config['dataset']['training_label_size']
                        else:
                            N = config['dataset']['training_data_size']
                    except:
                        pass

                    try:
                        epsilon = config['model'][ml][l]['epsilon']
                        if (epsilon[0] == 'Default'):
                            epsilon = epsilon + ('|', min(0.5 * C/N, 1.),)
                        if (epsilon[0] == 'factor'):
                            epsilon = epsilon + ('|', min(epsilon[1] * C/N, 1.),)
                        s_model[ml][l]['epsilon'] = epsilon
                    except:
                        pass

            self._setting['========== Config ==========']['total_mini_batch_size']\
                = config['config']['mini_batch_size']*MPI.COMM_WORLD.Get_size()

            self._setting.write()

    def write_results(self, classification, loglikelihood, config, run):
        """
        Write the given results into file.
        """
        if (self._RESULTS_OUTPUT and (MPI.COMM_WORLD.Get_rank() == 0)):
            filename = './output/%s/%s' % (
                self._txtfoldername,
                self._txtfilename + ' - Results.txt')
            self._results_file = open(filename,'a')
            for params in reversed(config.changed_parameters().keys()):
                self._results_file.write(
                    str(config.changed_parameters()[params][run]))
                self._results_file.write('\t')
            self._results_file.write('%f' % (classification))
            if self._LIKELIHOOD_OUTPUT:
                self._results_file.write('\t%f\n' % (loglikelihood))
            else:
                self._results_file.write('\n')
            self._results_file.close()

    def write_online_results(self, model, config, dataset, loglikelihood=None):
        """
        Calculate test error and likelihood during training and write
        into file.
        """
        if (self._RESULTS_ONLINE_OUTPUT
            and (model.MultiLayer[-1].get_iteration() % \
                 self._RESULTS_EVERY_N_ITERATIONS == 0)):
            # Calculate Test Error
            test_error = model.test(dataset.get_test_data(),
                                    dataset.get_test_label())
            MPI.COMM_WORLD.Barrier()
            # Calculate LogLikelihood
            if self._LIKELIHOOD_OUTPUT:
                if (loglikelihood is None):
                    loglikelihood = model.loglikelihood()
            MPI.COMM_WORLD.Barrier()
            # Output
            if (MPI.COMM_WORLD.Get_rank() == 0):
                filename = './output/%s/%s' % (
                    self._txtfoldername,
                    self._txtfilename + ' - Online Results.txt')
                self._results_online_file = open(filename,'a')
                self._results_online_file.write(
                    str(model.MultiLayer[-1].get_iteration()) + '\t')
                for params in reversed(config.changed_parameters().keys()):
                    self._results_online_file.write(
                        str(config.changed_parameters()[params][model._run])+'\t')
                self._results_online_file.write('%f' % (test_error))
                if self._LIKELIHOOD_OUTPUT:
                    self._results_online_file.write('\t%f' % (loglikelihood))
                self._results_online_file.write('\n')
                self._results_online_file.close()

    def write_loglikelihood(self, model, nmultilayer, loglikelihood=None):
        """
        Write loglikelihood to file.
        If the likelihood is not passed to the function, it is first
        calculated.
        """
        if (self._LIKELIHOOD_ONLINE_OUTPUT
            and (model.MultiLayer[nmultilayer].get_iteration() %\
                 self._LIKELIHOOD_EVERY_N_ITERATIONS == 0)):
            if (loglikelihood is None):
                loglikelihood = model.loglikelihood()
            if (MPI.COMM_WORLD.Get_rank() == 0):
                filename = './output/%s/%s' % (
                    self._txtfoldername,
                    self._txtfilename + ' - LogLikelihood.txt')
                self._out_file_llh = open(filename,'a')
                self._out_file_llh.write(
                    '%d\t%.10f\n' % (
                        model.MultiLayer[nmultilayer].get_iteration(),
                        loglikelihood))
                self._out_file_llh.close()