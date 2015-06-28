# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

from configobj import ConfigObj
import copy
import numpy as np

class Config(object):
    '''
    Config class, which imports and processes the network's config file
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self._changedparams = {}

    def import_config(self, configname):
        """
        Imports the config files and converts them to a single config
        file for each run.
        """
        self._configname = configname
        try:
            configfolder = configname[0:configname.rindex('/')+1]
            configname = configname[configname.rindex('/')+1:]
        except:
            configfolder = ''
        # store config files as dictionary
        self._config = {}
        # main config
        self._config['config'] = ConfigObj(
            './config/' + configfolder + configname + '.ini',
            unrepr=True)
        # dataset
        self._config['dataset'] = ConfigObj(
            './config/' + configfolder + 'dataset/' + \
            self._config['config']['DataSet'] + '.ini',
            unrepr=True)
        # model
        self._config['model'] = ConfigObj(
            './config/' + configfolder + 'model/' + \
            self._config['config']['Model'] + '.ini',
            unrepr=True)
        # output
        self._config['output'] = ConfigObj(
            './config/' + configfolder + 'output/' + \
            self._config['config']['Output'] + '.ini',
            unrepr=True)

        # store parameters, that change during different runs in separate
        # config dictonaries for each run
        self._converted = []
        for nrun in xrange(self._config['config']['Runs']):
            self._converted.append(copy.deepcopy(self._config))

        # parameters, that are allowed to change between runs
        conversionlist = {'config':('mini_batch_size',),
                          'dataset':('training_data_size','training_label_size',
                                     'test_size'),
                          'model':('Iterations','A','C','epsilon'),
                          'output':()
                          }

        for k in conversionlist['config']:
            converted = self._convert(self._config['config'][k], k)
            for nrun in xrange(self._config['config']['Runs']):
                self._converted[nrun]['config'][k] = converted[nrun]

        for k in conversionlist['dataset']:
            converted = self._convert(self._config['dataset'][k], k)
            for nrun in xrange(self._config['config']['Runs']):
                self._converted[nrun]['dataset'][k] = converted[nrun]

        for k in conversionlist['output']:
            converted = self._convert(self._config['output'][k], k)
            for nrun in xrange(self._config['config']['Runs']):
                self._converted[nrun]['output'][k] = converted[nrun]

        for k in conversionlist['model']:
            for ml in self._config['model'].keys():
                if ml[0:10] == 'MultiLayer':
                    mlname = ml[10:]
                else:
                    mlname = '0'
                name = mlname + '_' + k
                try:
                    converted = self._convert(
                        self._config['model'][ml][k],
                        name)
                    for nrun in xrange(self._config['config']['Runs']):
                        self._converted[nrun]['model'][ml][k] = converted[nrun]
                except:
                    pass
                for layer in self._config['model'][ml].keys():
                    if layer[0:15] == 'ProcessingLayer':
                        lname = '.' + layer[15:]
                    elif layer[0:10] == 'InputLayer':
                        lname = '.0'
                    else:
                        lname = ''
                    name = mlname + lname + '_' + k
                    try:
                        value = self._config['model'][ml][layer][k]
                        converted = self._convert(value,name)
                        for nrun in xrange(self._config['config']['Runs']):
                            # check for 'factor' keyword:
                            if (type(value) == tuple):
                                if value[0] == "factor":
                                    self._converted[nrun]['model'][ml][layer][k] = \
                                        ["factor", converted[nrun]]
                                else:
                                    self._converted[nrun]['model'][ml][layer][k] = \
                                        converted[nrun]
                            else:
                                self._converted[nrun]['model'][ml][layer][k] = \
                                    converted[nrun]
                    except:
                        pass

    def name(self):
        """
        return name of config
        """
        return self._configname

    def get(self, nthrun=None):
        """
        return the config file for given run
        """
        if (nthrun is None):
            return self._config
        else:
            if ((nthrun >= 0) and (nthrun <= self._config['config']['Runs'])):
                return self._converted[nthrun]
            else:
                return self._config

    def _convert(self, value, name=None):
        if (type(value) == tuple):
            if (value[0] == "factor"):
                value = value[1:]
            if (value[0] == "inc"):
                # interpret format as (initial, delta, every_nth_run)
                # and increase the value of the parameter every
                # 'every_nth_run' by 'delta' starting at 'initial'
                # e.g.: 'inc', 10,5,2 ->
                # [#run:value] = 1:10; 2:10; 3:15; 4:15: 5:20; 6:20; etc.
                if ( (len(value) >= 2) and (len(value) <= 4) ):
                    # can take between 1 and 3 parameters
                    if (len(value) >= 3):
                        initial = value[1]
                        delta = value[2]
                        if (len(value) == 4):
                            every_nth_run = value[3]
                        else:
                            # if only 'initial' and 'delta' are given, set
                            # 'every_nth_run' to 1
                            every_nth_run = 1
                        converted = []
                        for nthrun in xrange(self._config['config']['Runs']):
                            converted.append(initial + delta * (nthrun//every_nth_run))
                        self._changedparams[name] = converted
                    else:
                        # if only 'initial' is given, set 'delta' and
                        # 'every_nth_run' to 1
                        converted = [value[1]]*self._config['config']['Runs']
                else:
                    print 'CONFIG ERROR: wrong number of entries: ', value
            elif (value[0] == 'runs'):
                # interpret format as [value1,runs1,value2,runs2,...],
                # where the parameter is set to 'value[#]' for
                # 'runs[#]' runs.
                # e.g.; 'runs', 10,2,25,3,40,1 ->
                # [#run:value] = 1:10; 2:10; 3:25; 4:25; 5:25, 6:40
                # the total number of runs must be equal to the number
                # of 'Runs' in config-file
                if (len(value[1:])%2 == 0):
                    if (np.sum(value[2::2]) == self._config['config']['Runs']):
                        converted = []
                        for nvalue, nruns in zip(value[1::2],value[2::2]):
                            for _ in xrange(nruns):
                                converted.append(nvalue)
                        self._changedparams[name] = converted
                    else:
                        print 'CONFIG ERROR: wrong number of runs: ', value
                else:
                    print 'CONFIG ERROR: wrong number of entries: ', value
            elif (value[0] == 'list'):
                #interpret format as [value1,value2,value3,value4,...] where
                #the parameter is set to 'value[#]' for run[#].
                #e.g.; [10,10,25,40,30] -> [#run:value] = 1;10; 2:10; 3:25; 4:40; 5:30
                #the total number of values must be equal to the number of 'Runs' in config-file
                if (len(value[1:]) == self._config['config']['Runs']):
                    converted = []
                    for nvalue in value[1:]:
                        converted.append(nvalue)
                    self._changedparams[name] = converted
                else:
                    print 'CONFIG ERROR: number of values must equal the number of runs:', value

        else:
            converted = [value]*self._config['config']['Runs']
        return converted

    def changed_parameters(self):
        return self._changedparams