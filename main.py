# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

"""Module docstring.


"""
from utils.sysfunctions import mem_usage
from utils.parallel import pprint
pprint('Start (RAM Usage: %0.2f MB)' % mem_usage())

# system imports
pprint('0.0.0 - System Imports', end='')
import sys
from mpi4py import MPI
pprint(' (%0.2f MB)' % mem_usage())

# custom imports
pprint('0.0.1 - Custom Imports')
from classes.model.Model import Model
from classes.input.DataSet import DataSet
from classes.config.Config import Config
from classes.output.Output import Output

# MPI definitions
pprint('0.1 - MPI Definitions', end='')
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
processor_name = MPI.Get_processor_name()
pprint(' (%0.2f MB)' % mem_usage())

# config import
pprint('0.2 - Config Import', end='')
config = Config()
if (len(sys.argv) == 2):
    try:
        config.import_config(str(sys.argv[1]))
    except:
        pprint('\nERROR: No valid config file provided')
        sys.exit()
else:
    pprint('\nERROR: main.py takes one (1) argument, ', end='')
    pprint('%d arguments were given.' % (len(sys.argv)-1))
    sys.exit()
pprint(' (%0.2f MB)' % mem_usage())
pprint(config.get()['config']['foldername'])

# dataset declaration
pprint('0.3 - Dataset Declaration', end='')
dataset = DataSet()
pprint(' (%0.2f MB)' % mem_usage())

# create output files
pprint('0.4 - Create Output Files', end='')
output = Output()
output.read_config(config.get(0))
if (rank == 0):
    output.create_output_files(config)
    output.write_setting(config.get())
pprint(' (%0.2f MB)' % mem_usage())

# build model
pprint('0.5 - Build Model', end='')
NN = Model()
NN.build(config.get(0))
pprint(' (%0.2f MB)' % mem_usage())

# repeat for the given number of independent runs:
for nrun in xrange(config.get(0)['config']['Runs']):
    pprint('Run %d' % (nrun + 1))

    # reset model (for the first run, this has no effect)
    pprint('%d.0.0 - Reset Model' % (nrun +1), end='')
    NN.clear()
    NN._run = nrun
    pprint(' (%0.2f MB)' % mem_usage())

    # import dataset(s) and distribute evenly between processes
    # (this actually only defines the used data indeces)
    pprint('%d.0.1 - Import Dataset' % (nrun +1), end='')
    if (output._RESULTS_OUTPUT and output._RESULTS_ONLINE_OUTPUT):
        dsets = ('train','test')
    else:
        dsets = ('train',)
    dataset.import_dataset(comm, config.get(nrun))
    for dset in dsets:
        if (rank == 0):
            dataset.pick_new_subset(config.get(nrun), dset, run=nrun)
            output.save_data_index(nrun+1, dataset, dset)
        dataset.distribute_set(comm, dset)
    pprint(' (%0.2f MB)' % mem_usage())

    # load the actual data into RAM
    pprint('%d.0.2 - Load Dataset into RAM' % (nrun +1), end='')
    for dset in dsets:
        dataset.load_set(dset)
    comm.Barrier()
    pprint(' (%0.2f MB)' % mem_usage())

    # print 'I am processor %s' % processor_name,
    # print 'rank %d'%rank + ' in group of %d processes.' %size,
    # print '#Images = %d, #Labels: %d' % (dataset.get_train_data().shape[0],
    #                                      dataset.number_of_train_labels())

    # train the multilayers one after another
    for nmultilayer in xrange(NN.number_of_multilayers()):
        pprint('%d.%d - MultiLayer %d' % (nrun+1,nmultilayer+1,nmultilayer+1))

        # intialize multilayer
        pprint('%d.%d.0 - Initialization' % (nrun+1,nmultilayer+1))
        NN.initialize_multilayer(nmultilayer,config.get(nrun),dataset)
        comm.Barrier()

        # save visualized input
        pprint('%d.%d.1 - Visualize Input' % (nrun+1,nmultilayer+1), end='')
        output.visualize_inputs(NN,0,config)
        pprint(' (%0.2f MB)' % mem_usage())

        # training
        pprint('%d.%d.2 - Training' % (nrun+1,nmultilayer+1))
        NN.train(nmultilayer,output,config,dataset)

        # save the learned weights
        pprint('%d.%d.3 - Save Weights' % (nrun+1,nmultilayer+1), end='')
        output.save_weights(NN.MultiLayer[nmultilayer])
        pprint(' (%0.2f MB)' % mem_usage())
    comm.Barrier()
    dataset.delete_set('train')

    # calculate log-likelihood of training data
    if output._LIKELIHOOD_OUTPUT:
        loglikelihood = NN.loglikelihood()
        pprint('LogLikelihood: %f' % (loglikelihood))
    else:
        loglikelihood = 0.
    comm.Barrier()

    # calculate test error
    if output._RESULTS_OUTPUT:
        if not output._RESULTS_ONLINE_OUTPUT:
            if (rank == 0):
                dataset.pick_new_subset(config.get(nrun), dset='test', run=nrun)
            dataset.distribute_set(comm,'test')
            dataset.load_set('test')
            output.save_data_index(nrun+1, dataset, 'test')
        test_error, _ = NN.test(dataset.get_test_data(),dataset.get_test_label())
        # test_likelihood = NN.loglikelihood((dataset.get_test_data(),dataset.get_test_label()))
        dataset.delete_set('test')
        comm.Barrier()
        pprint('Test Error [%%]: %f' % test_error)
        output.write_results(test_error, loglikelihood, config, nrun)# , test_likelihood
    comm.Barrier()

comm.Barrier()
pprint('Done.')