# Copyright (C) 2015, Dennis Forster <forster@fias.uni-frankfurt.de>
#
# LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
# ACADEMIC FREE LICENSE (AFL) v3.0.
#

import os
from mpi4py import MPI
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize_all_weights(output, model, nmultilayer, config, first=0, last=20, figure=1, show=False, save=True, ion=True):
    # Optimized for two processing layers
    if ( (output._PICTURE_OUTPUT == True)
         and (MPI.COMM_WORLD.Get_rank() == 0)
         and ( (model.MultiLayer[nmultilayer].get_iteration() % output._PICTURE_EVERY_N_ITERATIONS == 0)
               or (model.MultiLayer[nmultilayer].get_iteration() == model.MultiLayer[nmultilayer].get_iterations()) ) ):

        Layer = model.MultiLayer[nmultilayer].Layer

        # create figure if not given
        if (issubclass(type(figure), matplotlib.figure.Figure)):
            pass
        elif (issubclass(type(figure), int)):
            figure = plt.figure(figure)
        else:
            figure = plt.figure()
        figure.clf()

        if ( (last is None) or (last > Layer[1].C) ):
            last = Layer[1].C

        # plot all given images on sub-plots
        cols = int(np.ceil(math.sqrt(last-first)))
        rows = int(np.ceil((last-first)/float(cols)))

        #for squares data set:
        #cols = 1
        #rows = last-first

        NLAYERS = model.MultiLayer[nmultilayer].number_of_layers()

        width_ratios = []
        # 1: 1/N1 : 1/N2 = N1N2 : N2 : N1
        ratio = 1
        for nlayer in xrange(2,NLAYERS):
            ratio *= Layer[nlayer].get_weights().shape[0]
        for _ in xrange(cols):
            for nlayer in xrange(1,NLAYERS):
                if (nlayer == 1):
                    width_ratios.append(ratio)
                else:
                    width_ratios.append(ratio/Layer[nlayer].get_weights().shape[0])
        npixels_width = np.ceil(np.sqrt(Layer[1].D[0]))
        for nlayer in xrange(2,NLAYERS):
            npixels_width += np.ceil(np.sqrt(Layer[1].D[0]))/Layer[nlayer].get_weights().shape[0]
        npixels_width *= cols

        npixels_height = np.ceil(math.sqrt(Layer[1].D[0]))*rows

        scale = 2 #adjust for higher resolution

        pixel_width = scale*npixels_width + (NLAYERS-1)*cols+1
        pixel_height = scale*npixels_height + (rows+1)
        gs = gridspec.GridSpec(rows, (NLAYERS-1)*cols, width_ratios=width_ratios)
        text_space = 0.45
        # the spacing has some problems which require the arbitrary factors 2. and 2.14 in 'right', 'top' and 'wspace', 'hspace'
        gs.update(left=1./pixel_width, right=1.-2.*1./pixel_width, bottom = text_space/float(rows), top = 1.-2.*1./pixel_height, wspace = 2.14*((NLAYERS-1)*cols+1)/(scale*npixels_width), hspace = 1.2*text_space)
        figure.set_figwidth(pixel_width/100)
        figure.set_figheight(pixel_height/100)
        figure.set_facecolor('white')
        all_img_2D = [(Layer[nlayer].get_weights()) for nlayer in xrange(1,NLAYERS)]
        vocabulary_file = open('./data-sets/20Newsgroups/vocabulary.txt', 'r')
        vocabulary = []
        for line in vocabulary_file:
            vocabulary.append(line[0:-1]) # omits the '\n' at the end of each line
        vocabulary = np.asarray(vocabulary)
        try:
            label_names = config.get()['dataset']['classes']
        except:
            label_file = open('./data-sets/20Newsgroups/label_names.txt', 'r')
            label_names = []
            for line in label_file:
                label_names.append(line[0:-1]) # omits the '\n' at the end of each line
            label_names = np.asarray(label_names)

        # ymax = np.power(2,np.ceil(np.log2(np.max(all_img_2D[0][first:last,:]))))
        for nimage in xrange(first,last):
            for nlayer in xrange(1,NLAYERS):
                if (nlayer == 1):
                    # for some reason this produces a memory leak in combination with imshow:
                    #img_2D = Layer[nlayer].get_weights()[nimage,:]
                    img_2D = all_img_2D[nlayer-1][nimage,:]
                    index = np.argsort(img_2D)[::-1][0:20]
                    np.set_printoptions(threshold=np.nan)
                    width = 0.8  # bar width
                    ax = plt.subplot(gs[nimage*(NLAYERS-1)-first+nlayer-1])
                    xTickMarks = vocabulary[index]
                    ax.set_xticks(np.arange(index.shape[0])+0.5*width)
                    xtickNames = ax.set_xticklabels(xTickMarks)
                    plt.setp(xtickNames, rotation=90, fontsize=16)
                    ax.set_yticks([])
                    # plt.ylim([0, ymax])
                    figure_title = label_names[np.argmax(all_img_2D[nlayer][:,nimage])] \
                        + '\np(k|c) = ' + str(np.round(np.max(all_img_2D[nlayer][:,nimage])/np.sum(all_img_2D[nlayer][:,nimage])*100.)) + '%'\
                        + '\np(c|k) = ' + str(np.round(np.max(all_img_2D[nlayer][:,nimage])*100.)) + '%'
                    plt.title(figure_title, y=0.65, fontsize=20)
                    plt.axis('on')
                    ax.bar(np.arange(index.shape[0]), img_2D[index], width=width)
                else:
                    img_2D = all_img_2D[nlayer-1][:,nimage]
                    img_2D = np.reshape(img_2D,
                                        (img_2D.shape[0],1),
                                        order='C')
                    plt.subplot(gs[nimage*(NLAYERS-1)-first+nlayer-1])
                    plt.axis('off')
                    plt.imshow(img_2D, cmap="Greys", interpolation="nearest", aspect='auto')
        plt.ioff()
        if (save == True):
            if not os.path.exists('./output/%s/pictures/' % output._txtfoldername):
                os.makedirs('./output/%s/pictures/' % output._txtfoldername)
            filename = './output/%s/pictures/%s - Run%d - M%d - %d.png' %(output._txtfoldername, output._txtfilename, model.MultiLayer[nmultilayer].run(), nmultilayer+1, model.MultiLayer[nmultilayer].get_iteration())
            plt.savefig(filename,facecolor=figure.get_facecolor())
        if (show == True):
            if (ion == True):
                plt.ion()
            plt.draw()
            plt.show()
        else:
            figure.clf()
            plt.clf()
            plt.close()