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

def visualize_inputs(output, model, nmultilayer, config, first=0, last=100, figure=1, show=False, save=True, ion=True):
    if ( output._PICTURE_OUTPUT and (MPI.COMM_WORLD.Get_rank() == 0) ):

        Layer = model.MultiLayer[nmultilayer].Layer[0]

        # create figure if not given
        if (issubclass(type(figure), matplotlib.figure.Figure)):
            pass
        elif (issubclass(type(figure), int)):
            figure = plt.figure(figure)
        else:
            figure = plt.figure()
        figure.clf()

        if ( (last is None) or (last > Layer.get_input_data().shape[0]) ):
            last = Layer.get_input_data().shape[0]

        # plot all given images on sub-plots
        cols = int(np.ceil(math.sqrt(last-first)))
        rows = int(np.ceil((last-first)/float(cols)))
        img_2D = Layer.get_input_data()[first:last,:]
        pixel_width = config.get()['dataset']['PIXEL_WIDTH']
        pixel_height = config.get()['dataset']['PIXEL_HEIGHT']
        D = Layer.get_input_data().shape[1]
        try: #Grayscale Image
            img_2D = np.append(img_2D,np.zeros(shape=(last-first, pixel_width*pixel_height-D)),axis=1)
            img_2D = np.reshape(img_2D,(last-first,pixel_height,pixel_width),order='C')
            grayscale = True
        except: #RGB Image
            img_2D = np.reshape(img_2D,(last-first,pixel_height,pixel_width,Layer.get_input_data().shape[2]),order='C')
            grayscale = False

        scale = 4 #adjust for higher resolution
        #scale = int(800/(np.ceil(math.sqrt(D))*cols))
        gs_pixel_width = float(scale*pixel_width*cols + (cols+1))
        gs_pixel_height = float(scale*pixel_height*rows + (rows+1))
        gs = gridspec.GridSpec(rows, cols)
        # unfortunately, the spacing seems to not be exact, but to introduce arbitrary deviations, which have to be compensated
        # gs.update(left=1./gs_pixel_width, right=1.-1./gs_pixel_width, bottom = 1./gs_pixel_height, top = 1.-2.*1./gs_pixel_height, wspace = 1.*(cols+1)/float(scale*pixel_width*cols), hspace = 2.14*(rows+1)/float(scale*pixel_height*rows))
        gs.update(left=1./gs_pixel_width, right=1.-1./gs_pixel_width, bottom = 1./gs_pixel_height, top = 1.-2.*1./gs_pixel_height, wspace = 2.14*(cols+1)/float(scale*pixel_width*cols), hspace = 2.14*(rows+1)/float(scale*pixel_height*rows))
        # for squares data set:
        #gs.update(left=1./gs_pixel_width, right=1.-2.*1./gs_pixel_width, bottom = 1./gs_pixel_height, top = 1.-2.25/gs_pixel_height, wspace = (cols+1)/float(scale*np.ceil(math.sqrt(D))*cols), hspace = 1.65*(rows+1)/float(scale*np.ceil(math.sqrt(D))*rows))

        figure.set_figwidth(gs_pixel_width/100)
        figure.set_figheight(gs_pixel_height/100)
        figure.set_facecolor('black')
        for h in xrange(last-first):
            figure.add_subplot(plt.subplot(gs[h]))
            plt.axis('off')
            if grayscale:
                plt.imshow(img_2D[h], cmap="Greys", interpolation="nearest", aspect='auto')
                #plt.imshow(img_2D[h], cmap="jet", interpolation="nearest", aspect='auto')
            else:
                plt.imshow(img_2D[h], interpolation="nearest", aspect='auto')
        plt.ioff()
        if save:
            if not os.path.exists('./output/%s/pictures/' % output._txtfoldername):
                os.makedirs('./output/%s/pictures/' % output._txtfoldername)
            filename = './output/%s/pictures/%s - Input %d-%d.png' %(output._txtfoldername, output._txtfilename, first+1, last)
            plt.savefig(filename,facecolor=figure.get_facecolor())
        if show:
            if ion:
                plt.ion()
            plt.draw()
            plt.show()
        else:
            plt.close(figure)

def visualize_weights(output, model, nmultilayer, nlayer, config, first=0, last=100, figure=1, show=False, save=True, ion=True):
    # TODO: check for same memory leak as was in VisualizeAllWeights!
    if ( output._PICTURE_OUTPUT
         and (MPI.COMM_WORLD.Get_rank() == 0)
         and ( (model.MultiLayer[nmultilayer].get_iteration() % output._PICTURE_EVERY_N_ITERATIONS == 0)
               or (model.MultiLayer[nmultilayer].get_iteration() == model.MultiLayer[nmultilayer].get_iterations()) ) ):

        Layer = model.MultiLayer[nmultilayer].Layer[nlayer]

        # create figure if not given
        if (issubclass(type(figure), matplotlib.figure.Figure)):
            pass
        elif (issubclass(type(figure), int)):
            figure = plt.figure(figure)
        else:
            figure = plt.figure()
        figure.clf()

        if ( (last is None) or (last > Layer.GetNumberOfNeurons()) ):
            last = Layer.GetNumberOfNeurons()

        D = Layer.get_input_data().shape[1]
        if nlayer == 1:
            pixel_width = config.get()['dataset']['PIXEL_WIDTH']
            pixel_height = config.get()['dataset']['PIXEL_HEIGHT']
        else:
            pixel_width = np.ceil(math.sqrt(Layer.D[0]))
            pixel_height = np.ceil(math.sqrt(Layer.D[0]))            

        # plot all given images on sub-plots
        cols = np.ceil(math.sqrt(last-first))
        rows = np.ceil((last-first)/cols)
        try:
            #Grayscale Image
            img_2D = np.append(Layer.get_weights()[first:last,:],np.zeros(shape=(last-first,pixel_width*pixel_height-Layer.D[0])),axis=1)
            img_2D = np.reshape(img_2D,(last-first,pixel_width,pixel_height),order='C')
        except:
            #RGB Image
            img_2D = np.append(Layer.get_weights()[first:last,:],np.zeros(shape=(last-first,pixel_width*pixel_height-Layer.D[0],Layer.get_weights().shape[2])),axis=1)
            img_2D = np.reshape(img_2D,(last-first,pixel_width,pixel_height,Layer.get_weights().shape[2]),order='C')
        #gs_pixel_width = 5.*np.ceil(math.sqrt(Layer.D[0]))*cols + (cols-1.)
        #gs_pixel_height = 5.*np.ceil(math.sqrt(Layer.D[0]))*rows + (rows-1.)

        scale = int(800/(np.ceil(math.sqrt(Layer.D[0]))*cols))
        gs_pixel_width = scale*pixel_width*cols + (cols-1.)
        gs_pixel_height = scale*pixel_height*rows + (rows-1.)

        figure.set_figwidth(gs_pixel_width/100)
        figure.set_figheight(gs_pixel_height/100)
        figure.set_facecolor('black')
        for h in xrange(last-first):
            figure.add_subplot(rows, cols, h+1)
            plt.axis('off')
            plt.subplots_adjust(left = 0., right = 1., bottom = 0., top = 1., wspace = 2.*(cols-1.)/gs_pixel_width, hspace = 3.*(rows-1.)/gs_pixel_height)
            plt.imshow(img_2D[h], cmap="Greys", interpolation="nearest", aspect='auto')
        plt.ioff()
        if (save == True):
            if not os.path.exists('./output/%s/pictures/' % output._txtfoldername):
                os.makedirs('./output/%s/pictures/' % output._txtfoldername)
            filename = './output/%s/pictures/%s - Run%d - M%dL%d - %d.png' %(output._txtfoldername, output._txtfilename, model.MultiLayer[nmultilayer].run(), nmultilayer+1, nlayer, model.MultiLayer[nmultilayer].get_iteration())
            plt.savefig(filename,facecolor=figure.get_facecolor())
        if (show == True):
            if (ion == True):
                plt.ion()
            plt.draw()
            plt.show()
        else:
            plt.close(figure)


def visualize_all_weights(output, model, nmultilayer, config, first=0, last=100, figure=1, show=False, save=True, ion=True):
    # Optimized for two processing layers
    if ( output._PICTURE_OUTPUT
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

        pixel_width = []
        pixel_height = []
        for nlayer in xrange(1,NLAYERS):
            if nlayer == 1 and nmultilayer == 0:
                pixel_width.append(config.get()['dataset']['PIXEL_WIDTH'])
                pixel_height.append(config.get()['dataset']['PIXEL_HEIGHT'])
            elif nlayer == NLAYERS-1 and nmultilayer == 0:
                pixel_width.append(1)
                pixel_height.append(Layer[nlayer].C)
            else:
                pixel_width.append(np.ceil(math.sqrt(Layer[nlayer].D[0])))
                pixel_height.append(np.ceil(math.sqrt(Layer[nlayer].D[0])))
        npixels_width = pixel_width[0]
        for nlayer in xrange(2,NLAYERS):
            npixels_width += pixel_width[0]/Layer[nlayer].get_weights().shape[0]
        npixels_width *= cols

        npixels_height = pixel_height[0]*rows
        scale = max(4, np.ceil(np.max(pixel_height)/float(pixel_height[0]))) #adjust for higher resolution

        gs_pixel_width = scale*npixels_width + (NLAYERS-1)*cols+1
        gs_pixel_height = scale*npixels_height + (rows+1)
        gs = gridspec.GridSpec(rows, (NLAYERS-1)*cols, width_ratios=width_ratios)
        # the spacing has some problems which require the arbitrary factors 2. and 2.14 in 'right', 'top' and 'wspace', 'hspace'
        #gs.update(left=1./gs_pixel_width, right=1.-2.*1./gs_pixel_width, bottom = 1./gs_pixel_height, top = 1.-2.*1./gs_pixel_height, wspace = 2.14*((NLAYERS-1)*cols+1)/(scale*npixels_width), hspace = 2.14*(rows+1)/float((scale*npixels_height)))
        # gs.update(left=1./gs_pixel_width, right=1.-1./gs_pixel_width, bottom = 1./gs_pixel_height, top = 1.-2.*1./gs_pixel_height, wspace = 1.*((NLAYERS-1)*cols+1)/(scale*npixels_width), hspace = 2.14*(rows+1)/float((scale*npixels_height)))
        gs.update(left=1./gs_pixel_width, right=1.-1./gs_pixel_width, bottom = 1./gs_pixel_height, top = 1.-2.*1./gs_pixel_height, wspace = 2.14*((NLAYERS-1)*cols+1)/(scale*npixels_width), hspace = 2.14*(rows+1)/float((scale*npixels_height)))
        # for C10:
        #gs.update(left=1./gs_pixel_width, right=1.-2.*1./gs_pixel_width, bottom = 1./gs_pixel_height, top = 1.-2.*1./gs_pixel_height, wspace = 1.*((NLAYERS-1)*cols+1)/(scale*npixels_width), hspace = 1.*(rows+1)/float((scale*npixels_height)))
        # for squares data set:
        #gs.update(left=1./gs_pixel_width, right=1.-1./float(gs_pixel_width), bottom = 1./float(gs_pixel_height), top = 1.-2./float(gs_pixel_height), wspace = 1.*((NLAYERS-1)*cols+1)/(scale*npixels_width), hspace = 1.65*(rows+1)/float((scale*npixels_height)))
        figure.set_figwidth(gs_pixel_width/100)
        figure.set_figheight(gs_pixel_height/100)
        figure.set_facecolor('black')
        all_img_2D = [(Layer[nlayer].get_weights()) for nlayer in xrange(1,NLAYERS)]

        # # Limits for colormap. If these are not given the colormap of each
        # # subplot is scaled independently
        # vmin = []
        # vmax = []
        # for nlayer in xrange(1,NLAYERS):
        #     vmin.append(np.min(all_img_2D[nlayer-1]))
        #     vmax.append(np.max(all_img_2D[nlayer-1]))

        for nimage in xrange(first,last):
            for nlayer in xrange(1,NLAYERS):
                if (nlayer == 1):
                    # for some reason this produces a memory leak in combination with imshow:
                    #img_2D = Layer[nlayer].get_weights()[nimage,:]
                    img_2D = all_img_2D[nlayer-1][nimage,:]
                    # try:
                    #Grayscale Image
                    img_2D = np.append(img_2D,np.zeros(shape=(pixel_height[nlayer-1]*pixel_width[nlayer-1]-img_2D.shape[0])),axis=0)
                    img_2D = np.reshape(img_2D,(pixel_height[nlayer-1],pixel_width[nlayer-1]),order='C')
                    # except:
                        # try:
                        #     #RGB Image
                        #     #-- TODO: implement np.append for RGB image
                        #     img_2D = np.reshape(img_2D,(pixel_width[nlayer-1],pixel_height[nlayer-1],Layer.get_weights().shape[2]),order='C')
                        # except:
                        #     pass
                else:
                    img_2D = Layer[nlayer].get_weights()[:,nimage]
                    img_2D = np.reshape(img_2D,(img_2D.shape[0],1),order='C')

                #figure.add_subplot(plt.subplot(gs[nimage*(NLAYERS-1)-first+nlayer-1]))
                plt.subplot(gs[nimage*(NLAYERS-1)-first+nlayer-1])
                plt.axis('off')
                plt.imshow(img_2D, cmap="Greys", interpolation="nearest", aspect='auto')
                # plt.imshow(img_2D, cmap="jet", interpolation="nearest", aspect='auto')

                # if nlayer == 1:
                #     plt.imshow(img_2D, cmap="Greys", interpolation="nearest", aspect='auto')
                # else:
                #     # If vmin and vmax is not given, the colormap of each
                #     # subplot is saled independently. This can be helpful
                #     # to better see the class belonging of each patch when
                #     # only very few labels are given.
                #     # plt.imshow(img_2D, cmap="Greys", interpolation="nearest",
                #     #     aspect='auto', vmin=0, vmax=vmax[nlayer-1])
                #     plt.imshow(img_2D, cmap="jet", interpolation="nearest",
                #         aspect='auto', vmin=0, vmax=1)
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
