# The NeSi algorithm - A Quick Start Guide

----------

## 1. Prerequisites
- Python 2.7.x
- 'numpy' and 'scipy', both available at [http://www.scipy.org/](http://www.scipy.org/)
- 'matplotlib' for graphical outputs, available at [http://matplotlib.org/](http://matplotlib.org/)
- 'mpi4py' 
- 'theano', availabe at [http://deeplearning.net/software/theano/](http://deeplearning.net/software/theano/)

Before running the algorithm, the data sets have to be downloaded and put into the /data-sets/ folder:

### MNIST
Go to [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) or download the files directly via:

- [http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
- [http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
- [http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
- [http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

Extract these files into the folder './data-sets/MNIST/'.
There should now be four files in that folder:

- t10k-images.idx3-ubyte
- t10k-labels.idx1-ubyte
- train-images.idx3-ubyte
- train-labels.idx1-ubyte
  
### 20 Newsgroups
Go to [http://qwone.com/~jason/20Newsgroups/](http://qwone.com/~jason/20Newsgroups/) and download the 'bydate' Matlab/Octave version, or directly via [http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz](http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz) and extract the content into the folder './data-sets/20 Newsgroups/'.
There should now be the following folders and files in that folder:

- label_names.txt
- vocabulary.txt
- 20news-bydate/
	- matlab/
    	- train.data
      	- train.label
      	- train.map
      	- test.data
      	- test.label
      	- test.map

----------
     
## 2. Execution and Configuration
To start the NeSi algorithm, `main.py` has to be executed with a main configuration file provided,
which links to further configuration files for the data set, the neural network model and the output
options (which have to exist in their corresponding folders).

For GPU execution run:
`
THEANO_FLAGS='device=gpu' python main.py [config_name]
`

To reproduce the experiments of the paper `[config_name]` has the form `[dataset]/[algorithm]-L[#Labels]`, with
    
    [dataset]: 'MNIST' or '20\ Newsgroups'
    [algorithm]: 'r-NeSi', 'r+-NeSi', 'ff-NeSi' or 'ff+-NeSi'
    [#Labels]: 10,100,600,1000,3000,60000 for MNIST or 20,40,200,800,2000,11269 for 20 Newsgroups
    
  e.g. to run the r+-NeSi algorithm on MNIST with 100 labels on a GPU execute

  `THEANO_FLAGS='device=gpu' python main.py MNIST/r+-NeSi-L100`

  e.g. to run the ff-NeSi algorithm on 20 Newsgroups with 800 labels on a GPU execute

  `THEANO_FLAGS='device=gpu' python main.py 20\ Newsgroups/ff-NeSi-L800`
  
For a fast test run, use `[dataset]/[algorithm]-test` as `[config_name]`.
  
The results will be stored in the ./output folder.
The free parameters of the algorithms can be set by editing the model configuration files under ./config/[dataset]/model/[algorithm].ini
