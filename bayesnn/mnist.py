"""
  wrapper code for MNIST experiment
  Tianqi Chen
"""
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import sys
import random
import nncfg
import numpy as np
import nnet

# load MNIST dataset
def load(digits, dataset = "training", path = "."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in xrange(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

# default parameters
def cfg_param():
    param = nnet.NNParam()
    param.init_sigma = 0.01
    # input size, for mnist, it is 28*28
    param.input_size = 28 * 28
    # number of output class
    param.num_class = 10
    param.eta = 0.1
    param.mdecay = 0.1
    param.wd = 0.00002
    param.batch_size = 500
    return param

def run_exp( param ):
    np.random.seed( param.seed )
    net = nncfg.create_net( param )
    print 'network configure end, start loading data ...'

    # load in data 
    train_images, train_labels = load( range(10), 'training', param.path_data )
    test_images , test_labels  = load( range(10), 'testing' , param.path_data )

    # create a batch data
    # nbatch: batch size
    # doshuffle: True, shuffle the data 
    # scale: 1.0/256 scale by this factor so all features are in [0,1]
    train_xdata, train_ylabel  = nncfg.create_batch( train_images, train_labels, param.batch_size, True, 1.0/256.0 )
    test_xdata , test_ylabel   = nncfg.create_batch( test_images , test_labels, param.batch_size, True, 1.0/256.0 )
    
    # split validation set
    ntrain = train_xdata.shape[0]    
    nvalid = 10000
    assert nvalid % param.batch_size == 0
    nvalid = nvalid / param.batch_size
    valid_xdata, valid_ylabel = train_xdata[0:nvalid], train_ylabel[0:nvalid]
    train_xdata, train_ylabel = train_xdata[nvalid:ntrain], train_ylabel[nvalid:ntrain]
    
    # setup evaluator
    evals = []
    evals.append( nnet.NNEvaluator( net, train_xdata, train_ylabel, param, 'train' ))
    evals.append( nnet.NNEvaluator( net, valid_xdata, valid_ylabel, param, 'valid' ))
    evals.append( nnet.NNEvaluator( net, test_xdata ,  test_ylabel, param, 'test'  ))    
    
    # set parameters
    param.num_train = train_ylabel.size
    print 'loading end,%d train,%d valid,%d test, start update ...' % ( train_ylabel.size, valid_ylabel.size, test_ylabel.size )
        
    for it in xrange( param.num_round ):
        param.set_round( it )
        net.update_all( train_xdata, train_ylabel )
        sys.stderr.write( '[%d]' % it )
        for ev in evals:
            ev.eval( it, sys.stderr )
        sys.stderr.write('\n')            
    print 'all update end'
