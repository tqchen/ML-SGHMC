"""
  Implementation of neutral network 
  network configurations
  Tianqi Chen
"""
import nnet
import nnupdater
import numpy as np

class NNFactory:
    def __init__( self, param ):
        self.param = param
    def create_updater( self, w, g_w, sg_w ):
        if self.param.updater == 'sgd':
            return nnupdater.SGDUpdater( w, g_w, self.param )
        elif self.param.updater == 'sghmc' or self.param.updater == 'sgld':
            if self.param.updater == 'sgld':
                self.param.mdecay = 1.0
            return nnupdater.SGHMCUpdater( w, g_w, self.param )
        elif self.param.updater == 'nag':
            return nnupdater.NAGUpdater( w, g_w, self.param )
        else:
            raise 'NNConfig', 'unknown updater'

    def create_hyperupdater( self, updaterlist ):
        if self.param.hyperupdater == 'none':
            return []
        elif self.param.hyperupdater == 'gibbs-joint':
            return [ nnupdater.HyperUpdater( self.param, updaterlist ) ]
        elif self.param.hyperupdater == 'gibbs-sep':
            return [ nnupdater.HyperUpdater( self.param, [u] ) for u in updaterlist ]
        else:
            raise 'NNConfig', 'unknown hyperupdater'

    def create_olabel( self ):
        param = self.param
        if param.out_type == 'softmax':
            return np.zeros((param.batch_size),'int8')
        else:
            return np.zeros((param.batch_size),'float32')

    def create_outlayer( self, o_node, o_label ):
        param = self.param
        if param.out_type == 'softmax':
            return nnet.SoftmaxLayer( o_node, o_label )
        elif param.out_type == 'linear':
            return nnet.RegressionLayer( o_node, o_label, param ) 
        elif param.out_type == 'logistic':
            return nnet.RegressionLayer( o_node, o_label, param )
        else:
            raise 'NNConfig', 'unknown out_type'
        
def softmax( param ):
    factory = NNFactory( param )
    # setup network for softmax
    i_node = np.zeros( (param.batch_size, param.input_size), 'float32' )
    o_node = np.zeros( (param.batch_size, param.num_class), 'float32' )
    o_label = factory.create_olabel()

    nodes = [ i_node, o_node ]
    layers = [ nnet.FullLayer( i_node, o_node, param.init_sigma, param.rec_gsqr() )  ]

    layers+= [ factory.create_outlayer( o_node, o_label ) ]
    net = nnet.NNetwork( layers, nodes, o_label, factory ) 
    return net

def mlp2layer( param ):
    factory = NNFactory( param )
    # setup network for 2 layer perceptron
    i_node = np.zeros( (param.batch_size, param.input_size), 'float32' )
    o_node = np.zeros( (param.batch_size, param.num_class), 'float32' )
    h1_node = np.zeros( (param.batch_size, param.num_hidden), 'float32' )
    h2_node = np.zeros_like( h1_node )
    o_label = factory.create_olabel()

    nodes = [ i_node, h1_node, h2_node, o_node ]
    layers = [ nnet.FullLayer( i_node, h1_node, param.init_sigma, param.rec_gsqr() )  ]
    layers+= [ nnet.ActiveLayer( h1_node, h2_node, param.node_type )   ]
    layers+= [ nnet.FullLayer( h2_node, o_node, param.init_sigma, param.rec_gsqr() )  ]
    layers+= [ factory.create_outlayer( o_node, o_label ) ]

    net = nnet.NNetwork( layers, nodes, o_label, factory )    
    return net

def mlp3layer( param ):
    factory = NNFactory( param )
    # setup network for 2 layer perceptron
    i_node = np.zeros( (param.batch_size, param.input_size), 'float32' )
    o_node = np.zeros( (param.batch_size, param.num_class), 'float32' )
    h1_node = np.zeros( (param.batch_size, param.num_hidden), 'float32' )
    h2_node = np.zeros_like( h1_node )

    h1_node2 = np.zeros( (param.batch_size, param.num_hidden2), 'float32' )
    h2_node2 = np.zeros_like( h1_node2 )
    
    o_label = factory.create_olabel()

    nodes = [ i_node, h1_node, h2_node, h1_node2, h2_node2, o_node ]

    layers = [ nnet.FullLayer( i_node, h1_node, param.init_sigma, param.rec_gsqr() )  ]
    layers+= [ nnet.ActiveLayer( h1_node, h2_node, param.node_type )   ]

    layers+= [ nnet.FullLayer( h2_node, h1_node2, param.init_sigma, param.rec_gsqr() )  ]
    layers+= [ nnet.ActiveLayer( h1_node2, h2_node2, param.node_type )   ]

    layers+= [ nnet.FullLayer( h2_node2, o_node, param.init_sigma, param.rec_gsqr() )  ]

    layers+= [ factory.create_outlayer( o_node, o_label ) ]
    return net

def create_net( param ):
    if param.net_type == 'mlp2':
        return mlp2layer( param )
    if param.net_type == 'mlp3':
        return mlp3layer( param )
    elif param.net_type == 'softmax':
        return softmax( param )
    else:
        raise 'NNConfig', 'unknown net_type'

# create a batch data from existing training data
# nbatch: batch size
# doshuffle: whether shuffle data first befor batch
# scale: scale the feature by scale
def create_batch( images, labels, nbatch, doshuffle=False, scale=1.0 ):
    if labels.shape[0] % nbatch != 0:
        print '%d data will be dropped during batching' % (labels.shape[0] % nbatch)
    nsize = labels.shape[0] / nbatch * nbatch
    assert images.shape[0] == labels.shape[0]

    if doshuffle:
        ind = range( images.shape[0] )
        np.random.shuffle( ind )
        images, labels = images[ind], labels[ind]

    images = images[ 0 : nsize ];
    labels = labels[ 0 : nsize ];
    xdata = np.float32( images.reshape( labels.shape[0]/nbatch, nbatch, images[0].size ) ) * scale
    ylabel = labels.reshape( labels.shape[0]/nbatch, nbatch )    
    return xdata, ylabel
