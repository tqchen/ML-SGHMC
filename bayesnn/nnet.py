"""
  Implementation of neural network 
  Core implementations
  Tianqi Chen
"""
import numpy as np
import sys

# Full connected layer
# note: all memory are pre-allocated, always use a[:]= instead of a= in assignment
class FullLayer:
    def __init__( self, i_node, o_node, init_sigma, rec_gsqr = False ):
        assert i_node.shape[0] == o_node.shape[0]
        self.rec_gsqr = rec_gsqr
        # node value
        self.i_node = i_node
        self.o_node = o_node
        # weight
        self.o2i_edge = np.float32( np.random.randn( i_node.shape[1], o_node.shape[1] ) * init_sigma )
        self.o2i_bias = np.zeros( o_node.shape[1], 'float32' ) 
        # gradient
        self.g_o2i_edge = np.zeros_like( self.o2i_edge )
        self.g_o2i_bias = np.zeros_like( self.o2i_bias )
        # gradient square
        self.sg_o2i_edge = np.zeros_like( self.o2i_edge )
        self.sg_o2i_bias = np.zeros_like( self.o2i_bias )
        if self.rec_gsqr:
            self.i_square = np.zeros_like( self.i_node )
            self.o_square = np.zeros_like( self.o_node )

    def forward( self, istrain = True ):
        # forward prop, node value to o_node
        self.o_node[:] = np.dot( self.i_node, self.o2i_edge ) + self.o2i_bias

    def backprop( self, passgrad = True ):
        # backprop, gradient is stored in o_node
        # divide by batch size
        bscale = 1.0 / self.o_node.shape[0]
        self.g_o2i_edge[:] = bscale * np.dot( self.i_node.T, self.o_node )
        self.g_o2i_bias[:] = np.mean( self.o_node, 0 )
        
        # record second moment of gradient if needed
        if self.rec_gsqr:
            self.o_square[:] = np.square( self.o_node )
            self.i_square[:] = np.square( self.i_node )
            self.sg_o2i_edge[:] = bscale * np.dot( self.i_square.T, self.o_square )
            self.sg_o2i_bias[:] = np.mean( self.o_square, 0 )
        
        # backprop to i_node if necessary
        if passgrad:
            self.i_node[:] = np.dot( self.o_node, self.o2i_edge.T )
            
    def params( self ):
        # return a reference list of parameters
        return [ (self.o2i_edge, self.g_o2i_edge, self.sg_o2i_edge), (self.o2i_bias,self.g_o2i_bias,self.sg_o2i_bias) ]

class ActiveLayer:
    def __init__( self, i_node, o_node, n_type = 'relu' ):
        assert i_node.shape[0] == o_node.shape[0]
        # node value
        self.n_type = n_type
        self.i_node = i_node
        self.o_node = o_node

    def forward( self, istrain = True ):
        # also get gradient ready in i node
        if self.n_type == 'relu':
            self.o_node[:] = np.maximum( self.i_node, 0.0 )
            self.i_node[:] = np.sign( self.o_node )
        elif self.n_type == 'tanh':
            self.o_node[:] = np.tanh( self.i_node )
            self.i_node[:] = ( 1.0 - np.square(self.o_node) )
        elif self.n_type == 'sigmoid':
            self.o_node[:] = 1.0 / ( 1.0 + np.exp( - self.i_node ) )
            self.i_node[:] = self.o_node * (1.0 - self.o_node)
        else:
            raise 'NNConfig', 'unknown node_type'
        
    def backprop( self, passgrad = True ):
        if passgrad:
            self.i_node[:] *= self.o_node;
            
    def params( self ):
        return []

class SoftmaxLayer:
    def __init__( self, i_node, o_label ):
        assert i_node.shape[0] == o_label.shape[0]
        assert len( o_label.shape ) == 1
        self.i_node  = i_node
        self.o_label = o_label

    def forward( self, istrain = True ):        
        nbatch = self.i_node.shape[0]
        self.i_node[:] = np.exp( self.i_node - np.max( self.i_node, 1 ).reshape( nbatch, 1 ) )
        self.i_node[:] = self.i_node / np.sum( self.i_node, 1 ).reshape( nbatch, 1 )

        
    def backprop( self, passgrad = True ):
        if passgrad:
            nbatch = self.i_node.shape[0]
            for i in xrange( nbatch ):
                self.i_node[ i, self.o_label[i] ] -= 1.0 
    def params( self ):
        return []

class RegressionLayer:
    def __init__( self, i_node, o_label, param ):
        assert i_node.shape[0] == o_label.shape[0]
        assert i_node.shape[0] == o_label.size
        assert i_node.shape[1] == 1
        self.i_tmp  = np.zeros_like( i_node )
        self.n_type = param.out_type
        self.i_node  = i_node
        self.o_label = o_label
        self.param = param
        self.base_score = None

    def init_params( self ):
        if self.base_score != None:
            return
        param = self.param
        self.scale = param.max_label - param.min_label;
        self.min_label = param.min_label
        self.base_score = (param.avg_label - param.min_label) / self.scale
        if self.n_type == 'logistic':
            self.base_score = - math.log( 1.0 / self.base_score - 1.0 );
        print 'range=[%f,%f], base=%f' %( self.min_label, param.max_label, param.avg_label )
    def forward( self, istrain = True ):     
        self.init_params()
        nbatch = self.i_node.shape[0]
        self.i_node[:] += self.base_score
        if self.n_type == 'logistic':
            self.i_node[:] = 1.0 / ( 1.0 + np.exp( -self.i_node ) )

        self.i_tmp[:] = self.i_node[:]

        # transform to approperiate output
        self.i_node[:] = self.i_node * self.scale + self.min_label
        
    def backprop( self, passgrad = True ):
        if passgrad:
            nbatch = self.i_node.shape[0]
            label = (self.o_label.reshape( nbatch, 1 ) - self.min_label) / self.scale
            self.i_node[:] = self.i_tmp[:] - label
            #print np.sum( np.sum( (label - self.i_tmp[:])**2 ) )
            
    def params( self ):
        return []

class NNetwork:
    def __init__( self, layers, nodes, o_label, factory ):
        self.nodes   = nodes
        self.o_label = o_label
        self.i_node = nodes[0]
        self.o_node = nodes[-1]
        self.layers = layers
        self.weights = []
        self.updaters = []
        for l in layers:
            self.weights += l.params()            
        for w, g_w, sg_w in self.weights:
            assert w.shape == g_w.shape and w.shape == sg_w.shape
            self.updaters.append( factory.create_updater( w, g_w, sg_w ) )

        self.updaters = factory.create_hyperupdater( self.updaters ) + self.updaters

    def update( self, xdata, ylabel ):
        self.i_node[:] = xdata
        for i in xrange( len(self.layers) ):
            self.layers[i].forward( True )

        self.o_label[:] = ylabel
        for i in reversed( xrange( len(self.layers) ) ):
            self.layers[i].backprop( i!= 0 )
        for u in self.updaters:
            u.update()

    def update_all( self, xdatas, ylabels ):
        for i in xrange( xdatas.shape[0] ):            
            self.update( xdatas[i], ylabels[i] )
        for u in self.updaters:
            u.print_info()

    def predict( self, xdata ):
        self.i_node[:] = xdata
        for i in xrange( len(self.layers) ):
            self.layers[i].forward( False )
        return self.o_node

# evaluator to evaluate results 
class NNEvaluator:
    def __init__( self, nnet, xdatas, ylabels, param, prefix='' ):
        self.nnet = nnet
        self.xdatas  = xdatas
        self.ylabels = ylabels
        self.param = param
        self.prefix = prefix
        nbatch, nclass = nnet.o_node.shape
        assert xdatas.shape[0] == ylabels.shape[0]
        assert nbatch == xdatas.shape[1]
        assert nbatch == ylabels.shape[1]
        self.o_pred  = np.zeros( ( xdatas.shape[0], nbatch, nclass ), 'float32'  )
        self.rcounter = 0
        self.sum_wsample = 0.0

    def __get_alpha( self ):
        if self.rcounter < self.param.num_burn:
            return 1.0
        else:
            self.sum_wsample += self.param.wsample
            return self.param.wsample / self.sum_wsample
        
    def eval( self, rcounter, fo ):
        self.rcounter = rcounter
        alpha = self.__get_alpha()        
        self.o_pred[:] *= ( 1.0 - alpha )
        sum_bad  = 0.0
        sum_loglike = 0.0
       
        for i in xrange( self.xdatas.shape[0] ):
            self.o_pred[i,:] += alpha * self.nnet.predict( self.xdatas[i] )
            y_pred = np.argmax( self.o_pred[i,:], 1 )            
            sum_bad += np.sum(  y_pred != self.ylabels[i,:] )
            for j in xrange( self.xdatas.shape[1] ):
                sum_loglike += np.log( self.o_pred[ i , j, self.ylabels[i,j] ] )

        ninst = self.ylabels.size
        fo.write( ' %s-err:%f %s-nlik:%f' % ( self.prefix, sum_bad/ninst, self.prefix, -sum_loglike/ninst) )

# Model parameter
class NNParam:
    def __init__( self ):
        # network type
        self.net_type = 'mlp2'
        self.node_type = 'sigmoid'
        self.out_type  = 'softmax'
        #------------------------------------
        # learning rate
        self.eta = 0.01
        # momentum decay
        self.mdecay = 0.1
        # weight decay, 
        self.wd = 0.0
        # number of burn-in round, start averaging after num_burn round
        self.num_burn = 1000
        # mini-batch size used in training
        self.batch_size = 500
        # initial gaussian standard deviation used in weight init
        self.init_sigma = 0.001
        # random number seed
        self.seed = 0
        # weight updating method
        self.updater = 'sgd'
        # temperature: temp=0 means no noise during sampling(MAP inference)
        self.temp = 1.0
        # start sampling weight after this round
        self.start_sample = 1
        #----------------------------------
        # hyper parameter sampling
        self.hyperupdater = 'none'
        # when to start sample hyper parameter
        self.start_hsample = 1
        # Gamma(alpha, beta) prior on regularizer
        self.hyper_alpha = 1.0
        self.hyper_beta  = 1.0        
        # sample hyper parameter each gap_hsample over training data
        self.gap_hsample = 1
        #-----------------------------------
        # adaptive learning rate and momentum
        # by default, no need to set these settings
        self.delta_decay = 0.0
        self.start_decay = None
        self.alpha_decay = 1.0
        self.decay_momentum = 0
        self.init_eta = None
        self.init_mdecay = None        
        #-----------------------
        # following things are not set by user
        # sample weight
        self.wsample = 1.0        
        # round counter
        self.rcounter = 0       

    # how many steps before resample hyper parameter
    def gap_hcounter( self ):
        return int(self.gap_hsample * self.num_train / self.batch_size)

    # adapt learning rate and momentum, if necessary
    def adapt_decay( self, rcounter ):
        # adapt decay ratio
        if self.init_eta == None:
            self.init_eta = self.eta
            self.init_mdecay = self.mdecay
        self.wsample = 1.0
        if self.start_decay == None:
            return

        d_eta = 1.0 * np.power( 1.0 + max( rcounter - self.start_decay, 0 ) * self.alpha_decay, - self.delta_decay )
        assert d_eta - 1.0 < 1e-6 and d_eta > 0.0
        
        if self.decay_momentum != 0:
            d_mom = np.sqrt( d_eta )
            self.wsample = d_mom
        else:
            d_mom = 1.0
            self.wsample = d_eta

        self.eta = d_eta * self.init_eta
        self.mdecay = d_mom * self.init_mdecay
        
    # set current round 
    def set_round( self, rcounter ):
        self.rcounter = rcounter
        self.adapt_decay( rcounter )
        if self.updater == 'sgld':
            assert np.abs( self.mdecay - 1.0 ) < 1e-6

    # get noise level for sampler
    def get_sigma( self ):
        if self.mdecay - 1.0 > -1e-5 or self.updater == 'sgld':
            scale = self.eta / self.num_train
        else:
            scale = self.eta * self.mdecay / self.num_train
        return np.sqrt( 2.0 * self.temp * scale ) 
    
    # whether we need to sample weight now
    def need_sample( self ):
        if self.start_sample == None:
            return False
        else:
            return self.rcounter >= self.start_sample

    # whether we need to sample hyper parameter now
    def need_hsample( self ):
        if self.start_hsample == None:
            return False
        else:
            return self.rcounter >= self.start_hsample

    # whether the network need to provide second moment of gradient
    def rec_gsqr( self ):
        return False

