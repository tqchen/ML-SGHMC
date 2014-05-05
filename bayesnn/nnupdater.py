"""
  Implementation of neutral network 
  parameter update method
  Tianqi Chen
"""
import numpy as np
import sys

# updater that performs SGD update given weight parameter
class SGDUpdater:
    def __init__( self, w, g_w, param ):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like( w )
    def print_info( self ):
        return
    def update( self ):
        param = self.param
        self.m_w[:] *= ( 1.0 - param.mdecay )
        self.m_w[:] += (-param.eta) * ( self.g_w + self.wd * self.w )
        self.w[:]   += self.m_w
        
# updater that performs update given weight parameter using SGHMC/SGLD
class SGHMCUpdater:
    def __init__( self, w, g_w, param ):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like( w )
    def print_info( self ):
        return
    def update( self ):
        param = self.param
        self.m_w[:] *= ( 1.0 - param.mdecay )
        self.m_w[:] += (-param.eta) * ( self.g_w + self.wd * self.w )
        if param.need_sample():
            self.m_w[:] += np.random.randn( self.w.size ).reshape( self.w.shape ) * param.get_sigma()
        self.w[:]   += self.m_w
        

# updater that performs NAG(nestrov's momentum) update given weight parameter
class NAGUpdater:
    def __init__( self, w, g_w, param ):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like( w )
        self.m_old = np.zeros_like( w )
    def print_info( self ):
        return
    def update( self ):
        param = self.param
        momentum = 1.0 - param.mdecay
        self.m_old[:] = self.m_w
        self.m_w[:] *= momentum
        self.m_w[:] += (-param.eta) * ( self.g_w + self.wd * self.w )
        if param.need_sample():
            self.m_w[:] += np.random.randn( self.w.size ).reshape( self.w.shape ) * param.get_sigma()
        self.w[:]   += (1.0+momentum) * self.m_w - momentum * self.m_old

# Hyper Parameter Gibbs Gamma sampler for regularizer update 
class HyperUpdater:
    def __init__( self, param, updaterlist  ):
        self.updaterlist = updaterlist
        self.param = param
        self.scounter = 0
    # update hyper parameters
    def update( self ):
        param = self.param
        if not param.need_hsample():
            return

        self.scounter += 1
        if self.scounter % param.gap_hcounter() != 0:
            return
        else:
            self.scounter = 0
        
        sumsqr = sum( np.sum( u.w * u.w ) for u in self.updaterlist )
        sumcnt = sum( u.w.size for u in self.updaterlist )
        alpha = param.hyper_alpha + 0.5 * sumcnt
        beta  = param.hyper_beta + 0.5 * sumsqr
        
        if param.temp < 1e-6:
            # if we are doing MAP, take the mode, note: normally MAP adjust is not as well as MCMC
            plambda = max( alpha - 1.0, 0.0 ) / beta
        else:
            plambda = np.random.gamma( alpha, 1.0 / beta )

        # set new weight decay
        wd = plambda / param.num_train
        
        for u in self.updaterlist:
            u.wd = wd

        ss = ','.join( str(u.w.shape) for u in self.updaterlist )
        print 'hyperupdate[%s]:plambda=%f,wd=%f' % ( ss, plambda, wd )
        sys.stdout.flush()

    def print_info( self ):
        return
