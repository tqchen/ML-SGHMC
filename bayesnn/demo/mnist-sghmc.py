# demo for mnist experiment
import sys
sys.path.append('..')
import mnist

# get default prameters
param = mnist.cfg_param()

# set necessary parameters we want

param.batch_size = 500
# number of total rounds run
param.num_round = 800

param.num_hidden = 100

# change the following line to PATH/TO/MNIST dataset
param.path_data = '../../../../../data/image/mnist/'

# network type
param.net_type = 'mlp2'
# updating method
param.updater  = 'sghmc'
# hyper parameter sampling: use gibbs for each parameter group
param.hyperupdater = 'gibbs-sep'
# number of burn-in round, start averaging after num_burn round
param.num_burn = 50

# learning rate
param.eta=0.1
# alpha, momentum decay
param.mdecay=0.01

# run the experiment
mnist.run_exp( param )
