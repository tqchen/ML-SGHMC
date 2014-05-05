bayesnn
=======

Simple numpy implementation of (Bayesian) Neural Network 


Instructions
=======

Running MNIST experiment

* change the param.path_data in demo/mnist-sghmc.py to your dataset
* python mnist-sghmc.py 2>log.txt
* note: the evaluation statistics comes out from stderr, which is redirected into log.txt in last command

How to adapting to other classification experiment:

* check mnist.py, which is specific to mnist dataset experiment
* get a copy, change the function cfg_param and run_exp
* basically we need a rewritten version of mnist.py and demo/mnist-sghmc.py
