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

UPDATE NOTES
============
For people who are interested in using this code. We found that experiments of bayesian NN on SGHMC is not statistically significant in terms ofthe performance of Bayesian-NN vs SGDMOM method (Thanks to Daniel Seita) by change different random seed. We also tried follow up experiments on CIFAR and ImageNet dataset which suggests the sampling averging is harder to catch up due to variance in the sampling. This doesn't take away from the merit of SGHMC itself as the point of sampling algorithm, but does suggest that bayesian averaging is only as good as optimization based method on this setting.

