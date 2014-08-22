This is folder contains code for Bayesian matrix factorization based on SGHMC and SGLD sampler.

To run the experiment
(1) cd code; make;
(2) cd code/tools;make
(3) download ml-1m dataset from movielens,  change Makefile DATA path to the unziped path of ml-1m
(4) ./mkall.sh will make all the fold's input file
(5) cd run; use runSGHMC.sh to run the experiment

Acknowledgement: This code is based on code-base of [SVDFeature](svdfeature.apexlab.org) project.
