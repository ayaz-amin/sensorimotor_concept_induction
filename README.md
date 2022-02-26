# Sensorimotor Concept Induction for Generalised Robot Behaviour via Probabilistic Modelling

## What it is
This is a collection of scripts for running three different machine learning algorithms (naive Bayes, neural network and gradient boosting) on a simulated multisensor robot. The algorithms are trained and tested on data sampled from the simulator.

## How to get started
Make sure Python 3.7.4 is installed. The following libraries are needed to execute the scripts:
- Pytorch (version >= 1.8.0)
- Matplotlib (version >= 2.2.3)
- Numpy (version >= 1.19.1)
- XGBoost (version >= 1.5.2)

This code has been tested on Windows 10, but the libraries used are avaiable and supported on MacOS and Linux.

## Files
- ```experiment_nb.py```: Experiment using naive Bayes classifier (the probabilistic model)
- ```experiment_nn.py``` Experiment using neural network classifier
- ```experiment_xgb.py``` Experiment using gradient boosting
- ```src/model.py``` The multisensor robot simulator
