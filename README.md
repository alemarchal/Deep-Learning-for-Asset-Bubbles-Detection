# Deep-Learning-for-Asset-Bubbles-Detection

## Introduction

This repo implements the methodology of the paper **[Deep Learning for Asset Bubbles Detection](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3531154)**.

The main file *Classification_regimes_vol.m* can be decomposed into 4 important parts:


(1) Training an LSTM network using simulated data that mimic stylised facts of equity price time-series.
The network is trained to solve a classification problem with two classes: {bubble, no bubble} for each individual return.


(2) Evaluate the performance of the network on out-of-sample simulated data.
This part is relatively insensitive to the network architecture.
This evaluation is quite slow, not because of the network but because of the benchmark method (Parametric Estimator) that performs an optimization for each new data point.

(3) Deploy the network on real market data.


(4) Implement a trading strategy based on the bubble classification of the previous point.
This part is sensitive to the chosen network architecture. However, it might be due to the fact that we imprecisely estimate the fundamental value of the long leg instead of a sensitive classification.

If you use another frequency than 2-min, you will either have to rescale your returns or train a new network.


## Basic example

Find a toy example showing the performance of our methodology over a parametric estimator (PE) in the file *Example_drawback_PE.m*.

## Hyperparameters

Multiple architectures are provided.

The networks are in general bidirectional, unless explicitely specified in the file name.
The files are called *net_XXX_YYY.mat* where *XXX* refers to the number of (bi)-LSTM layers and *YYY* the number of units.

To examine the hyperparameters in details, please load the network of your choice and execute *analyzeNetwork(net)* in the Matlab console.
This will give all you information on the number of layers, units and activation functions that were used.

We tried a few architectures at random (for instance add an LSTM layer or increase the number of units) but the classification performance (network accuracy) stays similar.

The trading strategy however appears more sensitive to the choice of hyperparameters.
