# Changes made by: Laurentiu-Cristian Duca:
# - use iris data set instead of mnist data.
# Original source: MatthieuCourbariaux, BSD3 license.

from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net

import binary_ops

#from pylearn2.datasets.mnist import MNIST
import iris_data
#from pylearn2.utils import serial

from collections import OrderedDict

optimizer=None
#exception_verbosity=high

if __name__ == "__main__":
    
    # BN parameters
    #batch_size = 100
    batch_size = 10
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    #num_units = 4096
    n_inputs_per_sample = 4
    n_neurons_per_hiddenlayer = 32
    print("n_neurons_per_hiddenlayer = "+str(n_neurons_per_hiddenlayer))
    n_hidden_layers = 2
    print("n_hidden_layers = "+str(n_hidden_layers))
    n_output_neurons = 3

    # Training parameters
    #num_epochs = 1000
    num_epochs = 2000
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    LR_start = .003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "mnist_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    #print('Loading MNIST dataset...')
    #train_set = MNIST(which_set= 'train', start=0, stop = 50, center = False)
    #valid_set = MNIST(which_set= 'train', start=50, stop = 60, center = False)
    #test_set = MNIST(which_set= 'test', start = 0, stop = 10, center = False)
    #test_set = MNIST(which_set= 'train', start = 60, stop = 70, center = False)
    print('Loading iris dataset...')
    (X_train, y_train), (X_test, y_test) = iris_data.load_data()

    max_train = np.amax(X_train)
    max_test = np.amax(X_test)
    if(max_train > max_test):
        maximum = max_train
    else:
        maximum = max_test
    print("maximum=", maximum)
    X_train /= maximum
    X_test /= maximum
    X_train = X_train * 2 - 1
    y_train = y_train * 2 - 1
    X_test = X_test * 2 - 1
    y_test = y_test * 2 - 1
    print('X_train type=', type(X_train), 'X_train[:] type=',type(X_train[:]))
    #print(X_train[:])
    #print(y_train[:])
    #print(X_test[:])
    #print(y_test[:])
    #print('train_set=', type(train_set), 'valid_set=', type(valid_set), 'test_set=', type(test_set))
    #print('train_set.X=', type(train_set.X), 'valid_set.X=', type(valid_set.X), 'test_set.X=', type(test_set.X))
    #print('train_set.y=', type(train_set.y), 'valid_set.y=', type(valid_set.y), 'test_set.y=', type(test_set.y))

    # bc01 format    
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    #train_set.X = 2* train_set.X.reshape(-1, 1, 28, 28) - 1.
    #valid_set.X = 2* valid_set.X.reshape(-1, 1, 28, 28) - 1.
    #test_set.X = 2* test_set.X.reshape(-1, 1, 28, 28) - 1.
    
    # flatten targets
    #train_set.y = np.hstack(train_set.y)
    #valid_set.y = np.hstack(valid_set.y)
    #test_set.y = np.hstack(test_set.y)
    #print('train_set.y=', type(train_set.y), 'valid_set.y=', type(valid_set.y), 'test_set.y=', type(test_set.y))
    
    # Onehot the targets
    #train_set.y = np.float32(np.eye(10)[train_set.y])    
    #valid_set.y = np.float32(np.eye(10)[valid_set.y])
    #test_set.y = np.float32(np.eye(10)[test_set.y])
    
    # for hinge loss
    #train_set.y = 2* train_set.y - 1.
    #valid_set.y = 2* valid_set.y - 1.
    #test_set.y = 2* test_set.y - 1.

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    #input = T.tensor4('inputs')
    input = T.matrix('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
            #shape=(None, 1, 28, 28),
            shape=(None, n_inputs_per_sample),
            input_var=input)
            
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
    
    for k in range(n_hidden_layers):

        mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=n_neurons_per_hiddenlayer)                  
        
        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
    
    mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                #num_units=10)
                num_units=n_output_neurons)
    
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    binary_net.train(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train, y_train,
            X_train, y_train,
            X_test, y_test,
            #train_set.X,train_set.y,
            #valid_set.X,valid_set.y,
            #test_set.X,test_set.y,
            save_path,
            shuffle_parts)


    # Load parameters
    with np.load('mnist_parameters.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #for i in range(len(f.files)):
    #    print('arr_%d ' % i)
    lasagne.layers.set_all_param_values(mlp, param_values)

    f= open("nn-binary.bin","wb")
    f.write(bytearray([n_inputs_per_sample]))
    f.write(bytearray([n_hidden_layers]))
    f.write(bytearray([n_neurons_per_hiddenlayer]))
    f.write(bytearray([n_output_neurons]))

    # Binarize the weights and show parameters values
    params = lasagne.layers.get_all_params(mlp)
    for param in params:
        # print param.name
        if param.name == "W":
            param.set_value(binary_ops.SignNumpy(param.get_value()))
        print('param.name=', param.name)
        print(param.get_value())
        f.write(np.float32(param.get_value()))
    f.close()
