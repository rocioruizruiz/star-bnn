# Changes made by: Laurentiu-Cristian Duca
# Original source: MatthieuCourbariaux, BSD3 license.

import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

def SignNumpy(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)

def SignTheano(x):
    return T.cast(2.*T.ge(x,0)-1., theano.config.floatX)


