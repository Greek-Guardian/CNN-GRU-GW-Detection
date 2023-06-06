from __future__ import print_function
from matplotlib import pyplot as plt
plt.switch_backend('agg')
#matplotlib inline
import numpy as np
import pandas as pd

from IPython.display import display, HTML
import tensorflow as tf
import numpy as np
import pandas as pd
from SampleFileTools1 import SampleFile
import random as ran
import h5py

strains_d = dict()
signals_d = dict()
f1 = h5py.File('/root/lzd2/RealEvents/GW151226/L-L1_LOSC_4_V2-1135136334-32_pretreatmented.hdf5', 'r')
list(f1.keys())
strains = []

strains_d['h1'] = f1['strain']['Strain'][()]
# for det in ['h1','l1','v1']:
#     strains_d[det] = f1['injection_samples']['{0}_strain'.format(det)][()]
#     signals_d[det] = f1['injection_parameters']['{0}_signal'.format(det)][()]
strains.append(strains_d)

strains = {
        'h1':np.concatenate([x['h1'] for x in strains]),
        # 'l1':np.concatenate([x['l1'] for x in strains]),
        # 'v1':np.concatenate([x['v1'] for x in strains])
    }

f1.close()

# print('strains shape:', strains['h1'].shape)

def normalize_test(a):
    a = a[67379:68403] # bad performance
    # a = a[66979:68003] # no performance
    # a = a[67179:68203] # bad bad performance
    # a = a[67279:68303] # bad performance

    def down_samplize(signal):
        '''4096Hz to 2048Hz'''
        tmp = []
        for i in range(len(signal)):
            if i % 2 == 0:
                tmp.append(signal[i])
        return tmp
    dataset = down_samplize(a)
    from scipy import signal
    dataset=signal.resample(a,512)
    # print('dataset shape:', len(dataset))  

    new_array = []
#        dataset = dataset[1536:2048]
    maximum = np.max(dataset)
    minimum = np.abs(np.min(dataset))
    for j in range(512):
        if(dataset[j] > 0):
            dataset[j] = dataset[j]/maximum
        else:
            dataset[j] = dataset[j]/minimum
#        dataset = dataset+gauss
    new_array.append(dataset)
    return new_array

l1_test_new = normalize_test(strains['h1'])

print('l1_test_new shape:', len(l1_test_new))
print('l1_test_new[0] shape:', len(l1_test_new[0]))

from numpy import array
# split a univariate sequence into samples
def split_sequence(sequence_noisy, n_steps):
    X = list()
    for i in range(len(sequence_noisy)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence_noisy)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence_noisy[i:end_ix]
        X.append(seq_x)
    return array(X)


# choose a number of time steps
n_steps = 4
X_test_noisy = []
X_test_pure = []

X_noisy = l1_test_new[0]
X_noisy = np.pad(X_noisy, (4, 4), 'constant', constant_values=(0, 0))
# split into samples
X = split_sequence(X_noisy, n_steps)
X_test_noisy.append(X)
    
X_test_noisy = np.asarray(X_test_noisy)

print('x_test_noisy shape: ', X_test_noisy.shape)

num_time_periods, num_detectors = 4, 1

input_shape = (num_time_periods*num_detectors)

X_test_noisy = X_test_noisy.reshape(X_test_noisy.shape[0], 516, input_shape)

print('x_test_noisy shape:', X_test_noisy.shape)
print('input_shape:', input_shape)

X_test_noisy = X_test_noisy.astype("float32")

#from keras import backend as K
def fractal_tanimoto_loss(y_true, y_pred, depth=0, smooth=1e-6):
    x = y_true
    y = y_pred
#    x_norm = K.sum(x)
#    y_norm = K.sum(y)
#    x = x/x_norm
#    y = y/y_norm
    depth = depth+1
    scale = 1./len(range(depth))
    
    def inner_prod(y, x):
        prod = y*x
        prod = tf.math.reduce_sum(prod, axis=1)
        
        return prod
    
    def tnmt_base(x, y):

        tpl  = inner_prod(y,x)
        tpp  = inner_prod(y,y)
        tll  = inner_prod(x,x)


        num = tpl + smooth
        denum = 0.0
        result = 0.0
        for d in range(depth):
            a = 2.**d
            b = -(2.*a-1.)

            denum = denum + tf.math.reciprocal( a*(tpp+tll) + b *tpl + smooth)
#            denum = ( a*(tpp+tll) + b *tpl + smooth)
#            result = tf.math.reduce_mean((result + (num/denum)), axis=0)

        result =  num * denum * scale
#        result = result
        return  result*scale
    
    
    l1 = tnmt_base(x,y)
#        l2 = self.tnmt_base(1.-preds, 1.-labels)

#        result = 0.5*(l1+l2)
    result = l1
    
    return  1. - result

    #from keras.models import load_model
 
# load model
model = tf.keras.models.load_model('/root/lzd2/outputs/exp2-gru3-1200epoch-pretreatdata/trained_model.h5', custom_objects={'fractal_tanimoto_loss': fractal_tanimoto_loss})
# summarize model.
model.summary()

decoded_signals = model.predict(X_test_noisy, batch_size=2000)

plt.plot(decoded_signals[0])
plt.savefig('/root/lzd2/RealEvents/exp2-gru3-1200epoch-pretreatdata/denoised.png')
plt.show()