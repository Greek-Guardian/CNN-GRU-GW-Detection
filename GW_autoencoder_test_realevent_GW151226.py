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

test_file_path = '/root/lzd2/RealEvents/O1-GW151012/L-L1_LOSC_4_V2-1128678884-32-seconds16-44_pretreatmented.hdf5'
model_path = '/root/lzd2/RealEvents/exp5-100gru3-2000epoch_SNR10-20/trained_model.h5'
output_path = '/root/lzd2/RealEvents/exp5-100gru3-2000epoch_SNR5-9/apply_to_real_data/SNR10_20'
save_picture_path = output_path + '/denoised_exp5-100gru3-2000epoch-O1-GW151012.png'
results_snr_20_path = output_path + '/DenoisedResults_exp5-100gru3-2000epoch-O1-GW151012.hdf'


strains_d = dict()
signals_d = dict()
f1 = h5py.File(test_file_path, 'r')
list(f1.keys())
strains = []

strains_d = f1['strain']['Strain'][()]
# for det in ['h1','l1','v1']:
#     strains_d[det] = f1['injection_samples']['{0}_strain'.format(det)][()]
#     signals_d[det] = f1['injection_parameters']['{0}_signal'.format(det)][()]
strains.append(strains_d)

strains = {
        # 'h1':np.concatenate([x for x in strains]),
        'l1':np.concatenate([x for x in strains]),
        # 'v1':np.concatenate([x['v1'] for x in strains])
    }

f1.close()

# print('strains shape:', strains.shape)

def normalize_test(a):
    a = a[33259:33770] # O1-GW151012
    # a = a[33689:34201] # O1-GW151226
    # a = a[33586:34097] # O1-GW170104
    # a = a[33361:33873] # O1-GW170608

    def down_samplize(signal):
        '''4096Hz to 2048Hz'''
        tmp = []
        for i in range(len(signal)):
            if i % 2 == 0:
                tmp.append(signal[i])
        return tmp
    # dataset = down_samplize(a)
    dataset = a
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

l1_test_new = normalize_test(strains['l1'])

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
model = tf.keras.models.load_model(model_path, custom_objects={'fractal_tanimoto_loss': fractal_tanimoto_loss})
# summarize model.
model.summary()

decoded_signals = model.predict(X_test_noisy, batch_size=2000)

f1 = h5py.File(results_snr_20_path, 'w')
f1.create_dataset('denoised_signals', data=decoded_signals)
f1.create_dataset('raw_signals', data=X_test_noisy)


# plt.plot(X_test_noisy[0])
plt.plot(decoded_signals[0])
plt.savefig(save_picture_path)
plt.show()