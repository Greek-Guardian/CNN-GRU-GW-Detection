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
from config import CFG

cfg_train = CFG.get('train')
n_samples_per_signal = cfg_train.get('n_samples_per_signal')
num_test_samples = cfg_train.get('num_test_samples')
num_noise_training_samples = cfg_train.get('num_noise_training_samples')

test_file_path = '/root/lzd2/Generated-Samples/v3/5000InjectionSamples_5000NoiseSamples_SNR10-20.hdf'
model_path = '/root/lzd2/RealEvents/exp5-100gru3-2000epoch_SNR10-20/trained_model.h5'
output_path = '/root/lzd2/RealEvents/exp5-100gru3-2000epoch_SNR5-9/apply_to_generated_samples/train_dataset'
save_picture_path = output_path + '/denoised_exp5-100gru3-2000epoch_v3_SNR10-20.png'
results_snr_20_path = output_path + '/DenoisedResults_exp5-100gru3-2000epoch_v3_SNR10-20.hdf'
load_injection_or_noise = 1

strains_d = dict()
signals_d = dict()
f1 = h5py.File(test_file_path, 'r')
# f1 = h5py.File('/root/lzd2/GravitationalWave/BBH_sample_files/default_snr.hdf', 'r')
list(f1.keys())
strains = []
signals = []

# load injection 1, noise 2
if load_injection_or_noise == 1:
    strains_d['l1'] = f1['injection_samples']['{0}_strain'.format('l1')][()]
    signals_d['l1'] = f1['injection_parameters']['{0}_signal'.format('l1')][()]
else:
    # strain = np.vstack([strain, df['noise_samples']['l1_strain'][0:num_noise_training_samples]])
    # signal = np.vstack([signal, np.full((num_noise_training_samples, n_samples_per_signal), 1e-6)])
    strains_d['l1'] = f1['noise_samples']['l1_strain'][()]
    signals_d['l1'] = np.full((num_noise_training_samples, n_samples_per_signal), 1e-6)

# for det in ['h1','l1','v1']:
#     strains_d[det] = f1['injection_samples']['{0}_strain'.format(det)][()]
#     signals_d[det] = f1['injection_parameters']['{0}_signal'.format(det)][()]
strains.append(strains_d)
signals.append(signals_d)

strains = {
        # 'h1':np.concatenate([x['h1'] for x in strains]),
        'l1':np.concatenate([x['l1'] for x in strains]),
        # 'v1':np.concatenate([x['v1'] for x in strains])
    }

signals = {
        # 'h1':np.concatenate([x['h1'] for x in signals]),
        'l1':np.concatenate([x['l1'] for x in signals]),
        # 'v1':np.concatenate([x['v1'] for x in signals])
    }


def normalize_test(a):
    new_array = []
    for i in range(num_test_samples):
        dataset = a[i]
#        dataset = dataset[1536:2048]
        maximum = np.max(dataset)
        minimum = np.abs(np.min(dataset))
        for j in range(n_samples_per_signal):
            if(dataset[j] > 0):
                dataset[j] = dataset[j]/maximum
            else:
                dataset[j] = dataset[j]/minimum
#        dataset = dataset+gauss
        new_array.append(dataset)
    return new_array

l1_test_new = normalize_test(strains['l1'])

print('l1_test_new shape: ', len(l1_test_new))
print('l1_test_new[0] shape: ', len(l1_test_new[0]))

#h1_pure_new = normalize_new(h1_pure_new)
l1_test_pure_new = normalize_test(signals['l1'])

from numpy import array
# split a univariate sequence into samples
def split_sequence(sequence_noisy,sequence_pure,n_steps):
    X, y = list(), list()
    for i in range(len(sequence_noisy)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence_noisy)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence_noisy[i:end_ix], sequence_pure[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# choose a number of time steps
n_steps = 4
X_test_noisy = []
X_test_pure = []

for i in range(num_test_samples):
    X_noisy = l1_test_new[i]
    X_pure = l1_test_pure_new[i]
    X_noisy = np.pad(X_noisy, (4, 4), 'constant', constant_values=(0, 0))
    # print('X_noisy shape: ', X_noisy.shape)
    X_pure = np.pad(X_pure, (4, 4), 'constant', constant_values=(0, 0))
    # split into samples
    X, y = split_sequence(X_noisy, X_pure, n_steps)
    X_test_noisy.append(X)
    X_test_pure.append(y)
    
X_test_noisy = np.asarray(X_test_noisy)
X_test_pure = np.asarray(X_test_pure)

print('x_test_noisy shape: ', X_test_noisy.shape)
print('x_test_pure shape: ', X_test_pure.shape)

num_time_periods, num_detectors = 4, 1

input_shape = (num_time_periods*num_detectors)

X_test_noisy = X_test_noisy.reshape(X_test_noisy.shape[0], 516, input_shape)
X_test_pure = X_test_pure.reshape(X_test_pure.shape[0], 516, 1)

print('x_test_noisy shape:', X_test_noisy.shape)
print('input_shape:', input_shape)

print('x_test_pure shape:', X_test_pure.shape)
print('input_shape:', input_shape)

X_test_noisy = X_test_noisy.astype("float32")
X_test_pure = X_test_pure.astype("float32")

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

score = model.evaluate(X_test_noisy, X_test_pure, verbose=1, batch_size=2000)


f2 = h5py.File(results_snr_20_path, 'w')
f2.create_dataset('denoised_signals', data=decoded_signals)
f2.create_dataset('raw_signals', data=X_test_noisy)
f2.create_dataset('pure_signals', data=X_test_pure)
l1_snr = f1['injection_parameters']['l1_snr'][()]
h1_snr = f1['injection_parameters']['h1_snr'][()]
scale_factor = f1['injection_parameters']['scale_factor'][()]
snr = np.zeros(num_test_samples)
for i in range(num_test_samples):
    snr[i] = scale_factor[i] * np.sqrt(l1_snr[i]*l1_snr[i] + h1_snr[i]*h1_snr[i])
f1.close()
f2.create_dataset('raw_signal_snr', data=snr[0:num_test_samples])

print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])

if load_injection_or_noise == 2:
    plt.plot(X_test_noisy[0])
    plt.plot(X_test_pure[0])
    plt.plot(decoded_signals[0])
    plt.savefig(save_picture_path)
    plt.show()
else:
    plt.plot(X_test_pure[0])
    plt.plot(decoded_signals[0])
    plt.savefig(save_picture_path)
    plt.show()
