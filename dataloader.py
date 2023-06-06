# -*- coding: utf-8 -*-
"""Data Loader"""

# from SampleFileTools1 import SampleFile
import pandas as pd
import h5py
import sys
from config import CFG
import numpy as np

class DataLoader:
    """Data Loader class"""
    
    def __init__(self, det, data):
        
        self.det = det
        self.data = data
       
    def load_data(self, data_config):
        """Loads dataset from path"""
        
        # Check training or testing data
        if(self.data == 'train'):
            df = h5py.File(data_config.path_train, 'r')
        elif(self.data == 'test'):
            df = h5py.File(data_config.path_test, 'r')
        
        # Obtain data for a given detector
        a = CFG.get('train')
        # num_training_samples = a.get('num_training_samples')
        num_injection_training_samples = a.get('num_injection_training_samples')
        num_noise_training_samples = a.get('num_noise_training_samples')
        n_samples_per_signal = a.get('n_samples_per_signal')

        num_test_samples = a.get('num_test_samples')

        if(self.data == 'train'):
            if(self.det == 'Hanford'):
                strain = df['injection_samples']['h1_strain'][0:num_injection_training_samples]#[0:10]
                signal = df['injection_parameters']['h1_signal'][0:num_injection_training_samples]#[0:10]
                strain = np.vstack([strain, df['noise_samples']['h1_strain'][0:num_noise_training_samples]])
                signal = np.vstack([signal, np.full((num_noise_training_samples, n_samples_per_signal), 1e-6)])
                # strain.vstack(df['noise_samples']['h1_strain'][0:num_noise_training_samples])
                # signal.vstack(np.full(num_noise_training_samples, n_samples_per_signal))
                
            elif(self.det == 'Livingston'):
                strain = df['injection_samples']['l1_strain'][0:num_injection_training_samples]#[0:10]
                signal = df['injection_parameters']['l1_signal'][0:num_injection_training_samples]#[0:10]
                strain = np.vstack([strain, df['noise_samples']['l1_strain'][0:num_noise_training_samples]])
                signal = np.vstack([signal, np.full((num_noise_training_samples, n_samples_per_signal), 1e-6)])
                # print(strain.shape)
                # print(signal)
                # strain.vstack(df['noise_samples']['l1_strain'][0:num_noise_training_samples])
                # signal.vstack(np.full(num_noise_training_samples, n_samples_per_signal))
                
            elif(self.det == 'Virgo'):
                strain = df['injection_samples']['v1_strain'][0:num_injection_training_samples]#[0:10]
                signal = df['injection_parameters']['v1_signal'][0:num_injection_training_samples]#[0:10]
                strain = np.vstack([strain, df['noise_samples']['v1_strain'][0:num_noise_training_samples]])
                signal = np.vstack([signal, np.full((num_noise_training_samples, n_samples_per_signal), 1e-6)])
                # strain.vstack(df['noise_samples']['v1_strain'][0:num_noise_training_samples])
                # signal.vstack(np.full(num_noise_training_samples, n_samples_per_signal))
                
            else:
                sys.exit('Detector not available. Quitting.')
        else:
            if(self.det == 'Hanford'):
                strain = df['injection_samples']['h1_strain'][0:num_test_samples]#[0:10]
                signal = df['injection_parameters']['h1_signal'][0:num_test_samples]#[0:10]
                
            elif(self.det == 'Livingston'):
                strain = df['injection_samples']['l1_strain'][0:num_test_samples]#[0:10]
                signal = df['injection_parameters']['l1_signal'][0:num_test_samples]#[0:10]
                
            elif(self.det == 'Virgo'):
                strain = df['injection_samples']['v1_strain'][0:num_test_samples]#[0:10]
                signal = df['injection_parameters']['v1_signal'][0:num_test_samples]#[0:10]
                
            else:
                sys.exit('Detector not available. Quitting.')
        
        df.close()
        
        return strain, signal
    
    