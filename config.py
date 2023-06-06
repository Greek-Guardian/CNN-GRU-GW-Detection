# -*- coding: utf-8 -*-
# Author: Chayan Chatterjee
# Last modified: 26th November 2021

"""Model config in json format"""

CFG = {
    "data": {
        "path_train": "/root/lzd2/Generated-Samples/v3/5000InjectionSamples_5000NoiseSamples_SNR10-20.hdf",
        # "path_train": "/root/lzd2/Generated-Samples/v1/sample2023-04-30-15-37-36.hdf",
        # "path_test": "/root/lzd2/RealEvents/GW151226/L-L1_LOSC_4_V2-1135136334-32.hdf5",
        # "path_train": "/root/lzd2/GravitationalWave/BBH_sample_files/default_snr.hdf",
        # "path_test": "/root/lzd2/Generated-Samples/v2/sample2023-04-30-16-01-13.hdf",
        "path_test": "/root/lzd2/Generated-Samples/v3/20InjectionSamples_20NoiseSamples_SNR19-20.hdf",
        
    },
    "train": {
        # Checked
        # num_training_samples = num_injection_training_samples + num_noise_training_samples
        "num_training_samples": 5000,
        "num_injection_training_samples": 5000,
        "num_noise_training_samples": 0,
        # Checked
        "num_test_samples": 20,
        # TODO
        # TODO
        # TODO
        "detector": 'Livingston', # 'Hanford'/'Livingston'/'Virgo'
        # Checked
        "n_samples_per_signal": 512, # 0.25 sec signals are used sampled at 2048 Hz
        # Checked
        "batch_size": 500,        # Checked
        "epoches": 2000,
        # TODO
        "depth": 0,
        # TODO
        "train_from_checkpoint": False,
        # TODO
        "checkpoint_path": '/root/lzd2/RealEvents/exp5-100gru3-2000epoch/Saved_checkpoint/tmp_0x35ca3849/ckpt-1.index', # if train_from_checkpoint == True
        # Checked
        "optimizer": {
            "type": "adam"
        },
    },
    "model": {
        "input": [516,4],
        "layers": {
            "CNN_layer_1": 32,
            "CNN_layer_2": 16,
            "GRU_layer_1": 100,
            "GRU_layer_2": 100,
            "GRU_layer_3": 100,
            "Output_layer": 1,
            "kernel_size": 1,
            "pool_size": 2,
            "learning_rate": 1e-3
        },
    }
}
