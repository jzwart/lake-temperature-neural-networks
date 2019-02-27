#!/usr/bin/python
# -*- coding: utf-8 -*-

# Usage: python run_job.py <job>
# where <job> is job number from 0 to (number of elements in model_params.py - 1)

import sys
import numpy as np
import apply_pgnn

# read the job number from the command-line arguments
job = int(sys.argv[1])
print('job: %s' % job)

# read and format the parameters corresponding to that job number
model_params = np.load('model_params.npy').item()
rep_name = list(model_params.keys())[job]
params = model_params[rep_name]
print('rep_name: %s' % rep_name)
print('state_size: %s' % params['state_size'])
print('pretrain: %s' % params['pretrain'])

#import pandas as pd
#geometry = pd.read_csv('tf_tuning/data/nhd_1097324_geometry.csv')
#meteo = pd.read_csv('tf_tuning/data/nhd_1097324_meteo.csv')
#temperatures = pd.read_feather('tf_tuning/data/nhd_1097324_temperatures.feather', nthreads=1)
#test_train = pd.read_feather('tf_tuning/data/nhd_1097324_test_train.feather')

# use the parameters to pretrain or train a model
apply_pgnn.apply_pgnn(
    input_size = 9,
    phy_size = 10,
    n_steps = 50,
    batch_size = 500,
    learning_rate = 0.008 if params['pretrain'] else 0.005,
    state_size = params['state_size'], # Step 1: try number of input drivers, and half and twice that number
    ec_threshold = 24, # TODO: calculate for each lake: take GLM temperatures, calculate error between lake energy and the fluxes, take the maximum as the threshold. Can tune from there, but it usually doesn’t change much from the maximum
    plam = 1.0, # TODO: implement depth-density constraint in model
    elam = 0.001, # original mille lacs values were 0.0005, 0.00025
    # TODO: should have L1 and L2 norm weights in this list, implemented in tf_graph

    x_full_file = 'processed_features.npy',
    x_raw_full_file = 'features.npy',
    diag_full_file = 'diag.npy',
    depth_areas_file = 'depth_areas.npy',
    label_file = 'labels.npy' if params['pretrain'] else 'Obs_temp_mille.npy', # labels.npy is GLM outputs, Obs is real temp data
    mask_file = '' if params['pretrain'] else 'Obs_mask_mille.npy', # mask can be all 1's ('') for GLM, which has no missing values
    train_range = range(5000, 6500) if params['pretrain'] else range(10000, 12500), # for Mille Lacs we're using the later part of the data for training, previous years for testing
    test_range = range(0, 5000) if params['pretrain'] else range(5000, 10000), # not really sure why train and test are different for pretrain and train
    data_path = './mille/data',

    tot_epochs = 300,
    min_epochs_test = 150 if params['pretrain'] else 100,
    min_epochs_save = 290,
    restore_path = params['restore_path'],
    save_path = params['save_path'],
    max_to_keep = 5 if params['pretrain'] else 3,
    save_preds = False if params['pretrain'] else True
)
