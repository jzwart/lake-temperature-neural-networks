# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:06:57 2018

@author: aappling
"""

from __future__ import print_function, division
import random
import sys
import datetime as dt
import numpy as np
sys.path.append('2_model/src')
import tf_graph
import tf_train

def apply_pgnn(
        phase = 'pretrain',
        learning_rate = 0.005,
        state_size = 8, # Step 1: try number of input drivers, and half and twice that number
        ec_threshold = 24, # TODO: calculate for each lake: take GLM temperatures, calculate error between lake energy and the fluxes, take the maximum as the threshold. Can tune from there, but it usually doesn’t change much from the maximum
        dd_lambda = 0, # TODO: implement depth-density constraint in model
        ec_lambda = 0.025, # original mille lacs values were 0.0005, 0.00025
        l1_lambda = 0.05,
        data_file = '1_format/tmp/pgdl_inputs/nhd_1099476.npz',
        sequence_offset = 100,
        max_batch_obs = 50000,
        n_epochs = 2, # 200 is usually more than enough
        min_epochs_test = 0,
        min_epochs_save = 2, # later is recommended (runs quickest if ==n_epochs)
        restore_path = '', #'2_model/out/EC_mille/pretrain',
        save_path = '2_model/tmp/nhd_1099476/pretrain'
        # TODO: should have L1 and L2 norm weights in this list, implemented in tf_graph
    ):
    """Train (or pretrain) a PGRNN, optionally save the weights+biases and/or predictions, and return the predictions

    Args:
        learning_rate: NN learning rate
        state_size: Number of units in each cell's hidden state
        ec_threshold: Energy imbalance beyond which NN will be penalized
        dd_lambda: PRESENTLY IGNORED. Depth-density penalty lambda, a hyperparameter that needs manual tuning. Multiplier to depth-density loss when adding to other losses.
        ec_lambda: Energy-conservation penalty lambda, another hyperparameter that needs manual tuning. Multiplier to energy-conservation loss when adding to other losses. Could set ec_lambda=0 if we wanted RNN only, no EC component
        l1_lambda: L1-regularization penalty lambda, another hyperparameter that needs manual tuning
        data_file: Filepath for the one file per lake that contains all the data.
        sequence_offset: Number of observations by which each data sequence in inputs['predict.features'] is offset from the previous sequence. Used to reconstruct a complete prediction sequence without duplicates.
        max_batch_obs: Upper limit on number of individual temperature predictions (date-depth-split combos) per batch. True batch size will be computed as the largest number of completes sequences for complete depth profiles that fit within this max_batch_size.
        n_epochs: Total number of epochs to run during training. Needs to be larger than the n epochs needed for the model to converge
        min_epochs_test: Minimum number of epochs to run through before computing test losses
        min_epochs_save: Minimum number of epochs to run through before considering saving a checkpoint (must be >= min_epochs_test)
        restore_path: Path to restore a model from, or ''
        save_path: Path (directory) to save a model to. Will always be saved as ('checkpoint_%s' %>% epoch)
    """

    # Track runtime
    start_time = dt.datetime.now()

    random.seed(9001)

    # %% Load data
    print('Loading data...')
    inputs = np.load(data_file)
    # Compute desirable number of sequences per batch based on max_batch_obs and
    # the number of observations in each sequence (depths*dates)
    obs_per_seq = inputs['train.labels'][:,:,0].size # get the num obs in the first sequence (same for every sequence)
    seq_per_batch = np.int(np.floor(max_batch_obs / obs_per_seq))

    # %% Build graph
    print('Building graph...')
    train_op, total_loss, rmse_loss, ec_loss, l1_loss, pred, x, y, m, unsup_inputs, unsup_phys_data = tf_graph.build_tf_graph(
            inputs['train.labels'].shape[1], inputs['train.features'].shape[2], state_size,
            inputs['unsup.physics'].shape[2], inputs['colnames.physics'], inputs['geometry'],
            ec_threshold, dd_lambda, ec_lambda, l1_lambda, seq_per_batch, learning_rate)

    # %% Train model
    print('Training model...')
    x_unsup = inputs['unsup.features']
    p_unsup = inputs['unsup.physics']
    x_pred = inputs['predict.features']
    if phase == 'tune':
        x_train = inputs['tune_train.features']
        y_train = inputs['tune_train.labels']
        m_train = inputs['tune_train.mask']
        x_test = inputs['tune_test.features']
        y_test = inputs['tune_test.labels']
        m_test = inputs['tune_test.mask']
    elif phase == 'pretrain':
        x_train = inputs['pretrain.features']
        y_train = inputs['pretrain.labels']
        m_train = inputs['pretrain.mask']
        x_test = inputs['pretrain.features'] # test on training data
        y_test = inputs['pretrain.labels']
        m_test = inputs['pretrain.mask']
    elif phase == 'train':
        x_train = inputs['train.features']
        y_train = inputs['train.labels']
        m_train = inputs['train.mask']
        x_test = inputs['test.features']
        y_test = inputs['test.labels']
        m_test = inputs['test.mask']
    else:
        print("Error: unrecognized phase '%s'"% phase)

    train_stats, test_loss_rmse, preds = tf_train.train_tf_graph(
            train_op, total_loss, rmse_loss, ec_loss, l1_loss, pred, x, y, m, unsup_inputs, unsup_phys_data,
            x_train, y_train, m_train, x_unsup, p_unsup, x_test, y_test, m_test, x_pred,
            sequence_offset=sequence_offset, seq_per_batch=seq_per_batch, n_epochs=n_epochs, min_epochs_test=min_epochs_test, min_epochs_save=min_epochs_save,
            restore_path=restore_path, save_path=save_path)

    # Track runtime, part 2
    end_time = dt.datetime.now()
    run_time = end_time - start_time

    # Save the model diagnostics
    stat_save_file = '%s/stats.npz' % save_path
    np.savez_compressed(stat_save_file, train_stats=train_stats, test_loss_rmse=test_loss_rmse,
                        start_time=start_time, end_time=end_time, run_time=run_time)
    print("  Diagnostics saved to %s" % stat_save_file)

    return(train_stats, test_loss_rmse, preds)
    # %% Inspect predictions

    # prd has dimensions [depths*batches, n timesteps per batch, 1]
    # xiaowei has a separate script that combines batches into a single time series. he uses the first occurence of each prediction as the final value because it's better (has more preceding info to build on)

    # visualize prd_EC, see Create_sparse_data.py
    #prd = np.load('prd_EC_mille.npy')
