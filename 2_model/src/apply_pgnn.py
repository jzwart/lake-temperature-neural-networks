# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:06:57 2018

@author: aappling
"""

from __future__ import print_function, division
import random
import sys
import numpy as np
sys.path.append('2_model/src')
import tf_graph
import tf_train

def apply_pgnn(
        phase = 'pretrain',
        learning_rate = 0.005,
        state_size = 14, # Step 1: try number of input drivers, and half and twice that number
        ec_threshold = 24, # TODO: calculate for each lake: take GLM temperatures, calculate error between lake energy and the fluxes, take the maximum as the threshold. Can tune from there, but it usually doesn’t change much from the maximum
        plam = 0.15, # TODO: implement depth-density constraint in model
        elam = 0.025, # original mille lacs values were 0.0005, 0.00025
        data_file = '1_format/tmp/pgdl_inputs/nhd_1099476.npz',
        max_batch_obs = 50000,
        n_epochs = 5, # 200 is usually more than enough
        min_epochs_test = 0,
        min_epochs_save = 5, # later is recommended
        restore_path = '', #'2_model/out/EC_mille/pretrain',
        save_path = '2_model/tmp/nhd_1099476',
        max_to_keep = 1,
        save_preds = True
        # TODO: should have L1 and L2 norm weights in this list, implemented in tf_graph
    ):
    """Train (or pretrain) a PGRNN, optionally save the weights+biases and/or predictions, and return the predictions

    Args:
        learning_rate: NN learning rate
        state_size: Number of units in each cell's hidden state
        ec_threshold: Energy imbalance beyond which NN will be penalized
        plam: PRESENTLY IGNORED. physics (depth-density) constraint lambda, multiplier to physics loss when adding to other losses
        elam: energy constraint lambda, multiplier to energy balance loss when adding to other losses
        data_file: Filepath for the one file per lake that contains all the data.
        max_batch_obs: Upper limit on number of individual temperature predictions (date-depth-split combos) per batch. True batch size will be computed as the largest number of completes sequences for complete depth profiles that fit within this max_batch_size.
        n_epochs: Total number of epochs to run during training. Needs to be larger than the n epochs needed for the model to converge
        min_epochs_test: Minimum number of epochs to run through before computing test losses
        min_epochs_save: Minimum number of epochs to run through before considering saving a checkpoint (must be >= min_epochs_test)
        restore_path: Path to restore a model from, or ''
        save_path: Path (directory) to save a model to. Will always be saved as ('checkpoint_%s' %>% epoch)
        max_to_keep: Max number of checkpoints to keep in the save_path
        save_preds: Logical - should test predictions be saved (and ovewritten) each time we save a checkpoint?
    """

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
    train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data = tf_graph.build_tf_graph(
            inputs['train.labels'].shape[1], inputs['train.features'].shape[2], state_size,
            inputs['unsup.physics'].shape[2], inputs['colnames.physics'], inputs['geometry'],
            ec_threshold, plam, elam, seq_per_batch, learning_rate)

    # %% Train model
    print('Training model...')
    if phase == 'tune':
        train_stats, test_loss_RMSE, preds = tf_train.train_tf_graph(
                train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data,
                inputs['tune_train.features'], inputs['tune_train.labels'], inputs['tune_train.mask'],
                inputs['unsup.features'], inputs['unsup.physics'],
                inputs['tune_test.features'], inputs['tune_test.labels'], inputs['tune_test.mask'], # test on training data
                inputs['predict.features'],
                seq_per_batch=seq_per_batch, n_epochs=n_epochs, min_epochs_test=min_epochs_test, min_epochs_save=min_epochs_save,
                restore_path=restore_path, save_path=save_path, max_to_keep=max_to_keep, save_preds=save_preds)
    elif phase == 'pretrain':
        train_stats, test_loss_RMSE, preds = tf_train.train_tf_graph(
                train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data,
                inputs['pretrain.features'], inputs['pretrain.labels'], inputs['pretrain.mask'],
                inputs['unsup.features'], inputs['unsup.physics'],
                inputs['pretrain.features'], inputs['pretrain.labels'], inputs['pretrain.mask'], # test on training data
                inputs['predict.features'],
                seq_per_batch=seq_per_batch, n_epochs=n_epochs, min_epochs_test=min_epochs_test, min_epochs_save=min_epochs_save,
                restore_path=restore_path, save_path=save_path, max_to_keep=max_to_keep, save_preds=save_preds)
    elif phase == 'train':
        train_stats, test_loss_RMSE, preds = tf_train.train_tf_graph(
                train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data,
                inputs['train.features'], inputs['train.labels'], inputs['train.mask'],
                inputs['unsup.features'], inputs['unsup.physics'],
                inputs['test.features'], inputs['test.labels'], inputs['test.mask'],
                inputs['predict.features'],
                seq_per_batch=seq_per_batch, n_epochs=n_epochs, min_epochs_test=min_epochs_test, min_epochs_save=min_epochs_save,
                restore_path=restore_path, save_path=save_path, max_to_keep=max_to_keep, save_preds=save_preds)
    else:
        print("Error: unrecognized phase '%s'"% phase)

    # %% Save model diagnostics
    stats_save_file = '%s/preds.npy' % save_path
    np.savez_compressed(stats_save_file, train_stats=train_stats, test_loss_RMSE=test_loss_RMSE)
    print("  Diagnostics saved to %s" % stats_save_file)

    return(preds)
    # %% Inspect predictions

    # prd has dimensions [depths*batches, n timesteps per batch, 1]
    # xiaowei has a separate script that combines batches into a single time series. he uses the first occurence of each prediction as the final value because it's better (has more preceding info to build on)

    # visualize prd_EC, see Create_sparse_data.py
    #prd = np.load('prd_EC_mille.npy')
