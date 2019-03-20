# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:06:57 2018

@author: aappling
"""

from __future__ import print_function, division
import random
import prep_data
import tf_graph
import tf_train

def apply_pgnn(
        input_size = 9,
        phy_size = 10,
        n_steps = 50,
        batch_size = 500,
        learning_rate = 0.005,
        state_size = 8,
        ec_threshold = 24,
        plam = 0.15,
        elam = 0.00025,

        x_full_file = 'processed_features.npy',
        x_raw_full_file = 'features.npy',
        diag_full_file = 'diag.npy',
        depth_areas_file = 'depth_areas.npy',
        label_file = 'Obs_temp_mille.npy',
        mask_file = 'Obs_mask_mille.npy',
        train_range = range(10000, 12500), # for Mille Lacs we're using the later part of the data for training, previous years for testing
        test_range = range(10000),
        data_path = 'data',

        tot_epochs = 300,
        min_epochs_test = 0,
        min_epochs_save = 250,
        restore_path = './out/EC_mille/pretrain',
        save_path = './out/EC_mille',
        max_to_keep = 3,
        save_preds = True
    ):
    """Train (or pretrain) a PGRNN, optionally save the weights+biases and/or predictions, and return the predictions

    Args:
        input_size: # Number of features in input. Must match between pretrain and train
        phy_size: Number of features in physics data (unnormalized input). Must match between pretrain and train
        n_steps: Number of timesteps within a window. should be even number and divide evenly into batch_size. Must match between pretrain and train
        batch_size: Number of dates (timesteps) within one batch. We're not breaking epochs into batches, but we do prep data in the shape of batches stacked vertically, such that we could use training batches later if desired. Should divide evenly into training and test range lengths.
        learning_rate: NN learning rate
        state_size: Number of units in each cell's hidden state
        ec_threshold: Energy imbalance beyond which NN will be penalized
        plam: PRESENTLY IGNORED. physics (depth-density) constraint lambda, multiplier to physics loss when adding to other losses
        elam: energy constraint lambda, multiplier to energy balance loss when adding to other losses

        x_full_file: Filename of normalized inputs (climate, possibly GLM)
        x_raw_full_file: Filename of raw (unnormalized) inputs
        diag_full_file: Filename of supplemental ("diagnostic") inputs for unsupervised training
        depth_areas_file: Filename of np array of depth areas. The length of this array is used to determine number of depths
        label_file: Filename of labels (temperature observations for training; GLM predictions for pretraining)
        mask_file: Filename of label masks (0 where temp obs/GLM pred unavailable, 1 where available)
        train_range: Date/timestep range or indices to select from the input files for training. Use a smaller range (2500) unless you have large-memory (~32Gb) GPU in which case 7500 is OK.
        test_range: Date/timestep range or indices to select from the input files for testing.
        data_path: Path to the data files. './data' is recommended.

        tot_epochs: Total number of epochs to run during training. Needs to be larger than the n epochs needed for the model to converge
        min_epochs_test: Minimum number of epochs to run through before computing test losses
        min_epochs_save: Minimum number of epochs to run through before considering saving a checkpoint (must be >= min_epochs_test)
        restore_path: Path to restore a model from, or ''
        save_path: Path (directory) to save a model to. Will always be saved as ('checkpoint_%s' %>% epoch)
        max_to_keep: Max number of checkpoints to keep in the save_path
        save_preds: Logical - should test predictions be saved (and ovewritten) each time we save a checkpoint?
    """

    print('printing status')
    print('restore_path: %s' % restore_path)
    print('save_path: %s' % save_path)

    random.seed(9001)

    # %% Compute constants

    # Compute info shared across multiple cells below
    N_sec = (int(batch_size/n_steps)-1)*2+1 # number of 50% overlapping sections (windows) per batch

    # %% Load and prepare data
    x_train_1, y_train_1, m_train_1, x_f, p_f, x_test, y_test, m_test, depth_areas = prep_data.prep_data(
            N_sec, n_steps, batch_size, input_size, phy_size,
            x_full_file = x_full_file,
            x_raw_full_file = x_raw_full_file,
            diag_full_file = diag_full_file,
            depth_areas_file = depth_areas_file,
            label_file = label_file,
            mask_file = mask_file,
            train_range = train_range,
            test_range = test_range,
            data_path = data_path)

    # %% Build graph
    train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data = tf_graph.build_tf_graph(
            n_steps, input_size, state_size, phy_size, depth_areas,
            ec_threshold, plam, elam, N_sec, learning_rate)

    # %% Train model
    prd = tf_train.train_tf_graph(
            train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data,
            x_train_1, y_train_1, m_train_1, x_f, p_f, x_test, y_test, m_test,
            tot_epochs=tot_epochs, min_epochs_test=min_epochs_test, min_epochs_save=min_epochs_save,
            restore_path=restore_path, save_path=save_path, max_to_keep=max_to_keep, save_preds=save_preds)

    return(prd)
    # %% Inspect predictions

    # prd has dimensions [depths*batches, n timesteps per batch, 1]
    # xiaowei has a separate script that combines batches into a single time series. he uses the first occurence of each prediction as the final value because it's better (has more preceding info to build on)

    # visualize prd_EC, see Create_sparse_data.py
    #prd = np.load('prd_EC_mille.npy')
