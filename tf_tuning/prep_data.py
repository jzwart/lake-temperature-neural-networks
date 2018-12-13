# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:17:36 2018

@author: aappling
"""

import numpy as np

def prep_data(
        N_sec, n_steps, batch_size, input_size, phy_size,
        x_full_file = 'processed_features.npy',
        x_raw_full_file = 'features.npy',
        diag_full_file = 'diag.npy',
        depth_areas_file = 'depth_areas.npy',
        label_file = 'Obs_temp_mille.npy',
        mask_file = 'Obs_mask_mille.npy',
        train_range = range(10000, 12500),
        test_range = range(10000),
        data_path='data'):
    """Read and format data
    
    Args:
        N_sec: Number of sections to arrange vertically within the input for a batch (after splitting into windows of width n_steps)
        n_steps: Number of timesteps in the time window represented by the LSTM (shorter than total number of timesteps in the dataset or even the batch)
        batch_size: Number of dates (timesteps) within one batch. We're not breaking epochs into batches, but we do prep data in the shape of batches stacked vertically, such that we could use training batches later if desired. Should divide evenly into training and test range lengths.
        input_size: Number of features in the input data (drivers) used for supervised learning
        phy_size: Number of features in the input data (drivers) used for UNsupervised learning. These data should be unnormalized.
        x_full_file: Filename of normalized inputs (climate, possibly GLM)
        x_raw_full_file: Filename of raw (unnormalized) inputs
        diag_full_file: Filename of supplemental inputs for unsupervised training
        depth_areas_file: Filename of np array of depth areas. The length of this array is used to determine number of depths
        label_file: Filename of labels (temperature observations for training; GLM predictions for pretraining)
        mask_file: Filename of label masks (0 where temp obs/GLM pred unavailable, 1 where available)
        train_range: Date/timestep range or indices to select from the input files for training. Use a smaller range (2500) unless you have large-memory (~32Gb) GPU in which case 7500 is OK.
        test_range: Date/timestep range or indices to select from the input files for testing. 
        data_path: Path to the data files. './data' is recommended.    
    """
    
    # Read in the predictor data (scaled and unscaled climate variables, mask, etc.)
    x_full = np.load('%s/%s' % (data_path, x_full_file))
    x_raw_full = np.load('%s/%s' % (data_path, x_raw_full_file))
    diag_full = np.load('%s/%s' % (data_path, diag_full_file))
    phy_full = np.concatenate((x_raw_full[:,:,:-2],diag_full),axis=2)
    depth_areas = np.load('%s/%s' % (data_path, depth_areas_file))
    n_depths = depth_areas.size
    
    # Read in the observation data (temperature)
    label = np.load('%s/%s' % (data_path, label_file)) # Observations
    if mask_file == '':
        mask = np.ones([label.shape[0],label.shape[1]])*1.0 # no masking
    else:
        mask = np.load('%s/%s' % (data_path, mask_file))
     
    # Separate training sets from test sets
    x_tr = x_full[:,train_range,:] # was 5000:12500. use a smaller dataset for testing on alison's memory-poor GPU
    y_tr = label[:,train_range]
    p_tr = phy_full[:,train_range,:]
    m_tr = mask[:,train_range]
    
    x_te = x_full[:,test_range,:]
    y_te = label[:,test_range]
    p_te = phy_full[:,test_range,:]
    m_te = mask[:,test_range]
    
    num_train_batches = np.int(np.ceil(train_range.__len__()/batch_size)) # __len__() requires python 3
    x_train_1 = np.zeros([n_depths*N_sec*num_train_batches, n_steps, input_size])
    y_train_1 = np.zeros([n_depths*N_sec*num_train_batches, n_steps])
    p_train_1 = np.zeros([n_depths*N_sec*num_train_batches, n_steps, phy_size])
    m_train_1 = np.zeros([n_depths*N_sec*num_train_batches, n_steps])
    
    num_test_batches = np.int(np.ceil(test_range.__len__()/batch_size)) # __len__() requires python 3
    x_test = np.zeros([n_depths*N_sec*num_test_batches, n_steps, input_size])
    y_test = np.zeros([n_depths*N_sec*num_test_batches, n_steps])
    p_test = np.zeros([n_depths*N_sec*num_test_batches, n_steps, phy_size])
    m_test = np.zeros([n_depths*N_sec*num_test_batches, n_steps])
    
    # rearrange data by batches and segments=windows within each batch,
    # going from a horizontal arrangement (depths x dates [x num features])
    # to a more vertical arrangement (depths*n_windows_per_batch*n_batches x n_dates_per_window [x num features]).
    # NOTE that we don't actually use batches in the NN training, but this arrangement
    # makes it so we could (because windows don't cross over between batches, even though
    # they can overlap within a batch)
    for i in range(1,N_sec+1): # iterate over the windows (sections)
        
        for j in range(num_train_batches):
            # pick out one training batch (subset by dates)
            tr_1_range = range((j)*batch_size, (j+1)*batch_size)
            x_tr_1 = x_tr[:,tr_1_range,:]
            y_tr_1 = y_tr[:,tr_1_range]
            p_tr_1 = p_tr[:,tr_1_range,:]
            m_tr_1 = m_tr[:,tr_1_range]
            
            # move the ith window from that training batch into its own vertical segment
            horiz_window_inds = range(int((i-1)*n_steps/2), int((i+1)*n_steps/2)) # window centered on i*n_steps
            vert_batch_start = j*n_depths*N_sec # move that window down to the next open vertical chunk
            vert_window_inds = range((i-1)*n_depths + vert_batch_start, i*n_depths + vert_batch_start)
            x_train_1[vert_window_inds,:,:] = x_tr_1[:,horiz_window_inds,:]
            y_train_1[vert_window_inds,:] = y_tr_1[:,horiz_window_inds]
            p_train_1[vert_window_inds,:,:] = p_tr_1[:,horiz_window_inds,:]
            m_train_1[vert_window_inds,:] = m_tr_1[:,horiz_window_inds]
        
        for j in range(num_test_batches):
            te_1_range = range((j)*batch_size, (j+1)*batch_size)
            x_te_1 = x_te[:,te_1_range,:]
            y_te_1 = y_te[:,te_1_range]
            p_te_1 = p_te[:,te_1_range,:]
            m_te_1 = m_te[:,te_1_range]
            
            horiz_window_inds = range(int((i-1)*n_steps/2), int((i+1)*n_steps/2))
            vert_batch_start = j*n_depths*N_sec
            vert_window_inds = range((i-1)*n_depths+vert_batch_start, i*n_depths+vert_batch_start)
            x_test[vert_window_inds,:,:] = x_te_1[:,horiz_window_inds,:]
            y_test[vert_window_inds,:] = y_te_1[:,horiz_window_inds]
            p_test[vert_window_inds,:,:] = p_te_1[:,horiz_window_inds,:]
            m_test[vert_window_inds,:] = m_te_1[:,horiz_window_inds]
    
    
    # prepare the unsupervised learning data, for which we can pass in predictors adn 
    x_f = np.concatenate((x_test,x_train_1),axis=0)
    p_f = np.concatenate((p_test,p_train_1),axis=0)
    
    return(x_train_1, y_train_1, m_train_1, x_f, p_f, x_test, y_test, m_test, depth_areas)