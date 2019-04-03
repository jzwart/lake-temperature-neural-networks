# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:05:09 2018

@author: aappling
"""

import tensorflow as tf
import physics as phy

def build_tf_graph(
        n_steps, input_size, state_size,
        phy_size, colnames_physics, depth_areas,
        ec_threshold, dd_lambda, ec_lambda, l1_lambda, seq_per_batch, learning_rate):
    """Builds a tensorflow graph for the PRGNN model

    Args:
        n_steps: Number of timesteps in the time window represented by the LSTM (shorter than total number of timesteps in the dataset or even the batch)
        input_size: Number of features in the input data (drivers) used for supervised learning
        state_size: Number of units in each LSTM cell's hidden state
        phy_size: Number of features in the input data (drivers) used for UNsupervised learning. These data should be unnormalized.
        colnames_physics: Numpy array of the column names for the feature dimension of the physics array
        ec_threshold: Tolerance for energy imbalance before any penalization occurs
        dd_lambda: Depth-density penalty lambda, a hyperparameter that needs manual tuning. PRESENTLY IGNORED (no physics constraint)
        ec_lambda: Energy-conservation penalty lambda, another hyperparameter that needs manual tuning. Could set ec_lambda=0 if we wanted RNN only, no EC component
        l1_lambda: L1-regularization penalty lambda, another hyperparameter that needs manual tuning
        seq_per_batch: Number of sections to arrange vertically within the input for a batch (after splitting into windows of width n_steps)
        learning_rate: Learning rate
    """

    # Clear the slate
    tf.reset_default_graph()

    # Graph input/output. n timesteps is less than the total number of timesteps at the lake; it's just the number of dates in one sequence
    x = tf.placeholder("float", [None, n_steps, input_size])
    y = tf.placeholder("float", [None, n_steps])
    m = tf.placeholder("float", [None, n_steps])

    # Define LSTM cells
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.0) # define a single LSTM cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(state_size, forget_bias=1.0, name='basic_lstm_cell') # define a single LSTM cell
    # dynamic_rnn is faster for long timeseries, compared to tf.nn.static_rnn. the default namespace for tf.nn.dynamic_rnn in "rnn"; this becomes importatin in line 367
    state_series_x, current_state_x = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32) # add multiple LSTM cells to the network. uses the dimensions of x to choose the dimensions of the LSTM network

    # Output layer
    n_classes = 1 # Number of output values (probably 1 for a single depth-specific, time-specific prediction)
    with tf.variable_scope('output'):
        output_weights = tf.get_variable('weights', [state_size, n_classes], tf.float32,
                                         initializer=tf.random_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable('bias', [n_classes], tf.float32,
                                      initializer=tf.constant_initializer(0.0))

    # Define how LSTM and output layer are connected to make predictions
    pred=[]
    for i in range(n_steps):
        tp1 = state_series_x[:,i,:] #state_series_x dimensions are [n depths, n timesteps, n hidden units]
        pt = tf.matmul(tp1, output_weights) + output_bias
        pred.append(pt) # can't do pred[i] = pt because tensorflow wouldn't like it

    pred = tf.stack(pred,axis=1) # converts the preds list to a tensorflow array. [n depths, n timesteps, 1] (the last dimension isn'd really needed; we'll remove it later)
    pred_s = tf.reshape(pred,[-1,1]) # collapse from 3D [depths, timesteps, 1] to 2D [depths * timesteps, 1]
    y_s = tf.reshape(y,[-1,1])

    # Compute cost as RMSE with masking (the tf.where call replaces pred_s-y_s
    # with 0 when y_s is nan; num_y_s is a count of just those non-nan observations)
    # so we're only looking at predictions with corresponding observations available
    num_y_s = tf.cast(tf.count_nonzero(~tf.is_nan(y_s)), tf.float32)
    zero_or_error = tf.where(tf.is_nan(y_s), tf.zeros_like(y_s), pred_s - y_s)
    rmse_loss = tf.sqrt(tf.reduce_sum(tf.square(zero_or_error)) / num_y_s)
    # the only other cost function we might reasonably consider besides RMSE might be MAE...but they're quite similar

    ### EC penalization (using unsupervised learning, which allows us to use a larger dataset than the above supervised learning can

    #unsup = unsupervised. this is a placeholder for all the drivers relevant to the EC computation
    # this is a second phase of training, called semi-supervised learning, where we only apply one part of the loss function (the EC part).
    # so we still learn something, if not as much as we learned from the days with observations
    unsup_inputs = tf.placeholder("float", [None, n_steps, input_size]) #tf.float32 [the 30 years of data, n timesteps per window, n drivers]
    # the variable_scope part needs to match the variable_scope we used before, rnn, so that tf.nn.dynamic_rnn(lstm_cell) uses and
    # trains the same lstm parameters as before
    with tf.variable_scope("rnn", reuse=True) as scope_sp:
        state_series_xu, current_state_xu = tf.nn.dynamic_rnn(lstm_cell, unsup_inputs, dtype=tf.float32, scope=scope_sp)

    pred_u=[] # same as above for the supervised part
    for i in range(n_steps):
        tp2 = state_series_xu[:,i,:] # state_series_xu is the state series from the unsupervised input
        pt2 = tf.matmul(tp2, output_weights) + output_bias
        pred_u.append(pt2)

    pred_u = tf.stack(pred_u,axis=1)
    pred_u = tf.reshape(pred_u,[-1,n_steps])

    # unsupervised physics data include the non-normalized versions of the drivers (for the LSTM we wanted the normalized, mean=0, sd=1 version)
    unsup_phys_data = tf.placeholder("float", [None, n_steps, phy_size]) #tf.float32
    n_depths = depth_areas.shape[0]

    ec_loss = phy.calculate_ec_loss(
        unsup_inputs,
        pred_u,
        unsup_phys_data,
        depth_areas,
        n_depths,
        ec_threshold,
        seq_per_batch,
        colnames_physics)

    # Regularization loss
    # select and compute L1 loss on the trainable weights. L1 because Jared has found it's better than L2,
    # and weights not biases because it's often counterproductive to regularize weights
    regularizable_variables = [tf.reshape(tf_var, [-1]) for tf_var in tf.trainable_variables() if not ('bias' in tf_var.name)]
    l1_loss = tf.reduce_sum(tf.abs(tf.concat(regularizable_variables, axis=0)))

    # Compute total costs as weighted sum of the cost components
    total_loss = rmse_loss + ec_lambda*ec_loss + l1_lambda*l1_loss # dd_lambda*(dd_loss + dd_loss_unsup)

    # Define the gradients and optimizer
    tvars = tf.trainable_variables()
    grads = tf.gradients(total_loss, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    return(train_op, total_loss, rmse_loss, ec_loss, l1_loss, pred, x, y, m, unsup_inputs, unsup_phys_data)
