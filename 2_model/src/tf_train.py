# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:16:35 2018

@author: aappling
"""

import os
import numpy as np
import tensorflow as tf

def train_tf_graph(
        train_op, total_loss, rmse_loss, ec_loss, l1_loss, pred, x, y, m, unsup_inputs, unsup_phys_data,
        x_train, y_train, m_train, x_unsup, p_unsup, x_test, y_test, m_test, x_pred,
        sequence_offset, seq_per_batch, n_epochs=300, min_epochs_test=0, min_epochs_save=300,
        restore_path='', save_path='./out/EC_mille/pretrain'):
    """Trains a tensorflow graph for the PRGNN model

    Args:
        train_op, rmse_loss, ec_loss, l1_loss, pred, x, y, m, unsup_inputs, unsup_phys_data: tf tensors
        x_train: Input data for supervised learning
        y_train: Observations for supervised learning
        m_train: Observation mask for supervised learning
        x_unsup: Input data for unsupervised learning
        p_unsup: Observations for unsupervised learning (p = physics)
        x_test: Input data for testing
        y_test: Observations for testing
        m_test: Observation mask for testing
        x_pred: Input data for final prediction
        sequence_offset: Number of observations by which each data sequence in inputs['predict.features'] is offset from the previous sequence. Used to reconstruct a complete prediction sequence without duplicates.
        seq_per_batch: Number of sequences to include per batch
        n_epochs: Total number of epochs to run during training. Needs to be larger than the n epochs needed for the model to converge
        min_epochs_test: Minimum number of epochs to run through before computing test losses
        min_epochs_save: Minimum number of epochs to run through before considering saving a checkpoint (must be >= min_epochs_test)
        restore_path: Path to restore a model from, or ''
        save_path: Path (directory) to save a model to. Will always be saved as ('checkpoint_%s' %>% epoch)
    """

    # Make sure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Compute the number of sequences in each batch
    n_super_seqs = y_train.shape[2]
    batches_per_epoch = np.int(np.floor(n_super_seqs / seq_per_batch))

    # Reshape the test data. We'll reshape the training data below, but the test
    # data should be the same (complete) shape every time.
    x_test_seqs = np.transpose(x_test, (3,0,1,2))
    y_test_seqs = np.transpose(y_test, (2,0,1))
    m_test_seqs = np.transpose(m_test, (2,0,1))
    x_test_reshaped = x_test_seqs.reshape((x_test.shape[0]*x_test.shape[3], x_test.shape[1], x_test.shape[2]))
    y_test_reshaped = y_test_seqs.reshape((y_test.shape[0]*y_test.shape[2], y_test.shape[1]))
    m_test_reshaped = m_test_seqs.reshape((m_test.shape[0]*m_test.shape[2], m_test.shape[1]))

    # Reshape the prediction data; as with test data, just once is plenty
    x_pred_seqs = np.transpose(x_pred, (3,0,1,2))
    x_pred_reshaped = x_pred_seqs.reshape((x_pred.shape[0]*x_pred.shape[3], x_pred.shape[1], x_pred.shape[2]))

    # Prepare a NN saver
    saver = tf.train.Saver(max_to_keep=1) # tells tf to prepare to save the graph we've defined so far

    # Initialize arrays to hold the training progress information
    train_stat_names = ('loss_total', 'loss_RMSE', 'loss_EC', 'loss_L1') # 'loss_DD',
    train_stats = np.full((n_epochs, batches_per_epoch, len(train_stat_names)), np.nan, dtype=np.float64)
    test_loss_rmse = np.full((n_epochs), np.nan, dtype=np.float64)

    with tf.Session() as sess:

        # If using pretrained model, reload it now
        if restore_path != '':
            latest_ckp = tf.train.latest_checkpoint(restore_path)
            #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
            #print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
            saver.restore(sess, latest_ckp)

        # Initialize the weights and biases
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):

            # Choose all the indices at once (for the epoch) at random
            super_seq_indices = np.random.randint(0, high=x_train.shape[3]-1, size=seq_per_batch*batches_per_epoch) # supervised
            unsup_seq_indices = np.random.randint(0, high=x_unsup.shape[3]-1, size=seq_per_batch*batches_per_epoch) # unsupervised
            # Note that the number of unsupervised sequences likely exceed the number
            # of supervised sequences in the full dataset, but we're still only selecting
            # seq_per_batch sequences for unsup as well as super, where seq_per_batch
            # is meant to [nearly] cover the number of supervised sequences in the dataset.
            # Thus, we're accepting that each "epoch" will see just about all of the
            # supervised data but only a subset of the unsupervised data. With enough epochs
            # and randomness, we'll eventually cover all the unsupervised and supervised data,
            # but we might be stretching the definition of "epoch" here. Still, it seems reasonable
            # to balance the amount of unsupervised data going into each epoch, especially since
            # there are many more non-NaN "observations" per sequence in the unsupervised data
            # already (because the supervised data, when training on real observations, have many holes).

            for batch in range(batches_per_epoch):

                # Pick out the sequences assigned to this batch
                batch_super_seq_indices = super_seq_indices[(batch*seq_per_batch):((batch+1)*seq_per_batch)]
                batch_unsup_seq_indices = unsup_seq_indices[(batch*seq_per_batch):((batch+1)*seq_per_batch)]

                # Subset the data into the selected sequences by index; resulting np.arrays are transposed to:
                # x.shape = (sequence, depth, date, feature),
                # y.shape and m.shape = (sequence, depth, date),
                # which is actually convenient for the .reshape() calls that follow.
                x_train_seqs = np.array([x_train[:,:,:,s] for s in batch_super_seq_indices])
                y_train_seqs = np.array([y_train[:,:,s]   for s in batch_super_seq_indices])
                m_train_seqs = np.array([m_train[:,:,s]   for s in batch_super_seq_indices])
                x_unsup_seqs = np.array([x_unsup[:,:,:,s] for s in batch_unsup_seq_indices])
                p_unsup_seqs = np.array([p_unsup[:,:,:,s] for s in batch_unsup_seq_indices])

                # Combine the first two dimensions (sequence, depth) into 1 ([s1d1, s1d2, ..., s2d1, s2d2, ...])
                x_train_batch = x_train_seqs.reshape((x_train.shape[0]*seq_per_batch, x_train.shape[1], x_train.shape[2]))
                y_train_batch = y_train_seqs.reshape((y_train.shape[0]*seq_per_batch, y_train.shape[1]))
                m_train_batch = m_train_seqs.reshape((m_train.shape[0]*seq_per_batch, m_train.shape[1]))
                x_unsup_batch = x_unsup_seqs.reshape((x_unsup.shape[0]*seq_per_batch, x_unsup.shape[1], x_unsup.shape[2]))
                p_unsup_batch = p_unsup_seqs.reshape((p_unsup.shape[0]*seq_per_batch, p_unsup.shape[1], p_unsup.shape[2]))

                # Train on training set, including both supervised and unsupervised data
                _, loss_total, loss_rmse, loss_ec, loss_l1 = sess.run( # this is where you feed in data and get output from the graph
                        # tell sess.run which tensorflow variables to update during each epoch.
                        # train_op is important because that contains the gradients for all the costs and variables.
                        # all others are optional to track here.
                        [train_op, total_loss, rmse_loss, ec_loss, l1_loss],
                        # define the inputs to the function
                        feed_dict = {
                                x: x_train_batch,
                                y: y_train_batch,
                                m: m_train_batch,
                                unsup_inputs: x_unsup_batch,
                                unsup_phys_data: p_unsup_batch
                })

                # Store stats
                train_stats[epoch, batch, :] = [loss_total, loss_rmse, loss_ec, loss_l1]

                # Report training losses
                print(
                    "Epoch " + str(epoch) \
                    + ", batch " + str(batch) \
                    + " training losses: Total " + "{:.4f}".format(loss_total) \
                    + ", RMSE " + "{:.4f}".format(loss_rmse) \
                    + ", EC " + "{:.4f}".format(loss_ec) \
                    + ", L1 " + "{:.4f}".format(loss_l1))

            # Calculate, store, and report RMSE-only loss for test set if & when requested
            if epoch >= min_epochs_test - 1:
                test_loss_rmse[epoch] = sess.run(
                        # note that this first arg doesn't include train_op, so we're not updating the model
                        # now that we're applying to the test set
                        rmse_loss,
                        feed_dict = {
                                x: x_test_reshaped,
                                y: y_test_reshaped,
                                m: m_test_reshaped
                })
                print(
                    "  Test RMSE: {:.4f}".format(test_loss_rmse[epoch]))

            # Save the model state if & when requested
            if epoch >= min_epochs_save - 1:
                model_save_file = saver.save(sess, "%s/model" % save_path)
                print("  Model state saved to %s.*" % model_save_file)

            # Generate temperature predictions (prd) for the full dataset after last epoch
            if epoch == n_epochs - 1:
                preds_raw = sess.run(pred, feed_dict = {x: x_pred_reshaped})
                # Reshape the raw predictions into a single depth-by-time matrix of best predictions
                n_depths, n_dates, n_seqs = (x_pred.shape[0], x_pred.shape[1], x_pred.shape[3]) # get some dimensions
                start_best_dates = n_dates - sequence_offset # start index of the best preds in each sequence
                preds_init = preds_raw[0:n_depths, 0:start_best_dates, 0] # the not-great but only-available initial preds
                preds_last = preds_raw[0:(n_depths*n_seqs), start_best_dates:n_dates, 0] \
                    .reshape((n_seqs, n_depths, sequence_offset)) \
                    .transpose((1 ,0, 2)) \
                    .reshape(n_depths, sequence_offset * n_seqs) # the best preds from every sequence, reshaped into matrix
                preds_best = np.concatenate((preds_init, preds_last), axis=1) # combo of init and last
                # Save
                pred_save_file = '%s/preds.npz' % save_path
                np.savez_compressed(pred_save_file, preds_raw=preds_raw, preds_best=preds_best)
                print("  Predictions saved to %s" % pred_save_file)

    return(train_stats, test_loss_rmse, preds_best)
