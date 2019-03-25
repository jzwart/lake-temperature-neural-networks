# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:16:35 2018

@author: aappling
"""

import numpy as np
import tensorflow as tf

def train_tf_graph(
        train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data,
        x_train, y_train, m_train, x_unsup, p_unsup, x_test, y_test, m_test,
        seq_per_batch, n_epochs=300, min_epochs_test=0, min_epochs_save=250,
        restore_path='', save_path='./out/EC_mille/pretrain', max_to_keep=3, save_preds=True):
    """Trains a tensorflow graph for the PRGNN model

    Args:
        train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data: tf tensors
        x_train: Input data for supervised learning
        y_train: Observations for supervised learning
        m_train: Observation mask for supervised learning
        x_unsup: Input data for unsupervised learning
        p_unsup: Observations for unsupervised learning (p = physics)
        x_test: Inputs data for testing
        y_test: Observations for testing
        m_test: Observation mask for testing
        seq_per_batch: Number of sequences to include per batch
        n_epochs: Total number of epochs to run during training. Needs to be larger than the n epochs needed for the model to converge
        min_epochs_test: Minimum number of epochs to run through before computing test losses
        min_epochs_save: Minimum number of epochs to run through before considering saving a checkpoint (must be >= min_epochs_test)
        restore_path: Path to restore a model from, or ''
        save_path: Path (directory) to save a model to. Will always be saved as ('checkpoint_%s' %>% epoch)
        max_to_keep: Max number of checkpoints to keep in the save_path
        save_preds: Logical - should test predictions be saved (and ovewritten) each time we save a checkpoint?
    """

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

    # Prepare a NN saver
    saver = tf.train.Saver(max_to_keep=max_to_keep) # tells tf to prepare to save the graph we've defined so far

    # Initialize merr to a flag value so we know to replace it with the first computed test loss
    merr = -1

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
                _, loss, rc, ec = sess.run( # this is where you feed in data and get output from the graph
                        # tell sess.run which tensorflow variables to update during each epoch.
                        # train_op is important because that contains the gradients for all the costs and variables.
                        # all others are optional to track here.
                        [train_op, cost, r_cost, unsup_loss],
                        # define the inputs to the function
                        feed_dict = {
                                x: x_train_batch,
                                y: y_train_batch,
                                m: m_train_batch,
                                unsup_inputs: x_unsup_batch,
                                unsup_phys_data: p_unsup_batch
                })

                # Report training losses
                print(
                    "Epoch " + str(epoch) \
                    + ", batch " + str(batch) \
                    + ", loss_train " + "{:.4f}".format(loss) \
                    + ", Rc " + "{:.4f}".format(rc) \
                    + ", Ec " + "{:.4f}".format(ec))

            if epoch >= min_epochs_test - 1:
                # Calculate loss and temperature predictions (prd) for test set
                loss_test, prd = sess.run(
                        # note that this first arg doesn't include train_op, so we're not updating the model
                        # now that we're applying to the test set
                        [r_cost, pred],
                        feed_dict = {
                                x: x_test_reshaped,
                                y: y_test_reshaped,
                                m: m_test_reshaped
                })

                # Initialize merr if needed
                if merr == -1:
                    merr = loss_test

                # Save the predictions and/or model state if this is the minimum loss we've seen
                if merr > loss_test:
                    merr = loss_test
                    if epoch >= min_epochs_save - 1: # (and only do this saving if we're already past some # of epochs)
                        save_file = saver.save(sess, "%s/checkpoint_%03d" % (save_path, epoch))
                        print("  Model saved to %s" % save_file)
                        if save_preds:
                            np.save('%s/preds.npy' % save_path, prd)

                # Report test losses
                print(
                    "  loss_test " + "{:.4f}".format(loss_test) \
                    + ", min_loss " + "{:.4f}".format(merr) )

    return prd
