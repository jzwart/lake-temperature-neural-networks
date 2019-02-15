# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:16:35 2018

@author: aappling
"""

import numpy as np
import tensorflow as tf

def train_tf_graph(
        train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data,
        x_train_1, y_train_1, m_train_1, x_f, p_f, x_test, y_test, m_test,
        tot_epochs=300, min_epochs_test=0, min_epochs_save=250,
        restore_path='', save_path='./out/EC_mille/pretrain', max_to_keep=3, save_preds=True):
    """Trains a tensorflow graph for the PRGNN model
    
    Args:
        train_op, cost, r_cost, pred, unsup_loss, x, y, m, unsup_inputs, unsup_phys_data: tf tensors
        x: Input data for supervised learning
        y: Observations for supervised learning
        m: Observation mask for supervised learning
        x_f: Input data for unsupervised learning
        p_f: Observations for unsupervised learning (p = physics)
        x_test: Inputs data for testing
        y_test: Observations for testing
        m_test: Observation mask for testing
        tot_epochs: Total number of epochs to run during training. Needs to be larger than the n epochs needed for the model to converge
        min_epochs_test: Minimum number of epochs to run through before computing test losses
        min_epochs_save: Minimum number of epochs to run through before considering saving a checkpoint (must be >= min_epochs_test)
        restore_path: Path to restore a model from, or ''
        save_path: Path (directory) to save a model to. Will always be saved as ('checkpoint_%s' %>% epoch)
        max_to_keep: Max number of checkpoints to keep in the save_path
        save_preds: Logical - should test predictions be saved (and ovewritten) each time we save a checkpoint?
    """
    
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
    
        for epoch in range(tot_epochs):
            # Train on training set, including both supervised and unsupervised data
            _, loss, rc, ec = sess.run( # this is where you feed in data and get output from the graph
                    [train_op, cost, r_cost, unsup_loss], # tells sess.run which tensorflow variables to update during each epoch. train_op is important because that contains the gradients for all the costs and variables. all others are optional to track here. a, b, and c are debugging variables xiaowei wanted
                    feed_dict = { # inputs to the function
                            x: x_train_1, # now these have a realized first dimension, the num depths * num windows, which is 24*300 for Mille Lacs
                            y: y_train_1,
                            m: m_train_1,
                            unsup_inputs: x_f,
                            unsup_phys_data: p_f
            })
            
            # Report training losses
            print(
                "Step " + str(epoch) \
                + ", loss_train " + "{:.4f}".format(loss) \
                + ", Rc " + "{:.4f}".format(rc) \
                + ", Ec " + "{:.4f}".format(ec))
    
            if epoch >= min_epochs_test - 1:
                # Calculate loss and temperature predictions (prd) for test set
                loss_test, prd = sess.run(
                        [r_cost, pred], # note that this first arg doesn't include train_op, so we're not updating the model now that we're applying to the test set
                        feed_dict = {
                                x: x_test,
                                y: y_test,
                                m: m_test
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
