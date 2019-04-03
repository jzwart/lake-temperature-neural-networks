#!/usr/bin/python
# -*- coding: utf-8 -*-

# Usage: python run_job.py <job>
# where <job> is job number from 0 to (number of elements in model_params.py - 1)

import sys
import argparse
sys.path.append('2_model/src')
import apply_pgnn

parser = argparse.ArgumentParser()
parser.add_argument('--phase')
parser.add_argument('--learning_rate')
parser.add_argument('--state_size')
parser.add_argument('--ec_threshold')
parser.add_argument('--dd_lambda')
parser.add_argument('--ec_lambda')
parser.add_argument('--l1_lambda')
parser.add_argument('--data_file')
parser.add_argument('--sequence_offset')
parser.add_argument('--max_batch_obs')
parser.add_argument('--n_epochs')
parser.add_argument('--min_epochs_test')
parser.add_argument('--min_epochs_save')
parser.add_argument('--restore_path')
parser.add_argument('--save_path')
args = parser.parse_args()
print(args)

# read the function arguments from the command-line and use them to run the
#  model once (one pretraining, training, or hyperparameter tuning run)
apply_pgnn.apply_pgnn(
        phase = args.phase,
        learning_rate = float(args.learning_rate),
        state_size = int(args.state_size),
        ec_threshold = float(args.ec_threshold),
        dd_lambda = float(args.dd_lambda),
        ec_lambda = float(args.ec_lambda),
        l1_lambda = float(args.l1_lambda),
        data_file = args.data_file,
        sequence_offset = int(args.sequence_offset),
        max_batch_obs = int(args.max_batch_obs),
        n_epochs = int(args.n_epochs),
        min_epochs_test = int(args.min_epochs_test),
        min_epochs_save = int(args.min_epochs_save),
        restore_path = args.restore_path,
        save_path = args.save_path
)
