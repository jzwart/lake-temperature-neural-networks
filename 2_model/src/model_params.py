# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:55:26 2018

@author: aappling
"""

import numpy as np

model_params = dict(
    pretrain_1a = dict(
        state_size = 5,
        pretrain = True,
        restore_path = '',
        save_path = './mille/out/pretrain_1a'
    ),
    pretrain_1b = dict(
        state_size = 9,
        pretrain = True,
        restore_path = '',
        save_path = './mille/out/pretrain_1b'
    ),
    pretrain_1c = dict(
        state_size = 18,
        pretrain = True,
        restore_path = '',
        save_path = './mille/out/pretrain_1c'
    ),
    train_1a = dict(
        state_size = 5, # n*0.5
        pretrain = False,
        restore_path = './mille/out/pretrain_1a',
        save_path = './mille/out/train_1a'
    ),
    train_1b = dict(
        state_size = 9, # equal to input size n
        pretrain = False,
        restore_path = './mille/out/pretrain_1b',
        save_path = './mille/out/train_1b'
    ),
    train_1c = dict(
        state_size = 18, # n*2
        pretrain = False,
        restore_path = './mille/out/pretrain_1c',
        save_path = './mille/out/train_1c'
    )
)

np.save('model_params.npy', model_params)
