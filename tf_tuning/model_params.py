# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:55:26 2018

@author: aappling
"""

import numpy as np

model_params = dict(
    pretrain_1a = dict(
        state_size = 5
    ),
    pretrain_1b = dict(
        state_size = 9
    ),
    pretrain_1c = dict(
        state_size = 18
    ),
    train_1a = dict(
        state_size = 5 # n*0.5
    ),
    train_1b = dict(
        state_size = 9 # equal to input size n
    ),
    train_1c = dict(
        state_size = 18 # n*2
    )
)

np.save('model_params.npy', model_params)
