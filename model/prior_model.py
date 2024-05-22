# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from networks.conditional_unet_1D import *


class PriorModel:
    def __init__(self, prior=None, action_dim=1, pred_horizon=1):
        self.prior = prior
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

    def sample(self, cond):
        states = cond

        if self.prior is None:
            if type(states) is np.array:
                states_torch = torch.as_tensor(states)
            else:
                states_torch = states
            prior_sample = torch.randn((states_torch.shape[0], self.pred_horizon, self.action_dim))
        else:
            if type(states) is torch.Tensor:
                states_np = states.cpu().numpy()
            else:
                states_np = states
            prior_sample = self.prior.sample_prior(states_np)
            prior_sample = torch.as_tensor(prior_sample)

        return prior_sample