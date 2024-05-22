# @markdown ### **Imports**
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from dataset.load_dataset import load_dataset, set_model_prior
from model.stochastic_interpolants import StochasticInterpolants
from model.vae import VAEModel

import pandas as pd
import util.util as util
import json

import os
import argparse


def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--diffuse_step", type=int, default=5)

    parser.add_argument("--task_name", type=str, default='door_human', help="task name")
    parser.add_argument("--test_type", type=str, default='reward')
    parser.add_argument("--sde_type", type=str, default='vs')
    parser.add_argument("--data_size", type=int, default=2500)
    parser.add_argument("--prior_policy", type=str, default='heuristic', help="task name")
    parser.add_argument("--beta_max", type=float, default=0.03)
    parser.add_argument("--interpolant_type", type=str, default='power3', help="{heuristic, gaussian, cvae}")

    parser.add_argument("--gpu", type=int, default=0, help="set only if you wish to run on a particular device")

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device = 'cuda' if opt.gpu is None else f'cuda:{opt.gpu}'

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    return opt


def save_csv(fileroot, filename, df):
    df.to_csv(fileroot + filename, index=False)


if __name__ == "__main__":
    opt = create_training_options()
    opt.pretrain = True

    opt.model_name = 'si'
    # Load config file for training
    spec_file = os.path.join(os.path.join('./dataset/config', opt.task_name))
    args = util.load_experiment_specifications(spec_file, opt.model_name + '.json')
    args['diffuse_params']['interpolant_type'] = opt.interpolant_type
    args['diffuse_params']['beta_max'] = opt.beta_max

    # Load dataset
    dataset_args = {'task_name': opt.task_name,
                    'data_size': opt.data_size}
    dataset = load_dataset(dataset_args)

    ckpt_path = os.path.join(f'./results/train/{opt.task_name}/{opt.model_name}_{opt.prior_policy}', util.make_model_dir(args, opt))

    # Create model
    model_args = args['diffuse_params']
    model_args['prior_policy'] = opt.prior_policy
    model_args['action_dim'] = dataset.action_dim
    model_args['action_horizon'] = dataset.pred_horizon
    model_args['obs_dim'] = dataset.obs_dim
    model_args['obs_horizon'] = dataset.obs_horizon
    model_args['pretrain'] = True
    model_args['ckpt_path'] = ckpt_path

    if opt.model_name == 'si':
        model_args['sde_type'] = opt.sde_type
        model = StochasticInterpolants(model_args)
    else:
        raise NotImplementedError
    model.load_model(model_args=model_args, device=opt.device)

    # Set model prior
    prior_args = dataset_args
    prior_args['model_name'] = 'si'
    prior_args['device'] = opt.device
    prior_args['prior_policy'] = model_args['prior_policy']
    prior_args['seed'] = opt.seed
    model = set_model_prior(model, prior_args)

    # Test
    result_info = dataset.test_reward(model, diffuse_step=opt.diffuse_step, num_episode=50, device=opt.device)


