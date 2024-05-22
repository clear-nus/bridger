from dataset.d4rl_state import D4RLDataset
from model.prior_model import PriorModel
import os
import util.util as util


def set_model_prior(model, prior_args):
    dataset = load_dataset(dataset_args=prior_args)
    if prior_args['prior_policy'] == 'heuristic':
            prior_model = PriorModel(prior=dataset, pred_horizon=dataset.pred_horizon, action_dim=dataset.action_dim)
    elif prior_args['prior_policy'] == 'gaussian':
        prior_model = PriorModel(prior=None, pred_horizon=dataset.pred_horizon, action_dim=dataset.action_dim)
    elif prior_args['prior_policy'] == 'cvae':
        from model.vae import VAEModel
        spec_file = os.path.join(os.path.join('./dataset/config', prior_args["task_name"]))
        args = util.load_experiment_specifications(spec_file, 'vae' + '.json')

        model_spec_names = args['diffuse_params']['net_type']
        model_args = args['diffuse_params']
        dir_name = str(prior_args['seed']) + '_' + 'vae' + '_' + str(prior_args["data_size"]) + '_' + model_spec_names
        ckpt_path = os.path.join(f'./results/train/{prior_args["task_name"]}/vae', dir_name)
        model_args['action_dim'] = dataset.action_dim
        model_args['action_horizon'] = dataset.pred_horizon
        model_args['obs_dim'] = dataset.obs_dim
        model_args['obs_horizon'] = dataset.obs_horizon
        model_args['pretrain'] = True
        model_args['ckpt_path'] = ckpt_path
        prior_model = VAEModel(model_args)
        prior_model.load_model(model_args=model_args, device=prior_args['device'])
    else:
        raise NotImplementedError
    model.prior_model = prior_model

    return model


def load_dataset(dataset_args):
    if dataset_args['task_name'] == 'franka_kitchen_mix':
        dataset = D4RLDataset(env_name='kitchen-mixed-v0', pred_horizon=16, obs_horizon=2, action_horizon=8, data_size=dataset_args['data_size'])
    elif dataset_args['task_name'] == 'door_human':
        dataset = D4RLDataset(env_name='door-human-v0', data_size=dataset_args['data_size'])
    elif dataset_args['task_name'] == 'pen_human':
        dataset = D4RLDataset(env_name='pen-human-v0', data_size=dataset_args['data_size'])
    elif dataset_args['task_name'] == 'relocate_human':
        dataset = D4RLDataset(env_name='relocate-human-v0', data_size=dataset_args['data_size'])
    elif dataset_args['task_name'] == 'hammer_human':
        dataset = D4RLDataset(env_name='hammer-human-v0', data_size=dataset_args['data_size'])
    else:
        raise NotImplementedError
    return dataset
