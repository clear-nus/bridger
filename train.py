# @markdown ### **Imports**
import torch
import numpy as np
from tqdm.auto import tqdm
from dataset.load_dataset import load_dataset, set_model_prior
from model.stochastic_interpolants import StochasticInterpolants
from model.vae import VAEModel

import util.util as util
from util.logger import Logger
import json

import os
from pathlib import Path
import argparse

RESULT_DIR = Path("results")


def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=250)

    parser.add_argument("--model_name", type=str, default='si', help="model name")
    parser.add_argument("--task_name", type=str, default='door_human', help="task name")
    parser.add_argument("--data_size", type=int, default=2500)

    parser.add_argument("--beta_max", type=float, default=0.03)
    parser.add_argument("--interpolant_type", type=str, default='power3', help="task name")

    parser.add_argument("--pretrain", action="store_true", help="use pretrained model")
    parser.add_argument("--prior_policy", type=str, default='heuristic', help="{heuristic, gaussian, cvae}")

    parser.add_argument("--gpu", type=int, default=1, help="set only if you wish to run on a particular device")

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device = 'cuda' if opt.gpu is None else f'cuda:{opt.gpu}'

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    return opt


if __name__ == "__main__":
    opt = create_training_options()

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
    os.makedirs(ckpt_path, exist_ok=True)

    # Create model
    model_args = args['diffuse_params']
    model_args['prior_policy'] = opt.prior_policy
    model_args['action_dim'] = dataset.action_dim
    model_args['action_horizon'] = dataset.pred_horizon
    model_args['obs_dim'] = dataset.obs_dim
    model_args['obs_horizon'] = dataset.obs_horizon
    model_args['pretrain'] = opt.pretrain
    model_args['ckpt_path'] = ckpt_path

    if opt.model_name == 'si':
        model = StochasticInterpolants(model_args)
        if opt.offline_prior:
            pass
        else:
            prior_args = dataset_args
            prior_args['model_name'] = opt.model_name
            prior_args['device'] = opt.device
            prior_args['prior_policy'] = model_args['prior_policy']
            prior_args['seed'] = opt.seed
            model = set_model_prior(model, prior_args)
    elif opt.model_name == 'vae':
        model = VAEModel(model_args)
    else:
        raise NotImplementedError
    model.load_model(model_args=model_args, device=opt.device)

    # Set writer and logger
    writer_args = args['logger_params']
    writer_args['log_dir'] = ckpt_path
    writer_args['name'] = 'logs'
    writer_args['global_rank'] = 0

    writer = util.build_log_writer(writer_args)
    log = Logger(0, writer_args['log_dir'])

    # Creat optimizer
    optimizer_args = args['train_params']
    optimizer_args['load'] = False

    optimizer, sched = util.build_optimizer_sched(optimizer_args, model.net, log)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=optimizer_args['batch_size'],
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    model.net.train()
    for it in tqdm(range(optimizer_args['num_itr']), desc="Epoch", unit="item"):
        for nbatch in dataloader:
            loss_args = {'prior_policy': model_args['prior_policy']}
            loss, loss_info = model.get_loss(nbatch, loss_args, opt.device)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            model.ema.update()
        if sched is not None: sched.step()

        if it % opt.log_freq == 0:
            model.log_info(writer, log, loss_info, optimizer, it, optimizer_args['num_itr'])

        if it % opt.save_freq == 0:
            model.save_model(ckpt_path=ckpt_path, itr=it)
            torch.save({
                "optimizer": optimizer.state_dict(),
                "sched": sched.state_dict() if sched is not None else sched,
            }, os.path.join(ckpt_path,  "optimizer.pt"))

            file_path = os.path.join(ckpt_path,  "static.json")
            with open(file_path, 'w') as json_file:
                json.dump(args, json_file)

            opt_json = json.dumps(vars(opt))
            file_path = os.path.join(ckpt_path, "dynamic.json")
            with open(file_path, 'w') as json_file:
                json.dump(opt_json, json_file)
    writer.close()