# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from matplotlib.patches import Arrow
import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.optim import AdamW, lr_scheduler
import util.distributed_util as dist_util
import numpy as np
import random
import json

import argparse


def make_model_dir(args, opt):
    seed_name = str(opt.seed)
    model_name = args['model_name']
    data_size = str(opt.data_size)
    if model_name == 'si':
        model_spec_names = args['diffuse_params']['net_type']
        model_spec_names += '_' + args['diffuse_params']['interpolant_type']
        model_spec_names += '_' + args['diffuse_params']['gamma_type']
        model_spec_names += '_' + str(args['diffuse_params']['beta_max'])
    elif model_name == 'ddpm':
        model_spec_names = args['diffuse_params']['net_type']
        model_spec_names += '_' + str(args['diffuse_params']['interval_train'])
    elif model_name == 'residual':
        model_spec_names = args['diffuse_params']['net_type']
    elif model_name == 'vae':
        model_spec_names = args['diffuse_params']['net_type']
    else:
        raise NotImplementedError

    dir_name = seed_name + '_' + model_name + '_' + data_size + '_' + model_spec_names
    return dir_name


def dict_to_argparser(input_dict):
    parser = argparse.ArgumentParser(description='Command line arguments')

    for key, value in input_dict.items():
        parser.add_argument(f'--{key}', type=type(value), default=value, help=f'Description for {key}')

    return parser.parse_args()


def indicator_function(condition):
    # Create a tensor of zeros with the same shape as the condition
    result = torch.zeros_like(condition, dtype=torch.float32)

    # Set the elements to 1 where the condition is satisfied
    result[condition] = 1.0

    return result


def draw_line(ax, action, title):
    ax.plot(action[0, :, 0], action[0, :, 1], label='Line 1', marker='o')
    ax.plot(action[1, :, 0], action[1, :, 1], label='Line 2', marker='v')
    ax.plot(action[2, :, 0], action[2, :, 1], label='Line 3', marker='P')
    ax.plot(action[3, :, 0], action[3, :, 1], label='Line 1', marker='1')
    ax.plot(action[4, :, 0], action[4, :, 1], label='Line 2', marker='2')
    ax.plot(action[5, :, 0], action[5, :, 1], label='Line 3', marker='3')
    doors = np.stack([np.array([-0.6, 0.0]),
                      np.array([-0.2, 0.0]),
                      # np.array([0.0, 0.0]),
                      np.array([0.35, 0.0]),
                      np.array([0.8, 0.0])], axis=0)
    ax.scatter(doors[:, 0], doors[:, 1], label='actions', color='b', marker='o', s=10)

    # arrow = Arrow(action[5, -2, 0], action[5, -2, 1], action[5, -1, 0] - action[5, -2, 0], action[5, -1, 1] - action[5, -2, 1],
    #               width=0.1, color='r', alpha=0.5)
    # ax.add_patch(arrow)

    ax.set_title(title)
    # ax.set_xlabel('X-coordinate')
    # ax.set_ylabel('Y-coordinate')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    # ax.grid()
    # ax.legend()
    # ax.show()


def draw_scatter(ax, state, action, state_marker_size=1, action_marker_size=1, title='GT'):
    # Create a scatter plot
    # x = state[:, 0]
    # y = state[:, 0] * 0.0
    # ax.scatter(x, y, label='state', color='r', marker='o', s=state_marker_size)

    x = action[:, 0, 0]
    y = action[:, 0, 1]

    # Create a scatter plot
    ax.scatter(x, y, label='actions', color='b', marker='o', s=action_marker_size)
    # ax.hist2d(x, y, bins=10, cmap='Blues')
    # ax.hist(y, bins=30, density=True, alpha=0.6, color='g')

    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    ax.set_title(title)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    # ax.legend()


def load_experiment_specifications(experiment_directory, specifications_filename):
    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"params.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()


def build_optimizer_sched(opt, net, log):
    optim_dict = {"lr": opt['lr'], 'weight_decay': opt['l2_norm']}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt['lr_gamma'] < 1.0:
        sched_dict = {"step_size": opt['lr_step'], "gamma": opt['lr_gamma']}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt['load']:
        checkpoint = torch.load(opt['load'], map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt['load']}!")
        else:
            log.warning(f"[Opt] Ckpt {opt['load']} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt['load']}!")
        else:
            log.warning(f"[Opt] Ckpt {opt['load']} has no lr sched!")

    return optimizer, sched


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def setup_loader(dataset, batch_size, num_workers=4):
    loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

    while True:
        yield from loader


class BaseWriter(object):
    def __init__(self, opt):
        self.rank = opt['global_rank']

    def add_scalar(self, step, key, val):
        pass  # do nothing

    def add_image(self, step, key, image):
        pass  # do nothing

    def close(self): pass


class WandBWriter(BaseWriter):
    def __init__(self, opt):
        super(WandBWriter, self).__init__(opt)
        if self.rank == 0:
            assert wandb.login(key=opt['wandb_api_key'])
            wandb.init(dir=str(opt['log_dir']), project="i2sb", entity=opt['wandb_user'], name=opt['name'], config=vars(opt))

    def add_scalar(self, step, key, val):
        if self.rank == 0: wandb.log({key: val}, step=step)

    def add_image(self, step, key, image):
        if self.rank == 0:
            # adopt from torchvision.utils.save_image
            image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            wandb.log({key: wandb.Image(image)}, step=step)


class TensorBoardWriter(BaseWriter):
    def __init__(self, opt):
        super(TensorBoardWriter, self).__init__(opt)
        if self.rank == 0:
            run_dir = os.path.join(opt['log_dir'] + '/', opt['name'])
            os.makedirs(run_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        if self.rank == 0: self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        if self.rank == 0:
            image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
            self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        if self.rank == 0: self.writer.close()


def build_log_writer(writer_args):
    if writer_args['log_writer'] == 'tensorboard':
        return TensorBoardWriter(writer_args)
    else:
        return BaseWriter(writer_args)  # do nothing


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]
