# @markdown ### **Dataset**
# @markdown
# @markdown Defines `PushTStateDataset` and helper functions
# @markdown
# @markdown The dataset class
# @markdown - Load data (obs, action) from a zarr storage
# @markdown - Normalizes each dimension of obs and action to [-1,1]
# @markdown - Returns
# @markdown  - All possible segments with length `pred_horizon`
# @markdown  - Pads the beginning and the end of each episode with repetition
# @markdown  - key `obs`: shape (obs_horizon, obs_dim)
# @markdown  - key `action`: shape (pred_horizon, action_dim)
import collections
import gym
import d4rl

from dataset.dataset_util import *
from heuristic_policies.relocate_policy import BatchRelocatePolicy
from heuristic_policies.hammer_policy import HammerPolicy
from heuristic_policies.door_policy import DoorPolicy


ENV_NAMES = ['relocate-human-v0', 'door-human-v0', 'pen-human-v0', 'hammer-human-v0']


class D4RLDataset(torch.utils.data.Dataset):
    def __init__(self, env_name, data_size=1000,
                 pred_horizon=64, obs_horizon=4, action_horizon=48):
        self.env_name = env_name
        self.env = gym.make(env_name)
        # read from zarr dataset
        dataset_root = d4rl.qlearning_dataset(self.env)
        """
        D4RL dataset, with keys:
            observations, actions, next_observations, rewards, terminals
        Mainly for Adroit human datasets
        """

        # todo convert to sequence data
        """
        train_data = {
            "action": [data_size, action_horizon, action_dim],
            "state": ...
            ...   
        }
        obs -> state
        """

        self.train_data = {
                # (N, action_dim)
                'action': dataset_root['actions'][:],
                'prior_action': dataset_root['actions'][:],  # TODO: implement priors & obs

                # (N, obs_dim)
                'obs': dataset_root['observations'][:],
                'noise_obs': dataset_root['observations'][:]
            }

        episode_ends = np.where(dataset_root["terminals"])[0]
        if len(episode_ends) == 0:
            episode_ends = np.array([dataset_root["terminals"].shape[0] - 1])
        self.episode_ends = episode_ends

        self.data_size = self.train_data['action'].shape[0] if data_size > self.train_data['action'].shape[0] else data_size

        indices = create_sample_indices(
            data_size=self.data_size,
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)

        # create equal spaced segments of full data seq
        indices = get_indices_for_segmented_seq(full_indices=indices, data_size=self.data_size)

        self.indices = indices
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        self.obs_dim = dataset_root['observations'].shape[-1]
        self.action_dim = dataset_root['actions'].shape[-1]

        if env_name == 'door-human-v0':
            self.heuristic_policy = DoorPolicy(interpolation_num=128)
        elif env_name == 'relocate-human-v0':
            self.heuristic_policy = BatchRelocatePolicy(interpolation_num=100)
        elif env_name == 'hammer-human-v0':
            self.heuristic_policy = HammerPolicy(interpolation_num=80)
        else:
            self.heuristic_policy = None

    def __len__(self):
        # all possible segments of the dataset
        return self.data_size

    def __getitem__(self, idx):
        idx = idx if idx < len(self.indices) else len(self.indices) - 1
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['state'] = nsample['obs'][:self.obs_horizon, :]
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]  # TODO: check if need to trim noisy
        return nsample

    def test_reward(self, model, num_episode, diffuse_step, device):
        # limit enviornment interaction to 200 steps before termination
        if self.env_name == 'kitchen-mixed-v0':
            max_steps = 330
        else:
            max_steps = 475
        # use a seed >200 to avoid initial states seen in the training dataset
        # envs.seed(100000)

        env = self.env
        avg_reward = []
        max_reward = []
        avg_cost = []
        for i in range(num_episode):
            # get first observation
            obs = env.reset()

            step_count = 0
            # keep a queue of last 2 steps of observations
            obs_deque = collections.deque(
                [obs] * self.obs_horizon, maxlen=self.obs_horizon)

            # save visualization and rewards
            # imgs = [env.render(mode='rgb_array')]
            rewards = list()
            costs = list()
            images = []
            done = False

            for j in range(10000):
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = obs_seq

                # device transfer
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float)
                # infer action
                with torch.no_grad():
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
                    if model.prior_policy == 'cvae':
                        prior_action = model.prior_model.sample(obs_cond).float().to(device)
                    elif model.prior_policy == 'heuristic':
                        prior_action = model.prior_model.sample(nobs[-1].unsqueeze(0)).float().to(device)
                    else:
                        prior_action = model.prior_model.sample(obs_cond).float().to(device)
                    model_actions = model.sample(x_prior=prior_action, cond=obs_cond, diffuse_step=diffuse_step)

                    # unnormalize action
                    naction = model_actions.detach().cpu().numpy()
                    naction = naction[0]
                    action_pred = naction

                # only take action_horizon number of actions
                start = self.obs_horizon - 1
                end = start + self.action_horizon
                action = action_pred[start:end, :]

                # execute action_horizon number of steps
                # without replanning
                for k in range(len(action)):
                    # stepping envs
                    obs, reward, done, info = env.step(action[k])

                    step_count += 1

                    # image = env.render(mode='rgb_array')
                    # images += [image]
                    # import cv2
                    # # image = np.zeros((10, 10, 3))
                    # cv2.imshow('test', image)
                    # cv2.waitKey(10)

                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    # print(reward)
                if step_count > max_steps:
                    break

            # print out the maximum target coverage
            print('Score: ', np.array(rewards).sum())
            print('AVG Score: ', np.array(rewards).mean())
            print('Cost', np.array(costs).mean())

    def sample_prior(self, states):
        batched_action = self.heuristic_policy.get_action(states)  # dim: (batch_size, pred_horizon, action_dim)
        return batched_action