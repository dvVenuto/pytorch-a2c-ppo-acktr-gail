import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from skimage.transform import resize

from gatedpixelcnn_bonus import PixelBonus


#from visualize import visdom_plot

import tensorflow as tf

imresize = resize

def update_tf_wrapper_args(args, tf_flags):
    """
    take input command line args to DQN agent and update tensorflow wrapper default
    settings
    :param args:
    :param FLAGS:
    :return:
    """
    # doesn't support boolean arguments
    to_parse = args.wrapper_args
    if to_parse:
        for kwarg in to_parse:
            keyname, val = kwarg.split('=')
            if keyname in ['ckpt_path', 'data_path', 'samples_path', 'summary_path']:
                # if directories don't exist, make them
                if not os.path.exists(val):
                    os.makedirs(val)
                tf_flags.update(keyname, val)
            elif keyname in ['data', 'model']:
                tf_flags.update(keyname, val)
            elif keyname in ['mmc_beta']:
                tf_flags.update(keyname, float(val))
            else:
                tf_flags.update(keyname, int(val))
    return tf_flags

class DotDict(object):
    def __init__(self, dict):
        self.dict = dict

    def __getattr__(self, name):
        return self.dict[name]

    def update(self, name, val):
        self.dict[name] = val

    # can delete this later
    def get(self, name):
        return self.dict[name]

FLAGS = DotDict({
    'img_height': 42,
    'img_width': 42,
    'channel': 1,
    'data': 'mnist',
    'conditional': False,
    'num_classes': None,
    'filter_size': 3,
    'init_fs': 7,
    'f_map': 16,
    'f_map_fc': 16,
    'colors': 8,
    'parallel_workers': 1,
    'layers': 3,
    'epochs': 25,
    'batch_size': 16,
    'model': '',
    'data_path': 'data',
    'ckpt_path': 'ckpts',
    'samples_path': 'samples',
    'summary_path': 'logs',
    'restore': True,
    'nr_resnet': 1,
    'nr_filters': 32,
    'nr_logistic_mix': 5,
    'resnet_nonlinearity': 'concat_elu',
    'lr_decay': 0.999995,
    'lr': 0.00005,
    'num_ds': 1,
    'nameDemonstrator' : 'None',
})


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    step_count = 0
    img_scale=1

    if (bool(args.useNeural)):
        FLAGS = update_tf_wrapper_args(args, utils.gatedpixelcnn_bonus.FLAGS)
        tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        pixel_bonus = PixelBonus(FLAGS, sess)
        tf.initialize_all_variables().run(session=sess)

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # print(obs)
            psc_add = 0
            if args.useNeural:
                for i in obs[0]:
                    psc_add += pixel_bonus.bonus(i, step_count)
                    step_count += 1
                psc_add = psc_add / 12
            else:
                psc_add = 0

            step += 1

            # print(psc_add)

            """
            for info in infos:
                if 'episode' in info.keys():
                    print(reward)
                    episode_rewards.append(info['episode']['r'])
            """

            # FIXME: works only for environments with sparse rewards
            for idx, eps_done in enumerate(done):
                if eps_done:
                    episode_rewards.append(reward[idx])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            psc_add = torch.FloatTensor([0.0])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, psc_add)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts,0,0)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    if args.useNeural:
        pixel_bonus.save_model(str(args.nameDemonstrator), step)
        print("Neural model has been successfully saved and named %s" % str(args.nameDemonstrator))


if __name__ == "__main__":
    main()
