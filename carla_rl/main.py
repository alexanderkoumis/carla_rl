#!/usr/bin/env python3
'''
Carla PPO1

Based on:
    https://github.com/openai/baselines/blob/master/baselines/ppo1/run_robotics.py
    https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
    https://github.com/carla-simulator/carla/blob/master/PythonAPI/automatic_control.py

'''

import baselines.common.tf_util
import gym

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import robotics_arg_parser
from baselines.ppo1 import mlp_policy, pposgd_simple
from mpi4py import MPI


import carla_env


def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=256, num_hid_layers=3)


def train(env_id, num_timesteps, seed):

    sess = baselines.common.tf_util.single_threaded_session()
    sess.__enter__()

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = gym.make('carla-v0')

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_actorbatch=2048,
        clip_param=0.2,
        entcoeff=0.0,
        optim_epochs=5,
        optim_stepsize=3e-4,
        optim_batchsize=None,
        gamma=0.99,
        lam=0.95,
        schedule='linear',
    )

    env.close()


def main():
    args = robotics_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
