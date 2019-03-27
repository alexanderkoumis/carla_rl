# #!/usr/bin/env python3
# '''
# Carla DQN

# Based on:
#     https://raw.githubusercontent.com/keon/deep-q-learning/master/ddqn.py
#     https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
#     https://github.com/carla-simulator/carla/blob/master/PythonAPI/automatic_control.py

# '''
# import random
# import os

# import gym
# import numpy as np
# import pygame

# import carla_env
# from agent import DQNAgent


# EPISODES = 5000

# def main():

#     env = gym.make('carla-v0')
#     state_size = env.image_size_net_chans
#     action_size = len(env.action_space)
#     agent = DQNAgent(state_size, action_size)

#     done = False
#     batch_size = 10

#     try:

#         for episode in range(EPISODES):
#             state = env.reset(render=True)
#             score = 0.0
#             for time in range(10000):
#                 env.render()
#                 action = agent.act(state)
#                 next_state, reward, done = env.step(action)

#                 if done:
#                     reward = -15
#                 else:
#                     if abs(reward) < 0.5:
#                         continue

#                 score += reward
#                 agent.remember(state, action, reward, next_state, done)
#                 state = next_state
#                 if done:
#                     agent.update_target_model()
#                     print('episode: {}/{}, score: {:.5}, e: {}'.format(
#                         episode, EPISODES, score, agent.epsilon))
#                     break
#                 if len(agent.memory) > batch_size:
#                     agent.replay(batch_size)
#             if episode % 10 == 0:
#                 agent.save(os.path.join('..', 'models', 'carla-ddqn.h5'))

#     finally:

#         env.world.destroy()


# if __name__ == '__main__':
#     main()




#!/usr/bin/env python3

"""From https://github.com/openai/baselines/blob/master/baselines/ppo1/run_robotics.py"""

import baselines.common.tf_util
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import robotics_arg_parser
from baselines.ppo1 import mlp_policy, pposgd_simple



def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=256, num_hid_layers=3)


def train(env_id, num_timesteps, seed):

    sess = baselines.common.tf_util.single_threaded_session()
    sess.__enter__()

    workerseed = seed + 10000
    set_global_seeds(workerseed)

    env = gym.make('carla-v0')

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_actorbatch=2048,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=256,
        gamma=0.99, lam=0.95, schedule='linear',
    )

    env.close()


def main():
    args = robotics_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()