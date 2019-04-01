#!/usr/bin/env python3
'''
Carla PPO1

Based on:
    https://github.com/openai/baselines/blob/master/baselines/ppo1/run_robotics.py
    https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
    https://github.com/carla-simulator/carla/blob/master/PythonAPI/automatic_control.py

'''

import gym
import carla_env
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy 
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1


def train():

    ## create a gym env
    env = gym.make('carla-v0')

    ## vectorize the environment
    env = DummyVecEnv([lambda: env])

    ## define a MLP policy with 2 layers of size 256 with tanh activation
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256])

    model = PPO1(MlpPolicy, env, policy_kwargs=policy_kwargs, timesteps_per_actorbatch=500, verbose=1, tensorboard_log='data/tensorboard_log/')

    model.learn(total_timesteps=50000000)
    model.save('data/ppo1_carla')

    del model

def main():
    train()


if __name__ == '__main__':
    main()
