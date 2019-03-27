from itertools import product

import carla
import cv2
import gym
import numpy as np
import pygame
from gym import error, spaces, utils
from gym.utils import seeding

from carla_env.world import World

ob_space = env.observation_space
ac_space = env.action_space

class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['fpv', 'follow']}

    def __init__(self):

        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(4.0)
        self.world = World(client.get_world())

        num_samples = 20

        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(num_samples,), dtype=np.float32)

        self.action_space = spaces.Box(
            np.array([-1,0]), np.array([1,1]))

    def step(self, action, straight=False):

        print(action)

        self.world.world.tick()

        if straight:
            act_steer, act_throttle = 0.0, 1.0
        else:
            act_steer, act_throttle = self.action_space[action]
        
        control = carla.VehicleControl()
        control.steer = act_steer
        control.throttle = act_throttle
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        self.world.vehicle.apply_control(control)
        next_state, next_state_resized = self.world.get_frame()

        done = len(self.world.collision_sensor.history) > 0

        vel = self.world.vehicle.get_velocity()
        reward = np.linalg.norm([vel.x, vel.y, vel.z])

        return next_state_resized, reward, done
            

    def reset(self, render=False):
        while True:
            try:
                self.world.restart()
                break
            except:
                print('WARNING!!!!!! EXCEPTION SPAWNING')
                pass
        for _ in range(35):
            self.step(0, True)
            if render:
                self.render()
        return self.world.get_frame()[1]


    def render(self, mode='fpv'):
        image = self.world.get_frame()[0]
        cv2.imshow('Image', (image*255).astype(np.uint8))
        cv2.waitKey(1)
