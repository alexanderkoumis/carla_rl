import time

import carla
import cv2
import gym
import numpy as np
import pygame
from gym import error, spaces, utils
from gym.utils import seeding

from carla_env.world import World


class CarlaEnv(gym.Env):


    metadata = {'render.modes': ['fpv', 'follow']}


    def __init__(self):

        num_samples = 80

        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(6.0)
        self.world = World(client.get_world(), num_samples)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_samples*3,), dtype=np.float32)

        self.action_space = spaces.Box(
            np.array([-1, 0]), np.array([1, 1]))


    def step(self, action=(0.0, 1.0)):

        act_steer, act_throttle = action
        
        control = carla.VehicleControl()
        control.steer = float(act_steer)
        control.throttle = float(act_throttle)
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        self.world.vehicle.apply_control(control)
        self.world.world.tick()
        self.world.world.wait_for_tick()
        next_state = self.world.get_state()

        done = len(self.world.collision_sensor.history) > 0
        done = np.int(done)

        vel_y, vel_x = self.world.get_velocity()

        reward = -10 if done else np.abs(vel_y)

        return next_state, reward, done, {}


    def reset(self):
        while True:
            try:
                self.world.restart()
                break
            except Exception as exc:
                print('Exception spawning: {}'.format(exc))
                pass
        for _ in range(35):
            self.step()
        return self.world.get_state()


    def render(self, mode='fpv'):
        image = self.world.get_depth_frame()
        cv2.imshow('Image', (image*255).astype(np.uint8))
        cv2.waitKey(1)
