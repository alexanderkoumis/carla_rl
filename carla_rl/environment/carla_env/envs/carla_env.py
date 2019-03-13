from itertools import product

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
    image_size_display = (1280, 720)
    image_size_net = (160, 90)
    image_size_net_chans = (160, 90, 3)
    # image_size_net = (80, 45)
    # image_size_net_chans = (80, 45, 3)

    def __init__(self):

        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(4.0)
        self.world = World(client.get_world())

        actions_steering = np.linspace(-1, 1, 5)
        actions_throttle = np.linspace(0, 1, 3)
        self.action_space = list(
            product(actions_steering, actions_throttle))
        
        self.cool = False

    def step(self, action, straight=False):

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
