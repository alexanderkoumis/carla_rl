import random
import re

import carla
import cv2
import numpy as np

from carla_env.camera_manager import CameraManager
from carla_env.collision_sensor import CollisionSensor


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]



class World(object):


    def __init__(self, carla_world):
        self.world = carla_world

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.map = self.world.get_map()
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self.weather_presets = find_weather_presets()
        self.weather_index = 0
        self.restart()


    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the vehicle.
        if self.vehicle is not None:
            spawn_point = self.vehicle.get_transform()
            # spawn_point.location.z += 8.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()

            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[1]
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        while self.vehicle is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[1]
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle)
        self.camera_manager = CameraManager(self.vehicle)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

        return self.get_frame()


    def next_weather(self, reverse=False):
        self.weather_index += -1 if reverse else 1
        self.weather_index %= len(self.weather_presets)
        preset = self.weather_presets[self.weather_index]
        self.vehicle.get_world().set_weather(preset[0])


    def get_frame(self):
        image = self.camera_manager.surface_np
        # TODO: Get this from carla_env.py
        # image_size_net = (160, 90)
        image_size_net = (80, 45)
        image_resized = cv2.resize(image, image_size_net)
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        return image, image_resized[:, :, np.newaxis].astype(float)


    def tick(self, clock):
        pass


    def render(self, display):
        self.camera_manager.render(display)
    

    def destroy(self):
        actors = [self.collision_sensor.sensor, self.camera_manager.sensor, self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()
