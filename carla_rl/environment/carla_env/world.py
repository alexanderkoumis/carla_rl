import random
import re
import time

import carla
import cv2
import numpy as np

from agents.navigation.roaming_agent import RoamingAgent

from carla_env.camera_manager import CameraManager
from carla_env.collision_sensor import CollisionSensor


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]



class World(object):


    def __init__(self, carla_world, num_samples):

        self.num_samples = num_samples

        self.world = carla_world

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.map = self.world.get_map()
        self.vehicle = None
        self.autopilot_agent = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self.weather_presets = find_weather_presets()
        self.weather_index = 0
        self.spawn_points = self.map.get_spawn_points()
        self.create_vehicle()
        # self.restart()


    def restart(self):

        # self.destroy_vehicles()

        # reset the transform of the vehicle
        # spawn_point = random.choice(self.spawn_points)
        spawn_point = self.spawn_points[0]
        self.vehicle.set_transform(spawn_point)

        # wait for the car to settle
        for _ in range(50):
            self.world.tick()
            if self.vehicle.get_velocity().z == 0.0:
                break

        # reset the local planner to purge all old waypoints
        self.autopilot_agent._local_planner.init_controller(opt_dict={})
        self.autopilot_agent._local_planner.set_speed(150)

        # clear collision sensor history
        self.collision_sensor.history.clear()

        return self.get_state()

    def create_vehicle(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 3
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        blueprint.set_attribute('role_name', 'hero')
        blueprint.set_attribute('sticky_control', 'False')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the vehicle.
        if self.vehicle is not None:
            spawn_point = self.vehicle.get_transform()
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()

            spawn_point = random.choice(self.spawn_points)
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        while self.vehicle is None:
            spawn_point = random.choice(self.spawn_points)
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        self.autopilot_agent = RoamingAgent(self.vehicle)
        ## set the target speed of the autopilot
        self.autopilot_agent._local_planner.set_speed(150)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle)
        self.camera_manager = CameraManager(self.vehicle)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)


    def get_velocity(self):

        yaw_global = np.radians(self.vehicle.get_transform().rotation.yaw)

        rotation_global = np.array([
            [np.sin(yaw_global),  np.cos(yaw_global)],
            [np.cos(yaw_global), -np.sin(yaw_global)]
        ])

        v = self.vehicle.get_velocity()
        velocity_global = np.array([v.y, v.x])
        velocity_local = rotation_global.T @ velocity_global

        return velocity_local[0], velocity_local[1]

    def get_acceleration(self):
        yaw_global = np.radians(self.vehicle.get_transform().rotation.yaw)

        rotation_global = np.array([
            [np.sin(yaw_global),  np.cos(yaw_global)],
            [np.cos(yaw_global), -np.sin(yaw_global)]
        ])

        acc = self.vehicle.get_acceleration()
        acceleration_global = np.array([acc.y, acc.x])
        acceleration_local = rotation_global.T @ acceleration_global

        return acceleration_local[0], acceleration_local[1]


    def next_weather(self, reverse=False):
        self.weather_index += -1 if reverse else 1
        self.weather_index %= len(self.weather_presets)
        preset = self.weather_presets[self.weather_index]
        self.vehicle.get_world().set_weather(preset[0])


    def tick(self, clock):
        pass


    def render(self, display):
        self.camera_manager.render(display)


    def get_depth_frame(self):
        return self.camera_manager.surface_depth


    def get_state(self):
        state = self.depth_to_points(self.get_depth_frame())
        return state


    def depth_to_points(self, image_depth):
        rows, cols = image_depth.shape[:2]
        step_size = cols // self.num_samples
        row = rows // 2
        idxs = [(r, col) for col in range(0, cols, step_size) for r in [row-20, row, row+20]]
        elems = np.array([np.mean(image_depth[row, col]) for (row, col) in idxs])
        return elems


    def get_autopilot_control(self):
        control = self.autopilot_agent.run_step()
        return control


    def destroy(self):
        actors = [self.collision_sensor.sensor, self.camera_manager.sensor, self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()


    def destroy_vehicles(self):
        for actor in self.world.get_actors():
            if isinstance(actor, carla.libcarla.Vehicle):
                actor.destroy()
        time.sleep(0.5)