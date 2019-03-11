import weakref

import carla
import numpy as np
import pygame

from carla import ColorConverter as cc

# from carla_env.envs.carla_env.CarlaEnv import image_size
# TODO: Get this from somewhere else
image_size = (1280, 720)


class CameraManager(object):

    def __init__(self, parent_actor):
        self.sensor = None
        self.surface = None
        self.surface_np = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.parent = parent_actor
        self.recording = False
        self.camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=0.8, z=1.7))]
        self.transform_index = 2
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)']]
        world = self.parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(image_size[0]))
                bp.set_attribute('image_size_y', str(image_size[1]))
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self.camera_transforms)
        self.sensor.set_transform(self.camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = (True if self.index is None
            else self.sensors[index][0] != self.sensors[self.index][0])

        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None

            self.sensor = self.parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self.camera_transforms[self.transform_index],
                attach_to=self.parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            print(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        print('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        pass


    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.surface_np = array
        self.surface = pygame.surfarray.make_surface(array[:, :, ::-1].swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame_number)
