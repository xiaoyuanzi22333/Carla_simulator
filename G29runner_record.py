#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

from tqdm import tqdm

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# Recorder = False

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

control_speed_loader = []
control_brake_loader = []
control_throttle_loader = []
control_gear_loader = []
control_steer_loader = []

surface_depth_loader = []
surface_RGB_loader = []
surface_Lidar_loader = []
# gnss = (lat, long)
client_Gnss_loader = []
# imu = (acc, gyr)
client_imu_loader = []
# radar = dict{point: data}
client_radar_loader = []
# frame_count, elapsed_seconds, delta_seconds, platform_timestamp
client_time_loader = []

class World(object):
    def __init__(self, carla_world, hud: 'HUD', actor_filter):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        
        # init the gamma
        self._gamma = 2.2
        
        # init two addtional sensors here 
        self.imu_sensor = None
        self.radar_sensor = None
        
        # init the recorder controller
        self.recorder = False
        
        # apply additional camera
        self.camera_depth = None
        self.camera_Lidar = None
        self.camera_RGB = None
        # apply additional camera
        
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        # self.world.on_tick(hud.on_world_tick)
        weak_self = weakref.ref(self)
        self.world.on_tick(lambda world_snapshot: World.data_record(weak_self,world_snapshot))
        # self.world.on_tick(lambda world_snapshot: World.data_record(self,world_snapshot))

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        # =================================
        # apply RGB and Depth camera config
        cam_depth_index = self.camera_depth.index if self.camera_depth is not None else 1
        cam_depth_pos_index = self.camera_depth.transform_index if self.camera_depth is not None else 0

        cam_RGB_index = self.camera_RGB.index if self.camera_RGB is not None else 0
        cam_RGB_pos_index = self.camera_RGB.transform_index if self.camera_RGB is not None else 0

        cam_Lidar_index = self.camera_Lidar.index if self.camera_Lidar is not None else 8
        cam_Lidar_pos_index = self.camera_Lidar.transform_index if self.camera_Lidar is not None else 0
        # apply RGB and Depth and Lidar camera config
        # =================================
        
        
        
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        
        # two addtional Sensor settings
        self.imu_sensor = IMUSensor(self.player)
        self.radar_sensor = RadarSensor(self.player)
        
        
        # =============================
        # add addtional camera to world
        self.camera_depth = CameraManager(self.player, self.hud, self._gamma)
        self.camera_depth.transform_index = cam_depth_pos_index
        self.camera_depth.set_sensor(cam_depth_index, notify=False, recorder="depth")

        self.camera_RGB = CameraManager(self.player, self.hud, self._gamma)
        self.camera_RGB.transform_index = cam_RGB_pos_index
        self.camera_RGB.set_sensor(cam_RGB_index, notify=False, recorder="RGB")

        self.camera_Lidar = CameraManager(self.player, self.hud, self._gamma)
        self.camera_Lidar.transform_index = cam_Lidar_pos_index
        self.camera_Lidar.set_sensor(cam_Lidar_index, notify=False, recorder="Lidar")
        # add addtional camera to world
        # =============================
        
        
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        
        # self.world.on_tick(lambda world_snapshot: World.data_record(weak_self,world_snapshot.timestamp))

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])
        
        
    def data_record(weak_self, world_snapshot):
        self = weak_self()
        if not self:
            return
        timestamp = world_snapshot.timestamp
        if self.recorder:
            print("recording data")
            client_time_loader.append((timestamp.frame_count,\
                timestamp.elapsed_seconds,timestamp.delta_seconds,timestamp.platform_timestamp))
            surface_depth_loader.append(self.camera_depth.surface)
            surface_Lidar_loader.append(self.camera_Lidar.surface)
            surface_RGB_loader.append(self.camera_RGB.surface)
            client_Gnss_loader.append((self.gnss_sensor.lat, self.gnss_sensor.lon))
            client_imu_loader.append((self.imu_sensor.accelerometer, self.imu_sensor.gyroscope))
            client_radar_loader.append(self.radar_sensor.radar_measure)
            
            v = self.player.get_velocity()
            speed = round(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2), 1)
            control_speed_loader.append(speed)
            
            c = self.player.get_control()
            control_brake_loader.append(c.brake)
            control_throttle_loader.append(c.throttle)
            control_gear_loader.append(c.gear)
            control_steer_loader.append(c.steer)
            
            
            
        self.hud.on_world_tick(timestamp)
    
    # def data_record(self, world_snapshot):
    #     self.hud.on_world_tick(world_snapshot.timestamp)
        
        
    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            
            # two addtional sensors
            self.imu_sensor.sensor,
            self.radar_sensor.sensor,
            
            # additional camera
            self.camera_depth.sensor,
            self.camera_RGB.sensor,
            self.camera_Lidar.sensor,
            
            ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self.timer = 0
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        # print()
        # self._steer_idx = int(
        #     self._parser.get('G29 Driving Force Racing Wheel', 'steering_wheel'))
        # self._throttle_idx = int(
        #     self._parser.get('G29 Driving Force Racing Wheel', 'throttle'))
        # self._brake_idx = int(self._parser.get('G29 Driving Force Racing Wheel', 'brake'))
        # self._reverse_idx = int(self._parser.get('G29 Driving Force Racing Wheel', 'reverse'))
        # self._handbrake_idx = int(
        #     self._parser.get('G29 Driving Force Racing Wheel', 'handbrake'))
        
        
        self._steer_idx = 0
        self._throttle_idx = 2
        self._brake_idx = 3
        self._reverse_idx = 4
        self._handbrake_idx = 3



    def parse_events(self, world: World, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()
                elif event.button == 7:
                    world.recorder = True
                    self.timer = time.time()
                # R2 pressed, stop recording and save data
                elif event.button == 6:
                    world.recorder = False
                    print(len(surface_depth_loader))
                    print(len(surface_Lidar_loader))
                    print(len(surface_RGB_loader))
                    print(len(client_Gnss_loader))
                    print(len(client_imu_loader))
                    print(len(client_radar_loader))
                    print(len(control_brake_loader))
                    print(len(control_gear_loader))
                    print(len(control_steer_loader))
                    print(len(control_throttle_loader))
                    print(len(control_speed_loader))

                    print(time.time() - self.timer)
                    
                    root = './output_test_left'
                    if not os.path.exists(root):
                        os.mkdir(root)
                        
                    date_name = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
                    sub_folder = root + '/' + date_name
                    if not os.path.exists(sub_folder):
                        os.mkdir(sub_folder)
                        
                    print('saving data')
                    np.save(sub_folder + "/Gnss_data.npy", client_Gnss_loader)
                    np.save(sub_folder + "/imu_data.npy", client_imu_loader)
                    np.save(sub_folder + "/time_data.npy", client_time_loader)
                    np.save(sub_folder + "/radar_data.npy", client_radar_loader)
                    np.save(sub_folder + "/speed_data.npy", control_speed_loader)
                    np.save(sub_folder + "/brake_data.npy", control_brake_loader)
                    np.save(sub_folder + "/throttle_data.npy", control_throttle_loader)
                    np.save(sub_folder + "/gear_data.npy", control_gear_loader)
                    np.save(sub_folder + '/steer_data.npy', control_steer_loader)
                    # print(sub_folder + "/Gnss_data.npy")     
                    
                    # print('saving imgs')
                    # for idx,item in enumerate(tqdm(surface_depth_loader)):
                    #      pygame.image.save(item, root + '/depth_image'+str(idx)+'.png')
                    # for idx,item in enumerate(tqdm(surface_RGB_loader)):
                    #      pygame.image.save(item, root + '/RGB_image'+str(idx)+'.png')
                    # for idx,item in enumerate(tqdm(surface_Lidar_loader)):
                    #      pygame.image.save(item, root + '/Lidar_image'+str(idx)+'.png')
                    
                    
                    surface_depth_loader.clear()
                    surface_Lidar_loader.clear()
                    surface_RGB_loader.clear()
                    client_Gnss_loader.clear()
                    client_imu_loader.clear()
                    client_radar_loader.clear()
                    client_time_loader.clear()
                    control_brake_loader.clear()
                    control_gear_loader.clear()
                    control_throttle_loader.clear()
                    control_speed_loader.clear()
                    control_steer_loader.clear()
                    
                    print("done!")
                # R3 pressed, stop recording and abandon data
                elif event.button == 10:
                    world.recorder = False
                    surface_depth_loader.clear()
                    surface_Lidar_loader.clear()
                    surface_RGB_loader.clear()
                    client_Gnss_loader.clear()
                    client_imu_loader.clear()
                    client_radar_loader.clear()
                    client_time_loader.clear()
                    control_brake_loader.clear()
                    control_gear_loader.clear()
                    control_throttle_loader.clear()
                    control_speed_loader.clear()
                    control_steer_loader.clear()
                
                #  自动档手动档切换(手柄切换并控制)
                if isinstance(self._control, carla.VehicleControl):
                    if event.button == 5:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.button>=12 and event.button<=17:
                        self._control.gear = int(event.button) - 11
                    elif self._control.manual_gear_shift and event.button == 18:
                        self._control.gear = -1
            
            # 手动档空档设置
            elif event.type == pygame.JOYBUTTONUP:
                if isinstance(self._control, carla.VehicleControl):
                    if self._control.manual_gear_shift and event.button >=12 and event.button <= 18:
                        self._control.gear = 0

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                
                # 自动档手动挡切换(键盘切换并控制)
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                
                

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 0.5  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


surface_depth_loader = []
surface_RGB_loader = []
surface_Lidar_loader = []
# gnss = (lat, long)
client_Gnss_loader = []
# imu = (acc, gyr)
client_imu_loader = []
# radar = dict{point: data}
client_radar_loader = []
# frame_count, elapsed_seconds, delta_seconds, platform_timestamp
client_time_loader = []
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        # add recorder
        self.recorder = False

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
        # print(Recorder)
        # if Recorder:
        #     print(timestamp)

    def tick(self, world: World, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        # world.world.on_tick(lambda world_snapshot: print(world_snapshot))



    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        
        
# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)

# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================
    
        
Radar_data_loader = {}
Radar_time_loader = {}
class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.recorder = False
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        self.radar_measure = None

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))
        
        
    def start_recorder(self):
        self.recorder = True
    
    def end_recorder(self):
        self.recorder = False
    
    
    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))
        radar_measurement = {}
        for idx,detect in enumerate(radar_data):
            radar_dict = {}
            # print(idx)
            # print(detect)
            radar_dict["velocity"] = detect.velocity
            radar_dict["azimuth"] = detect.azimuth
            radar_dict["altitude"] = detect.altitude
            radar_dict["depth"] = detect.depth
            radar_measurement[idx] = radar_dict
        self.radar_measure = radar_measurement
        
        if self.recorder:
            print(radar_data)
            Radar_data_loader[radar_data.frame] = radar_measurement
            Radar_time_loader[radar_data.frame] = radar_data.timestamp
            
            
            

        # current_rot = radar_data.transform.rotation
        # for detect in radar_data:
        #     azi = math.degrees(detect.azimuth)
        #     alt = math.degrees(detect.altitude)
        #     # The 0.25 adjusts a bit the distance so the dots can
        #     # be properly seen
        #     fw_vec = carla.Vector3D(x=detect.depth - 0.25)
        #     carla.Transform(
        #         carla.Location(),
        #         carla.Rotation(
        #             pitch=current_rot.pitch + alt,
        #             yaw=current_rot.yaw + azi,
        #             roll=current_rot.roll)).transform(fw_vec)

        #     def clamp(min_v, max_v, value):
        #         return max(min_v, min(value, max_v))

        #     norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
        #     r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
        #     g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
        #     b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
        #     self.debug.draw_point(
        #         radar_data.transform.location + fw_vec,
        #         size=0.075,
        #         life_time=0.06,
        #         persistent_lines=False,
        #         color=carla.Color(r, g, b))        


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

depth_data_loader = []
depth_time_loader = {}
RGB_data_loader = []
RGB_time_loader = {}

class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self.image = None
        
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        
        # data collection recorder
        self.recorder = False
        # self.depth_data_loader = []
        # self.RGB_data_loader = []
        
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False, recorder=""):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            if recorder == "":
                self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
            elif recorder == "depth":
                self.sensor.listen(lambda image: CameraManager.depth_img_process(weak_self, image))
            elif recorder == "RGB":
                self.sensor.listen(lambda image: CameraManager.RGB_img_process(weak_self, image))
            elif recorder == "Lidar":
                self.sensor.listen(lambda image: CameraManager.Lidar_img_process(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    # start the data recorder
    def start_recorder(self):
        self.recorder = True
        self.hud.notification('data recorder start')
    
    # end the data recorder
    def end_recorder(self):
        self.recorder = False
        self.hud.notification('data recorder end')
        
    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        
        # print(image)
        
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out1/%08d' % image.frame)
            # image = image.get_color_coded_flow()
            # image.save_to_disk('_out2/%08d' % image.frame)

    def depth_img_process(weak_self, image):
        # print("listenning")
        self = weak_self()
        if not self:
            return
        
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        # print(self.surface)
        
        if self.recorder:
            print("process depth img")
            print(image)
            depth_data_loader.append(image)
            depth_time_loader[image.frame] = image.timestamp
            
            
            
    
    def RGB_img_process(weak_self,image):
        self = weak_self()
        if not self:
            return
        
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        if self.recorder:
            print("process RGB img")
            print(image)
            RGB_data_loader.append(image)
            RGB_time_loader[image.frame] = image.timestamp
            
            
    def Lidar_img_process(weak_self,image):
        self = weak_self()
        if not self:
            return

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
        lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        self.surface = pygame.surfarray.make_surface(lidar_img)

        if self.recorder:
            print("process RGB img")
            print(image)
            RGB_data_loader.append(image)
            RGB_time_loader[image.frame] = image.timestamp
    


# class CameraManager(object):
#     def __init__(self, parent_actor, hud):
#         self.sensor = None
#         self.surface = None
#         self._parent = parent_actor
#         self.hud = hud
#         self.recording = False
#         self._camera_transforms = [
#             carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
#             carla.Transform(carla.Location(x=1.6, z=1.7))]
#         self.transform_index = 1
#         self.sensors = [
#             ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
#             ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
#             ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
#             ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
#             ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
#             ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
#                 'Camera Semantic Segmentation (CityScapes Palette)'],
#             ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
#         world = self._parent.get_world()
#         bp_library = world.get_blueprint_library()
#         for item in self.sensors:
#             bp = bp_library.find(item[0])
#             if item[0].startswith('sensor.camera'):
#                 bp.set_attribute('image_size_x', str(hud.dim[0]))
#                 bp.set_attribute('image_size_y', str(hud.dim[1]))
#             elif item[0].startswith('sensor.lidar'):
#                 bp.set_attribute('range', '50')
#             item.append(bp)
#         self.index = None

#     def toggle_camera(self):
#         self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
#         self.sensor.set_transform(self._camera_transforms[self.transform_index])

#     def set_sensor(self, index, notify=True):
#         index = index % len(self.sensors)
#         needs_respawn = True if self.index is None \
#             else self.sensors[index][0] != self.sensors[self.index][0]
#         if needs_respawn:
#             if self.sensor is not None:
#                 self.sensor.destroy()
#                 self.surface = None
#             self.sensor = self._parent.get_world().spawn_actor(
#                 self.sensors[index][-1],
#                 self._camera_transforms[self.transform_index],
#                 attach_to=self._parent)
#             # We need to pass the lambda a weak reference to self to avoid
#             # circular reference.
#             weak_self = weakref.ref(self)
#             self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
#         if notify:
#             self.hud.notification(self.sensors[index][2])
#         self.index = index

#     def next_sensor(self):
#         self.set_sensor(self.index + 1)

#     def toggle_recording(self):
#         self.recording = not self.recording
#         self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

#     def render(self, display):
#         if self.surface is not None:
#             display.blit(self.surface, (0, 0))

#     @staticmethod
#     def _parse_image(weak_self, image):
#         self = weak_self()
#         if not self:
#             return
#         if self.sensors[self.index][0].startswith('sensor.lidar'):
#             points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
#             points = np.reshape(points, (int(points.shape[0] / 4), 4))
#             lidar_data = np.array(points[:, :2])
#             lidar_data *= min(self.hud.dim) / 100.0
#             lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
#             lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
#             lidar_data = lidar_data.astype(np.int32)
#             lidar_data = np.reshape(lidar_data, (-1, 2))
#             lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
#             lidar_img = np.zeros(lidar_img_size)
#             lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
#             self.surface = pygame.surfarray.make_surface(lidar_img)
#         else:
#             image.convert(self.sensors[self.index][1])
#             array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#             array = np.reshape(array, (image.height, image.width, 4))
#             array = array[:, :, :3]
#             array = array[:, :, ::-1]
#             self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
#         if self.recording:
#             image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter)
        controller = DualControl(world, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
