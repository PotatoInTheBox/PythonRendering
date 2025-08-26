

import numpy as np
from renderable_object import RenderableObject
from shaders import skybox_fragment_shader, skybox_vertex_shader


def scene_all():
    # Load only objects needed for this scene
    monkey = RenderableObject.load_new_obj("./models/blender_monkey.obj")
    name = RenderableObject.load_new_obj("./models/name.obj")
    ship = RenderableObject.load_new_obj("./models/ship.obj")
    fox = RenderableObject.load_new_obj("./models/fox.obj", texture_filepath="./textures/colMap.bytes")
    fox_sitting = RenderableObject.load_new_obj("./models/foxSitting.obj", texture_filepath="./textures/colMap.bytes")
    cloud = RenderableObject.load_new_obj("./models/cloud.obj")
    dragon = RenderableObject.load_new_obj("./models/dragon.obj")
    floor = RenderableObject.load_new_obj("./models/floor.obj", texture_filepath="./textures/uvGrid.bytes")
    dave = RenderableObject.load_new_obj("./models/dave.obj", texture_filepath="./textures/daveTex.bytes")
    skybox = RenderableObject.load_new_obj("./models/skybox.obj", texture_filepath="./textures/skyboxTex.bytes")
    skybox.vertex_shader = skybox_vertex_shader
    skybox.fragment_shader = skybox_fragment_shader

    # Transform objects
    monkey.transform.translate([-3,0,-1])
    name.transform.scale([1.8]*3)
    ship.transform.translate([3,-1,1])
    cloud.transform.translate([1, 10, 4])
    fox.transform.translate([2, -4, 5])
    fox_sitting.transform.translate([0, 0.25, 2])
    dragon.transform.translate([-1, -4, 5])
    dragon.transform.rotate([0,np.pi,0])
    floor.transform.translate([0,-0.5,2])
    dave.transform.translate([-1.0,2,0])
    skybox.transform.scale([900]*3)

    return [monkey, name, ship, fox, cloud, fox_sitting, dragon, floor, dave, skybox]


def scene_fox_floor_sky():
    fox_sitting = RenderableObject.load_new_obj("./models/foxSitting.obj", texture_filepath="./textures/colMap.bytes")
    floor = RenderableObject.load_new_obj("./models/floor.obj", texture_filepath="./textures/uvGrid.bytes")
    skybox = RenderableObject.load_new_obj("./models/skybox.obj", texture_filepath="./textures/skyboxTex.bytes")
    skybox.vertex_shader = skybox_vertex_shader
    skybox.fragment_shader = skybox_fragment_shader

    fox_sitting.transform.translate([0, 0.25, 2])
    floor.transform.translate([0,-0.5,2])
    skybox.transform.scale([900]*3)

    return [fox_sitting, floor, skybox]


def scene_sky_only():
    skybox = RenderableObject.load_new_obj("./models/skybox.obj", texture_filepath="./textures/skyboxTex.bytes")
    skybox.vertex_shader = skybox_vertex_shader
    skybox.fragment_shader = skybox_fragment_shader
    skybox.transform.scale([900]*3)
    return [skybox]

def scene_ship_only():
    ship = RenderableObject.load_new_obj("./models/ship.obj")
    return [ship]