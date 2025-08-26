#!/usr/bin/python
# render.py

# The goal of this project is to make a python renderer that can simulate drawing operations.
# This is for practice only.
# We will aim to simulate how lines, squares, and pixels are drawn on a grid.
# (the renderer can toggle between passthrough mode, where it uses the original drawing methods,
# and simulated mode, where it updates an RGB buffer instead)

# WASD to move forward, left, backwards, and right
# Space to go up
# C to go down
# Dragging mouse allows the camera to look left/right/up/down
# V to toggle wireframe
# F to toggle faces
# Z to toggle z buffer view

import profiler
from profiler import Profiler
from renderable_object import RenderableObject
from object_frame_data import ObjectFrameData
from texture import Texture
from transform import Transform
from debug_window import DebugWindow
from config import Config
from config import ConfigEntry
from config import global_config
import vertex_helpers as v
import debug as debug
from shaders import VertexInput, skybox_fragment_shader, skybox_vertex_shader
from shaders import VertexOutput
from shaders import FragmentInput
import scenes

import pygame
import sys
import ctypes
import cv2
from skimage.draw import line
import numpy as np
from typing import Any, List, Optional, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass
# Make windows not scale this window (pixels do have to be perfect)
if sys.platform == "win32":
    # USING_WINDOWS = False
    USING_WINDOWS = True
    ctypes.windll.user32.SetProcessDPIAware()
else:
    USING_WINDOWS = False
# Prepare the config data
# TODO move as many settings as possible into the config
render_config = global_config
debug_win = DebugWindow()

# area_debug = []

# ========== Camera settings ==========
debug_win.create_slider_input_float("CAMERA_SPEED", render_config.camera_speed, min_val=0.001, max_val=1)
debug_win.create_slider_input_float("CAMERA_SENSITIVITY", render_config.camera_sensitivity, min_val=0.01, max_val=20)

# ========== Object initialization ==========
# RENDER_OBJECTS = scenes.scene_all()
# RENDER_OBJECTS = scenes.scene_ship_only()
RENDER_OBJECTS = scenes.scene_all()

# Documenting... We can also use .transform.set_rotation() and .transform.set_scale()

# ========== Wave Settings ==========
DO_WAVE_SHADER = False
debug_win.create_slider_input_float("WAVE_AMPLITUDE", render_config.wave_amplitude, min_val=0.01, max_val=3)  # bigger number = taller wave
debug_win.create_slider_input_float("WAVE_PERIOD", render_config.wave_period, min_val=0.01, max_val=20)  # bigger number = shorter wave
debug_win.create_slider_input_float("WAVE_SPEED", render_config.wave_speed, min_val=0.001, max_val=0.1)  # The speed/increment of the wave, based on frame count

# Camera Settings
debug_win.create_debug_label("Camera Position", render_config.camera_position)
debug_win.create_debug_label("Camera Rotation", render_config.camera_rotation)


# ========== Performance metrics ==========
ENABLE_PROFILER = True
profiler.enabled_profiler = ENABLE_PROFILER
FRAME_LOG_INTERVAL = 60  # log once per 60 frames

# ========== Pixel outline effect ==========
DRAW_PIXEL_BORDER = True  # Toggle to draw a border around pixels
PIXEL_BORDER_SIZE = 1  # Size of the pixel border

# ========== Rendering modes ==========

# ========== Common Colors ==========
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (50, 50, 50)
COLOR_PINK = (255, 105, 180)
COLOR_SLATE_BLUE = (50, 50, 120)

# ========== Global variables ==========
angle = 0

mouse_x = 0
mouse_y = 0

draw_z_buffer = False
do_draw_faces = True
draw_lines = False

frame_count = 0  # count frames rendered so far

hover_triangle_index = -1

    

# Fragment output is simply either the sub-buffer we write to (np.array). Or possibly
# an array in the future. In the future I may want to do it differently.
# Such as possibly outputing as an array or having an opacity value somehow.

@dataclass(slots=True)
class FragmentBuffer():
    sub_rgb_buffer: NDArray[np.uint8]       # (H, W, 3) RGB values
    sub_zbuffer: NDArray[np.float32]        # (H, W) depth values
    update_mask: NDArray[np.bool_]          # (H, W) coverage mask
    position_map: NDArray[np.float32]       # (H, W, 3) interpolated positions
    normals_map: NDArray[np.float32]|None   # (H, W, 3) interpolated normals
    uv_map: NDArray[np.float32]|None        # (H, W, 2) interpolated UVs

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-16)

def get_ortho_projection_matrix(left, right, bottom, top, near, far):
    proj = np.zeros((4, 4))
    proj[0,0] = 2 / (right - left)
    proj[1,1] = 2 / (top - bottom)
    proj[2,2] = -2 / (far - near)
    proj[0,3] = -(right + left) / (right - left)
    proj[1,3] = -(top + bottom) / (top - bottom)
    proj[2,3] = -(far + near) / (far - near)
    proj[3,3] = 1
    return proj

def get_projection_matrix(fov, aspect, near, far):
    # top = np.tan(fov / 2) * near
    # bottom = -top
    # right = top * aspect
    # left = -right
    # return get_ortho_projection_matrix(left, right, bottom, top, near=near, far=far)
    f = 1 / np.tan(fov / 2)
    proj = np.zeros((4, 4))
    proj[0,0] = f / aspect
    proj[1,1] = f
    proj[2,2] = (far + near) / (near - far)
    proj[2,3] = (2 * far * near) / (near - far)
    proj[3,2] = -1
    return proj

def point_in_triangle(pt, v0, v1, v2):
    # Barycentric technique for 2D point-in-triangle
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(pt, v0, v1) < 0.0
    b2 = sign(pt, v1, v2) < 0.0
    b3 = sign(pt, v2, v0) < 0.0

    return ((b1 == b2) and (b2 == b3))

class Renderer:
    def __init__(self) -> None:
        self.width = render_config.screen_width.val
        self.height = render_config.screen_height.val
        # ========== Screen settings ==========
        debug_win.create_debug_label("SCREEN_WIDTH", render_config.screen_width)  # How much width should the window have?
        debug_win.create_debug_label("SCREEN_HEIGHT", render_config.screen_height)  # How much height should the window have?
        debug_win.create_slider_input_int("CELL_SIZE", render_config.cell_size, min_val=1, max_val=20, on_change=lambda _: self.initialize_render_buffers())  # How many pixels big is each raster cell?
        # In degrees, how much can the camera see from left to right?
        debug_win.create_slider_input_float("FOV", render_config.fov, min_val=10, max_val=170, on_change=lambda _: self.initialize_projection_matrix())

        self.initialize_render_scene()
        pygame.init()
        pygame.display.set_caption("Renderer")
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.running = True
    
    def initialize_render_scene(self):
        self.initialize_render_buffers()
        self.initialize_start_positions()
        # Load and create our object which we will render
        self.objects = RENDER_OBJECTS
    
    def initialize_render_buffers(self):
        self.grid_size_x = self.width // render_config.cell_size.val
        self.grid_size_y = self.height // render_config.cell_size.val
        self.cell_size_x = render_config.cell_size.val
        self.cell_size_y = render_config.cell_size.val
        self.create_empty_rgb_buffer()
        self.create_empty_z_buffer()
        self.initialize_projection_matrix()
    
    def initialize_start_positions(self):
        # The camera will need to face -z. So we need to push the camera towards positive z.
        # This is because our object will be at 0.
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        # The camera is facing towards positive z.
    
    def initialize_projection_matrix(self):
        self.projection_matrix = get_projection_matrix(fov=np.radians(render_config.fov.val),aspect=self.grid_size_x/self.grid_size_y,near=0.1,far=1000)
    
    def create_empty_rgb_buffer(self):
        self.rgb_buffer = np.zeros((self.grid_size_y, self.grid_size_x, 3), dtype=np.uint8)
        self.rgb_buffer[:] = COLOR_SLATE_BLUE
    
    def create_empty_z_buffer(self):
        self.z_buffer = np.full((self.grid_size_y, self.grid_size_x), np.inf, dtype=np.float32)

    def _is_bounded(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.grid_size_x and 0 <= y < self.grid_size_y

    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED, width: int = 1) -> None:
        # self.bresenhams_algorithm_draw_line(start, end, color)
        self.draw_line_opencv(start, end, color, width)
        
        
    def draw_line_opencv(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED, width: int = 1) -> None:
        """
        Draws a line on the RGB buffer using OpenCV.
        
        NOTE: It is not z-buffer aware. So it will simply force a line anywhere.

        Args:
            start: (x1, y1) starting pixel coordinates.
            end: (x2, y2) ending pixel coordinates.
            color: RGB color tuple (R, G, B).
            width: Thickness of the line in pixels.
        """
        # Ensure color and buffer are in correct format
        x1, y1 = map(int, np.round(start))
        x2, y2 = map(int, np.round(end))
        cv2.line(self.rgb_buffer, (x1, y1), (x2, y2), color, thickness=width)
    
    def draw_line_skimage(self, start: Tuple[int, int, float], end: Tuple[int, int, float], color: Tuple[int, int, int] = COLOR_RED, width: int = 1) -> None:
        """
        Draw a z-buffer-aware line on the RGB buffer.
        
        NOTE: Currently overwrites the z buffer for some reason
        This means all line drawing should be done before drawing faces.

        Args:
            start: (x, y, z) starting point.
            end: (x, y, z) ending point.
            color: RGB color tuple.
            width: Not used (line thickness unsupported in z-buffer mode).
        """
        # Extract coordinates and depth
        x1, y1, z1 = start
        x2, y2, z2 = end

        # Round to nearest int for rasterization
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))
        
        # Early out: completely outside buffer bounds
        h, w = self.rgb_buffer.shape[:2]
        if (x1 < 0 and x2 < 0) or (y1 < 0 and y2 < 0) or \
        (x1 >= w and x2 >= w) or (y1 >= h and y2 >= h):
            return  # Entire line is outside the viewport

        # Get all pixel positions along the line
        rr, cc = line(y1, x1, y2, x2)
        num_points = len(rr)

        # Interpolate Z values along the line
        z_values = np.linspace(z1, z2, num_points)

        # Clip to buffer bounds
        mask = (rr >= 0) & (rr < self.rgb_buffer.shape[0]) & \
            (cc >= 0) & (cc < self.rgb_buffer.shape[1])
        rr, cc, z_values = rr[mask], cc[mask], z_values[mask]

        # Apply Z-buffer test
        update_mask = z_values < self.z_buffer[rr, cc]
        # Prioritize the line in the z buffer ever so slightly to reduce z fighting
        self.z_buffer[rr[update_mask], cc[update_mask]] = -z_values[update_mask] - 0.0005
        self.rgb_buffer[rr[update_mask], cc[update_mask]] = color

    # https://medium.com/geekculture/bresenhams-line-drawing-algorithm-2e0e953901b3
    @Profiler.timed()
    def bresenhams_algorithm_draw_line(self, start, end, color):
        is_x_flipped = False  # for handling slope < 0
        is_x_dominant_axis = False  # for handling slope > 1
        
        x1 = start[0]
        x2 = end[0]
        y1 = start[1]
        y2 = end[1]
        
        # Handle x1 > x2 edge case (swap (x2, y2) and (x1, y1))
        if x1 > x2:
            # flip start and end
            tx, ty = x1, y1
            x1, y1 = x2, y2
            x2, y2 = tx, ty
        
        # Handle slope < 0
        if y2 - y1 < 0:
            # change (x1, y1) to (x1, -y1) and (x2, y2) to (x2, -y2)
            y1 *= -1
            y2 *= -1
            is_x_flipped = True
        
        # Handle slope > 1
        if y2 - y1 > x2 - x1:
            # exchange x and y values. So x1 becomes y1
            t2, t1 = x2, x1
            x2, x1 = y2, y1
            y2, y1 = t2, t1
            is_x_dominant_axis = True
        
        dx = (x2 - x1)
        dy = (y2 - y1)
        p = 2 * dy - dx
        x = x1
        y = y1
        
        while x <= x2:
            out_x = x
            out_y = y
            # undo "slope > 1" edge case
            if is_x_dominant_axis:
                tmp = out_x
                out_x = out_y
                out_y = tmp
            # undo "slope < 0" edge case
            out_y = out_y if not is_x_flipped else out_y * -1

            self.draw_pixel((int(out_x), int(out_y)), color)
            x += 1
            if p < 0:
                p += 2 * dy
            else:
                p += 2 * (dy - dx)
                y += 1

    @Profiler.timed()
    def draw_square(self, top_left: Tuple[int, int], size: int, color: Tuple[int, int, int] = COLOR_RED) -> None:
        for y in range(top_left[1], top_left[1] + size):
            for x in range(top_left[0], top_left[0] + size):
                if self._is_bounded((x, y)):
                    self.rgb_buffer[y][x] = color

    @Profiler.timed()
    def draw_pixel(self, position: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED) -> None:
        x, y = position
        if self._is_bounded((x,y)):
            self.rgb_buffer[y][x] = color

    @Profiler.timed()
    def draw_circle(self, center: Tuple[int, int], radius: int, color: Tuple[int, int, int] = COLOR_RED) -> None:
        sqrt_limit = radius**2
        for y in range(center[1] - radius, center[1] + radius):
            for x in range(center[0] - radius, center[0] + radius):
                if self._is_bounded((x, y)) and (y - center[1])**2 + (x - center[0])**2 < sqrt_limit:
                    self.rgb_buffer[y][x] = color
    
    def _prepare_triangle(self, p1, p2, p3):
        p1 = (p1[0], p1[1], -p1[2])
        p2 = (p2[0], p2[1], -p2[2])
        p3 = (p3[0], p3[1], -p3[2])
        area = self._edge_fn(p1, p2, p3)
        return area != 0
    
    def _compute_pixel_grid(self, p1, p2, p3):
        min_x = max(int(min(p1[0], p2[0], p3[0])), 0)
        max_x = min(int(max(p1[0], p2[0], p3[0])), self.grid_size_x - 1)
        min_y = max(int(min(p1[1], p2[1], p3[1])), 0)
        max_y = min(int(max(p1[1], p2[1], p3[1])), self.grid_size_y - 1)
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        return min_x, max_x, min_y, max_y, *np.meshgrid(xs, ys)

    @Profiler.timed()
    def _compute_barycentrics(self, X, Y, p1, p2, p3):
        area = self._edge_fn(p1, p2, p3)
        inv_area = 1 / area
        A0, B0, C0 = self._edge_coeffs(p2, p3)
        A1, B1, C1 = self._edge_coeffs(p3, p1)
        A2, B2, C2 = self._edge_coeffs(p1, p2)
        w0 = A0 * X + B0 * Y + C0
        w1 = A1 * X + B1 * Y + C1
        w2 = A2 * X + B2 * Y + C2
        mask = ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        return w0, w1, w2, inv_area, mask

    def _edge_fn(self, a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    def _edge_coeffs(self, p1, p2):
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0]*p2[1] - p1[1]*p2[0]
        return A, B, C
    
    # ChatGPT pipeline:
    # Vertex Shader – transforms per-vertex data (positions, normals, UVs) from object space 
    # → clip space (or equivalent space for projection).

    # Primitive Assembly – groups vertices into triangles/lines/points based on the draw call.

    # Clipping – removes primitives (or portions) outside the view volume.

    # Perspective Divide – converts from clip coords → normalized device coords (NDC).

    # Viewport Transform – maps NDC → screen coordinates.

    # Back-face Culling – discards triangles facing away from the camera (optional, could also happen earlier).

    # Rasterization – converts primitives into fragments via barycentric interpolation.

    # Fragment Shader – runs per fragment, producing the final color/depth values.

    # Output Merger – writes the final values to the framebuffer.

    def draw_triangle(self, p1: Tuple[int,int], p2: Tuple[int, int], p3: Tuple[int, int], color: Tuple[int, int, int] = COLOR_WHITE):
        self.draw_line(p1, p2, color)
        self.draw_line(p2, p3, color)
        self.draw_line(p3, p1, color)
    
    def draw_triangle_with_z(self, p1: Tuple[int, int, float], p2: Tuple[int, int, float], p3: Tuple[int, int, float], color: Tuple[int, int, int] = COLOR_WHITE):
        self.draw_line_skimage(p1, p2, color)
        self.draw_line_skimage(p2, p3, color)
        self.draw_line_skimage(p3, p1, color)

    def append_new_triangle_to_v_buffer(self, original_vertices: np.ndarray, vertices: list[np.ndarray], indices: list[np.ndarray], tri: np.ndarray, idx_A: int, idx_B: int, idx_C: int):
        vertices.extend(tri)
        future_i = len(original_vertices) + len(indices)
        new_tri_idx = np.array([future_i + idx_C, future_i + idx_A, future_i + idx_B], dtype=np.int64)
        indices.append(new_tri_idx)
    
    # This one is a doozy. I need to do way too many things in one function.
    # I need it to receive interpolated values and apply them to any given
    # triangle, then it needs to form a new triangle (by interpolating the
    # vertices).
    @Profiler.timed()
    def clip_append_new(self, original_vertices: np.ndarray, original_indices: np.ndarray, vertices: list, indices: list, clip_idx: int, prev_idx: int, next_idx: int, frac_prev, frac_next, i: int, do_quad=True):
        tri_indices = original_indices[i]
        tri = original_vertices[tri_indices]
        
        v_clipped = tri[clip_idx]
        v_next = tri[next_idx]
        v_prev = tri[prev_idx]
        
        # Basically, we need to make our new triangles, easy enough.
        new_vert_prev = v.lerp(v_clipped, v_prev, frac_prev)
        new_vert_next = v.lerp(v_clipped, v_next, frac_next)
        
        # Then we append them to the list of vertices
        future_i = len(original_vertices) + len(vertices)
        tri_prev_clip_idx = future_i
        tri_next_clip_idx = future_i + 1
        vertices.append(new_vert_prev)
        vertices.append(new_vert_next)
        
        # Now we need to construct two new triangles.
        # We can reuse the position of the previous two for one of them.
        # And reuse the position of one vertex for the other one.
        
        if do_quad:
            new_tri_idx_1 = (tri_indices[prev_idx], np.int32(tri_prev_clip_idx), tri_indices[next_idx])
            new_tri_idx_2 = (tri_prev_clip_idx, np.int32(tri_next_clip_idx), tri_indices[next_idx])
            indices.append(new_tri_idx_1)
            indices.append(new_tri_idx_2)
        else:
            new_tri_idx = (np.int32(tri_prev_clip_idx), tri_indices[clip_idx], np.int32(tri_next_clip_idx))
            indices.append(new_tri_idx)


    # Basically, the objective of this function is to grab our vertices and clip
    # them against the near plane
    # 
    @Profiler.timed()
    def clip_faces(self, obj_frame_data: ObjectFrameData, face_to_clip: np.ndarray):
        # TODO make sure we are only calculating ones we kept with faces_kept
        # faces that aren't facing us are not important.
        # TODO accumilate our new triangles into our buffer.
        # Basically, when we clip we will have new positions/normals/uvs so we need
        # to append them to the `obj_frame_data` vertices. We will also then
        # need to create a new triangle and add that to the `obj_frame_data` faces.
        # This means our buffer will need indices that point into the original vertex
        # buffer as well as a new one.
        v_buffer = v.VertexClipBuffer()
        for i in (face_to_clip):
            # process_triangle(i, faces[i])
            tri_indices = obj_frame_data.vertex_faces[i]
            tri = obj_frame_data.clip_space_vertices[tri_indices]
            
            near_clip = 0.01
            clip = (tri[:, 2] < near_clip)
            clip_count = clip.sum()
            if clip_count == 0:
                # # Lucky us, the issue is resolved (likely a different triangle
                # # already clipped for us). Let's accept this triangle now.
                # obj_frame_data.faces_kept[i] = True

                # TODO
                # Not lucky us. If this happens then we likely lost information
                raise Exception("You messed up. You modified a partially \
                                behind triangle earlier and now we can't \
                                properly rebuild this one!")
                pass
            elif clip_count == 1:
                clip_idx = np.where(clip)[0][0]
                index_next = (clip_idx + 1) % 3
                index_prev = (clip_idx - 1 + 3) % 3
                
                v_clipped = tri[clip_idx]
                v_next = tri[index_next]
                v_prev = tri[index_prev]
                
                frac_next = (near_clip - v_clipped[2]) / (v_next[2] - v_clipped[2])
                frac_prev = (near_clip - v_clipped[2]) / (v_prev[2] - v_clipped[2])
                
                self.clip_append_new(obj_frame_data.clip_space_vertices, obj_frame_data.vertex_faces, 
                                          v_buffer.positions, v_buffer.p_indices, 
                                          clip_idx, index_prev, index_next, 
                                          frac_prev, frac_next, i)
                
                v_buffer.face_normals.append(obj_frame_data.face_normals[i])
                v_buffer.face_normals.append(obj_frame_data.face_normals[i])
                
                if obj_frame_data.vertex_normals is not None and len(obj_frame_data.vertex_normals) > 0:
                    self.clip_append_new(obj_frame_data.vertex_normals, obj_frame_data.normal_faces, 
                                          v_buffer.normals, v_buffer.n_indices, 
                                          clip_idx, index_prev, index_next, 
                                          frac_prev, frac_next, i)
                
                # # build the new uv vertices (append)
                if obj_frame_data.uv_coords is not None and len(obj_frame_data.uv_coords) > 0:
                    self.clip_append_new(obj_frame_data.uv_coords, obj_frame_data.uv_faces, 
                                          v_buffer.uvs, v_buffer.t_indices, 
                                          clip_idx, index_prev, index_next, 
                                          frac_prev, frac_next, i)
                    
                pass
            elif clip_count == 2:  # shorten existing triangle
                # pseudocode...
                # figure out the vertex that will not be clipped
                # figure out the vertex after this one
                # figure out the vertex before this one
                no_clip_idx = np.where(~clip)[0][0]
                index_next = (no_clip_idx + 1) % 3
                index_prev = (no_clip_idx - 1 + 3) % 3
                
                v_not_clipped = tri[no_clip_idx]
                v_next = tri[index_next]
                v_prev = tri[index_prev]
                
                frac_next = (near_clip - v_not_clipped[2]) / (v_next[2] - v_not_clipped[2])
                frac_prev = (near_clip - v_not_clipped[2]) / (v_prev[2] - v_not_clipped[2])
                
                self.clip_append_new(obj_frame_data.clip_space_vertices, obj_frame_data.vertex_faces, 
                                          v_buffer.positions, v_buffer.p_indices, 
                                          no_clip_idx, index_prev, index_next, 
                                          frac_prev, frac_next, i, do_quad=False)
                
                v_buffer.face_normals.append(obj_frame_data.face_normals[i])
                
                if obj_frame_data.vertex_normals is not None and len(obj_frame_data.vertex_normals) > 0:
                    self.clip_append_new(obj_frame_data.vertex_normals, obj_frame_data.normal_faces, 
                                          v_buffer.normals, v_buffer.n_indices, 
                                          no_clip_idx, index_prev, index_next, 
                                          frac_prev, frac_next, i, do_quad=False)
                
                # # build the new uv vertices (append)
                if obj_frame_data.uv_coords is not None and len(obj_frame_data.uv_coords) > 0:
                    self.clip_append_new(obj_frame_data.uv_coords, obj_frame_data.uv_faces, 
                                          v_buffer.uvs, v_buffer.t_indices, 
                                          no_clip_idx, index_prev, index_next, 
                                          frac_prev, frac_next, i, do_quad=False)
                
                pass
            else:
                # This means that a non-culled triangle got here.
                # We need to make sure we have it culled earlier.
                # raise Exception("How did we get here?")
                pass
            pass
        # Now we need to build our new triangles
        if len(v_buffer.positions) != 0:
            obj_frame_data.clip_space_vertices = np.vstack([obj_frame_data.clip_space_vertices, np.array(v_buffer.positions, dtype=np.float64)])
        if obj_frame_data.vertex_normals is not None and len(v_buffer.normals) != 0:
            obj_frame_data.vertex_normals = np.vstack([obj_frame_data.vertex_normals, np.array(v_buffer.normals, dtype=np.float64)])
        if obj_frame_data.uv_coords is not None and len(v_buffer.uvs) != 0:
            obj_frame_data.uv_coords = np.vstack([obj_frame_data.uv_coords, np.array(v_buffer.uvs, dtype=np.float64)])
        if obj_frame_data.face_normals is not None and len(v_buffer.face_normals) != 0:
            obj_frame_data.face_normals = np.vstack([obj_frame_data.face_normals, np.array(v_buffer.face_normals, dtype=np.float64)])
        pass
    
        # Add indices
        if len(v_buffer.p_indices) != 0:
            obj_frame_data.vertex_faces = np.vstack([obj_frame_data.vertex_faces, np.array(v_buffer.p_indices, dtype=np.int32)])
        if len(v_buffer.n_indices) != 0:
            obj_frame_data.normal_faces = np.vstack([obj_frame_data.normal_faces, np.array(v_buffer.n_indices, dtype=np.int32)])
        if len(v_buffer.t_indices) != 0:
            obj_frame_data.uv_faces = np.vstack([obj_frame_data.uv_faces, np.array(v_buffer.t_indices, dtype=np.int32)])
        
        # Extend faces_kept
        obj_frame_data.faces_kept = np.concatenate([obj_frame_data.faces_kept, np.ones(len(v_buffer.p_indices), dtype=bool)])
    
        pass

    @Profiler.timed()
    def draw_polygons(self):
        """
        Main rendering pipeline function.

        Handles all steps:
            - View and model transformations
            - Vertex processing (e.g., wave shader)
            - Projection to clip and NDC space
            - Face culling, lighting, and screen conversion
            - Final drawing of faces
        """
        global angle
        
        # ========= Setup =========
        # Lights should go FROM the object TO the light.
        # This means the light will look like it's going down even though the matrix points up.
        light = np.array([0, 1, 0])

        view_matrix = v.compute_view_matrix()

        for r_object in self.objects:
            Profiler.profile_accumulate_start("draw_polygons: prepare")
            obj_frame_data = ObjectFrameData(r_object)
            # Copy will likely give us some overhead but I don't want to deal
            # with this right now.
            # TODO
            obj_frame_data.texture = r_object.texture
            obj_frame_data.uv_coords = r_object.uv_coords.copy()
            obj_frame_data.vertex_normals = r_object.normals.copy()
            obj_frame_data.vertex_faces = r_object.faces.copy()
            obj_frame_data.normal_faces = r_object.normal_faces.copy()
            obj_frame_data.uv_faces = r_object.uv_faces.copy()
            
            obj_frame_data.homogeneous_vertices = v.to_homogeneous_vertices(r_object)
            obj_frame_data.model_matrix = v.get_model_matrix(r_object, angle)
            m = obj_frame_data.model_matrix
            mv = view_matrix @ obj_frame_data.model_matrix
            mvp = self.projection_matrix @ view_matrix @ obj_frame_data.model_matrix
            Profiler.profile_accumulate_end("draw_polygons: prepare")
            
            Profiler.profile_accumulate_start("draw_polygons: vertex shader")
            # Load shader values
            vertex_in = VertexInput(m, view_matrix, self.projection_matrix, mv, mvp, obj_frame_data.homogeneous_vertices)
            if len(obj_frame_data.vertex_normals) > 0:
                vertex_in.normal = obj_frame_data.vertex_normals
            if len(obj_frame_data.uv_coords) > 0:
                vertex_in.uv = obj_frame_data.uv_coords
            
            # Run the shader
            vertex_out: VertexOutput = r_object.vertex_shader(vertex_in)
            
            # Extract shader values and place into buffer (via reference)
            # NOTE we don't want to use vertex_out in the code later on.
            # We simply use vertex_out to grab our data. Not to continue passing
            # it around.
            obj_frame_data.clip_space_vertices = vertex_out.clip_position
            obj_frame_data.world_space_vertices = vertex_out.world_position
            if vertex_out.face_normals is not None:
                obj_frame_data.face_normals = vertex_out.face_normals
            if vertex_out.world_normal is not None:
                obj_frame_data.world_normals = vertex_out.world_normal
            if vertex_out.uv is not None:
                obj_frame_data.uv_coords = vertex_out.uv
            

            Profiler.profile_accumulate_end("draw_polygons: vertex shader")

            Profiler.profile_accumulate_start("draw_polygons: cull")
            # TODO compute these in the vertex shader
            obj_frame_data.world_space_triangles = v.compute_world_triangles(obj_frame_data.world_space_vertices, r_object.faces)
            obj_frame_data.face_normals = v.compute_normals(obj_frame_data.world_space_triangles)
            
            # obj_frame_data.face_shade = v.compute_lighting(obj_frame_data.face_normals, light)
            obj_frame_data.faces_kept, partially_behind_z_idx = v.cull_faces(obj_frame_data.clip_space_vertices, r_object.faces) # type: ignore
            
            self.clip_faces(obj_frame_data, partially_behind_z_idx)
            
            Profiler.profile_accumulate_end("draw_polygons: cull")
            
            # filter out the faces we do not want to draw, keep only the ones we want to draw.
            Profiler.profile_accumulate_start("draw_polygons: assemble")
            self.assemble(obj_frame_data, obj_frame_data.vertex_faces, obj_frame_data.normal_faces, obj_frame_data.uv_faces, obj_frame_data.face_normals)
            Profiler.profile_accumulate_end("draw_polygons: assemble")
            
            V_ndc, inv_w = v.perspective_divide(obj_frame_data.clip_space_vertices)
            obj_frame_data.ndc_vertices = V_ndc
            obj_frame_data.inverse_w = inv_w
            
            # TODO ndc_to_screen is doing too much. It should not be creating
            # triangles.
            obj_frame_data.screen_space_triangles = v.ndc_to_screen(obj_frame_data.ndc_vertices, obj_frame_data.vertex_faces, self.grid_size_x, self.grid_size_y)
            
            # At this point we MUST go through the triangles individually
            # Loop through the triangles we are left with. No triangle means
            # no draw. (normal and uv optional)
            # TODO access screen space from something like `obj_frame_data.screen_space_vertices`
            for i, _ in enumerate(obj_frame_data.vertex_faces):
                triangle_screen_vertices = obj_frame_data.screen_space_triangles[i]
                triangle_normals = self.get_normal_triangle(obj_frame_data.vertex_normals, obj_frame_data.normal_faces, i)
                if obj_frame_data.uv_coords is not None:
                    triangle_uvs = self.get_uv_triangle(obj_frame_data.uv_coords, obj_frame_data.uv_faces, i)
                else:
                    triangle_uvs = None
                triangle_ws = self.get_normal_triangle(obj_frame_data.inverse_w, obj_frame_data.vertex_faces, i)
                frag_buffer = self.rasterize(triangle_screen_vertices, triangle_normals, triangle_uvs, triangle_ws)
                
                if do_draw_faces and frag_buffer is not None:
                    # self.fill_triangle_2(frag_buffer, r_object, obj_frame_data.texture)
                    Profiler.profile_accumulate_start("draw_polygons: fragment shader")
                    if obj_frame_data.face_normals is not None:
                        face_normal = obj_frame_data.face_normals[i]
                    else:
                        face_normal = np.array([0,0,0], dtype=np.float64)  # failsafe
                    f_in = FragmentInput(frag_buffer.position_map, face_normal, frag_buffer.normals_map, frag_buffer.uv_map, obj_frame_data.texture)
                    f_out: np.ndarray = r_object.fragment_shader(f_in)
                    Profiler.profile_accumulate_end("draw_polygons: fragment shader")
                    # ignore alpha channel for now
                    frag_buffer.sub_rgb_buffer[frag_buffer.update_mask] = (f_out[..., :3][frag_buffer.update_mask] * 255).astype(np.uint8)
                    if draw_lines:
                        self.draw_triangle(triangle_screen_vertices[0][0:2], triangle_screen_vertices[1][0:2], triangle_screen_vertices[2][0:2], COLOR_GREEN)
    
    @Profiler.timed()
    def bary_terpolate_2d(self, w0, w1, w2, triangle, triangle_ws):
        (x1, y1), (x2, y2), (x3, y3) = triangle
        inv_w1, inv_w2, inv_w3 = triangle_ws
        u_w = w0 * x1 * inv_w1 + w1 * x2 * inv_w2 + w2 * x3 * inv_w3
        v_w = w0 * y1 * inv_w1 + w1 * y2 * inv_w2 + w2 * y3 * inv_w3

        # Shared perspective correction factor
        inv_w = w0 * inv_w1 + w1 * inv_w2 + w2 * inv_w3

        # Apply perspective correction
        x = u_w / inv_w
        y = v_w / inv_w

        return np.stack((x, y), axis=-1)
    
    @Profiler.timed()
    def bary_terpolate_3d_small(self, w0, w1, w2, triangle, triangle_ws):
        """
        Barycentric interpolation optimized for small grids (H*W <= ~6).
        Returns result of shape (H, W, 3).
        """
        H, W = w0.shape

        # Unpack triangle and weights
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = triangle
        inv_w0, inv_w1, inv_w2 = triangle_ws

        # Pre-multiply weights
        w0_inv = w0 * inv_w0
        w1_inv = w1 * inv_w1
        w2_inv = w2 * inv_w2

        # Preallocate output
        result = np.empty((H, W, 3), dtype=np.float32)

        # Flat loop over small grid
        for i in range(H):
            for j in range(W):
                wi0 = w0_inv[i,j]
                wi1 = w1_inv[i,j]
                wi2 = w2_inv[i,j]
                inv_sum = wi0 + wi1 + wi2

                result[i,j,0] = (wi0*x1 + wi1*x2 + wi2*x3) / inv_sum
                result[i,j,1] = (wi0*y1 + wi1*y2 + wi2*y3) / inv_sum
                result[i,j,2] = (wi0*z1 + wi1*z2 + wi2*z3) / inv_sum

        return result

    # NOTE: as of commit e6b7c06 on my PC it takes ~3ms to process the profiler
    # out of the 23.5ms it took over 1540 calls.
    @Profiler.timed()
    def bary_terpolate_3d(self, w0, w1, w2, triangle, triangle_ws):
        
        H, W = w0.shape
        # area_debug.append(H * W)
        
        if W * H <= 10:
            return self.bary_terpolate_3d_small(w0, w1, w2, triangle, triangle_ws)
        
        Profiler.profile_accumulate_start("bary_terpolate_3d: unpack")
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = triangle
        inv_w0, inv_w1, inv_w2 = triangle_ws
        Profiler.profile_accumulate_end("bary_terpolate_3d: unpack")

        Profiler.profile_accumulate_start("bary_terpolate_3d: invert weights")
        w0_inv = w0 * inv_w0
        w1_inv = w1 * inv_w1
        w2_inv = w2 * inv_w2
        Profiler.profile_accumulate_end("bary_terpolate_3d: invert weights")

        Profiler.profile_accumulate_start("bary_terpolate_3d: create grid")
        u_w = w0_inv*x1 + w1_inv*x2 + w2_inv*x3
        v_w = w0_inv*y1 + w1_inv*y2 + w2_inv*y3
        w_w = w0_inv*z1 + w1_inv*z2 + w2_inv*z3
        Profiler.profile_accumulate_end("bary_terpolate_3d: create grid")
        
        # Sanity check to make sure our performance isn't being lost by a profiler
        Profiler.profile_accumulate_start("bary_terpolate_3d: do nothing")
        Profiler.profile_accumulate_end("bary_terpolate_3d: do nothing")

        Profiler.profile_accumulate_start("bary_terpolate_3d: create inverse grid")
        inv_w = w0_inv + w1_inv + w2_inv
        Profiler.profile_accumulate_end("bary_terpolate_3d: create inverse grid")
        
        Profiler.profile_accumulate_start("bary_terpolate_3d: create numpy stack")
        result = np.empty(w0.shape + (3,), dtype=np.float32)
        result[...,0] = u_w / inv_w
        result[...,1] = v_w / inv_w
        result[...,2] = w_w / inv_w
        Profiler.profile_accumulate_end("bary_terpolate_3d: create numpy stack")

        return result
    
    @Profiler.timed()
    def rasterize(self, triangle_screen_vertices, triangle_normals, triangle_uvs, triangle_ws):
        Profiler.profile_accumulate_start("rasterize: barycentric setup")
        p1, p2, p3 = triangle_screen_vertices
        area = self._edge_fn(p1, p2, p3)
        if area == 0: 
            Profiler.profile_accumulate_end("rasterize: barycentric setup")
            return None  # skip degenerate triangle

        # Bounding box and pixel grid
        min_x, max_x, min_y, max_y, X, Y = self._compute_pixel_grid(p1, p2, p3)
        if min_x > max_x or min_y > max_y: 
            Profiler.profile_accumulate_end("rasterize: barycentric setup")
            return None # Triangle is completely outside screen bounds

        # Barycentric weights
        w0, w1, w2, inv_area, mask = self._compute_barycentrics(X, Y, p1, p2, p3)
        w0_n, w1_n, w2_n = w0 * inv_area, w1 * inv_area, w2 * inv_area
        Profiler.profile_accumulate_end("rasterize: barycentric setup")
        
        Profiler.profile_accumulate_start("rasterize: buffer setup")
        # Z-depth buffer
        z = w0_n * p1[2] + w1_n * p2[2] + w2_n * p3[2]
        sub_zbuffer = self.z_buffer[min_y:max_y+1, min_x:max_x+1]
        sub_rgb = self.rgb_buffer[min_y:max_y+1, min_x:max_x+1]
        update_mask = mask & (z < sub_zbuffer)
        sub_zbuffer[update_mask] = z[update_mask]
        Profiler.profile_accumulate_end("rasterize: buffer setup")
        
        Profiler.profile_accumulate_start("rasterize: interpolate")
        pos_map = self.bary_terpolate_3d(w0_n, w1_n, w2_n, triangle_screen_vertices, triangle_ws)
        if triangle_normals is not None:
            # TODO can be optimized by just not calling it. Make absolutely sure that
            # flat shaded objects have `triangle_normals` set to none so we can skip
            # this expensive interpolation.
            normals_map = self.bary_terpolate_3d(w0_n, w1_n, w2_n, triangle_normals, triangle_ws)
        else: normals_map = None
        if triangle_uvs is not None:
            uv_map = self.bary_terpolate_2d(w0_n, w1_n, w2_n, triangle_uvs, triangle_ws)
        else: uv_map = None
        Profiler.profile_accumulate_end("rasterize: interpolate")
        
        Profiler.profile_accumulate_start("rasterize: build fragment buffer")
        fragment_buffer = FragmentBuffer(sub_rgb, sub_zbuffer, update_mask, pos_map, normals_map, uv_map)
        Profiler.profile_accumulate_end("rasterize: build fragment buffer")
        return fragment_buffer
    
    def get_normal_triangle(self, vertex_normals: NDArray[np.float64], normal_faces: NDArray[np.int32], index: int) -> Optional[NDArray[np.float64]]:
        if index < 0 or index >= len(normal_faces):
            return None
        tri_idx = normal_faces[index]
        if np.any(tri_idx < 0):
            return None
        try:
            return vertex_normals[tri_idx]
        except IndexError:
            return None
    
    def get_uv_triangle(self, uv_coords: NDArray[np.float64], uv_faces: NDArray[np.int32], index: int) -> Optional[NDArray[np.float64]]:
        if index < 0 or index >= len(uv_faces):
            return None
        tri_idx = uv_faces[index]
        if np.any(tri_idx < 0):
            return None
        try:
            return uv_coords[tri_idx]
        except IndexError:
            return None

    def assemble(self, obj_frame_data: ObjectFrameData, vertex_faces: np.ndarray, normal_faces: np.ndarray, uv_faces: np.ndarray, face_normals: np.ndarray):
        obj_frame_data.vertex_faces = vertex_faces[obj_frame_data.faces_kept]
        if normal_faces is not None and len(normal_faces) == len(obj_frame_data.faces_kept):
            obj_frame_data.normal_faces = normal_faces[obj_frame_data.faces_kept]
        if uv_faces is not None and len(uv_faces) == len(obj_frame_data.faces_kept):
            obj_frame_data.uv_faces = uv_faces[obj_frame_data.faces_kept]
        if face_normals is not None and len(face_normals) == len(obj_frame_data.faces_kept):
            obj_frame_data.face_normals = face_normals[obj_frame_data.faces_kept]

    @Profiler.timed("render_buffer")
    def render_buffer(self):
        global draw_z_buffer  # Toggle this to enable/disable Z buffer debug view
        # DEBUG_Z_MIN = -100  # how close we can see
        # DEBUG_Z_MAX = 100 # how far we can see
        if draw_z_buffer:
            # Normalize Z buffer to [0, 255] for display
            finite_z = self.z_buffer[np.isfinite(self.z_buffer)]
            if finite_z.size == 0:
                return  # Nothing to render

            z_min, z_max = finite_z.min(), finite_z.max()
            if z_min == z_max:
                z_max += 1e-5  # Prevent divide by zero
            
            z_range = z_max - z_min
            z_scaled = np.clip((z_max - self.z_buffer) / z_range, 0, 1)
            z_norm = (z_scaled * 205 + 50).astype(np.uint8)  # near = bright, far = dark
            z_norm[~np.isfinite(self.z_buffer)] = 0  # +inf → black (far away)

            z_gray = np.stack([z_norm] * 3, axis=-1)
            surface = pygame.surfarray.make_surface(z_gray.swapaxes(0, 1))
        else:
            surface = pygame.surfarray.make_surface(self.rgb_buffer.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))

        if DRAW_PIXEL_BORDER and self.cell_size_x > 2 and self.cell_size_y > 2:
            pixel_border_color = COLOR_DARK_GRAY
            for x in range(self.grid_size_x):
                pygame.draw.line(self.screen, pixel_border_color, (x * self.cell_size_x, 0), (x * self.cell_size_x, self.height), PIXEL_BORDER_SIZE)
            for y in range(self.grid_size_x):
                pygame.draw.line(self.screen, pixel_border_color, (0, y * self.cell_size_y), (self.width, y * self.cell_size_y), PIXEL_BORDER_SIZE)

    def run(self):
        while self.running:
            global frame_count
            frame_count += 1  # start of the next frame
            self.screen.fill((0, 0, 0))

            if True:
                self.draw_polygons()

                mx, my = pygame.mouse.get_pos()
                
                mx //= self.cell_size_x
                my //= self.cell_size_y

                global mouse_x
                mouse_x = mx
                global mouse_y
                mouse_y = my

                # Handle mouse click to cycle active point
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    # ====== Camera rotation stuff ======
                    if event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 1:  # Left click
                                self.dragging = True
                                self.last_mouse_pos = event.pos
                                global hover_triangle_index
                                print(f"Red triangle index is {hover_triangle_index}")
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            self.dragging = False
                    elif event.type == pygame.MOUSEMOTION and self.dragging:
                        x, y = event.pos
                        dx = x - self.last_mouse_pos[0]
                        dy = y - self.last_mouse_pos[1]
                        self.last_mouse_pos = (x, y)

                        render_config.camera_rotation.val[1] -= dx * 0.005 * render_config.camera_sensitivity.val  # yaw (Y axis)
                        render_config.camera_rotation.val[0] -= dy * 0.005 * render_config.camera_sensitivity.val  # pitch (X axis)
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_z:
                        global draw_z_buffer
                        draw_z_buffer = not draw_z_buffer
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                        global draw_lines
                        draw_lines = not draw_lines
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                        global do_draw_faces
                        do_draw_faces = not do_draw_faces
                keys = pygame.key.get_pressed()
                move_dir = np.array([0.0, 0.0, 0.0])
                if keys[pygame.K_LSHIFT]: 
                    move_boost = 5
                else: 
                    move_boost = 1
                if keys[pygame.K_w]: move_dir[2] -= 1
                if keys[pygame.K_s]: move_dir[2] += 1
                if keys[pygame.K_a]: move_dir[0] -= 1
                if keys[pygame.K_d]: move_dir[0] += 1
                if keys[pygame.K_SPACE]: move_dir[1] += 1
                if keys[pygame.K_c]: move_dir[1] -= 1
                global angle
                if keys[pygame.K_RIGHT]: angle += 0.05
                if keys[pygame.K_LEFT]: angle -= 0.05

                if np.linalg.norm(move_dir) > 0:
                    # move_dir = move_dir / np.linalg.norm(move_dir) * self.camera_speed
                    move_dir = move_dir / np.linalg.norm(move_dir) * render_config.camera_speed.val * move_boost

                    # Camera rotation to world space
                    pitch, yaw, roll = render_config.camera_rotation.val
                    # Rx = rotation_matrix_x(pitch)
                    # Ry = rotation_matrix_y(yaw)
                    # Rz = rotation_matrix_z(roll)
                    # R_cam = Rz @ Ry @ Rx
                    R_cam = Transform().with_rotation([pitch, yaw, roll])
                    move_world = R_cam @ Transform(translation=move_dir)
                    render_config.camera_position.val += move_world.get_matrix()[:3, 3]

            self.render_buffer()
            
            if frame_count % 60 == 0:
                Profiler.profile_accumulate_report(intervals=60)
            
            # Clear the RGB buffer for the next frame
            self.create_empty_rgb_buffer()
            self.create_empty_z_buffer()

            pygame.display.flip()
            self.clock.tick(60)
            if USING_WINDOWS:
                debug_win.render_ui()
                debug_win.apply_pending_updates()

        pygame.quit()
    

if __name__ == "__main__":
    renderer = Renderer()
    renderer.run()

