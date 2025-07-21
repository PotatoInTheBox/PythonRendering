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
from transform import Transform
from debug_window import DebugWindow
from config import Config
from config import ConfigEntry

import pygame
import sys
import ctypes
import cv2
from skimage.draw import line
import numpy as np
from typing import List, Tuple
# Make windows not scale this window (pixels do have to be perfect)
if sys.platform == "win32":
    ctypes.windll.user32.SetProcessDPIAware()
    
# Prepare the config data
# TODO move as many settings as possible into the config
render_config = Config()
debug_win = DebugWindow()

# ========== Camera settings ==========
debug_win.create_slider_input_float("CAMERA_SPEED", render_config.camera_speed, min_val=0.001, max_val=1)
debug_win.create_slider_input_float("CAMERA_SENSITIVITY", render_config.camera_sensitivity, min_val=0.01, max_val=20)

# ========== Object initialization ==========
MONKEY_OBJ = RenderableObject.load_new_obj("./models/blender_monkey.obj")
NAME_OBJ = RenderableObject.load_new_obj("./models/name.obj")
SHIP_OBJ = RenderableObject.load_new_obj("./models/ship.obj")

MONKEY_OBJ.transform = MONKEY_OBJ.transform.with_translation([-3,0,-1])  # we will have this on the left of our initial camera (slightly further)
NAME_OBJ.transform = NAME_OBJ.transform.with_translation([0,0,0])  # We will have this at the origin
NAME_SCALE = 1.8
NAME_OBJ.transform = NAME_OBJ.transform.with_scale([NAME_SCALE,NAME_SCALE,NAME_SCALE])
SHIP_OBJ.transform = SHIP_OBJ.transform.with_translation([3,-1,1])  # we will have this on the right of our initial camera (slightly closer) (slightly up)
# Documenting... We can also use .transform.set_rotation() and .transform.set_scale()

# ========== Wave Settings ==========
debug_win.create_slider_input_float("WAVE_AMPLITUDE", render_config.wave_amplitude, min_val=0.01, max_val=3)  # bigger number = taller wave
debug_win.create_slider_input_float("WAVE_PERIOD", render_config.wave_period, min_val=0.01, max_val=20)  # bigger number = shorter wave
debug_win.create_slider_input_float("WAVE_SPEED", render_config.wave_speed, min_val=0.001, max_val=0.1)  # The speed/increment of the wave, based on frame count

RENDER_OBJECTS = [MONKEY_OBJ, NAME_OBJ, SHIP_OBJ]  # all the objects we want rendered

# ========== Performance metrics ==========
ENABLE_PROFILER = True
profiler.enabled_profiler = ENABLE_PROFILER
FRAME_LOG_INTERVAL = 60  # log once per 60 frames

# ========== Pixel outline effect ==========
DRAW_PIXEL_BORDER = True  # Toggle to draw a border around pixels
PIXEL_BORDER_SIZE = 1  # Size of the pixel border

# ========== Common Colors ==========
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (50, 50, 50)
COLOR_PINK = (255, 105, 180)
COLOR_SLATE_BLUE = (50, 50, 120)

# ========== Rendering modes ==========

# ========== Global variables ==========
angle = 0

mouse_x = 0
mouse_y = 0

draw_z_buffer = False
draw_faces = True
draw_lines = False

frame_count = 0  # count frames rendered so far

hover_triangle_index = -1

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-16)

def get_projection_matrix(fov, aspect, near, far):
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
        self.camera_pos = render_config.CAMERA_START_POSITION.val
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        # The camera is facing towards positive z.
        self.camera_rot = render_config.CAMERA_START_ROTATION.val
    
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

    # 500x500 triangles cost 0.8ms to draw (not great)
    @Profiler.timed()
    def fill_triangle(self, p1: Tuple[float,float,float], p2: Tuple[float,float,float], p3: Tuple[float,float,float], color: Tuple[int, int, int] = COLOR_RED):
        def edge(a, b, c):
            # Computes twice the signed area of triangle abc.
            # Used to calculate barycentric coordinates and inside-triangle tests.
            return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
        
        area = edge(p1, p2, p3)  # Precompute this outside the loop
        if area == 0:
            return  # skip degenerate triangle
        inv_area = 1/ area

        # Compute integer bounding box of the triangle, clipped to screen size
        min_x = max(int(min(p1[0], p2[0], p3[0])), 0)
        max_x = min(int(max(p1[0], p2[0], p3[0])), self.grid_size_x - 1)
        min_y = max(int(min(p1[1], p2[1], p3[1])), 0)
        max_y = min(int(max(p1[1], p2[1], p3[1])), self.grid_size_y - 1)

        # width = max_x - min_x + 1
        # height = max_y - min_y + 1

        # if width <= 0 or height <= 0:
        #     return
        
        if min_x > max_x or min_y > max_y:
            return  # Triangle is completely outside screen bounds
        
        # Create grid of pixel centers for the bounding box
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        X, Y = np.meshgrid(xs, ys)  # X and Y arrays represent pixel coordinates
        
        # Compute edge function coefficients: E(x, y) = A*x + B*y + C
        # Each edge function checks whether a point is to the left or right of an edge
        def edge_coeffs(p1, p2):
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = p1[0]*p2[1] - p1[1]*p2[0]
            return A, B, C

        A0, B0, C0 = edge_coeffs(p2, p3)
        A1, B1, C1 = edge_coeffs(p3, p1)
        A2, B2, C2 = edge_coeffs(p1, p2)
        
        # Compute edge function values for all pixels at once (vectorized)
        w0 = A0 * X + B0 * Y + C0
        w1 = A1 * X + B1 * Y + C1
        w2 = A2 * X + B2 * Y + C2
        
        # Mask of pixels inside the triangle: all edge signs match (either all >= 0 or all <= 0)
        mask = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        
        # Compute barycentric coordinates by normalizing edge function values
        alpha = w0 * inv_area
        beta = w1 * inv_area
        gamma = w2 * inv_area
        
        # Interpolate z-values for all pixels (depth calculation)
        z = alpha * p1[2] + beta * p2[2] + gamma * p3[2]
        
        # Extract sub-region of z-buffer and color buffer that corresponds to the bounding box
        sub_zbuffer = self.z_buffer[min_y:max_y+1, min_x:max_x+1]
        sub_rgb = self.rgb_buffer[min_y:max_y+1, min_x:max_x+1]

        # Create mask for pixels that pass both the inside-triangle test and the z-buffer test
        update_mask = mask & (z < sub_zbuffer)
        
        # Update z-buffer and color buffer in one vectorized operation
        sub_zbuffer[update_mask] = z[update_mask]
        sub_rgb[update_mask] = color

    @Profiler.timed()
    def draw_triangle(self, p1: Tuple[int,int], p2: Tuple[int, int], p3: Tuple[int, int], color: Tuple[int, int, int] = COLOR_WHITE):
        self.draw_line(p1, p2, color)
        self.draw_line(p2, p3, color)
        self.draw_line(p3, p1, color)
    
    def draw_triangle_with_z(self, p1: Tuple[int, int, float], p2: Tuple[int, int, float], p3: Tuple[int, int, float], color: Tuple[int, int, int] = COLOR_WHITE):
        self.draw_line_skimage(p1, p2, color)
        self.draw_line_skimage(p2, p3, color)
        self.draw_line_skimage(p3, p1, color)
    
    @Profiler.timed()
    def apply_vertex_wave_shader(self, verticies: np.ndarray, amplitude: float, period: float, speed: float) -> np.ndarray:
        """
        Applies a sine wave transformation to all vertices in a mesh (vectorized).

        This function simulates a "wave shader" by offsetting the Y-coordinate
        of each vertex using a sine function. It operates on all vertices at once
        for efficiency using NumPy.

        Parameters:
            verticies (np.ndarray): 
                Array of shape (N, 4) containing homogeneous vertex positions.
            amplitude (float): 
                Maximum wave height. Larger values create taller waves.
            period (float): 
                Frequency of the wave. Larger values create shorter wavelengths.
            speed (float): 
                Speed factor controlling how the wave changes over time.

        Returns:
            np.ndarray:
                The transformed vertices with applied Y-axis wave offset (shape (N, 4)).

        Notes:
            - This function builds a unique translation matrix for each vertex
            (shape (N, 4, 4)) and applies it using batch matrix multiplication.
            - The wave displacement is computed as:
                offset = sin((x + frame_count * speed) * period) * amplitude
            - Vectorized using NumPy for efficiency (no Python loops).
        """
        # TODO abstract this further somehow. This is per vertex instead of an entire
        # object so it's a bit weird.
        # Add vertex wave shader
        x_pos = verticies[:, 0]
        # Calculate the wave offsets for all vertices at once
        offsets = np.sin((x_pos + frame_count * speed) * period) * amplitude  # shape (N,)
        # Build a translation matrix for each vertex: shape (N, 4, 4)
        translations = np.array([
            [[1, 0, 0, 0],
            [0, 1, 0, offset],
            [0, 0, 1, 0],
            [0, 0, 0, 1]] for offset in offsets
        ])
        # Multiply each vertex (shape (N,4,1)) by its translation matrix (N,4,4)
        V_h = verticies[:, :, None]  # (N,4,1)
        V_transformed = np.matmul(translations, V_h)  # (N,4,1)
        verticies = V_transformed[:, :, 0]  # (N,4)
        return verticies
    
    @Profiler.timed()
    def compute_view_matrix(self):
        """
        Computes the camera view matrix from camera rotation and position.

        Returns:
            np.ndarray:
                A 4x4 view transformation matrix (rotation and translation).
        """
        pitch, yaw, roll = self.camera_rot
        R_cam = Transform().with_rotation([pitch, yaw, roll])
        R_view = R_cam.get_matrix().T  # inverse of rotation matrix is transpose            

        # R_view must be 4d in order to be used in the matrix
        R_view_4d = np.eye(4)
        R_view_4d[:3, :3] = R_view[:3, :3]

        # Translation to move world relative to camera
        T_view = np.eye(4)
        camera_pos = np.array(self.camera_pos)
        T_view[:3, 3] = -camera_pos

        # View matrix = rotation * translation
        view_matrix = R_view_4d @ T_view
        return view_matrix
    
    @Profiler.timed()
    def prepare_vertices(self, obj: RenderableObject) -> np.ndarray:
        """
        Converts object vertices into homogeneous coordinates.

        Parameters:
            obj (RenderableObject):
                The object containing vertex positions as (N, 3).

        Returns:
            np.ndarray:
                Vertex array in homogeneous coordinates of shape (N, 4).
        """
        # Assuming: vertex_list = [(x, y, z), ...]
        V = np.array(obj.vertices)  # Shape: (N, 3)

        # Add homogeneous coordinate
        # TODO possibly implement column-vector
        V = np.hstack([V, np.ones((V.shape[0], 1))])  # Shape: (N, 4)
        return V
    
    @Profiler.timed()
    def get_model_matrix(self, obj: RenderableObject) -> np.ndarray:
        """
        Computes the model transformation matrix for the given object.

        Parameters:
            obj (RenderableObject):
                The renderable object whose transform is used.

        Returns:
            np.ndarray:
                A 4x4 model transformation matrix.
        """
        # ========= MODEL SPACE → WORLD SPACE =========
        # Create a matrix for rotating the model.
        R = Transform().with_rotation([angle, angle, 0])
        # Set the rotation of the matrix using our rotation matrix.
        obj.transform = obj.transform.with_rotation(R)
        # Use our newly completed matrix for future calculations.
        model_matrix = obj.transform.get_matrix()
        return model_matrix
    
    @Profiler.timed()
    def project_vertices(self, V_model: np.ndarray, model_matrix: np.ndarray, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """
        Projects vertices from model space into clip space.

        Parameters:
            V_model (np.ndarray):
                Homogeneous model-space vertices (N, 4).
            model_matrix (np.ndarray):
                Model transformation matrix (4x4).
            view_matrix (np.ndarray):
                View transformation matrix (4x4).
            projection_matrix (np.ndarray):
                Projection matrix (4x4).

        Returns:
            np.ndarray:
                Vertices in clip space (N, 4).
        """
        # Combine all transforms into a single 4x4 matrix
        M = projection_matrix @ view_matrix @ model_matrix

        # ========= MODEL → CLIP SPACE =========
        V_clip = (M @ V_model.T).T  # Shape: (N, 4)

        return V_clip
    
    @Profiler.timed()
    def cull_faces(self, V_clip: np.ndarray, faces: np.ndarray):
        """
        Performs frustum and backface culling on faces using clip-space coordinates.

        Parameters:
            V_clip (np.ndarray):
                Clip-space vertex positions (N, 4).
            faces (np.ndarray):
                Face indices referencing vertices (F, 3).

        Returns:
            np.ndarray:
                Boolean mask of length F, where True means face is kept.
        """
        verts = V_clip[faces]  # (F, 3, 4)

        # 1. ANY vertex behind camera (z <= 0) → drop
        behind_camera = (verts[..., 3] <= 0).any(axis=1)

        # 2. ALL vertices outside clip bounds → drop
        abs_coords = np.abs(verts[..., :3])  # (F, 3, 3)
        outside_clip = (abs_coords > verts[..., [3]]).all(axis=(1, 2))
        
        CLIP_ANY_OUTSIDE = False
        
        if CLIP_ANY_OUTSIDE:
            # Drop faces if ANY vertex is outside clip bounds
            outside_clip_any = (abs_coords > verts[..., [3]]).any(axis=(1, 2))
            outside_clip = outside_clip_any
        
        # Compute normals in clip space
        a = verts[:, 1, :3] / verts[:, 1, [3]] - verts[:, 0, :3] / verts[:, 0, [3]]
        b = verts[:, 2, :3] / verts[:, 2, [3]] - verts[:, 0, :3] / verts[:, 0, [3]]
        normals = np.cross(a, b)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        backfacing = normals[:, 2] < 0  # Facing -Z means away from camera → drop

        # Faces to keep
        faces_kept = ~(behind_camera | outside_clip | backfacing)
        return faces_kept
    
    @Profiler.timed()
    def perspective_divide(self, V_clip: np.ndarray) -> np.ndarray:
        """
        Converts vertices from clip space to normalized device coordinates (NDC).

        Parameters:
            V_clip (np.ndarray):
                Clip-space vertex positions (N, 4).

        Returns:
            np.ndarray:
                NDC positions (N, 3), after dividing by w.
        """
        # Perspective divide
        V_ndc = V_clip[:, :3] / V_clip[:, 3:4]  # Shape: (N, 3)
        return V_ndc

    @Profiler.timed()
    def compute_world_vertices(self, V_model: np.ndarray, model_matrix: np.ndarray) -> np.ndarray:
        """
        Transforms model-space vertices into world space.

        Parameters:
            V_model (np.ndarray):
                Homogeneous model-space vertices (N, 4).
            model_matrix (np.ndarray):
                Model transformation matrix (4x4).

        Returns:
            np.ndarray:
                World-space vertex positions (N, 3).
        """
        # ========= MODEL → WORLD FOR LIGHTING =========
        V_world = (model_matrix @ V_model.T).T[:, :3]  # apply model matrix, ignore w
        return V_world
    
    @Profiler.timed()
    def compute_world_triangles(self, V_world: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Retrieves triangles from world-space vertices using face indices.

        Parameters:
            V_world (np.ndarray):
                World-space vertices (N, 3).
            faces (np.ndarray):
                Face indices (F, 3).

        Returns:
            np.ndarray:
                Triangles in world space of shape (F, 3, 3).
        """
        tri_world = V_world[faces]
        return tri_world
    
    @Profiler.timed()
    def compute_normals(self, tri_world: np.ndarray) -> np.ndarray:
        """
        Computes normalized face normals for triangles.

        Parameters:
            tri_world (np.ndarray):
                Triangles in world space (F, 3, 3).

        Returns:
            np.ndarray:
                Unit normals for each face (F, 3).
        """
        # Precompute all normals of all faces
        a = tri_world[:,1] - tri_world[:,0]  # (N, 3)
        b = tri_world[:,2] - tri_world[:,0]  # (N, 3)
        normals = np.cross(a, b)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return normals
    
    @Profiler.timed()
    def compute_lighting(self, normals: np.ndarray, light: np.ndarray) -> np.ndarray:
        """
        Computes per-face lighting using Lambertian shading.

        Parameters:
            normals (np.ndarray):
                Normalized face normals (F, 3).
            light (np.ndarray):
                Directional light vector (3,).

        Returns:
            np.ndarray:
                Per-face colors based on light intensity (F, 3).
        """
        # precompute the brightness of all faces
        # Mapped from [-1,1] -> [0,1] so that faces not facing the light
        # still slightly light up.
        brightness = np.clip((normals @ light + 1) / 2, 0, 1)

        # precompute colors of all faces
        colors = (brightness[:, None] * COLOR_WHITE).astype(int)
        return colors
    
    @Profiler.timed()
    def ndc_to_screen(self, V_ndc: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Converts NDC coordinates of vertices to screen-space coordinates.

        Parameters:
            V_ndc (np.ndarray):
                Normalized device coordinates (N, 3).
            faces (np.ndarray):
                Face indices (F, 3).

        Returns:
            np.ndarray:
                Triangles in screen space of shape (F, 3, 3), where each vertex has (x, y, z).
        """
        faces = faces  # Shape (F, 3)
        tri_ndc_all = V_ndc[faces]  # Shape (F, 3, 3) —  F faces, 3 verts each, 3 coords each
        # ========= NDC → SCREEN SPACE =========
        # Precompute screen space
        # Extract x, y, z (NDC space)
        xy = tri_ndc_all[:, :, :2]  # (F, 3, 2)
        # z = tri_ndc_all[:, :, 2]    # (F, 3)
        z = -tri_ndc_all[:, :, 2]  # flip the z because somewhere in my logic i flipped it a long time ago

        # Convert x, y to screen coordinates
        grid_x = self.grid_size_x
        grid_y = self.grid_size_y
        xy_screen = np.empty_like(xy)
        xy_screen[:, :, 0] = ((xy[:, :, 0] + 1) * 0.5 * grid_x).astype(int)
        xy_screen[:, :, 1] = ((1 - (xy[:, :, 1] + 1) * 0.5) * grid_y).astype(int)

        # Combine x, y, z back
        tri_screen = np.dstack((xy_screen, z[..., None]))  # shape (F, 3, 3)
        
        return tri_screen
    
    @Profiler.timed()
    def draw_faces(self, tri_screen_all: np.ndarray, colors: np.ndarray, faces: np.ndarray):
        """
        Draws triangles on the screen with optional wireframe or filled faces.

        Parameters:
            tri_screen_all (np.ndarray):
                Screen-space triangles (F, 3, 3).
            colors (np.ndarray):
                Color for each face (F, 3).
            faces (np.ndarray):
                Face indices for drawing (F, 3).
        """
        global hover_triangle_index
        hover_triangle_index = -1
        # ========= DRAWING =========
        for i, face in enumerate(faces):
            tri_screen = tri_screen_all[i]
            if draw_lines:
                self.draw_triangle_with_z(tri_screen[0][0:3], tri_screen[1][0:3], tri_screen[2][0:3], COLOR_GREEN)
        for i, face in enumerate(faces):
            Profiler.profile_accumulate_start("draw_faces: project")

            # apply light to this face
            color = colors[i]

            tri_screen = tri_screen_all[i]

            # global hover_triangle_index
            if point_in_triangle((mouse_x, mouse_y), tri_screen[0], tri_screen[1], tri_screen[2]):
                color = COLOR_RED
                hover_triangle_index = i

            Profiler.profile_accumulate_end("draw_faces: project")
            Profiler.profile_accumulate_start("draw_faces: draw")

            if draw_faces or draw_z_buffer:
                self.fill_triangle(tri_screen[0], tri_screen[1], tri_screen[2], color) # type: ignore
            # if draw_lines:
            #     self.draw_triangle(tri_screen[0][0:2], tri_screen[1][0:2], tri_screen[2][0:2], COLOR_GREEN)

            Profiler.profile_accumulate_end("draw_faces: draw")

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

        view_matrix = self.compute_view_matrix()

        for r_object in self.objects:
            Profiler.profile_accumulate_start("draw_polygons: pre_compute")
            V_model = self.prepare_vertices(r_object)
            V_model = self.apply_vertex_wave_shader(
                V_model, 
                render_config.wave_amplitude.val, 
                render_config.wave_period.val, 
                render_config.wave_speed.val)
            model_matrix = self.get_model_matrix(r_object)
            V_clip = self.project_vertices(V_model, model_matrix, view_matrix, self.projection_matrix)
            faces_kept = self.cull_faces(V_clip, r_object.faces)
            V_ndc = self.perspective_divide(V_clip)

            V_world = self.compute_world_vertices(V_model, model_matrix)
            tri_world = self.compute_world_triangles(V_world, r_object.faces)
            normals = self.compute_normals(tri_world)
            colors = self.compute_lighting(normals, light)
            colors = colors[faces_kept]  # keep only faces kept

            faces = r_object.faces[faces_kept]   # keep only faces kept
            tri_screen_all = self.ndc_to_screen(V_ndc, faces)
            Profiler.profile_accumulate_end("draw_polygons: pre_compute")
            
            self.draw_faces(tri_screen_all, colors, faces)

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
                                self.last_mouse_pos = pygame.mouse.get_pos()
                                global hover_triangle_index
                                print(f"Red triangle index is {hover_triangle_index}")
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            self.dragging = False
                    elif event.type == pygame.MOUSEMOTION and self.dragging:
                        x, y = pygame.mouse.get_pos()
                        dx = x - self.last_mouse_pos[0]
                        dy = y - self.last_mouse_pos[1]
                        self.last_mouse_pos = (x, y)

                        self.camera_rot[1] -= dx * 0.005 * render_config.camera_sensitivity.val  # yaw (Y axis)
                        self.camera_rot[0] -= dy * 0.005 * render_config.camera_sensitivity.val  # pitch (X axis)
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_z:
                        global draw_z_buffer
                        draw_z_buffer = not draw_z_buffer
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                        global draw_lines
                        draw_lines = not draw_lines
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                        global draw_faces
                        draw_faces = not draw_faces
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
                    pitch, yaw, roll = self.camera_rot
                    # Rx = rotation_matrix_x(pitch)
                    # Ry = rotation_matrix_y(yaw)
                    # Rz = rotation_matrix_z(roll)
                    # R_cam = Rz @ Ry @ Rx
                    R_cam = Transform().with_rotation([pitch, yaw, roll])
                    move_world = R_cam @ Transform(translation=move_dir)
                    self.camera_pos += move_world.get_matrix()[:3, 3]

            self.render_buffer()
            
            if frame_count % 60 == 0:
                Profiler.profile_accumulate_report(intervals=60)
            
            # Clear the RGB buffer for the next frame
            self.create_empty_rgb_buffer()
            self.create_empty_z_buffer()

            pygame.display.flip()
            self.clock.tick(60)
            debug_win.render_ui()
            debug_win.apply_pending_updates()

        pygame.quit()

if __name__ == "__main__":
    renderer = Renderer()
    renderer.run()

# NOTE ChatGPT (4o for general templates) and Copilot (internally using GPT-4.1) was used in this project.
