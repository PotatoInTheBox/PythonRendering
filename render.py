#!/usr/bin/python

# The goal of this project is to make a python renderer that can simulate drawing operations.
# This is for practice only.
# We will aim to simulate how lines, squares, and pixels are drawn on a grid.
# (the renderer can toggle between passthrough mode, where it uses the original drawing methods,
# and simulated mode, where it updates an RGB buffer instead)

import pygame
import math
import time
import sys
import ctypes
import numpy as np
# Make windows not scale this window (pixels do have to be perfect)
if sys.platform == "win32":
    ctypes.windll.user32.SetProcessDPIAware()

PASSTHROUGH = False # Toggle between passthrough and simulated draw
DRAW_PIXEL_BORDER = True  # Toggle to draw a border around pixels
PIXEL_BORDER_SIZE = 1  # Size of the pixel border

angle = 0

mouse_x = 0
mouse_y = 0

draw_z_buffer = False
draw_faces = True
draw_lines = False

from typing import List, Tuple
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (50, 50, 50)
COLOR_PINK = (255, 105, 180)

OBJ_PATH = "./models/utahTeapot.obj"
CONUTER_CLOCKWISE_TRIANGLES = False
START_DISTANCE = 4.0
CAMERA_SPEED = 0.1
FRAME_LOG_INTERVAL = 60  # log once per 60 frames
frame_count = 0  # count frames rendered so far

# Accumulates total time spent in named segments
_profile_accumulators = {}
# keep track of our named profilers
_profile_timers = {}




class Object:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

def profile_accumulate_start(name: str):
    if name not in _profile_accumulators:
        _profile_accumulators[name] = [0.0, 0, None]  # [total_time, count, start_time]
    _profile_accumulators[name][2] = time.perf_counter()  # Reset start time

def profile_accumulate_end(name: str):
    if name not in _profile_accumulators or _profile_accumulators[name][2] is None:
        return  # ignore unmatched end
    start = _profile_accumulators[name][2]
    elapsed = time.perf_counter() - start
    _profile_accumulators[name][0] += elapsed
    _profile_accumulators[name][1] += 1
    _profile_accumulators[name][2] = None  # clear start

def profile_accumulate_report(intervals=1):
    print("\n////////==== Report Start ====\\\\\\\\\\\\\\\\")
    grand_total = sum(total for total, count, _ in _profile_accumulators.values())

    for name, (total, count, _) in _profile_accumulators.items():
        if count == 0:
            continue
        total_ms = total * 1000
        avg_ms = (total / (count / intervals)) * 1000
        percent = (total / grand_total) * 100 if grand_total > 0 else 0
        if percent >= 100:
            percent_str = "100%"
        elif percent >= 10:
            percent_str = f"{percent:4.1f}%"
        else:
            percent_str = f"{percent:4.2f}%"
        print(f"{percent_str} — {name}: {total_ms/intervals:.3f}ms total over {count/intervals} calls (avg {avg_ms/intervals:.3f}ms)")

    _profile_accumulators.clear()
    print("\\\\\\\\\\\\\\\\==== Report End   ====////////")

def timed(name=""):
    def wrapper(fn):
        def inner(*args, **kwargs):
            label = name or fn.__name__
            label = "f:" + label
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if label not in _profile_accumulators:
                _profile_accumulators[label] = [0.0, 0, None]
            _profile_accumulators[label][0] += elapsed
            _profile_accumulators[label][1] += 1
            return result
        return inner
    return wrapper


def profile_start(name: str, n=60):
    global frame_count
    if frame_count % n == 0:
        _profile_timers[name] = time.perf_counter()

def profile_end(name: str, n=60):
    global frame_count
    if frame_count % n == 0:
        if name in _profile_timers:
            elapsed = (time.perf_counter() - _profile_timers.pop(name)) * 1000
            print(f"{name}: {elapsed:.3f}ms")
        else:
            print(f"Warning: profile_end called for '{name}' without matching profile_start")

def load_obj(filepath):
    vertices = []
    triangles = []

    with open(filepath) as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if parts[0] == 'v':
                vertices.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'f':
                face = []
                for p in parts[1:4]:
                    v = p.split('/')[0]  # Always use the first part (vertex index)
                    face.append(int(v) - 1)
                triangles.append(tuple(face))

    return Object(vertices, triangles)
@timed()
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
@timed()
def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s,  c]])
@timed()
def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                    [ 0, 1, 0],
                    [-s, 0, c]])
@timed()
def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]])

# At startup we conver the verticies to values between -1 and 1.
def normalize_obj_vertices(vertices):
    v = np.array(vertices)  # Shape: (N, 3)
    min_vals = v.min(axis=0)
    max_vals = v.max(axis=0)
    center = (min_vals + max_vals) / 2
    scale = (max_vals - min_vals).max() / 2
    return (v - center) / scale

class Renderer:
    def __init__(self, width: int = 800, height: int = 800, grid_size: int = 200) -> None:
        pygame.init()
        self.width, self.height = width, height
        self.grid_size = grid_size
        self.cell_size = width // grid_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Renderer")
        self.clock = pygame.time.Clock()
        self.running = True

        # Initialize 2D RGB buffer
        self.rgb_buffer = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.z_buffer = np.full((self.grid_size, self.grid_size), -np.inf, dtype=np.float32)
        # Initialize 2D polygon buffer
        self.triangle_buffer = [
            [(12, 14), (18, 22), (15, 30)],
            [(20, 20), (30, 25), (25, 35)],
            [(40, 15), (45, 28), (35, 33)],
            [(60, 40), (70, 45), (65, 55)],
            [(50, 60), (55, 65), (45, 70)],
            [(30, 50), (35, 60), (25, 65)],
            [(80, 20), (85, 30), (75, 35)],
            [(15, 75), (20, 85), (10, 90)],
            [(70, 70), (80, 75), (75, 85)],
            [(90, 10), (95, 20), (85, 25)],
            [(65, 25), (70, 30), (60, 35)],
            [(22, 40), (28, 45), (24, 50)],
        ]

        self.object = load_obj(OBJ_PATH)
        self.object.vertices = normalize_obj_vertices(self.object.vertices)
        
        # The camera will need to face -z. So we need to push the camera towards positive z.
        # This is because our object will be at 0.
        self.camera_pos = [0.0,0.0,float(START_DISTANCE)]
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        # The camera is facing towards positive z.
        self.camera_rot = [0.0,0.0,0]
        self.camera_speed = CAMERA_SPEED
        self.projection_matrix = get_projection_matrix(fov=np.radians(90),aspect=1,near=0.1,far=1000)

    def _is_bounded(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED, width: int = 1) -> None:
        if PASSTHROUGH:
            pygame.draw.line(self.screen, color, start, end, width)
            return
        self.bresenhams_algorithm_draw_line(start, end, color)

    # https://medium.com/geekculture/bresenhams-line-drawing-algorithm-2e0e953901b3
    @timed()
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

            self.draw_pixel((out_x, out_y), color)
            x += 1
            if p < 0:
                p += 2 * dy
            else:
                p += 2 * (dy - dx)
                y += 1

    @timed()
    def draw_square(self, top_left: Tuple[int, int], size: int, color: Tuple[int, int, int] = COLOR_RED) -> None:
        if PASSTHROUGH:
            pygame.draw.rect(self.screen, color, (*top_left, size, size))
            return
        for y in range(top_left[1], top_left[1] + size):
            for x in range(top_left[0], top_left[0] + size):
                if self._is_bounded((x, y)):
                    self.rgb_buffer[y][x] = color

    @timed()
    def draw_pixel(self, position: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED) -> None:
        if PASSTHROUGH:
            self.screen.set_at(position, color)
            return
        x, y = position
        if self._is_bounded((x,y)):
            self.rgb_buffer[y][x] = color

    @timed()
    def draw_circle(self, center: Tuple[int, int], radius: int, color: Tuple[int, int, int] = COLOR_RED) -> None:
        if PASSTHROUGH:
            pygame.draw.circle(self.screen, color, center, radius)
            return

        sqrt_limit = radius**2
        for y in range(center[1] - radius, center[1] + radius):
            for x in range(center[0] - radius, center[0] + radius):
                if self._is_bounded((x, y)) and (y - center[1])**2 + (x - center[0])**2 < sqrt_limit:
                    self.rgb_buffer[y][x] = color

    # 500x500 triangles cost 0.8ms to draw (not great)
    @timed()
    def fill_triangle(self, p1: Tuple[float,float,float], p2: Tuple[float,float,float], p3: Tuple[float,float,float], color: Tuple[int, int, int] = COLOR_RED):
        if PASSTHROUGH:
            pygame.draw.polygon(self.screen, color, [p1, p2, p3])

        min_x = int(max(min(p1[0], p2[0], p3[0]), 0))
        max_x = int(min(max(p1[0], p2[0], p3[0]), self.grid_size - 1))
        min_y = int(max(min(p1[1], p2[1], p3[1]), 0))
        max_y = int(min(max(p1[1], p2[1], p3[1]), self.grid_size - 1))
        
        # Compute edge coefficients for: E(x, y) = A*x + B*y + C
        def edge_coeffs(p1, p2):
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = p1[0]*p2[1] - p1[1]*p2[0]
            return A, B, C

        A0, B0, C0 = edge_coeffs(p2, p3)
        A1, B1, C1 = edge_coeffs(p3, p1)
        A2, B2, C2 = edge_coeffs(p1, p2)

        for y in range(min_y, max_y + 1):
            # Start x from min_x
            w0 = A0 * min_x + B0 * y + C0
            w1 = A1 * min_x + B1 * y + C1
            w2 = A2 * min_x + B2 * y + C2

            # Precompute deltas
            dw0_dx = A0
            dw1_dx = A1
            dw2_dx = A2

            def edge(a, b, c):
                # Returns twice the signed area of triangle abc
                return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
            area = edge(p1, p2, p3)  # Precompute this outside the loop
            if area == 0:
                return  # skip degenerate triangle
            
            for x in range(min_x, max_x + 1):
                # Accept both types of faces. Can be optimized if only one is supported
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    # Normalize barycentric coordinates
                    alpha = w0 / area
                    beta = w1 / area
                    gamma = w2 / area

                    # Interpolate Z
                    z = alpha * p1[2] + beta * p2[2] + gamma * p3[2]
                    if z > self.z_buffer[y, x]:
                        self.rgb_buffer[y, x] = color
                        # Do a dumb interpolation (won't work for different perspectives)
                        self.z_buffer[y, x] = z
                w0 += dw0_dx
                w1 += dw1_dx
                w2 += dw2_dx

    @timed()
    def draw_triangle(self, p1: Tuple[int,int], p2: Tuple[int, int], p3: Tuple[int, int], color: Tuple[int, int, int] = COLOR_WHITE):
        if PASSTHROUGH:
            pygame.draw.lines(self.screen, color, True, [p1, p2, p3])
            return
        self.draw_line(p1, p2, color)
        self.draw_line(p2, p3, color)
        self.draw_line(p3, p1, color)

    @timed()
    def draw_polygons(self):
        profile_accumulate_start("draw_polygons: pre_compute")
        # === Setup ===
        if CONUTER_CLOCKWISE_TRIANGLES:
            light = np.array([0, -1, 0]) # for some reason I have to flip the light direction when the triangles are different
            # I guess that kinda makes sense. I'm getting the normals of the triangle, which if counter clockwise, will lead
            # to the triangle facing the opposite direction, and thus the light.
            # TODO Let's instead just change the light calculations to invert in the normal being flipped.
        else:
            light = np.array([0, 1, 0]) # The light is pointing towards positive y. This means down for us.
        camera_direction = np.array([0, 0, 1])

        global angle
        Rx = rotation_matrix_x(angle)
        Ry = rotation_matrix_y(angle)
        R = Ry @ Rx
        # angle += 0.01
        
        pitch, yaw, roll = self.camera_rot
        Rx = rotation_matrix_x(pitch)
        Ry = rotation_matrix_y(yaw)
        Rz = rotation_matrix_z(roll)
        R_cam = Rz @ Ry @ Rx  # camera rotation
        R_view = R_cam.T  # inverse of rotation matrix is transpose
        profile_accumulate_end("draw_polygons: pre_compute")
        
        # We are going to precompute as much as possible.
        # Basically, we are going to set up all the matrix calculations, make a
        # VERTEX list (or have it ready for numpy),
        # Then we apply all the vertex transformations using numpy.
        
        # The standard pipeline is as follows:
        # * Object space (won't deal with yet, treat it as world space, possibly going to apply scaling, transorms, and rotations in the future)
        # * World space (assume this is the start)
        # * View Space (rotated and transformed relative to the camera)
        # * Clip Space (perspective transformation applied)
        # * Normalized Device Coordinates (NDC) space (perspective divided by w)
        # obj space 
        # -> world space 
        # -> view space 
        # -> clip space 
        # -> normalized device coordinates (ndc) space 
        # -> screen space
        
        vertex_list = self.object.vertices
        
        # Assuming: vertex_list = [(x, y, z), ...]
        V = np.array(vertex_list)  # Shape: (N, 3)

        # Add homogeneous coordinate
        V = np.hstack([V, np.ones((V.shape[0], 1))])  # Shape: (N, 4)
        
        # Identity model matrix
        model_matrix = np.eye(4)
        
        # R_view must be 4d in order to be used in the matrix
        R_view_4d = np.eye(4)
        R_view_4d[:3, :3] = R_view
        
        # Translation to move world relative to camera
        T_view = np.eye(4)
        camera_pos = np.array(self.camera_pos)
        T_view[:3, 3] = -camera_pos
        
        # View matrix = rotation * translation
        view_matrix = R_view_4d @ T_view

        # Combine all transforms into a single 4x4 matrix
        M = self.projection_matrix @ view_matrix @ model_matrix  # or just view @ model if no projection yet

        # Transform all vertices in one go
        V_clip = (M @ V.T).T  # Shape: (N, 4)

        # Perspective divide
        V_ndc = V_clip[:, :3] / V_clip[:, 3:4]  # Shape: (N, 3)
        
        faces = np.array(self.object.faces)  # Shape (F, 3)
        tri_ndc_all = V_ndc[faces]  # Shape (F, 3, 3) —  F faces, 3 verts each, 3 coords each
        vertices = np.array(self.object.vertices)  # Shape (V, 3)

        tri_world_all = vertices[faces]  # Shape (F, 3, 3)

        profile_accumulate_start("draw_polygons: project_and_draw")
        for i in range(len(faces)):
            profile_accumulate_start("draw_polygons: project_and_draw: project")
  
            # === Frustum near-plane culling using clip.w (approximated here) ===
            if any(V_clip[j][3] <= 0 for j in faces[i]):
                profile_accumulate_end("draw_polygons: project_and_draw: project")
                continue
            
            # === NDC Space ===
            tri_ndc = tri_ndc_all[i]
            
            # === Backface culling ===
            a = tri_ndc[1][:2] - tri_ndc[0][:2]
            b = tri_ndc[2][:2] - tri_ndc[0][:2]
            screen_normal_z = a[0]*b[1] - a[1]*b[0]
            facing_camera = screen_normal_z < 0
            if facing_camera != CONUTER_CLOCKWISE_TRIANGLES:
                profile_accumulate_end("draw_polygons: project_and_draw: project")
                continue

            # === World-space triangle ===
            tri_world = tri_world_all[i]

            # === Compute face normal ===
            a = tri_world[1] - tri_world[0]
            b = tri_world[2] - tri_world[0]
            normal = normalize(np.cross(a, b))

            # === Lighting ===
            brightness = max(0, (np.dot(normal, light) + 1) / 2)
            color = tuple(int(brightness * c) for c in COLOR_WHITE)

            # === Convert to screen space ===
            tri_screen = [
                (
                    int((v[0] + 1) * 0.5 * self.grid_size),
                    int((1 - (v[1] + 1) * 0.5) * self.grid_size),
                    v[2]  # keep depth
                )
                for v in tri_ndc
            ]
            profile_accumulate_end("draw_polygons: project_and_draw: project")
            profile_accumulate_start("draw_polygons: project_and_draw: draw")
            if draw_faces or draw_z_buffer:
                self.fill_triangle(tri_screen[0], tri_screen[1], tri_screen[2], color) # type: ignore
            if draw_lines:
                self.draw_triangle(tri_screen[0][0:2], tri_screen[1][0:2], tri_screen[2][0:2], COLOR_GREEN)
            profile_accumulate_end("draw_polygons: project_and_draw: draw")
        profile_accumulate_end("draw_polygons: project_and_draw")


    @timed("render_buffer")
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
            z_scaled = np.clip((self.z_buffer - z_min) / z_range, 0, 1)
            z_norm = (z_scaled * 205 + 50).astype(np.uint8)  # near = dark, far = bright
            z_norm[~np.isfinite(self.z_buffer)] = 0  # Inf → black

            z_gray = np.stack([z_norm] * 3, axis=-1)
            surface = pygame.surfarray.make_surface(z_gray.swapaxes(0, 1))
        else:
            surface = pygame.surfarray.make_surface(self.rgb_buffer.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))

        if DRAW_PIXEL_BORDER:
            pixel_border_color = COLOR_DARK_GRAY
            for x in range(self.grid_size):
                pygame.draw.line(self.screen, pixel_border_color, (x * self.cell_size, 0), (x * self.cell_size, self.height), PIXEL_BORDER_SIZE)
            for y in range(self.grid_size):
                pygame.draw.line(self.screen, pixel_border_color, (0, y * self.cell_size), (self.width, y * self.cell_size), PIXEL_BORDER_SIZE)

    def run(self):
        
        while self.running:
            global frame_count
            frame_count += 1  # start of the next frame
            self.screen.fill((0, 0, 0))

            # Drawing demo here
            if True:
                # self.draw_line((0, 0), (self.width, self.height), (255, 0, 0))
                # self.draw_square((50, 50), 100, (0, 255, 0))
                # self.draw_pixel((0, 0), COLOR_RED)
                # self.draw_pixel((1, 1), COLOR_BLUE)
                # self.draw_pixel((1, 0), COLOR_GREEN)
                # self.draw_line((6, 6), (11, 8), COLOR_GREEN)
                # self.draw_line((2, 2), (5, 5), COLOR_WHITE)
                # self.draw_square((10,10), 5, COLOR_WHITE)
                # self.draw_circle((20, 8), 8, COLOR_GREEN)
                # self.fill_triangle((1*2, 12*2), (8*2, 9*2), (18*2, 15*2), COLOR_RED)
                # self.draw_polygons_2d()
                self.draw_polygons()
                
                # Spinning line (10px long from center)
                # cx, cy = self.grid_size // 2, self.grid_size // 2
                # length = 20
                # global angle
                # x2 = int(cx + length * math.cos(angle))
                # y2 = int(cy + length * math.sin(angle))
                # self.draw_line((cx, cy), (x2, y2), COLOR_WHITE)
                
                # angle += 0.01
                
                # Polygon follows mouse, left click cycles which point is moved
                if not hasattr(self, "poly_points"):
                    cx, cy = self.grid_size // 2, self.grid_size // 2
                    self.poly_points = [
                        [cx - 10, cy - 5],
                        [cx + 10, cy - 5],
                        [cx, cy + 10]
                    ]
                    self.active_point = 0

                mx, my = pygame.mouse.get_pos()
                
                mx //= self.cell_size
                my //= self.cell_size

                global mouse_x
                mouse_x = mx
                global mouse_y
                mouse_y = my
                
                # Move the active point to mouse position
                # self.poly_points[self.active_point][0] = mx
                # self.poly_points[self.active_point][1] = my

                # # Draw the polygon
                # self.fill_triangle(
                #     (self.poly_points[0][0], self.poly_points[0][1]),
                #     (self.poly_points[1][0], self.poly_points[1][1]),
                #     (self.poly_points[2][0], self.poly_points[2][1]),
                #     COLOR_WHITE
                # )

                # # Draw points for visual feedback
                # for idx, pt in enumerate(self.poly_points):
                #     color = COLOR_RED if idx == self.active_point else COLOR_BLUE
                #     self.draw_square((pt[0], pt[1]), 2, color)

                # Handle mouse click to cycle active point
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            # self.active_point = (self.active_point + 1) % 3
                            print(self.poly_points)
                    # ====== Camera rotation stuff ======
                    if event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 1:  # Left click
                                self.dragging = True
                                self.last_mouse_pos = pygame.mouse.get_pos()
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            self.dragging = False
                            print(f"Camera rotated with new vector ({self.camera_rot[0]}, {self.camera_rot[1]}, {self.camera_rot[2]})")
                    elif event.type == pygame.MOUSEMOTION and self.dragging:
                        x, y = pygame.mouse.get_pos()
                        dx = x - self.last_mouse_pos[0]
                        dy = y - self.last_mouse_pos[1]
                        self.last_mouse_pos = (x, y)

                        self.camera_rot[1] -= dx * 0.005  # yaw (Y axis)
                        self.camera_rot[0] -= dy * 0.005  # pitch (X axis)
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
                    move_dir = move_dir / np.linalg.norm(move_dir) * self.camera_speed * move_boost

                    # Camera rotation to world space
                    pitch, yaw, roll = self.camera_rot
                    Rx = rotation_matrix_x(pitch)
                    Ry = rotation_matrix_y(yaw)
                    Rz = rotation_matrix_z(roll)
                    R_cam = Rz @ Ry @ Rx
                    move_world = R_cam @ move_dir
                    self.camera_pos += move_world

            if not PASSTHROUGH:
                self.render_buffer()
            
            if frame_count % 60 == 0:
                profile_accumulate_report(intervals=60)
            
            # Clear the RGB buffer for the next frame
            self.rgb_buffer[:] = [50, 50, 120]
            self.z_buffer.fill(-np.inf)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    renderer = Renderer()
    renderer.run()

# NOTE ChatGPT (4o for general templates) and Copilot (internally using GPT-4.1) was used in this project.
