#!/usr/bin/python

# The goal of this project is to make a python renderer that can simulate drawing operations.
# This is for practice only.
# We will aim to simulate how lines, squares, and pixels are drawn on a grid.
# (the renderer can toggle between passthrough mode, where it uses the original drawing methods,
# and simulated mode, where it updates an RGB buffer instead)

import profiler
from profiler import Profiler
from renderable_object import RenderableObject

import pygame
import sys
import ctypes
import numpy as np
from typing import List, Tuple
# Make windows not scale this window (pixels do have to be perfect)
if sys.platform == "win32":
    ctypes.windll.user32.SetProcessDPIAware()

# ========== Screen settings ==========
SCREEN_WIDTH = 900  # How much width should the window have?
SCREEN_HEIGHT = 450  # How much height should the window have?
GRID_CELL_SIZE = 5  # How many pixels big is each raster cell?

# ========== Camera settings ==========
CAMERA_SPEED = 0.1
START_DISTANCE = 4.0
CAMERA_POSITION = [0.0,0.0,float(START_DISTANCE)]
CAMERA_ROTATION = [0.0,0.0,0]
FOV=90  # In degrees, how much can the camera see from left to right?

# ========== Object initialization ==========
MONKEY_OBJ = RenderableObject.load_new_obj("./models/blender_monkey.obj")
NAME_OBJ = RenderableObject.load_new_obj("./models/name.obj")
SHIP_OBJ = RenderableObject.load_new_obj("./models/ship.obj")

MONKEY_OBJ.transform.set_translation([-3,0,-1])  # we will have this on the left of our initial camera (slightly further)
NAME_OBJ.transform.set_translation([0,0,0])  # We will have this at the origin
NAME_SCALE = 1.8
NAME_OBJ.transform.set_scale([NAME_SCALE,NAME_SCALE,NAME_SCALE])
SHIP_OBJ.transform.set_translation([3,-1,1])  # we will have this on the right of our initial camera (slightly closer) (slightly up)
# Documenting... We can also use .transform.set_rotation() and .transform.set_scale()

RENDER_OBJECTS = [MONKEY_OBJ, NAME_OBJ, SHIP_OBJ]  # all the objects we want rendered

# ========== Performance metrics ==========
ENABLE_PROFILER = False
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
CONUTER_CLOCKWISE_TRIANGLES = False

# ========== Global variables ==========
angle = 0

mouse_x = 0
mouse_y = 0

draw_z_buffer = False
draw_faces = True
draw_lines = False

frame_count = 0  # count frames rendered so far

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

def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s,  c]])

def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                    [ 0, 1, 0],
                    [-s, 0, c]])

def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]])

class Renderer:
    def __init__(self, width: int = 900, height: int = 450, grid_size_x: int = 150, grid_size_y: int = 75) -> None:
        self.width, self.height = width, height
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.cell_size_x = width // grid_size_x
        self.cell_size_y = height // grid_size_y
        self.create_empty_rgb_buffer()
        self.create_empty_z_buffer()
        
        # The camera will need to face -z. So we need to push the camera towards positive z.
        # This is because our object will be at 0.
        self.camera_pos = CAMERA_POSITION
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        # The camera is facing towards positive z.
        self.camera_rot = CAMERA_ROTATION
        self.camera_speed = CAMERA_SPEED
        self.projection_matrix = get_projection_matrix(fov=np.radians(FOV),aspect=self.grid_size_x/self.grid_size_y,near=0.1,far=1000)
        # Load and create our object which we will render
        self.objects = RENDER_OBJECTS
        
        pygame.init()
        pygame.display.set_caption("Renderer")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.running = True
    
    def create_empty_rgb_buffer(self):
        self.rgb_buffer = np.zeros((self.grid_size_y, self.grid_size_x, 3), dtype=np.uint8)
        self.rgb_buffer[:] = COLOR_SLATE_BLUE
    
    def create_empty_z_buffer(self):
        self.z_buffer = np.full((self.grid_size_y, self.grid_size_x), -np.inf, dtype=np.float32)

    def _is_bounded(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.grid_size_x and 0 <= y < self.grid_size_y

    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED, width: int = 1) -> None:
        self.bresenhams_algorithm_draw_line(start, end, color)

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
            # Returns twice the signed area of triangle abc
            return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
        
        area = edge(p1, p2, p3)  # Precompute this outside the loop
        if area == 0:
            return  # skip degenerate triangle
        inv_area = 1/ area

        min_x = int(max(min(p1[0], p2[0], p3[0]), 0))
        max_x = int(min(max(p1[0], p2[0], p3[0]), self.grid_size_x - 1))
        min_y = int(max(min(p1[1], p2[1], p3[1]), 0))
        max_y = int(min(max(p1[1], p2[1], p3[1]), self.grid_size_y - 1))
        
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
            
            for x in range(min_x, max_x + 1):
                # Accept both types of faces. Can be optimized if only one is supported
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    # Normalize barycentric coordinates
                    alpha = w0 * inv_area
                    beta = w1 * inv_area
                    gamma = w2 * inv_area

                    # Interpolate Z
                    z = alpha * p1[2] + beta * p2[2] + gamma * p3[2]
                    if z > self.z_buffer[y, x]:
                        self.rgb_buffer[y, x] = color
                        # Do a dumb interpolation (won't work for different perspectives)
                        self.z_buffer[y, x] = z
                w0 += dw0_dx
                w1 += dw1_dx
                w2 += dw2_dx

    @Profiler.timed()
    def draw_triangle(self, p1: Tuple[int,int], p2: Tuple[int, int], p3: Tuple[int, int], color: Tuple[int, int, int] = COLOR_WHITE):
        self.draw_line(p1, p2, color)
        self.draw_line(p2, p3, color)
        self.draw_line(p3, p1, color)

    @Profiler.timed()
    def draw_polygons(self):
        for r_object in self.objects:
            Profiler.profile_accumulate_start("draw_polygons: pre_compute")
            # === Setup ===
            if CONUTER_CLOCKWISE_TRIANGLES:
                light = np.array([0, -1, 0]) # for some reason I have to flip the light direction when the triangles are different
                # I guess that kinda makes sense. I'm getting the normals of the triangle, which if counter clockwise, will lead
                # to the triangle facing the opposite direction, and thus the light.
                # TODO Let's instead just change the light calculations to invert in the normal being flipped.
            else:
                light = np.array([0, 1, 0]) # The light is pointing towards positive y. This means down for us.

            global angle
            Rx = rotation_matrix_x(angle)
            Ry = rotation_matrix_y(angle)
            R = Ry @ Rx

            # model_matrix = R_4d  # T @ R @ S
            r_object.transform.set_rotation(R)
            model_matrix = r_object.transform.get_matrix()

            pitch, yaw, roll = self.camera_rot
            Rx = rotation_matrix_x(pitch)
            Ry = rotation_matrix_y(yaw)
            Rz = rotation_matrix_z(roll)
            R_cam = Rz @ Ry @ Rx  # camera rotation
            R_view = R_cam.T  # inverse of rotation matrix is transpose
            
            # Assuming: vertex_list = [(x, y, z), ...]
            V = np.array(r_object.vertices)  # Shape: (N, 3)

            # Add homogeneous coordinate
            V = np.hstack([V, np.ones((V.shape[0], 1))])  # Shape: (N, 4)
            
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
            M = self.projection_matrix @ view_matrix @ model_matrix

            # Transform all vertices in one go
            V_clip = (M @ V.T).T  # Shape: (N, 4)

            # Perspective divide
            V_ndc = V_clip[:, :3] / V_clip[:, 3:4]  # Shape: (N, 3)
            
            faces = np.array(r_object.faces)  # Shape (F, 3)
            tri_ndc_all = V_ndc[faces]  # Shape (F, 3, 3) —  F faces, 3 verts each, 3 coords each

            V_world = (model_matrix @ V.T).T[:, :3]  # apply model matrix, ignore w
            tri_world_all = V_world[faces]
            
            # Precompute all normals of all faces
            a = tri_world_all[:,1] - tri_world_all[:,0]  # (N, 3)
            b = tri_world_all[:,2] - tri_world_all[:,0]  # (N, 3)
            normals = np.cross(a, b)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            
            # precompute the brightness of all faces
            brightness_all = np.clip((normals @ light + 1) / 2, 0, 1)
            
            # precompute colors of all faces
            color_all = (brightness_all[:, None] * COLOR_WHITE).astype(int)
            
            # Precompute frustrum culling
            # faces: (F, 3), V_clip: (V, 4)
            w_vals = V_clip[:, 3]  # (V,)

            # Gather w components per face vertex
            w_faces = w_vals[faces]  # (F, 3)

            # Check if any w <= 0 per face (near-plane culling)
            mask = np.any(w_faces <= 0, axis=1)  # (F,) True if face is culled

            # Cull faces upfront
            valid_faces_idx = np.nonzero(~mask)[0]
            
            # Precompute screen space
            # Extract x, y, z (NDC space)
            xy = tri_ndc_all[:, :, :2]  # (F, 3, 2)
            z = tri_ndc_all[:, :, 2]    # (F, 3)

            # Convert x, y to screen coordinates
            grid_x = self.grid_size_x
            grid_y = self.grid_size_y
            xy_screen = np.empty_like(xy)
            xy_screen[:, :, 0] = ((xy[:, :, 0] + 1) * 0.5 * grid_x).astype(int)
            xy_screen[:, :, 1] = ((1 - (xy[:, :, 1] + 1) * 0.5) * grid_y).astype(int)

            # Combine x, y, z back
            tri_screen_all = np.dstack((xy_screen, z[..., None]))  # shape (F, 3, 3)

            Profiler.profile_accumulate_end("draw_polygons: pre_compute")
            Profiler.profile_accumulate_start("draw_polygons: project_and_draw")
            # Backface culling seems a bit inaccurate right now
            # for i, face in enumerate(valid_faces_idx):
            for i, face in enumerate(faces):
                Profiler.profile_accumulate_start("draw_polygons: project_and_draw: project")
                Profiler.profile_accumulate_start("draw_polygons: project_and_draw: project: frustum culling")
                # === Frustum near-plane culling using clip.w (approximated here) ===
                if any(V_clip[j][3] <= 0 for j in faces[i]):
                    Profiler.profile_accumulate_end("draw_polygons: project_and_draw: project")
                    Profiler.profile_accumulate_end("draw_polygons: project_and_draw: project: frustum culling")
                    continue
                Profiler.profile_accumulate_end("draw_polygons: project_and_draw: project: frustum culling")

                # apply light to this face
                color = color_all[i]

                # === Convert to screen space ===
                tri_screen = tri_screen_all[i]
                Profiler.profile_accumulate_end("draw_polygons: project_and_draw: project")
                Profiler.profile_accumulate_start("draw_polygons: project_and_draw: draw")
                if draw_faces or draw_z_buffer:
                    self.fill_triangle(tri_screen[0], tri_screen[1], tri_screen[2], color) # type: ignore
                if draw_lines:
                    self.draw_triangle(tri_screen[0][0:2], tri_screen[1][0:2], tri_screen[2][0:2], COLOR_GREEN)
                Profiler.profile_accumulate_end("draw_polygons: project_and_draw: draw")
            Profiler.profile_accumulate_end("draw_polygons: project_and_draw")


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

            self.render_buffer()
            
            if frame_count % 60 == 0:
                Profiler.profile_accumulate_report(intervals=60)
            
            # Clear the RGB buffer for the next frame
            # self.rgb_buffer[:] = [50, 50, 120]
            # self.z_buffer.fill(-np.inf)
            self.create_empty_rgb_buffer()
            self.create_empty_z_buffer()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH//GRID_CELL_SIZE, SCREEN_HEIGHT//GRID_CELL_SIZE)
    renderer.run()

# NOTE ChatGPT (4o for general templates) and Copilot (internally using GPT-4.1) was used in this project.
