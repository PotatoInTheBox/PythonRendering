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

from typing import List, Tuple
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (50, 50, 50)
COLOR_PINK = (255, 105, 180)

FRAME_LOG_INTERVAL = 60  # log once per 60 frames
frame_count = 0  # count frames rendered so far
_profile_timers = {}  # keep track of our named profilers

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

def timed(name="", n=60):
    def wrapper(fn):
        def inner(*args, **kwargs):
            global frame_count
            result = fn(*args, **kwargs)
            if frame_count % n == 0:
                start = time.perf_counter()
                fn(*args, **kwargs)
                print(f"{name or fn.__name__}: {(time.perf_counter() - start) * 1000:.3f}ms")
            return result
        return inner
    return wrapper

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-16)

class Renderer:
    def __init__(self, width: int = 700, height: int = 700, grid_size: int = 100) -> None:
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

        
    def _is_bounded(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED, width: int = 1) -> None:
        if PASSTHROUGH:
            pygame.draw.line(self.screen, color, start, end, width)
            return

        self.bresenhams_algorithm_draw_line(start, end, color)

    # https://medium.com/geekculture/bresenhams-line-drawing-algorithm-2e0e953901b3
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

    def draw_square(self, top_left: Tuple[int, int], size: int, color: Tuple[int, int, int] = COLOR_RED) -> None:
        if PASSTHROUGH:
            pygame.draw.rect(self.screen, color, (*top_left, size, size))
            return
        for y in range(top_left[1], top_left[1] + size):
            for x in range(top_left[0], top_left[0] + size):
                if self._is_bounded((x, y)):
                    self.rgb_buffer[y][x] = color

    def draw_pixel(self, position: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED) -> None:
        if PASSTHROUGH:
            self.screen.set_at(position, color)
            return
        x, y = position
        if self._is_bounded((x,y)):
            self.rgb_buffer[y][x] = color

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
    def fill_triangle(self, p1: Tuple[int,int], p2: Tuple[int, int], p3: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED):
        if PASSTHROUGH:
            pygame.draw.polygon(self.screen, color, [p1, p2, p3])
        
        p0, p1, p2 = sorted([p1, p2, p3], key=lambda p: p[1])
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        for y in range(y0, y2 + 1):
            if y2 != y0:
                xa = int(x0 + (x2 - x0) * ((y - y0) / (y2 - y0)))
            else:
                xa = x0

            if y < y1 and y1 != y0:
                xb = int(x0 + (x1 - x0) * ((y - y0) / (y1 - y0)))
            elif y >= y1 and y2 != y1:
                xb = int(x1 + (x2 - x1) * ((y - y1) / (y2 - y1)))
            else:
                xb = x1

            from_x = max(min(xa, xb), 0)
            to_x = min(max(xa, xb), self.grid_size - 1)
            # profile_start("draw_triangle row")
            if 0 <= y < self.grid_size and from_x <= to_x:
                self.rgb_buffer[y, from_x:to_x+1] = color
            # profile_end("draw_triangle row")


        # TODO do a performance test (seems like fun)
        # I'll probably want to start putting profilers in place
        # Barycentric coordinate method to fill the triangle
        # Right now this method is less effective because we have to go over more pixels with large triangles
        # This becomes a better algorithm with smaller triangles.
        # min_x = max(min(p1[0], p2[0], p3[0]), 0)
        # max_x = min(max(p1[0], p2[0], p3[0]), self.grid_size - 1)
        # min_y = max(min(p1[1], p2[1], p3[1]), 0)
        # max_y = min(max(p1[1], p2[1], p3[1]), self.grid_size - 1)

        # for y in range(min_y, max_y + 1):
        #     for x in range(min_x, max_x + 1):
        #         # Inline edge functions
        #         w0 = (x - p2[0]) * (p3[1] - p2[1]) - (y - p2[1]) * (p3[0] - p2[0])
        #         w1 = (x - p3[0]) * (p1[1] - p3[1]) - (y - p3[1]) * (p1[0] - p3[0])
        #         w2 = (x - p1[0]) * (p2[1] - p1[1]) - (y - p1[1]) * (p2[0] - p1[0])

        #         if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
        #             self.rgb_buffer[y][x] = color

    def draw_triangle(self, p1: Tuple[int,int], p2: Tuple[int, int], p3: Tuple[int, int], color: Tuple[int, int, int] = COLOR_WHITE):
        if PASSTHROUGH:
            pygame.draw.lines(self.screen, color, True, [p1, p2, p3])
            return
        self.draw_line(p1, p2, color)
        self.draw_line(p2, p3, color)
        self.draw_line(p3, p1, color)
    
    def draw_polygons_2d(self):
        # So what we are going to do is draw polygons from a buffer/list containing polygons.
        # We want to scan every single pixel and draw if a polygon intersects.
        # Initially, it will be a 2d implementation, then it will move to 3d.
        for tri in self.triangle_buffer:
            self.fill_triangle(tri[0], tri[1], tri[2], COLOR_GREEN)
        
    def draw_polygons(self):
        # Hardcoded upside-down isosceles triangle in 3D
        # Top two points are farther (larger z), bottom is closer (smaller z)
        # triangle = [
        #     [40, 20, 2.0],  # top-left (far)
        #     [60, 20, 2.0],  # top-right (far)
        #     [50, 40, 0.5],  # bottom (close)
        # ]
        global mouse_x
        global mouse_y
        triangle = [
            [40, 20, 2.0],  # top-left (far)
            [60, 20, 2.0],  # top-right (far)
            [mouse_x, mouse_y, 0.5],  # bottom (close)
        ]
        

        # Placeholder: just draw a 2D projection for now
        # We'll handle shading and projection later
        p1 = (int(triangle[0][0]), int(triangle[0][1]))
        p2 = (int(triangle[1][0]), int(triangle[1][1]))
        p3 = (int(triangle[2][0]), int(triangle[2][1]))
        
        # I can reuse the same trick to figure out how shaded it is.
        # right now it is in 3d space. If I have a light source (say (0,-1,0) where -1 is up)
        # then, I can use the dot product to see how much the triangle face is pointing towards it.
        # However, this means I must get the face of the triangle (the direction). 
        # If I use the cross product on two points then I will get a direction vector.
        # I can then use this direction vector to do a dot product calculation with the light vector.
        
        # I'll do the dot product and cross product myself for practice.
        # Sike, I realized as I was doing the cross product that this is something I do not want to
        # do by hand or even type it in each time. I'd rather just tell a helper function to do a cross
        # product between two matricies.
        
        # Get the triangle face vector (eg. by doing a cross product of b and c while offsetting by a)
        a = np.subtract(triangle[1], triangle[0])
        b = np.subtract(triangle[2], triangle[0])
        triangle_face = normalize(np.cross(a, b))
        # It would be nice if i normalized it. This can be done by dividing all the numbers by the pythagoreas of all of them
        
        # Get the light source vector (given) inverted because my triangle is
        light = [0,1,0]
        
        # Get the dot product between the two (fairly simple)
        light_amount = np.dot(triangle_face, light)
        
        # dot product will give me a number between negative 1 and positive 1
        light_amount = (light_amount + 1)/2
        
        # cap it
        brightness = max(0, light_amount)
        
        # Apply the dot product to a whiteness level
        color = tuple(int(brightness * c) for c in COLOR_WHITE)
        self.fill_triangle(p1, p2, p3, color)

    @timed("render_buffer")
    def render_buffer(self):
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
                self.draw_pixel((0, 0), COLOR_RED)
                self.draw_pixel((1, 1), COLOR_BLUE)
                self.draw_pixel((1, 0), COLOR_GREEN)
                self.draw_line((6, 6), (11, 8), COLOR_GREEN)
                self.draw_line((2, 2), (5, 5), COLOR_WHITE)
                self.draw_square((10,10), 5, COLOR_WHITE)
                self.draw_circle((20, 8), 8, COLOR_GREEN)
                self.fill_triangle((1*2, 12*2), (8*2, 9*2), (18*2, 15*2), COLOR_RED)
                # self.draw_polygons_2d()
                self.draw_polygons()
                
                # Spinning line (10px long from center)
                cx, cy = self.grid_size // 2, self.grid_size // 2
                length = 20
                global angle
                x2 = int(cx + length * math.cos(angle))
                y2 = int(cy + length * math.sin(angle))
                self.draw_line((cx, cy), (x2, y2), COLOR_WHITE)
                
                angle += 0.01
                
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
            
            

            if not PASSTHROUGH:
                self.render_buffer()
            
            # Clear the RGB buffer for the next frame
            self.rgb_buffer.fill(0)
                

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    renderer = Renderer()
    renderer.run()

# NOTE ChatGPT (4o for general templates) and Copilot (internally using GPT-4.1) was used in this project.
