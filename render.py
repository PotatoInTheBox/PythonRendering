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
# Make windows not scale this window (pixels do have to be perfect)
if sys.platform == "win32":
    ctypes.windll.user32.SetProcessDPIAware()

PASSTHROUGH = False # Toggle between passthrough and simulated draw
DRAW_PIXEL_BORDER = True  # Toggle to draw a border around pixels
PIXEL_BORDER_SIZE = 1  # Size of the pixel border

angle = 0

from typing import List, Tuple
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (50, 50, 50)
COLOR_PINK = (255, 105, 180)

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
        self.rgb_buffer = [[(0, 0, 0) for _ in range(grid_size)] for _ in range(grid_size)]
        
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
    
    def draw_polygon(self, p1: Tuple[int,int], p2: Tuple[int, int], p3: Tuple[int, int], color: Tuple[int, int, int] = COLOR_RED):
        if PASSTHROUGH:
            pygame.draw.polygon(self.screen, color, [p1, p2, p3])
        
        # Sort the points by their x-coordinate (ascending)
        points_y = sorted([p1, p2, p3], key=lambda p: p[1])
        
        for y in range(points_y[0][1], points_y[2][1] + 1):
            from_x = 0
            to_x = 0
            
            def interpolate_int(from_num: int, to_num: int, blend_ratio: float, flip=False) -> int:
                if flip: return int(to_num - ((to_num - from_num)*blend_ratio))
                return int(from_num + ((to_num - from_num)*blend_ratio))
            def calc_blend_ratio(from_num: int, to_num: int, curr_num: int) -> float:
                if (to_num - from_num) == 0: return 0  # we don't really know if it is between [0-1]
                return (curr_num - from_num)/(to_num - from_num)
            def interpolate_blend(from_num: int, to_num: int, original_from: int, original_to: int, current_num: int, flip=False) -> int:
                return interpolate_int(from_num, to_num, calc_blend_ratio(original_from, original_to, current_num))
            
            
            # Now we interpolate
            if y < points_y[1][1]:
                xa = interpolate_blend(points_y[0][0], points_y[2][0], points_y[0][1], points_y[2][1], y)  # from smallest y to largest y
                xb = interpolate_blend(points_y[0][0], points_y[1][0], points_y[0][1], points_y[1][1], y)  # from small y to middle y
            else:
                xa = interpolate_blend(points_y[0][0], points_y[2][0], points_y[0][1], points_y[2][1], y)  # from smallest y to largest y
                xb = interpolate_blend(points_y[1][0], points_y[2][0], points_y[1][1], points_y[2][1], y)  # from middle y to large y
            
            # get start x and end x to draw to
            from_x = min(xa, xb)
            to_x = max(xa, xb)
            
            # draw the row
            for x in range(int(from_x), int(to_x)):
                if self._is_bounded((x,y)):
                    self.rgb_buffer[y][x] = color

            # debug draw points
            for p in points_y:
                if self._is_bounded((p[0],p[1])):
                    self.rgb_buffer[p[1]][p[0]] = COLOR_PINK


        # Barycentric coordinate method to fill the triangle
        # def edge(p1, p2, p):
        #     return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])

        # for y in range(smallest_y, largest_y + 1):
        #     for x in range(smallest_x, largest_x + 1):
        #     p = (x, y)
        #     w0 = edge(x2, x3, p)
        #     w1 = edge(x3, x1, p)
        #     w2 = edge(x1, x2, p)
        #     if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
        #         if self._is_bounded(p):
        #         self.rgb_buffer[y][x] = color

    def render_buffer(self):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = self.rgb_buffer[y][x]
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
        if DRAW_PIXEL_BORDER:
            pixel_border_color = COLOR_DARK_GRAY
            for x in range(self.grid_size):
                # Draw all the vertical lines
                pygame.draw.line(self.screen, pixel_border_color, (x * self.cell_size, 0), (x * self.cell_size, self.height), PIXEL_BORDER_SIZE)
            # Draw all the horizontal lines
            for y in range(self.grid_size):
                pygame.draw.line(self.screen, pixel_border_color, (0, y * self.cell_size), (self.width, y * self.cell_size), PIXEL_BORDER_SIZE)

    def run(self):
        
        while self.running:
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
                self.draw_polygon((1*2, 12*2), (8*2, 9*2), (18*2, 15*2), COLOR_RED)
                
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

                # Move the active point to mouse position
                self.poly_points[self.active_point][0] = mx
                self.poly_points[self.active_point][1] = my

                # Draw the polygon
                self.draw_polygon(
                    (self.poly_points[0][0], self.poly_points[0][1]),
                    (self.poly_points[1][0], self.poly_points[1][1]),
                    (self.poly_points[2][0], self.poly_points[2][1]),
                    COLOR_WHITE
                )

                # Draw points for visual feedback
                for idx, pt in enumerate(self.poly_points):
                    color = COLOR_RED if idx == self.active_point else COLOR_BLUE
                    self.draw_square((pt[0], pt[1]), 2, color)

                # Handle mouse click to cycle active point
                for event in pygame.event.get(pygame.MOUSEBUTTONDOWN):
                    if event.button == 1:  # Left click
                        self.active_point = (self.active_point + 1) % 3
                        print(self.poly_points)
                    if event.type == pygame.QUIT:
                        self.running = False
            
            

            if not PASSTHROUGH:
                self.render_buffer()
            
            # Clear the RGB buffer for the next frame
            self.rgb_buffer = [[(0, 0, 0) for _ in range(self.grid_size)] for _ in range(self.grid_size)]
                

            pygame.display.flip()
            self.clock.tick(60)

            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         self.running = False

        pygame.quit()

if __name__ == "__main__":
    renderer = Renderer()
    renderer.run()

# NOTE ChatGPT (4o for general templates) and Copilot (internally using GPT-4.1) was used in this project.
