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

class Renderer:
    def __init__(self, width: int = 1000, height: int = 1000, grid_size: int = 40) -> None:
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
                
                # Spinning line (10px long from center)
                cx, cy = self.grid_size // 2, self.grid_size // 2
                length = 20
                global angle
                x2 = int(cx + length * math.cos(angle))
                y2 = int(cy + length * math.sin(angle))
                self.draw_line((cx, cy), (x2, y2), COLOR_WHITE)
                
                angle += 0.025
            
                # Line pointing to mouse
                cx, cy = self.grid_size // 2, self.grid_size // 2
                mx, my = pygame.mouse.get_pos()
                mx //= self.cell_size
                my //= self.cell_size

                dx = mx - cx
                dy = my - cy
                dist = math.hypot(dx, dy)

                max_length = 20
                if dist > 0:
                    scale = min(max_length / dist, 1.0)
                    tx = cx + int(dx * scale)
                    ty = cy + int(dy * scale)
                    self.draw_line((cx, cy), (tx, ty), COLOR_WHITE)
            
            

            if not PASSTHROUGH:
                self.render_buffer()
            
            # Clear the RGB buffer for the next frame
            self.rgb_buffer = [[(0, 0, 0) for _ in range(self.grid_size)] for _ in range(self.grid_size)]
                

            pygame.display.flip()
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

        pygame.quit()

if __name__ == "__main__":
    renderer = Renderer()
    renderer.run()

# NOTE ChatGPT (4o for general templates) and Copilot (internally using GPT-4.1) was used in this project.
