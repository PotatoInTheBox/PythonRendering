#!/usr/bin/python

# The goal of this project is to make a python renderer that can simulate drawing operations.
# This is for practice only.
# We will aim to simulate how lines, squares, and pixels are drawn on a grid.
# (the renderer can toggle between passthrough mode, where it uses the original drawing methods,
# and simulated mode, where it updates an RGB buffer instead)

import pygame
import math
import time

PASSTHROUGH = False  # Toggle between passthrough and simulated draw
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
    def __init__(self, width: int = 400, height: int = 400, grid_size: int = 40) -> None:
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
        # Calculate step direction
        if end[0] - start[0] < 0:
            dx = -1
        else:
            dx = 1
        if end[1] - start[1] < 0:
            dy = -1
        else:
            dy = 1
        cur_y = start[1]
        for i, x in enumerate(range(start[0], end[0] + dx, dx)):
            slope = end[1] - start[1] if end[0] - start[0] == 0 else (end[1] - start[1]) / (end[0] - start[0])
            start_y = cur_y
            end_y = start_y + slope
            for y in range(int(start_y), int(end_y) + dy, dy):
                self.draw_pixel((x, y), color)
            cur_y = end_y

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

        pass

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
            if False:
                # self.draw_line((0, 0), (self.width, self.height), (255, 0, 0))
                # self.draw_square((50, 50), 100, (0, 255, 0))
                self.draw_pixel((0, 0), COLOR_RED)
                self.draw_pixel((1, 1), COLOR_BLUE)
                self.draw_pixel((1, 0), COLOR_GREEN)
                self.draw_line((6, 6), (11, 8), COLOR_GREEN)
                self.draw_line((2, 2), (5, 5), COLOR_WHITE)
                self.draw_square((10,10), 5, COLOR_WHITE)

            # Spinning line (10px long from center)
            cx, cy = self.grid_size // 2, self.grid_size // 2
            length = 20
            global angle
            x2 = int(cx + length * math.cos(angle))
            y2 = int(cy + length * math.sin(angle))
            self.draw_line((cx, cy), (x2, y2), COLOR_WHITE)
            
            angle += 0.01

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
