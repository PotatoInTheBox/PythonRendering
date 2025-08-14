# texture.py

import numpy as np
from numpy.typing import NDArray

def _create_texture_from_bytes(filepath: str) -> NDArray[np.float32]:
    with open(filepath, 'rb') as f:
        bytes_data = f.read()

    width = bytes_data[0] | (bytes_data[1] << 8)
    height = bytes_data[2] | (bytes_data[3] << 8)

    image = np.zeros((height, width, 3), dtype=np.float32)
    index = 4  # skip width and height bytes

    for y in range(height):
        for x in range(width):
            r = bytes_data[index + 0] / 255.0
            g = bytes_data[index + 1] / 255.0
            b = bytes_data[index + 2] / 255.0
            image[y, x] = [r, g, b]
            index += 3

    return image

class Texture:
    def __init__(self, filepath: str):
        self.image: NDArray[np.float32]  # shape: (width, height, 3)
        self.image = np.array(_create_texture_from_bytes(filepath), dtype=np.float32)
        self.width: int = self.image.shape[1]
        self.height: int = self.image.shape[0]

