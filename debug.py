# debug.py
# I inted to have this file be used while the debugger is running and actively
# pausing the scene. So that I can do stuff like poke into memory while it's
# running and see large swaths of data as an image.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

def draw_array(image: np.ndarray):
    h, w = image.shape[:2]
    fig, ax = plt.subplots()

    norm = Normalize(vmin=image.min(), vmax=image.max())
    im_display = None

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        im_display = ax.imshow(image.squeeze(), norm=norm)
    elif image.ndim == 3 and image.shape[2] == 2:
        img_norm = norm(image)
        im_display = ax.imshow(np.dstack([img_norm, np.zeros((h, w))]))
    elif image.ndim == 3 and image.shape[2] == 3:
        img_norm = norm(image)
        im_display = ax.imshow(img_norm)
    else:
        raise ValueError("Unsupported shape")

    rect = patches.Rectangle((0, 0), w-1, h-1, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')

    # Single-arg version for hover tool
    def format_coord(x: float, y: float) -> str:
        xi, yi = int(x + 0.5), int(y + 0.5)
        if 0 <= yi < h and 0 <= xi < w:
            val = image[yi, xi]
            return f"x={xi}, y={yi}, val={val}"
        return ""

    ax.format_coord = format_coord
    plt.show()