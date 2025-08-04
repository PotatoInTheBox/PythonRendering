# triangle.py

# TODO finish writing
import numpy as np

class Triangle:
    def __init__(self, screen_pos, ndc_pos, uv=None, inv_w=None):
        self.screen = np.array(screen_pos)  # shape (3, 2)
        self.ndc = np.array(ndc_pos)        # shape (3, 3), includes z
        self.uv = np.array(uv) if uv is not None else None  # shape (3, 2)
        self.inv_w = np.array(inv_w) if inv_w is not None else None  # shape (3,)