# triangle.py

# TODO finish writing
import numpy as np
from config import global_config
from profiler import Profiler

class Triangle:
    def __init__(self, screen_pos, ndc_pos, clip_pos,
                 uv=None, inv_w=None, texture=None,
                 base_color=None, normals=None):

        # --- Core vertex data ---
        self.screen = np.array(screen_pos, dtype=np.float32)   # (3, 2)
        self.ndc = np.array(ndc_pos, dtype=np.float32)         # (3, 3)
        self.clip = np.array(clip_pos, dtype=np.float32)       # (3, 4)
        self.inv_w = np.array(inv_w, dtype=np.float32) if inv_w is not None else None
        self.depth = self.ndc[:, 2]  # z after perspective divide

        # --- Appearance ---
        self.texture = texture
        self.base_color = np.array(base_color, dtype=np.float32) if base_color is not None else np.array([1.0, 1.0, 1.0])
        self.uv = np.array(uv, dtype=np.float32) if uv is not None else None
        self.normals = np.array(normals, dtype=np.float32) if normals is not None else None

        # --- Precompute for perspective-correct interpolation ---
        if uv is not None and inv_w is not None:
            self.u_over_w = self.uv[:, 0] * self.inv_w  # type: ignore ("None" will occur at this point)
            self.v_over_w = self.uv[:, 1] * self.inv_w  # type: ignore ("None" will occur at this point)
        else:
            self.u_over_w = None
            self.v_over_w = None

        if normals is not None and inv_w is not None:
            self.normals_over_w = self.normals * self.inv_w[:, None]  # type: ignore ("None" will occur at this point)
        else:
            self.normals_over_w = None

        # --- Precompute raster helpers ---
        self.area = self._compute_area()
        if self.area != 0:
            self.inv_area = 1.0 / self.area
            self.edge_coeffs = self._compute_edge_coeffs()
            self.bounding_box = self._compute_bounding_box()
        else:
            self.inv_area = 0
            self.edge_coeffs = ((0,0,0),(0,0,0),(0,0,0))
            self.bounding_box = (0,0,0,0)

    def _compute_area(self):
        a, b, c = self.screen
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    def _compute_edge_coeffs(self):
        def edge(p1, p2):
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = p1[0] * p2[1] - p1[1] * p2[0]
            return (A, B, C)
        p1, p2, p3 = self.screen
        return (edge(p2, p3), edge(p3, p1), edge(p1, p2))

    def _compute_bounding_box(self):
        min_x = max(int(np.floor(np.min(self.screen[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(self.screen[:, 0]))),  global_config.screen_width.val - 1)
        min_y = max(int(np.floor(np.min(self.screen[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(self.screen[:, 1]))), global_config.screen_height.val - 1)
        return (min_x, min_y, max_x, max_y)
    
# ---------------------------------------------------
# Rasterizer: fill a triangle (inner loop only)
# ---------------------------------------------------
@Profiler.timed()
def fill_triangle(tri: Triangle, z_buffer: np.ndarray, rgb_buffer: np.ndarray):
    if tri.area == 0:
        return  # degenerate

    # --- Prep bounding box grid ---
    min_x, min_y, max_x, max_y = tri.bounding_box
    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(xs, ys)

    # --- Edge functions ---
    (A0, B0, C0), (A1, B1, C1), (A2, B2, C2) = tri.edge_coeffs
    w0 = A0 * X + B0 * Y + C0
    w1 = A1 * X + B1 * Y + C1
    w2 = A2 * X + B2 * Y + C2

    # Inside mask
    mask = ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))

    # Barycentrics
    w0_n = w0 * tri.inv_area
    w1_n = w1 * tri.inv_area
    w2_n = w2 * tri.inv_area

    # --- Depth interpolation ---
    z = w0_n * tri.depth[0] + w1_n * tri.depth[1] + w2_n * tri.depth[2]

    # Sub-buffers
    sub_z = z_buffer[min_y:max_y+1, min_x:max_x+1]
    sub_rgb = rgb_buffer[min_y:max_y+1, min_x:max_x+1]

    # Z-test
    update_mask = mask & (z < sub_z)
    sub_z[update_mask] = z[update_mask]

    # --- Shading modes ---
    if tri.texture is not None and tri.uv is not None:
        # Perspective-correct interpolation using precomputed u/w, v/w, inv_w
        u_w = w0_n * tri.u_over_w[0] + w1_n * tri.u_over_w[1] + w2_n * tri.u_over_w[2]  # type: ignore
        v_w = w0_n * tri.v_over_w[0] + w1_n * tri.v_over_w[1] + w2_n * tri.v_over_w[2]  # type: ignore
        inv_w = w0_n * tri.inv_w[0] + w1_n * tri.inv_w[1] + w2_n * tri.inv_w[2]  # type: ignore

        u = u_w / inv_w
        v = v_w / inv_w

        tex_x = (u * tri.texture.width).astype(np.int32)
        tex_y = (v * tri.texture.height).astype(np.int32)
        tex_x = np.clip(tex_x, 0, tri.texture.width - 1)
        tex_y = np.clip(tex_y, 0, tri.texture.height - 1)

        sampled_color = tri.texture.image[tex_y, tex_x] * 255.0
        sub_rgb[update_mask] = sampled_color[update_mask]

    elif tri.normals is not None:
        # PSEUDOCODE: interpolate normals for smooth shading
        # normal = normalize( w0_n * tri.normals[0] + ... )
        # color = lambert_shade(normal, light_dir) * tri.base_color
        pass

    else:
        # Simple fake shading by dot product with a constant light dir
        # PSEUDOCODE:
        # light_dir = normalize(np.array([0,0,-1]))
        # face_normal = compute_face_normal(tri.screen)
        # intensity = max(0, dot(face_normal, light_dir))
        # sub_rgb[update_mask] = (tri.base_color * intensity * 255).astype(np.uint8)
        pass