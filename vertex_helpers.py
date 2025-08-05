from typing import Tuple
from profiler import Profiler
from transform import Transform
from config import COLOR_WHITE
from config import global_config
from renderable_object import RenderableObject

import numpy as np

render_config = global_config

@Profiler.timed()
def apply_vertex_wave_shader(verticies: np.ndarray, amplitude: float, period: float, speed: float, frame_count: int, angle: float) -> np.ndarray:
    """
    Applies a sine wave transformation to all vertices in a mesh (vectorized).

    This function simulates a "wave shader" by offsetting the Y-coordinate
    of each vertex using a sine function. It operates on all vertices at once
    for efficiency using NumPy.

    Parameters:
        verticies (np.ndarray): 
            Array of shape (N, 4) containing homogeneous vertex positions.
        amplitude (float): 
            Maximum wave height. Larger values create taller waves.
        period (float): 
            Frequency of the wave. Larger values create shorter wavelengths.
        speed (float): 
            Speed factor controlling how the wave changes over time.

    Returns:
        np.ndarray:
            The transformed vertices with applied Y-axis wave offset (shape (N, 4)).

    Notes:
        - This function builds a unique translation matrix for each vertex
        (shape (N, 4, 4)) and applies it using batch matrix multiplication.
        - The wave displacement is computed as:
            offset = sin((x + frame_count * speed) * period) * amplitude
        - Vectorized using NumPy for efficiency (no Python loops).
    """
    # TODO abstract this further somehow. This is per vertex instead of an entire
    # object so it's a bit weird.
    # Add vertex wave shader
    x_pos = verticies[:, 0]
    # Calculate the wave offsets for all vertices at once
    offsets = np.sin((x_pos + frame_count * speed) * period) * amplitude  # shape (N,)
    # Build a translation matrix for each vertex: shape (N, 4, 4)
    translations = np.array([
        [[1, 0, 0, 0],
        [0, 1, 0, offset],
        [0, 0, 1, 0],
        [0, 0, 0, 1]] for offset in offsets
    ])
    # Multiply each vertex (shape (N,4,1)) by its translation matrix (N,4,4)
    V_h = verticies[:, :, None]  # (N,4,1)
    V_transformed = np.matmul(translations, V_h)  # (N,4,1)
    verticies = V_transformed[:, :, 0]  # (N,4)
    return verticies

@Profiler.timed()
def compute_view_matrix():
    """
    Computes the camera view matrix from camera rotation and position.

    Returns:
        np.ndarray:
            A 4x4 view transformation matrix (rotation and translation).
    """
    pitch, yaw, roll = render_config.camera_rotation.val
    R_cam = Transform().with_rotation([pitch, yaw, roll])
    R_view = R_cam.get_matrix().T  # inverse of rotation matrix is transpose            

    # R_view must be 4d in order to be used in the matrix
    R_view_4d = np.eye(4)
    R_view_4d[:3, :3] = R_view[:3, :3]

    # Translation to move world relative to camera
    T_view = np.eye(4)
    camera_pos = np.array(render_config.camera_position.val)
    T_view[:3, 3] = -camera_pos

    # View matrix = rotation * translation
    view_matrix = R_view_4d @ T_view
    return view_matrix

@Profiler.timed()
def prepare_vertices(obj: RenderableObject) -> np.ndarray:
    """
    Converts object vertices into homogeneous coordinates.

    Parameters:
        obj (RenderableObject):
            The object containing vertex positions as (N, 3).

    Returns:
        np.ndarray:
            Vertex array in homogeneous coordinates of shape (N, 4).
    """
    # Assuming: vertex_list = [(x, y, z), ...]
    V = np.array(obj.vertices)  # Shape: (N, 3)

    # Add homogeneous coordinate
    # TODO possibly implement column-vector
    V = np.hstack([V, np.ones((V.shape[0], 1))])  # Shape: (N, 4)
    return V

@Profiler.timed()
def get_model_matrix(obj: RenderableObject, angle: float) -> np.ndarray:
    """
    Computes the model transformation matrix for the given object.

    Parameters:
        obj (RenderableObject):
            The renderable object whose transform is used.

    Returns:
        np.ndarray:
            A 4x4 model transformation matrix.
    """
    # ========= MODEL SPACE → WORLD SPACE =========
    # Create a matrix for rotating the model.
    ROTATED_MODEL = obj.transform.copy()
    ROTATED_MODEL.rotate([angle, angle, 0])
    # Use our newly completed matrix for future calculations.
    model_matrix = ROTATED_MODEL.get_matrix()
    return model_matrix

@Profiler.timed()
def project_vertices(V_model: np.ndarray, model_matrix: np.ndarray, view_matrix: np.ndarray, projection_matrix: np.ndarray):
    """
    Projects vertices from model space into clip space.

    Parameters:
        V_model (np.ndarray):
            Homogeneous model-space vertices (N, 4).
        model_matrix (np.ndarray):
            Model transformation matrix (4x4).
        view_matrix (np.ndarray):
            View transformation matrix (4x4).
        projection_matrix (np.ndarray):
            Projection matrix (4x4).

    Returns:
        np.ndarray:
            Vertices in clip space (N, 4).
    """
    # Combine all transforms into a single 4x4 matrix
    M = projection_matrix @ view_matrix @ model_matrix

    # ========= MODEL → CLIP SPACE =========
    V_clip: np.ndarray = (M @ V_model.T).T  # Shape: (N, 4)

    return V_clip

@Profiler.timed()
def cull_faces(V_clip: np.ndarray, faces: np.ndarray):
    """
    Performs frustum and backface culling on faces using clip-space coordinates.

    Parameters:
        V_clip (np.ndarray):
            Clip-space vertex positions (N, 4).
        faces (np.ndarray):
            Face indices referencing vertices (F, 3).

    Returns:
        np.ndarray:
            Boolean mask of length F, where True means face is kept.
    """
    verts = V_clip[faces]  # (F, 3, 4)

    # 1. ANY vertex behind camera (z <= 0) → drop
    behind_camera = (verts[..., 3] <= 0).any(axis=1)

    # 2. ALL vertices outside clip bounds → drop
    abs_coords = np.abs(verts[..., :3])  # (F, 3, 3)
    # outside_clip = (abs_coords > verts[..., [3]]).all(axis=(1, 2))
    
    x, y, z, w = verts[..., 0], verts[..., 1], verts[..., 2], verts[..., 3]

    # Check each plane
    outside_left   = (x < -w)
    outside_right  = (x >  w)
    outside_bottom = (y < -w)
    outside_top    = (y >  w)
    outside_near   = (z < -w)
    outside_far    = (z >  w)

    # Face fully outside if all vertices are outside the same plane
    fully_outside = (
        outside_left.all(axis=1) |
        outside_right.all(axis=1) |
        outside_bottom.all(axis=1) |
        outside_top.all(axis=1) |
        outside_near.all(axis=1) |
        outside_far.all(axis=1)
    )
    
    CLIP_ANY_OUTSIDE = False
    
    if CLIP_ANY_OUTSIDE:
        # Drop faces if ANY vertex is outside clip bounds
        outside_clip_any = (abs_coords > verts[..., [3]]).any(axis=(1, 2))
    
    # Compute normals in clip space
    a = verts[:, 1, :3] / verts[:, 1, [3]] - verts[:, 0, :3] / verts[:, 0, [3]]
    b = verts[:, 2, :3] / verts[:, 2, [3]] - verts[:, 0, :3] / verts[:, 0, [3]]
    normals = np.cross(a, b)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    
    backfacing = normals[:, 2] < 0  # Facing -Z means away from camera → drop

    # Faces to keep
    faces_kept = ~(behind_camera | fully_outside | backfacing)
    return faces_kept

@Profiler.timed()
def perspective_divide(V_clip: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """
    Converts vertices from clip space to normalized device coordinates (NDC).

    Parameters:
        V_clip (np.ndarray):
            Clip-space vertex positions (N, 4).

    Returns:
        np.ndarray:
            NDC positions (N, 3), after dividing by w.
    """
    # Perspective divide
    V_ndc = V_clip[:, :3] / V_clip[:, 3:4]  # Shape: (N, 3)
    inv_w = 1.0 / V_clip[:, 3:4]
    return V_ndc, inv_w


@Profiler.timed()
def compute_world_vertices(V_model: np.ndarray, model_matrix: np.ndarray) -> np.ndarray:
    """
    Transforms model-space vertices into world space.

    Parameters:
        V_model (np.ndarray):
            Homogeneous model-space vertices (N, 4).
        model_matrix (np.ndarray):
            Model transformation matrix (4x4).

    Returns:
        np.ndarray:
            World-space vertex positions (N, 3).
    """
    # ========= MODEL → WORLD FOR LIGHTING =========
    V_world = (model_matrix @ V_model.T).T[:, :3]  # apply model matrix, ignore w
    return V_world

@Profiler.timed()
def compute_world_triangles(V_world: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Retrieves triangles from world-space vertices using face indices.

    Parameters:
        V_world (np.ndarray):
            World-space vertices (N, 3).
        faces (np.ndarray):
            Face indices (F, 3).

    Returns:
        np.ndarray:
            Triangles in world space of shape (F, 3, 3).
    """
    tri_world = V_world[faces]
    return tri_world

@Profiler.timed()
def compute_normals(tri_world: np.ndarray) -> np.ndarray:
    """
    Computes normalized face normals for triangles.

    Parameters:
        tri_world (np.ndarray):
            Triangles in world space (F, 3, 3).

    Returns:
        np.ndarray:
            Unit normals for each face (F, 3).
    """
    # Precompute all normals of all faces
    a = tri_world[:,1] - tri_world[:,0]  # (N, 3)
    b = tri_world[:,2] - tri_world[:,0]  # (N, 3)
    normals = np.cross(a, b)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals

@Profiler.timed()
def compute_lighting(normals: np.ndarray, light: np.ndarray) -> np.ndarray:
    """
    Computes per-face lighting using Lambertian shading.

    Parameters:
        normals (np.ndarray):
            Normalized face normals (F, 3).
        light (np.ndarray):
            Directional light vector (3,).

    Returns:
        np.ndarray:
            Per-face colors based on light intensity (F, 3).
    """
    # precompute the brightness of all faces
    # Mapped from [-1,1] -> [0,1] so that faces not facing the light
    # still slightly light up.
    brightness = np.clip((normals @ light + 1) / 2, 0, 1)

    # precompute colors of all faces
    colors = (brightness[:, None] * COLOR_WHITE).astype(int)
    return colors

@Profiler.timed()
def ndc_to_screen(V_ndc: np.ndarray, faces: np.ndarray, grid_size_x: int, grid_size_y: int) -> np.ndarray:
    """
    Converts NDC coordinates of vertices to screen-space coordinates.

    Parameters:
        V_ndc (np.ndarray):
            Normalized device coordinates (N, 3).
        faces (np.ndarray):
            Face indices (F, 3).

    Returns:
        np.ndarray:
            Triangles in screen space of shape (F, 3, 3), where each vertex has (x, y, z).
    """
    faces = faces  # Shape (F, 3)
    tri_ndc_all = V_ndc[faces]  # Shape (F, 3, 3) —  F faces, 3 verts each, 3 coords each
    # ========= NDC → SCREEN SPACE =========
    # Precompute screen space
    # Extract x, y, z (NDC space)
    xy = tri_ndc_all[:, :, :2]  # (F, 3, 2)
    # z = tri_ndc_all[:, :, 2]    # (F, 3)
    z = -tri_ndc_all[:, :, 2]  # flip the z because somewhere in my logic i flipped it a long time ago

    # Convert x, y to screen coordinates
    grid_x = grid_size_x
    grid_y = grid_size_y
    xy_screen = np.empty_like(xy)
    xy_screen[:, :, 0] = ((xy[:, :, 0] + 1) * 0.5 * grid_x).astype(int)
    xy_screen[:, :, 1] = ((1 - (xy[:, :, 1] + 1) * 0.5) * grid_y).astype(int)

    # Combine x, y, z back
    tri_screen = np.dstack((xy_screen, z[..., None]))  # shape (F, 3, 3)
    
    return tri_screen