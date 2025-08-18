# shaders.py

from texture import Texture
import numpy as np
from dataclasses import dataclass
from profiler import Profiler
import debug

def sample(texture: Texture, uv: np.ndarray):
    u = uv[..., 0]  # shape (X, Y)
    v = uv[..., 1]  # shape (X, Y)
    tex_x = np.clip((u * texture.width).astype(np.int32), 0, texture.width - 1)
    tex_y = np.clip((v * texture.height).astype(np.int32), 0, texture.height - 1)
    sampled = texture.image[tex_y, tex_x]
    return sampled

# ========== Shader I/O ==========
# This unfortunarly looks like it will take a performance hit (possibly a x5
# compared to passing data as tuples).
# This is necessary however, because the hours spent on managing data and functions
# is getting out of hand. We must be able to pass arbritrary data around without
# having to specify the argument in every single function as it gets passed around.
# (God I love structs in C).
# https://www.reddit.com/r/learnpython/comments/1fn8u38/fastest_structlike_thing_in_python/
# Using dataclass for organized transfer of data in a performant way.
# https://stackoverflow.com/questions/55248416/python-3-which-one-is-faster-for-accessing-data-dataclasses-or-dictionaries
@dataclass(slots=True)
class VertexInput:
    """Data fed into the vertex shader for each vertex.
    Holds geometry position, surface normal, and texture UVs from the mesh.
    - Do not expect any value to be normalized."""
    worldMatrix: np.ndarray  # (4,4)
    """Object to World matrix for converting vertices into world space."""
    viewMatrix: np.ndarray  # (4,4)
    """World to View matrix for converting vertices into view space."""
    projectionMatrix: np.ndarray  # (4,4)
    """View to Projection matrix for converting vertices into clip space."""
    worldViewMatrix: np.ndarray  # (4,4)
    """Object to View matrix for converting vertices into view space.
    View space is where we have the world but rotated and moved relative
    to the position and rotation of or camera."""
    worldViewProjectionMatrix: np.ndarray  # (4,4)
    """Object to Projection matrix for converting vertices into clip space.
    This is where we have our vertices warped by the perspective view. So the
    triangles may not be entirely accurate."""
    position: np.ndarray  # (4,) homogeneous
    """The **MODEL/OBJECT** (x,y,z,w) position. w is optional and will usually not be passed
    unless there is a specific reason."""
    normal: np.ndarray|None = None    # (3,)
    """The (x,y,z) direction the surface normal if facing. Not guarenteed to be normalized.
    (For smooth shading)"""
    uv: np.ndarray|None = None        # (2,)
    """The (x,y) position on a texture that we should sample from. Not guarenteed to be normalized."""

@dataclass(slots=True)
class VertexOutput:
    """Data output from vertex shader to the rasterizer.
    Contains transformed positions, normals, and UVs for interpolation."""
    world_position: np.ndarray # (3,)
    """The (x,y,z) world position of this vertex."""
    clip_position: np.ndarray  # (4,)
    """The (x,y,z,w) clip position of this vertex. This can be gotten by applying
    the world_view_matrix"""
    normal: np.ndarray|None = None         # (3,)
    """The (x,y,z) direction the surface normal if facing. Not guarenteed to be normalized.
    (For smooth shading).
    Usually we want this in world space."""
    uv: np.ndarray|None = None             # (2,)
    """The (x,y) position on a texture that we should sample from. Not guarenteed to be normalized."""

@dataclass(slots=True)
class FragmentInput:
    """Data for the fragment shader after interpolation.
    Gives per-pixel world position, normal, and UVs for shading.
    * Note, perspective corrections are already applied."""
    world_position: np.ndarray # (3,)
    """`NOT IN USE`. This could probably help with point lighting. If I know where
    the pixel is located in space then I can get the vertex between the pixel position
    and point light, which can then be used to do the dot product with the normal to get
    our diffuse value."""
    face_normal: np.ndarray  # (3)
    """The (x,y,z) vertex representing the face of the triangle this pixel is part of.
    This will be useful for flat shading. NOTE this is not interpolated."""
    normal: np.ndarray|None = None         # (3,)
    """Interpolated (x,y,z) values of where the normal is pointing. For use in shading."""
    uv: np.ndarray|None = None             # (2,)
    """Interpolated (x,y) values of where we should sample the texture from."""
    texture: Texture|None = None
    """The texture object we should sample from."""
    

@Profiler.timed()
def default_vertex_shader(v: VertexInput) -> VertexOutput:
    """The default vertex shader will apply matrix transformations to translate
    out object into world space and clip space. Here we want to do the bare
    minimum to process the vertex."""
    # v.position = v.position.reshape(4) # make sure it is a 4d vertex
    
    # Calculate world positions for stuff like point lights, culling, etc.
    world = (v.worldMatrix @ v.position.T).T
    # Calculate clip positions so we can actually draw the triangles on the screen.
    clip = (v.worldViewProjectionMatrix @ v.position.T).T
    
    # Start writing out output.
    out = VertexOutput(world_position=world, clip_position=clip)
    
    # Only output normals to worldspace if we have any
    if v.normal is not None:
        out.normal = (v.worldMatrix[:3, :3] @ v.normal.T).T
    
    # Copy over the uv's. We aren't doing anything to them yet.
    # What would I even need to modify them for?
    out.uv = v.uv
    
    return out

@Profiler.timed()
def default_fragment_shader(f: FragmentInput) -> np.ndarray:
    """Writes a fragment with (r,g,b,a) values of our triangle.
    * The default fragment shader will try to paint the object to the screen.
    For shading if there is per fragment normals then we use that. Else we use
    the face_normals which was precomputed.
    * We will try to paint if and only if there is a texture and some uv
    coordinates we can pull from. Else we cannot do it and thus simply paint our
    object white (TODO or a color specified as input).
    * By the way. Due to the implementation, the fragment shader runs on ALL
    pixels within the triangle's bounds. This means that precisely HALF the
    pixels will get thrown away as they are not part of the triangle.
    (TODO maybe in the future optimize this? It would double the shader performance.)
    """
    
    # Calculate our diffusion
    # (We will hardcode a global ray light)
    global_light = np.array((0.0,1.0,0.0))
    light_color = np.array((1.0,0.95,0.95))  # slight red
    if f.normal is not None:
        diffusion_amount: np.ndarray = np.clip((np.dot(f.normal, global_light) + 1.0) / 2.0, 0, 1) # (1,). dot product
        diffuse = diffusion_amount[..., np.newaxis] * light_color  # (3,) as RGB. apply light color
    else:
        diffusion_amount = np.clip((np.dot(f.face_normal, global_light) + 1.0) / 2.0, 0, 1)
        diffuse = diffusion_amount * light_color
        H, W = f.world_position.shape[:2]
        diffuse = np.tile(diffuse, (H, W, 1))
    
    object_color = np.array((0.85, 0.9, 1.0))  # slight blue
    
    if f.texture is not None and f.uv is not None:
        sampled = sample(f.texture, f.uv)
        result = sampled * diffuse
    else:
        result = object_color * diffuse
    
    alpha = np.ones(result.shape[:2] + (1,), dtype=result.dtype)  # shape (H, W, 1)
    rgba = np.concatenate((result, alpha), axis=-1)  # shape (H, W, 4)
    return rgba

@Profiler.timed()
def skybox_vertex_shader(v: VertexInput) -> VertexOutput:
    """
    Skybox vertex shader:
    - Ignores camera translation.
    - Uses the model matrix for scaling/positioning.
    - Passes direction or UVs for texturing.
    """
    # Copy view matrix and zero translation
    rot_only = v.viewMatrix.copy()
    rot_only[:3, 3] = 0.0

    # Apply model matrix first (scales cube by 9000)
    world_pos = (v.worldMatrix @ v.position.T).T

    # Apply rotation-only view
    world_dir = (rot_only @ world_pos.T).T

    # Clip space
    clip = (v.projectionMatrix @ world_dir.T).T

    out = VertexOutput(world_position=world_dir, clip_position=clip)
    out.uv = v.uv
    return out

@Profiler.timed()
def skybox_fragment_shader(f: FragmentInput, brightness: float = 1.0) -> np.ndarray:
    """
    Fragment shader for rendering a skybox.
    - Ignores normals and lighting, just samples texture color.
    - Applies uniform brightness multiplier.
    """
    if f.texture is not None and f.uv is not None:
        uv = f.uv.copy()
        uv[...,1] = 1.0 - uv[...,1]
        sampled = sample(f.texture, uv)
        result = sampled * brightness
        # result = sample(f.texture, f.uv) * brightness
    else:
        result = np.array((0.0, 0.0, 0.0), dtype=np.float32)  # fallback black

    alpha = np.ones(result.shape[:2] + (1,), dtype=result.dtype)
    return np.concatenate((result, alpha), axis=-1)