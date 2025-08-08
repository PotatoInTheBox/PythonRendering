# object_frame_data.py

# temporary solution. I have a lot of data and it's disorganized.
# I need to put it somewhere since I'm getting tired of passing super large
# arguments.

from renderable_object import RenderableObject
import numpy as np
from typing import Optional
from numpy.typing import NDArray

from texture import Texture

# using property and setter to have more control over how we use the data
class ObjectFrameData:
    def __init__(self, renderable_object: "RenderableObject"):
        # Properties that cannot be None before being used.
        self.object: "RenderableObject" = renderable_object
        """Original object. Vertices and normals (if any) are at object space here."""
        self.model_matrix: NDArray[np.float64] = None # type: ignore  # shape (4, 4)
        """A `model matrix` to apply to the vertices and normal vertices.\n
        Usually, this will be in combination with the `view matrix` since we don't need
        world space (we can do our math in view space instead)."""
        self.homogeneous_vertices: NDArray[np.float64] = None # type: ignore
        """Convert our vertices to a `4d` shape. This allows it to have a `w` component
        which helps calculate the perspective divide."""
        self.world_space_vertices: NDArray[np.float64] = None # type: ignore
        """Vertices in world position. Similar to OpenGL with up being positive and down
        being negative. Right being positive and left being negative. Z follows the right-hand rule
        so Z is towards the camera and negative Z is away from the camera.
        Most calculations can be done in view space but world_space_vertices is very cheap to calculate
        so it's fine keeping it for now."""
        self.view_space_vertices: NDArray[np.float64] = None # type: ignore
        """View space vertices are like world space vertices but rotated and transformed relative
        to the camera. Positive Z at this point still points towards the camera."""
        self.clip_space_vertices: NDArray[np.float64] = None # type: ignore
        """Clip space has our vertices after the perspective transform. But without
        perspective divide. We cannot yet use it for drawing to the screen (perspective
        divide needed first). We can still use it for interpolating triangles (since
        their shape will still be the same). We can also use it to clip out of bounds
        components."""
        self.ndc_vertices: NDArray[np.float64] = None # type: ignore
        """NDC vertices (Normalized Device Coordinates) are vertices after perspective divide.
        They map onto a normalized screen. They do not keep their original shape so
        interpolating textures or shading will look wrong."""
        self.screen_space_triangles: NDArray[np.float64] = None # type: ignore
        """The actual triangles stored as 3 points relative to screen space."""
        self.faces_kept: NDArray[np.bool] = None # type: ignore
        """Faces currently in use in the pipeline (indices). As we progress, less faces
        will be used due to clipping, culling, etc. This can be used to grab the
        `faces` of a face/triangle array."""
        self.inverse_w: NDArray[np.float64] = None # type: ignore
        """Vertex inverse w data."""
        self.world_space_triangles: NDArray[np.float64] = None # type: ignore
        """TODO"""
        self.face_normals: NDArray[np.float64] = None # type: ignore
        """**FACE** normal (not to be confused with vertex_normals). Contains vertices
        which represent where the triangle/face is **POINTING**. Generally used for
        **back-face culling**, **flat shading**, and geometric tests."""
        self.vertex_normals: NDArray[np.float64] = None # type: ignore
        """**VERTEX** normal (not to be confused with face_normals). Contains vertices
        which represent averaged normals from **SURROUNDING** vertices. Generally used
        for **smooth shading**."""
        self.uv_coords: NDArray[np.float64] = np.array([], dtype=np.float64)
        """The uv coordinates for textures [0,1]. Used to sample textures, pixel per
        pixel, at the fragment shader stage."""
        self.vertex_faces: NDArray[np.int32] = None # type: ignore
        """An array of faces which consist of 3 indicies. These are in reference
        to the original verticies and thus must be the same size as the array
        they are pulling from."""
        self.uv_faces: NDArray[np.int32] = None # type: ignore
        """An array of faces which consist of 3 indicies. These are in reference
        to the original verticies and thus must be the same size as the array
        they are pulling from."""
        self.normal_faces: NDArray[np.int32] = None # type: ignore
        """An array of faces which consist of 3 indicies. These are in reference
        to the original verticies and thus must be the same size as the array
        they are pulling from."""
        
        # variables that can be None
        self.texture: Texture|None = None
        """Contains the texture class with the data used for sampling textures."""

        # More temporary data
        self.face_shade: NDArray[np.float64]
        """Temporary storage for colors assigned to each vertex (primarily for flat shading)."""
