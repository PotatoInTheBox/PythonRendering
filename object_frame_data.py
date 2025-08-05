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
        self._homogeneous_vertices: Optional[NDArray[np.float64]] = None
        self._model_matrix: Optional[NDArray[np.float64]] = None  # shape (4, 4)
        self._clip_space_vertices: Optional[NDArray[np.float64]] = None
        self._faces_kept: Optional[NDArray[np.bool]] = None
        self._ndc_vertices: Optional[NDArray[np.float64]] = None
        self._inverse_w: Optional[NDArray[np.float64]] = None
        self._world_vertices: Optional[NDArray[np.float64]] = None
        self._world_space_triangles: Optional[NDArray[np.float64]] = None
        self._normals: Optional[NDArray[np.float64]] = None
        self._faces: Optional[NDArray[np.int32]] = None
        self._uv_faces: Optional[NDArray[np.int32]] = None
        self._screen_space_triangles: Optional[NDArray[np.float64]] = None
        self._normal_faces: Optional[NDArray[np.int32]] = None
        
        # variables that can be None
        self.texture: Texture|None = None
        self.uv_coords: NDArray[np.float64] = np.array([], dtype=np.float64)
        
        # More temporary data
        self.colors: NDArray[np.float64]

    @property
    def homogeneous_vertices(self) -> NDArray[np.float64]:
        if self._homogeneous_vertices is None:
            raise RuntimeError("homogeneous_vertices accessed before being set")
        return self._homogeneous_vertices
    @homogeneous_vertices.setter
    def homogeneous_vertices(self, value: NDArray[np.float64]) -> None:
        self._homogeneous_vertices = value
    
    @property
    def model_matrix(self) -> NDArray[np.float64]:
        if self._model_matrix is None:
            raise RuntimeError("model_matrix accessed before being set")
        return self._model_matrix
    @model_matrix.setter
    def model_matrix(self, value: NDArray[np.float64]) -> None:
        self._model_matrix = value
    
    @property
    def clip_space_vertices(self) -> NDArray[np.float64]:
        if self._clip_space_vertices is None:
            raise RuntimeError("clip_space_vertices accessed before being set")
        return self._clip_space_vertices 
    @clip_space_vertices.setter
    def clip_space_vertices(self, value: NDArray[np.float64]) -> None:
        self._clip_space_vertices = value

    @property
    def faces_kept(self) -> NDArray[np.bool]:
        if self._faces_kept is None:
            raise RuntimeError("faces_kept accessed before being set")
        return self._faces_kept 
    @faces_kept.setter
    def faces_kept(self, value: NDArray[np.bool]) -> None:
        self._faces_kept = value
    
    @property
    def ndc_vertices(self) -> NDArray[np.float64]:
        if self._ndc_vertices is None:
            raise RuntimeError("ndc_vertices accessed before being set")
        return self._ndc_vertices
    @ndc_vertices.setter
    def ndc_vertices(self, value: NDArray[np.float64]) -> None:
        self._ndc_vertices = value
    
    @property
    def inverse_w(self) -> NDArray[np.float64]:
        if self._inverse_w is None:
            raise RuntimeError("inverse_w accessed before being set")
        return self._inverse_w 
    @inverse_w.setter
    def inverse_w(self, value: NDArray[np.float64]) -> None:
        self._inverse_w = value
    
    @property
    def world_vertices(self) -> NDArray[np.float64]:
        if self._world_vertices is None:
            raise RuntimeError("world_vertices accessed before being set")
        return self._world_vertices 
    @world_vertices.setter
    def world_vertices(self, value: NDArray[np.float64]) -> None:
        self._world_vertices = value
    
    @property
    def world_space_triangles(self) -> NDArray[np.float64]:
        if self._world_space_triangles is None:
            raise RuntimeError("world_space_triangles accessed before being set")
        return self._world_space_triangles 
    @world_space_triangles.setter
    def world_space_triangles(self, value: NDArray[np.float64]) -> None:
        self._world_space_triangles = value
    
    @property
    def normals(self) -> NDArray[np.float64]:
        if self._normals is None:
            raise RuntimeError("normals accessed before being set")
        return self._normals 
    @normals.setter
    def normals(self, value: NDArray[np.float64]) -> None:
        self._normals = value
    
    @property
    def faces(self) -> NDArray[np.int32]:
        if self._faces is None:
            raise RuntimeError("faces accessed before being set")
        return self._faces 
    @faces.setter
    def faces(self, value: NDArray[np.int32]) -> None:
        self._faces = value
    
    @property
    def uv_faces(self) -> NDArray[np.int32]:
        if self._uv_faces is None:
            raise RuntimeError("uv_faces accessed before being set")
        return self._uv_faces 
    @uv_faces.setter
    def uv_faces(self, value: NDArray[np.int32]) -> None:
        self._uv_faces = value

    @property
    def screen_space_triangles(self) -> NDArray[np.float64]:
        if self._screen_space_triangles is None:
            raise RuntimeError("screen_space_triangles accessed before being set")
        return self._screen_space_triangles
    @screen_space_triangles.setter
    def screen_space_triangles(self, value: NDArray[np.float64]) -> None:
        self._screen_space_triangles = value

    # @property
    # def texture(self) -> Texture:
    #     if self._texture is None:
    #         raise RuntimeError("texture accessed before being set")
    #     return self._texture
    # @texture.setter
    # def texture(self, value: Texture) -> None:
    #     self._texture = value

    @property
    def normal_faces(self) -> NDArray[np.int32]:
        if self._normal_faces is None:
            raise RuntimeError("normal_faces accessed before being set")
        return self._normal_faces
    @normal_faces.setter
    def normal_faces(self, value: NDArray[np.int32]) -> None:
        self._normal_faces = value
