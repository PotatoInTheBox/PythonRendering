# renderable_object.py

import numpy as np
from transform import Transform
from numpy.typing import NDArray
from texture import Texture
from shaders import default_vertex_shader
from shaders import default_fragment_shader

class RenderableObject:
    """
    A renderable object is an object that contains data such as verticies, triangles, normals, textures, etc.
    It is a object representation of what we read from an object file. Thus, the data shouldn't really be modified.
    Instead, consider writing modified data to a buffer or seperate object.
    
    Modifications are allowed though for ease of access, such as the transform variable.
    This is because it pertains to the object. But be aware that reusing this instance will have the effects
    apply to all other duplicates of this object as well.
    """
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normalize=True, name="UnnamedObject", uv_faces=[], texcoords=[], normals=[], normal_faces=[], texture_obj=None):
        self.vertices: NDArray[np.float64]  # (N, 3) float64
        self.vertices = np.array(vertices, dtype=np.float64)

        self.faces: NDArray[np.int32]  # (M, 3) int32
        self.faces = np.array(faces, dtype=np.int32)
        
        self.uv_faces: NDArray[np.int32]  # (M, 3) int32
        self.uv_faces = np.array(uv_faces, dtype=np.int32)

        self.uv_coords: NDArray[np.float64]  # (K, 2) float64
        self.uv_coords = np.array(texcoords, dtype=np.float64)
        
        self.normals: NDArray[np.float64]  # (N, 3) float64
        self.normals = np.array(normals, dtype=np.float64)
        
        self.normal_faces: NDArray[np.int32]  # (M, 3) int32
        self.normal_faces = np.array(normal_faces, dtype=np.int32)
        
        self.uv_coords: NDArray[np.float64]  # (K, 2) float64
        self.uv_coords = np.array(texcoords, dtype=np.float64)
        
        self.texture: Texture | None
        self.texture = texture_obj
        
        self.transform = Transform()
        self.name = name
        
        self.__has_warned_degenerate_triangles = False
        self.remove_degenerate_triangles()

        if normalize:
            # At startup we conver the verticies to values between -1 and 1.
            self.normalize()
        
        self.vertex_shader = default_vertex_shader
        self.fragment_shader = default_fragment_shader
        
        
    
    def remove_degenerate_triangles(self, eps=1e-12):
        """
        Removes degenerate faces (zero/near-zero area) and their correlated
        uv_faces and normal_faces.
        """
        if len(self.faces) == 0:
            return

        # Grab actual triangle vertices
        tri = self.vertices[self.faces]  # (M, 3, 3)

        # Compute cross product (face normals before normalization)
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        normals = np.cross(a, b)

        # Length = 2 * triangle area
        lengths = np.linalg.norm(normals, axis=1)

        # Mask for valid (non-degenerate) faces
        valid_mask = lengths > eps
        
        # Debug hook: if any degenerate triangles exist
        if self.__has_warned_degenerate_triangles == False and not np.all(valid_mask):
            self.__has_warned_degenerate_triangles = True
            print(f"Warning: Found at least 1 degenerate triangle in {self.name}! Please make sure all triangles have a non-zero area.")
            print("Degenerate triangles will be removed!")

        # Filter faces + correlated attributes
        self.faces = self.faces[valid_mask]
        if len(self.uv_faces) == len(valid_mask):
            self.uv_faces = self.uv_faces[valid_mask]
        if len(self.normal_faces) == len(valid_mask):
            self.normal_faces = self.normal_faces[valid_mask]

    def normalize(self):
        v = np.array(self.vertices)  # Shape: (N, 3)
        min_vals = v.min(axis=0)
        max_vals = v.max(axis=0)
        center = (min_vals + max_vals) / 2
        scale = (max_vals - min_vals).max() / 2
        self.vertices = (v - center) / scale
    
    def load_texture(self, filepath: str):
        self.texture = Texture(filepath)

    @staticmethod
    def parse_face(point_arr: list[str], reverse_faces: bool) -> tuple[list[tuple[int,int,int]],
                                                                    list[tuple[int,int,int]]|None,
                                                                    list[tuple[int,int,int]]|None]:
        faces = []
        uv_faces = []
        normal_faces = []
        
        do_uv = True
        do_normals = True
        
        if len(point_arr) < 3:
            raise Exception("Cannot build tri if less than 3 points are present!")
        
        for i in range(len(point_arr) - 2):
            v_face_tri = []
            uv_face_tri = []
            n_face_tri = []
            for p in [point_arr[0], point_arr[i + 1], point_arr[i + 2]]:
                face_data = p.split('/')
                
                # vertex index
                v_face_tri.append(int(face_data[0]) - 1)

                # uv index
                if do_uv and len(face_data) > 1 and face_data[1] != '':
                    uv_face_tri.append(int(face_data[1]) - 1)
                else:
                    do_uv = False

                # normal index
                if do_normals and len(face_data) > 2 and face_data[2] != '':
                    n_face_tri.append(int(face_data[2]) - 1)
                else:
                    do_normals = False

            if reverse_faces:
                v_face_tri.reverse()
                if do_uv:
                    uv_face_tri.reverse()
                if do_normals:
                    n_face_tri.reverse()

            faces.append(tuple(v_face_tri))
            if do_uv:
                uv_faces.append(tuple(uv_face_tri)) # type: ignore
            else:
                uv_faces = None
            if do_normals:
                normal_faces.append(tuple(n_face_tri)) # type: ignore
            else:
                normal_faces = None

        return faces, uv_faces, normal_faces

    @staticmethod
    def load_new_obj(filepath: str, reverse_faces=False, texture_filepath: str|None=None):
        """
        Load an OBJ file and optionally reverse triangle winding.

        Args:
            filepath (str): Path to the OBJ file.
            reverse_faces (bool): If True, reverse the order of vertices in each face.
        """
        vertices = []
        texcoords = []
        normals = []
        triangles = []
        all_uv_faces = []
        all_normal_faces = []

        with open(filepath) as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if parts[0] == 'v':
                    vertices.append(tuple(map(float, parts[1:4])))
                elif parts[0] == 'vt':
                    texcoords.append(tuple(map(float, parts[1:3])))
                elif parts[0] == 'vn':
                    normals.append(tuple(map(float, parts[1:4])))
                elif parts[0] == 'f':
                    faces, uv_faces, normal_faces = RenderableObject.parse_face(parts[1:], reverse_faces)
                    triangles.extend(faces)
                    if uv_faces is not None:
                        all_uv_faces.extend(uv_faces)
                    if normal_faces is not None:
                        all_normal_faces.extend(normal_faces)
            
            for idx, (face_idx, normal_idx) in enumerate(zip(triangles, all_normal_faces)):
                if -1 in normal_idx:
                    v0, v1, v2 = (np.array(vertices[i]) for i in face_idx)
                    n = np.cross(v1 - v0, v2 - v0)
                    norm_len = np.linalg.norm(n)
                    if norm_len != 0:
                        n /= norm_len
                    normals.append(tuple(n))
                    new_n_idx = len(normals) - 1
                    # NOTE updated normals will all have the same index.
                    # This is because they will all be facing the same direction.
                    # This has the effect of doing flat shading.
                    updated_normal_idx = tuple(new_n_idx if i == -1 else i for i in normal_idx)
                    all_normal_faces[idx] = updated_normal_idx

                            
            for index_tri in all_normal_faces:
                for index in index_tri:
                    if index == -1:
                        raise Exception("Not yet generated")
                    if index >= len(normals) or index < 0:
                        raise Exception("Normal pointing to invalid index")
        
        texture_obj: Texture|None = None
        if texture_filepath is not None:
            texture_obj = Texture(texture_filepath)

        renderable_object = RenderableObject(
            np.array(vertices),
            np.array(triangles),
            name=filepath,
            uv_faces=all_uv_faces,
            texcoords=np.array(texcoords),
            texture_obj=texture_obj,
            normals=np.array(normals),
            normal_faces=all_normal_faces
        )
        
        return renderable_object

    @staticmethod
    def from_data(vertices: np.ndarray, faces: np.ndarray, normalize=True):
        return RenderableObject(vertices, faces, normalize)