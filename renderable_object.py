# renderable_object.py

import numpy as np
from transform import Transform
from numpy.typing import NDArray
from texture import Texture

class RenderableObject:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normalize=True, name="UnnamedObject", uv_faces=[], texcoords=[], texture_obj=None):
        self.vertices: NDArray[np.float64]  # (N, 3) float64
        self.vertices = np.array(vertices, dtype=np.float64)

        self.faces: NDArray[np.int32]  # (M, 3) int32
        self.faces = np.array(faces, dtype=np.int32)
        
        self.uv_faces: NDArray[np.int32]  # (M, 3) int32
        self.uv_faces = np.array(uv_faces, dtype=np.int32)

        self.uv_coords: NDArray[np.float64]  # (K, 2) float64
        self.uv_coords = np.array(texcoords, dtype=np.float64)
        
        self.texture: Texture | None
        self.texture = texture_obj
        
        self.transform = Transform()
        self.name = name
        if normalize:
            # At startup we conver the verticies to values between -1 and 1.
            self.normalize()

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
    def parse_face(point_arr: list[str], reverse_faces: bool) -> tuple[list[tuple[int,int,int]],list[tuple[int,int,int]]]:
        faces = []
        uv_faces = []
        
        if len(point_arr) < 3:
            raise Exception("Cannot build tri if less than 3 points are present!")
        
        for i in range(len(point_arr) - 2):
            v_face_tri = []
            uv_face_tri = []
            for p in [point_arr[0], point_arr[i + 1], point_arr[i + 2]]:
                face_data = p.split('/')
                v_face = face_data[0]  # (vertex index)
                v_face_tri.append(int(v_face) - 1)

                if len(face_data) > 1:
                    uv_face = face_data[1]  # (uv index)
                    if uv_face != '':
                        uv_face_tri.append(int(uv_face) - 1)
                    else:
                        uv_face_tri.append(0)
                
                # TODO read normal index from file
                # n_face = face_data[2]  # (normal index)
            if reverse_faces:
                v_face_tri.reverse()
                uv_face_tri.reverse()
            faces.append(tuple(v_face_tri))
            uv_faces.append(tuple(uv_face_tri))
        return faces, uv_faces

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
        triangles = []
        all_uv_faces = []

        with open(filepath) as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if parts[0] == 'v':
                    vertices.append(tuple(map(float, parts[1:4])))
                elif parts[0] == 'vt':
                    texcoords.append(tuple(map(float, parts[1:3])))
                elif parts[0] == 'f':
                    faces, uv_faces = RenderableObject.parse_face(parts[1:], reverse_faces)
                    triangles.extend(faces)
                    all_uv_faces.extend(uv_faces)
        
        texture_obj: Texture|None = None
        if texture_filepath is not None:
            texture_obj = Texture(texture_filepath)

        return RenderableObject(np.array(vertices), np.array(triangles), name=filepath, uv_faces=all_uv_faces, texcoords=np.array(texcoords), texture_obj=texture_obj)

    @staticmethod
    def from_data(vertices: np.ndarray, faces: np.ndarray, normalize=True):
        return RenderableObject(vertices, faces, normalize)