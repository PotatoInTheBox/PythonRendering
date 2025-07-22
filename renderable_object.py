# renderable_object.py

import numpy as np
from transform import Transform

class RenderableObject:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normalize=True, name="UnnamedObject"):
        self.vertices = vertices
        self.faces = faces
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

    @staticmethod
    def parse_face(point_arr: list[str], reverse_faces: bool) -> list[tuple[int,int,int]]:
        faces = []
        
        if len(point_arr) < 3:
            raise Exception("Cannot build tri if less than 3 points are present!")
        
        for i in range(len(point_arr) - 2):
            face = []
            for p in [point_arr[0], point_arr[i + 1], point_arr[i + 2]]:
                v = p.split('/')[0]  # Always use the first part (vertex index)
                face.append(int(v) - 1)
            if reverse_faces:
                face.reverse()
            faces.append(tuple(face))
        return faces

    @staticmethod
    def load_new_obj(filepath: str, reverse_faces=False):
        """
        Load an OBJ file and optionally reverse triangle winding.

        Args:
            filepath (str): Path to the OBJ file.
            reverse_faces (bool): If True, reverse the order of vertices in each face.
        """
        vertices = []
        triangles = []

        with open(filepath) as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if parts[0] == 'v':
                    vertices.append(tuple(map(float, parts[1:4])))
                elif parts[0] == 'f':
                    faces = RenderableObject.parse_face(parts[1:], reverse_faces)
                    triangles.extend(faces)
                    

        return RenderableObject(np.array(vertices), np.array(triangles), name=filepath)

    @staticmethod
    def from_data(vertices: np.ndarray, faces: np.ndarray, normalize=True):
        return RenderableObject(vertices, faces, normalize)