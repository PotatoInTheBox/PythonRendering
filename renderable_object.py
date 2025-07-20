# renderable_object.py

import numpy as np
from transform import Transform

class RenderableObject:
    def __init__(self, vertices, faces, normalize=True):
        self.vertices = vertices
        self.faces = faces
        self.transform = Transform()
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
                    face = []
                    for p in parts[1:4]:
                        v = p.split('/')[0]  # Always use the first part (vertex index)
                        face.append(int(v) - 1)
                    if reverse_faces:
                        face.reverse()  # Reverse winding order
                    triangles.append(tuple(face))

        return RenderableObject(vertices, triangles)

    @staticmethod
    def from_data(vertices, faces, normalize=True):
        return RenderableObject(vertices, faces, normalize)