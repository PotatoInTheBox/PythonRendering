import numpy as np
from numpy.typing import NDArray

class Transform:
    def __init__(self):
        self._rotation = np.eye(4)
        self._scale = np.eye(4)
        self._translation = np.eye(4)
        self._cached = None
        self._dirty = True

    def set_rotation(self, R):
        R = np.array(R)
        if R.shape == (3,):  # Treat as Euler angles in radians (XYZ order)
            rx, ry, rz = R
            self._rotation = self._embed_3x3(self._euler_to_matrix(rx, ry, rz))
        elif R.shape == (3, 3):
            self._rotation = self._embed_3x3(R)
        elif R.shape == (4, 4):
            self._rotation = R
        else:
            raise ValueError("Rotation must be 3D vector, 3x3 or 4x4 matrix")
        self._dirty = True

    def set_scale(self, S):
        S = np.array(S)
        if S.shape == (3,):  # Per-axis scaling
            mat = np.eye(4)
            mat[0, 0], mat[1, 1], mat[2, 2] = S
            self._scale = mat
        elif S.shape == (3, 3):
            self._scale = self._embed_3x3(S)
        elif S.shape == (4, 4):
            self._scale = S
        else:
            raise ValueError("Scale must be 3D vector, 3x3 or 4x4 matrix")
        self._dirty = True

    def set_translation(self, T):
        T = np.array(T)
        if T.shape == (3,):
            mat = np.eye(4)
            mat[:3, 3] = T
            self._translation = mat
        elif T.shape == (4, 4):
            self._translation = T
        else:
            raise ValueError("Translation must be 3D vector or 4x4 matrix")
        self._dirty = True

    def get_rotation(self):
        return self._rotation

    def get_scale(self):
        return self._scale

    def get_translation(self):
        return self._translation

    def get_matrix(self) -> NDArray[np.float64]:
        if self._dirty:
            self._cached = self._translation @ self._rotation @ self._scale
            self._dirty = False
        return self._cached # type: ignore

    @staticmethod
    def _embed_3x3(mat3: NDArray) -> NDArray[np.float64]:
        mat4 = np.eye(4)
        mat4[:3, :3] = mat3
        return mat4

    @staticmethod
    def _euler_to_matrix(rx: float, ry: float, rz: float) -> NDArray[np.float64]:
        # Rotation matrices around X, Y, Z
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]])
        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                       [sz, cz, 0],
                       [0, 0, 1]])

        return Rz @ Ry @ Rx  # type: ignore # XYZ order