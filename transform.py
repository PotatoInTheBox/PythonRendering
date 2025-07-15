import numpy as np
from numpy.typing import NDArray

class Transform:
    def __init__(self, rotation=None, scale=None, translation=None):
        self._rotation = self._parse_rotation(rotation) if rotation is not None else np.eye(4)
        self._scale = self._parse_scale(scale) if scale is not None else np.eye(4)
        self._translation = self._parse_translation(translation) if translation is not None else np.eye(4)
        self._matrix = self._translation @ self._rotation @ self._scale

    def get_matrix(self) -> NDArray[np.float64]:
        return self._matrix

    def with_rotation(self, R) -> "Transform":
        return Transform(rotation=self._parse_rotation(R),
                         scale=self._scale,
                         translation=self._translation)

    def with_scale(self, S) -> "Transform":
        return Transform(rotation=self._rotation,
                         scale=self._parse_scale(S),
                         translation=self._translation)

    def with_translation(self, T) -> "Transform":
        return Transform(rotation=self._rotation,
                         scale=self._scale,
                         translation=self._parse_translation(T))

    def __matmul__(self, other: "Transform") -> "Transform":
        if not isinstance(other, Transform):
            return NotImplemented
        combined = self._matrix @ other._matrix
        return Transform(rotation=np.eye(4), scale=np.eye(4), translation=combined)  # combined as full matrix

    @staticmethod
    def _parse_rotation(R):
        if isinstance(R, Transform):
            return R._rotation
        R = np.array(R)
        if R.shape == (3,):
            rx, ry, rz = R
            return Transform._embed_3x3(Transform._euler_to_matrix(rx, ry, rz))
        elif R.shape == (3,3):
            return Transform._embed_3x3(R)
        elif R.shape == (4,4):
            return R
        else:
            raise ValueError("Rotation must be 3-vector, 3x3, or 4x4")

    @staticmethod
    def _parse_scale(S):
        if isinstance(S, Transform):
            return S._scale
        S = np.array(S)
        if S.shape == (3,):
            mat = np.eye(4)
            mat[0,0], mat[1,1], mat[2,2] = S
            return mat
        elif S.shape == (3,3):
            return Transform._embed_3x3(S)
        elif S.shape == (4,4):
            return S
        else:
            raise ValueError("Scale must be 3-vector, 3x3, or 4x4")

    @staticmethod
    def _parse_translation(T):
        if isinstance(T, Transform):
            return T._translation
        T = np.array(T)
        if T.shape == (3,):
            mat = np.eye(4)
            mat[:3,3] = T
            return mat
        elif T.shape == (4,4):
            return T
        else:
            raise ValueError("Translation must be 3-vector or 4x4")

    @staticmethod
    def _embed_3x3(mat3):
        mat4 = np.eye(4)
        mat4[:3,:3] = mat3
        return mat4

    @staticmethod
    def _euler_to_matrix(rx, ry, rz):
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz @ Ry @ Rx
    