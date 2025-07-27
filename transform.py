# transform.py

import numpy as np
from numpy.typing import NDArray

class Transform:
    """
    Represents a 3D affine transform composed of separate rotation, scale, and translation components.

    Internally stores each component as a 4x4 homogeneous transformation matrix.
    The full transform matrix is computed on demand as:
        Transform = Translation @ Rotation @ Scale

    Methods with_*(...) return a new Transform with the given component replaced.
    Methods rotate(...), scale(...), translate(...) modify this Transform *in-place* by composing
    the given transform *onto* the existing one.

    Use copy() to create a new independent copy.

    Supports matrix multiplication (@) to combine two Transforms by multiplying their full matrices.

    Rotation input formats accepted:
        - Euler angles (3-vector, radians)
        - 3x3 rotation matrix
        - 4x4 rotation matrix

    Scale input formats accepted:
        - 3-vector scale factors
        - 3x3 scale matrix
        - 4x4 scale matrix

    Translation input formats accepted:
        - 3-vector translation
        - 4x4 translation matrix

    Example usage:
        t = Transform()
        t.rotate([0, np.pi/2, 0])     # modifies t in place
        t.translate([1,0,0])
        t2 = t.copy()                 # independent copy of t
    """
    def __init__(self, rotation=None, scale=None, translation=None):
        self._rotation = self._parse_rotation(rotation) if rotation is not None else np.eye(4)
        self._scale = self._parse_scale(scale) if scale is not None else np.eye(4)
        self._translation = self._parse_translation(translation) if translation is not None else np.eye(4)

    def get_matrix(self) -> NDArray[np.float64]:
        """Compute and return the combined 4x4 transform matrix."""
        return self._translation @ self._rotation @ self._scale # type: ignore

    def with_rotation(self, R) -> "Transform":
        """Return a new Transform with rotation replaced by R."""
        rot = self._parse_rotation(R)
        return Transform(rotation=rot, scale=self._scale, translation=self._translation)

    def with_scale(self, S) -> "Transform":
        """Return a new Transform with scale replaced by S."""
        scale = self._parse_scale(S)
        return Transform(rotation=self._rotation, scale=scale, translation=self._translation)

    def with_translation(self, T) -> "Transform":
        """Return a new Transform with translation replaced by T."""
        trans = self._parse_translation(T)
        return Transform(rotation=self._rotation, scale=self._scale, translation=trans)

    def rotate(self, R) -> None:
        """
        In-place composition: applies rotation R *after* current rotation.
        """
        rot = self._parse_rotation(R)
        self._rotation = self._rotation @ rot

    def scale(self, S) -> None:
        """
        In-place composition: applies scaling S *after* current scale.
        """
        scale = self._parse_scale(S)
        self._scale = self._scale @ scale

    def translate(self, T) -> None:
        """
        In-place composition: applies translation T *after* current translation.
        """
        trans = self._parse_translation(T)
        self._translation = self._translation @ trans

    def copy(self) -> "Transform":
        """Return a deep copy of this Transform."""
        new_transform = Transform()
        new_transform._rotation = self._rotation.copy()
        new_transform._scale = self._scale.copy()
        new_transform._translation = self._translation.copy()
        return new_transform

    def __matmul__(self, other: "Transform") -> "Transform":
        """Combine two Transforms by multiplying their full matrices."""
        if not isinstance(other, Transform):
            return NotImplemented
        combined_mat = self.get_matrix() @ other.get_matrix()
        # Store combined matrix in translation component, leave others identity (no decomposition)
        return Transform(rotation=np.eye(4), scale=np.eye(4), translation=combined_mat)

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
    