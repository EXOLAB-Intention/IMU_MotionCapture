"""
Quaternion utility functions for kinematics calculations
Provides quaternion operations: multiplication, inversion, conversion to Euler angles, etc.
"""
import numpy as np

def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (pitch, roll, yaw) in degrees.

    Input quaternion format: [w, x, y, z].
    Returns angles in **order** [pitch, roll, yaw] (degrees), where
    - pitch: rotation about Y axis
    - roll:  rotation about X axis
    - yaw:   rotation about Z axis

    Supports input shapes (4,) or (N,4). Output shape is (3,) or (N,3)
    respectively. Uses clipping of the asin argument for numerical stability.
    """
    q = np.asarray(q, dtype=float)

    def _single(qv: np.ndarray) -> np.ndarray:
        if qv.shape != (4,):
            raise ValueError("Quaternion must have shape (4,)")

        w, x, y, z = qv
        norm = np.linalg.norm(qv)
        if norm == 0.0:
            raise ValueError("Zero-norm quaternion")
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Return in order [pitch, roll, yaw] (degrees)
        return np.degrees(np.array([pitch, roll, yaw]))

    if q.ndim == 1:
        return _single(q)
    elif q.ndim == 2:
        if q.shape[1] != 4:
            raise ValueError("Quaternion array must have shape (N,4)")

        norms = np.linalg.norm(q, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("Zero-norm quaternion found in input array")
        qn = (q.T / norms).T  # normalize each quaternion

        w = qn[:, 0]
        x = qn[:, 1]
        y = qn[:, 2]
        z = qn[:, 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        angles = np.vstack((pitch, roll, yaw)).T
        return np.degrees(angles)
    else:
        raise ValueError("Input quaternion must have shape (4,) or (N,4)")


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) to unit norm.
    Supports input shapes (4,) or (N,4). Returns array with same shape.
    """
    q = np.asarray(q, dtype=float)

    if q.ndim == 1:
        norm = np.linalg.norm(q)
        if norm == 0.0:
            raise ValueError("Cannot normalize zero-norm quaternion")
        return q / norm

    elif q.ndim == 2:
        norms = np.linalg.norm(q, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("Cannot normalize zero-norm quaternion in batch")
        return (q.T / norms).T

    else:
        raise ValueError("Input quaternion must have shape (4,) or (N,4)")


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (q1 * q2)

    Quaternions are expected in `[w, x, y, z]` format. Supports inputs of
    shape `(4,)` or `(N, 4)`. If one input is `(N,4)` and the other is `(4,)`,
    the latter will be broadcast to `(N,4)`. Returns the product with the
    same leading dimension as the broadcasted inputs.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    # Scalar (single quaternion) * scalar
    if q1.ndim == 1 and q2.ndim == 1:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    # Promote 1D to 2D for broadcasting convenience
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
    if q2.ndim == 1:
        q2 = q2[np.newaxis, :]

    if q1.shape[1] != 4 or q2.shape[1] != 4:
        raise ValueError("Quaternion arrays must have shape (4,) or (N,4)")

    # Broadcast along axis 0 if needed
    if q1.shape[0] != q2.shape[0]:
        if q1.shape[0] == 1:
            q1 = np.repeat(q1, q2.shape[0], axis=0)
        elif q2.shape[0] == 1:
            q2 = np.repeat(q2, q1.shape[0], axis=0)
        else:
            raise ValueError("Quaternion batch sizes must match or one must be of size 1")

    w1 = q1[:, 0]
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]

    w2 = q2[:, 0]
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.vstack((w, x, y, z)).T


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse.

    For quaternion q = [w, x, y, z], the inverse is q^{-1} = q* / ||q||^2,
    where q* is the conjugate [w, -x, -y, -z]. Supports inputs of shape
    (4,) or (N,4). Returns array with same leading dimension as input.
    """
    q = np.asarray(q, dtype=float)

    if q.ndim == 1: # Single quaternion
        if q.shape[0] != 4:
            raise ValueError("Quaternion must have shape (4,) or (N,4)")
        norm2 = float(np.dot(q, q))
        if norm2 == 0.0:
            raise ValueError("Cannot invert zero-norm quaternion")
        w, x, y, z = q
        conj = np.array([w, -x, -y, -z], dtype=float)
        return conj / norm2

    elif q.ndim == 2:   # Batch of quaternions
        if q.shape[1] != 4:
            raise ValueError("Quaternion array must have shape (N,4)")
        norms2 = np.sum(q * q, axis=1)
        if np.any(norms2 == 0.0):
            raise ValueError("Cannot invert quaternion with zero norm in batch")
        conj = q * np.array([1.0, -1.0, -1.0, -1.0])
        inv = (conj.T / norms2).T
        return inv

    else:
        raise ValueError("Input quaternion must have shape (4,) or (N,4)")


def compute_relative_orientation(
    q_proximal: np.ndarray, 
    q_distal: np.ndarray
) -> np.ndarray:
    """Compute relative orientation from proximal to distal segment.

    Computes q_rel such that q_distal = q_proximal * q_rel, therefore
    q_rel = q_proximal^{-1} * q_distal.

    Supports inputs of shape (4,) or (N,4) for either argument and
    performs broadcasting when one input is (N,4) and the other is (4,).
    The returned quaternion(s) are normalized.
    """
    qp = np.asarray(q_proximal, dtype=float)
    qd = np.asarray(q_distal, dtype=float)

    # Compute inverse of proximal (handles both (4,) and (N,4))
    qp_inv = quaternion_inverse(qp)

    # Compute relative orientation and normalize
    q_rel = quaternion_multiply(qp_inv, qd)
    return quaternion_normalize(q_rel)
