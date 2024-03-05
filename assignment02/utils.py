from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import math


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def length(v):
    return np.linalg.norm(v)


def angle_between(v1, v2):
    v1_u = normalize(v1)[:2]
    v2_u = normalize(v2)[:2]
    return np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))


def angle_between3d(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    cosine_angle = np.clip(dot_product / (magnitude_a * magnitude_b), -1, 1)
    # print(cosine_angle, a, b)
    angle_radians = np.arccos(cosine_angle)
    # print("radians", angle_radians)
    signed_angle = np.sign(dot_product) * angle_radians
    return signed_angle


def plot_nodes_trajs(nodes_by_frame: np.ndarray, name: Optional[str] = None):
    num_frames = nodes_by_frame.shape[0]
    num_nodes = nodes_by_frame.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])
    ax.set_zlim([-2.1, 2.1])
    ax.set_aspect("equal")

    (arm_plot,) = ax.plot([], [], [])
    traj_plots = [
        ax.plot(
            [],
            [],
            [],
        )[0]
        for _ in range(num_nodes - 1)
    ]

    def animate(i, arm_plot, traj_plots):
        arm_plot.set_data(nodes_by_frame[i][:, 0], nodes_by_frame[i][:, 1])
        arm_plot.set_3d_properties(nodes_by_frame[i][:, 2])
        for j in range(num_nodes - 1):
            traj_plots[j].set_data(nodes_by_frame[:i, j + 1, 0], nodes_by_frame[:i, j + 1, 1])
            traj_plots[j].set_3d_properties(nodes_by_frame[:i, j + 1, 2])

    ani = matplotlib.animation.FuncAnimation(
        fig, animate, frames=num_frames, fargs=(arm_plot, traj_plots), interval=16
    )
    if name is not None:
        ani.save(name, writer="pillow", fps=16)
    return ani


def rotate_point_2d(point: np.ndarray, angle: float):
    """Rotate a point (x, y) by a given angle (in radians)"""
    assert point.shape == (2,), f"the shape of points should be (2,) but get {point.shape}"
    x, y = point
    rotated_x = x * math.cos(angle) - y * math.sin(angle)
    rotated_y = x * math.sin(angle) + y * math.cos(angle)
    return np.array([rotated_x, rotated_y])


def rotate_point_3d(point: np.ndarray, axis: np.ndarray, angle: float):
    """Rotate a point (x, y, z) around axis by a given angle (in radians)"""
    assert point.shape == (3,), f"the shape of points should be (3,) but get {point.shape}"
    assert axis.shape == (3,), f"the shape of axis should be (3,) but get {axis.shape}"
    x, y, z = point
    u, v, w = normalize(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated_x = u * (u * x + v * y + w * z) * (1 - cos_a) + x * cos_a + (-w * y + v * z) * sin_a
    rotated_y = v * (u * x + v * y + w * z) * (1 - cos_a) + y * cos_a + (w * x - u * z) * sin_a
    rotated_z = w * (u * x + v * y + w * z) * (1 - cos_a) + z * cos_a + (-v * x + u * y) * sin_a
    return np.array([rotated_x, rotated_y, rotated_z])


def rotation_mat_3d(point: np.ndarray, axis: np.ndarray, angle: float):
    """Rotate a point (x, y, z) around axis by a given angle (in radians)"""
    assert point.shape == (3,), f"the shape of points should be (3,) but get {point.shape}"
    assert axis.shape == (3,), f"the shape of axis should be (3,) but get {axis.shape}"
    x, y, z = point
    u, v, w = normalize(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array(
        [
            [cos_a + u**2 * (1 - cos_a), u * v * (1 - cos_a) - w * sin_a, u * w * (1 - cos_a) + v * sin_a],
            [v * u * (1 - cos_a) + w * sin_a, cos_a + v**2 * (1 - cos_a), v * w * (1 - cos_a) - u * sin_a],
            [w * u * (1 - cos_a) - v * sin_a, w * v * (1 - cos_a) + u * sin_a, cos_a + w**2 * (1 - cos_a)],
        ]
    )
