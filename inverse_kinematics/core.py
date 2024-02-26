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
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))


def plot_nodes_trajs(nodes_by_frame: np.ndarray, name: Optional[str] = None):
    num_frames = nodes_by_frame.shape[0]
    num_nodes = nodes_by_frame.shape[1]

    fig, ax = plt.subplots()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    plt.axis("equal")

    (arm_plot,) = ax.plot([], [])
    traj_plots = [ax.plot([], [])[0] for _ in range(num_nodes - 1)]

    def animate(i, arm_plot, traj_plots):
        arm_plot.set_data(nodes_by_frame[i][:, 0], nodes_by_frame[i][:, 1])
        for j in range(num_nodes - 1):
            traj_plots[j].set_data(nodes_by_frame[:i, j + 1, 0], nodes_by_frame[:i, j + 1, 1])

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


def forward_kinematics(nodes: np.ndarray, delta_angles: np.ndarray):
    """Perform forward kinematics given bones and rotation angles"""
    assert nodes.shape[1] == 2, f"the shape of nodes should be (n, 2) but get {nodes.shape}"
    assert (
        delta_angles.shape[0] == nodes.shape[0]
    ), f"the shape of delta_angles should be ({ nodes.shape[0] }) but get {delta_angles.shape}"
    transformed_nodes = [nodes[0]]
    prev_node_pos = nodes[0]  # Initial point at the origin
    prev_angle = 0.0  # Initial angle

    for i in range(1, len(nodes)):
        global_node_pos = nodes[i]
        delta_angle = delta_angles[i]
        node_pos_offset = global_node_pos - nodes[i - 1]
        rotated_point = prev_node_pos + rotate_point_2d(node_pos_offset, delta_angle + prev_angle)
        transformed_nodes.append(rotated_point)
        prev_node_pos = rotated_point
        prev_angle += delta_angle
    return np.array(transformed_nodes)


def fabrik(nodes: np.ndarray, target_pos: np.ndarray, max_iter: int = 100, precision: float = 1e-5):
    # implement the FABRIK algorithm (Forward And Backward Reaching Inverse Kinematics)
    # we assume that the length of the bones are the same, bone_length = 1
    for _ in range(max_iter):
        forwarded_nodes = [target_pos]
        for i in reversed(range(len(nodes) - 1)):
            dir = normalize(nodes[i] - forwarded_nodes[-1])
            forwarded_node = forwarded_nodes[-1] + dir
            forwarded_nodes.append(forwarded_node)

        backwarded_nodes = [nodes[0]]
        for i in reversed(range(len(forwarded_nodes) - 1)):
            dir = normalize(forwarded_nodes[i] - backwarded_nodes[-1])
            backwarded_node = backwarded_nodes[-1] + dir
            backwarded_nodes.append(backwarded_node)
        nodes = backwarded_nodes
        if length(nodes[-1] - target_pos) < precision:
            break
    return nodes


def ccdik(nodes: np.ndarray, target_pos: np.ndarray, max_iter: int = 100, precision: float = 1e-5):
    # implement the CCD (Cyclic Coordinate Descent) algorithm
    for _ in range(max_iter):
        for i in reversed(range(0, len(nodes))):
            end_to_cur_dir = normalize(nodes[-1] - nodes[i - 1])
            target_to_cur_dir = normalize(target_pos - nodes[i - 1])
            rotate_angels = np.zeros(len(nodes))
            rotate_angels[i] = angle_between(end_to_cur_dir, target_to_cur_dir)
            nodes = forward_kinematics(nodes, rotate_angels)
        if length(nodes[-1] - target_pos) < precision:
            break
    return nodes
