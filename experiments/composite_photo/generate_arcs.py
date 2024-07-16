#!/bin/python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple

"""
generate_arcs.py
Created: Cameron Wolfe - 07/15/2024

Generates constant curvature arcs based on coordinates from composite photo.  Saves
arcs to images in the arcs folder.
"""
image_size = (1760, 1320)
arc_width = 16
feather = 2
arc_color = np.array([255, 0, 0], dtype=np.uint8)

# Base of spine
base_position = np.loadtxt(Path("base_position.dat"), delimiter=",")

# Positions of upper corners of discs
disc_positions = np.loadtxt(Path("disc_positions.dat"), delimiter=",")

arc_x = disc_positions[:, 0::2].mean(axis=1)
arc_y = disc_positions[:, 1::2].mean(axis=1)

arc_positions = np.zeros((2, 3))
arc_positions[0, :] = arc_x
arc_positions[1, :] = arc_y


def get_arc_center(x1, x2, y1, y2) -> np.ndarray:

    if x1 == x2:
        return np.array([np.inf, y1]), np.inf, y2 - y1

    y = y1
    x = (x2**2 - x1**2 + (y2 - y1) ** 2) / (2 * (x2 - x1))
    r = np.abs(x1 - x)
    theta = np.arccos(np.abs(x2 - x) / r)
    return np.array([x, y]), r, theta


def get_arc_positions(
    base_position: np.ndarray,
    arc_position: np.ndarray,
    num_segments: int = 4,
    num_positions: int = 100,
) -> np.ndarray:
    x1, y1 = base_position.tolist()
    x2, y2 = arc_position.tolist()

    arc_center, r, theta = get_arc_center(x1, x2, y1, y2)

    theta_arr = np.arange(num_positions + 1) / num_positions * theta * num_segments

    x = arc_center[0] + np.sign(x1 - arc_center[0]) * r * np.cos(theta_arr)
    y = arc_center[1] + np.sign(y2 - arc_center[1]) * r * np.sin(theta_arr)

    return x, y


def get_arc_pixels(
    base_position: np.ndarray,
    arc_position: np.ndarray,
    image_size: Tuple[int, int],
    num_segments: int = 4,
    arc_width: int = 16,
    feather: int = 2,
    color: np.ndarray = np.array([255, 0, 0]),
) -> np.ndarray:
    x1, y1 = base_position.tolist()
    x2, y2 = arc_position.tolist()

    arc_center, radius, theta = get_arc_center(x1, x2, y1, y2)
    y, x = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

    radius_vals = np.sqrt((x - arc_center[0]) ** 2 + (y - arc_center[1]) ** 2)
    valid_radius = np.logical_and(
        radius - arc_width / 2 <= radius_vals, radius_vals <= radius + arc_width / 2
    )

    theta_vals = np.arctan2(
        (y - arc_center[1]) * np.sign(y2 - arc_center[1]),
        (x - arc_center[0]) * np.sign(x2 - arc_center[0]),
    )
    valid_theta = np.logical_and(0 <= theta_vals, theta_vals <= num_segments * theta)

    valid_pixels = np.logical_and(valid_radius, valid_theta)

    red = np.zeros(image_size, dtype=np.uint8)
    green = np.zeros(image_size, dtype=np.uint8)
    blue = np.zeros(image_size, dtype=np.uint8)
    alpha = np.zeros(image_size, dtype=np.uint8)

    red[valid_pixels] = color[0]
    green[valid_pixels] = color[1]
    blue[valid_pixels] = color[2]
    alpha[valid_pixels] = 255

    red = np.expand_dims(red, axis=2)
    green = np.expand_dims(green, axis=2)
    blue = np.expand_dims(blue, axis=2)
    alpha = np.expand_dims(alpha, axis=2)

    canvas = np.concatenate([red, green, blue, alpha], axis=2)

    return canvas


plt.figure(0)
plt.plot([0, 1760, 1760, 0, 0], [0, 0, -1320, -1320, 0])
for i in range(arc_positions.shape[1]):
    canvas = get_arc_pixels(base_position, arc_positions[:, i], image_size)

    Image.fromarray(np.transpose(canvas, [1, 0, 2]), "RGBA").save(f"arcs/{i}.png")

    x, y = get_arc_positions(base_position, arc_positions[:, i])
    plt.figure(0)
    plt.plot(x, -y)
    plt.figure(1)
    plt.imshow(np.transpose(canvas, [1, 0, 2]), interpolation="nearest")

plt.figure(0)
ax = plt.gca()
ax.set_aspect("equal")
plt.title("Arcs in Plot")
plt.figure(1)
ax = plt.gca()
ax.set_aspect("equal")
plt.title("Arcs as Pixels in Image")
plt.show()
