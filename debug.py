# debug.py
# I inted to have this file be used while the debugger is running and actively
# pausing the scene. So that I can do stuff like poke into memory while it's
# running and see large swaths of data as an image.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

def draw_array(image: np.ndarray):
    h, w = image.shape[:2]
    fig, ax = plt.subplots()

    norm = Normalize(vmin=image.min(), vmax=image.max())
    im_display = None

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        im_display = ax.imshow(image.squeeze(), norm=norm)
    elif image.ndim == 3 and image.shape[2] == 2:
        img_norm = norm(image)
        im_display = ax.imshow(np.dstack([img_norm, np.zeros((h, w))]))
    elif image.ndim == 3 and image.shape[2] == 3:
        img_norm = norm(image)
        im_display = ax.imshow(img_norm)
    elif image.ndim == 3 and image.shape[2] == 4:
        # Treat as RGBA, normalize RGB channels, keep alpha as is
        rgb_norm = norm(image[..., :3])
        rgba = np.dstack([rgb_norm, image[..., 3]])
        im_display = ax.imshow(rgba)
    else:
        raise ValueError("Unsupported shape")

    rect = patches.Rectangle((0, 0), w-1, h-1, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')

    # Single-arg version for hover tool
    def format_coord(x: float, y: float) -> str:
        xi, yi = int(x + 0.5), int(y + 0.5)
        if 0 <= yi < h and 0 <= xi < w:
            val = image[yi, xi]
            return f"x={xi}, y={yi}, val={val}"
        return ""

    ax.format_coord = format_coord
    plt.show()

def plot_area(title, x_label, y_label, data):
    """
    Plot a 1D array of data with matplotlib.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(data, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_area_distribution(title, x_label, y_label, data):
    """
    Plot a histogram with 1:1 bins (each integer value gets its own bin).
    """
    if not data:
        return

    min_val = int(min(data))
    max_val = int(max(data))
    bins = range(min_val, max_val + 2)  # +2 to include last value

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_vertices_triangles(vertices: np.ndarray, triangles: np.ndarray):
    """
    vertices: (N, 4) or (N, 3) array of vertex positions (ignore 4th component)
    triangles: (M, 3) array of indices into vertices forming triangles
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x,y,z from vertices (ignore w if present)
    verts_xyz = vertices[:, :3]
    verts_xyz = verts_xyz[:, [0, 2, 1]]

    # Plot vertices
    ax.scatter(verts_xyz[:, 0], verts_xyz[:, 1], verts_xyz[:, 2], c='r', s=20) # type: ignore
    
    # Annotate each vertex with its index
    for i, (x, y, z) in enumerate(verts_xyz):
        ax.text(x, y, z, str(i), color='black', fontsize=8)

    # Plot filled triangles with random color at 10% opacity
    for tri in triangles:
        pts = verts_xyz[tri]
        tri_poly = Poly3DCollection([pts])
        tri_poly.set_facecolor(np.append(np.random.rand(3), 0.1))  # RGBA
        tri_poly.set_edgecolor('k')  # optional black edges
        ax.add_collection3d(tri_poly)

    # ax.view_init(elev=-70, azim=-60)

    ax.set_xlabel('X (left-right)')
    ax.set_ylabel('Z (forward-backward)')
    ax.set_zlabel('Y (up-down)')
    ax.set_title('3D Vertices and Triangles')
    plt.show()

def plot_vertices(vertices: np.ndarray):
    """
    vertices: (N, 4) or (N, 3) array of vertex positions (ignore 4th component)
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x,y,z (ignore w if present)
    verts_xyz = vertices[:, :3]
    verts_xyz = verts_xyz[:, [0, 2, 1]]

    # Plot vertices
    ax.scatter(verts_xyz[:, 0], verts_xyz[:, 1], verts_xyz[:, 2], c='r', s=20) # type: ignore

    # Annotate with index
    for i, (x, y, z) in enumerate(verts_xyz):
        ax.text(x, y, z, str(i), color='black', fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('3D Vertices')
    plt.show()