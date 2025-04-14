import os
import math
import pickle
import numpy as np
from brainrender import Scene
from vedo import Sphere, Line, Text3D
from brainrender.cameras import cameras
from tqdm import tqdm

# ==================== Helper: Add a Clipped Region ====================
def add_clipped_region(scene, region, side="left", alpha=0.1, color="red"):
    actor = scene.add_brain_region(region, alpha=alpha, color=color)
    origin = [5700, 0, 0]
    normal = [-1, 0, 0] if side == "left" else [1, 0, 0]
    clipped = actor.cut_with_plane(origin=origin, normal=normal)
    scene.add(clipped)
    scene.remove(actor)

# ==================== Helper: Add 3D Legend ====================
def add_3d_legend(scene, region_dict, start_pos=(-1000, 0, 10500), spacing=200):
    x, y, z = start_pos
    for i, (region, color) in enumerate(region_dict.items()):
        label = Text3D(region, pos=(x, y + i * spacing, z), s=130, c=color)
        label.follow_camera()
        label.scale([1, -1, 1])  # Flip Y-axis to fix mirrored text
        scene.add(label)

# ==================== Initialize Scene ====================
scene = Scene(title="Visualization of Spikes in the Mouse Brain")

# ==================== Add Brain Regions ====================
ctx_subregions = {
    "MOp": "red", "MOs": "tomato", "SSp": "orange", "SSs": "gold",
    "GU": "peru", "VISC": "darkkhaki", "AUDp": "seagreen", "AUDv": "mediumseagreen",
    "AUDd": "lightseagreen", "VISp": "mediumpurple", "VISl": "orchid", "VISal": "plum",
    "VISam": "blueviolet", "VISpm": "thistle", "RSP": "skyblue", "RSPv": "lightblue",
    "RSPd": "steelblue", "ACA": "violet", "ACAd": "lightsteelblue", "ACAv": "cornflowerblue",
    "PL": "pink", "ILA": "hotpink", "ORB": "lightcoral", "ORBm": "indianred",
    "ORBl": "firebrick", "ORBvl": "rosybrown", "FRP": "salmon", "PTLp": "lightgoldenrod",
    "TEa": "sandybrown", "ECT": "tan", "PERI": "navajowhite", "ENTl": "burlywood",
    "ENTm": "wheat", "DP": "khaki", "VISli": "slateblue", "VISrl": "mediumorchid"
}

for region, color in ctx_subregions.items():
    scene.add_brain_region(region, hemisphere="right", alpha=0.1, color=color)

# Add floating legend for cortex subregions
add_3d_legend(scene, ctx_subregions, start_pos=(-1000, 0, 10500), spacing=200)

# ==================== Compute Probe Trajectory ====================
def compute_probe_trajectory(entry_point_rl, entry_point_ap, vertical_angle, horizontal_angle, distance_advanced, num_points=200):
    theta_rad = math.radians(vertical_angle)
    phi_rad = math.radians(horizontal_angle)
    probe_path = np.linspace(0, distance_advanced, num_points)
    DV_z_path = probe_path * np.sin(theta_rad)
    RL_y_path = entry_point_rl + probe_path * np.cos(theta_rad) * np.sin(phi_rad)
    AP_x_path = entry_point_ap + probe_path * np.cos(theta_rad) * np.cos(phi_rad)
    return list(zip(RL_y_path, DV_z_path, AP_x_path))

# ==================== Add Probes ====================
entry_point_rl_1 = 2390
entry_point_ap_1 = 3200
vertical_angle_1 = 70
horizontal_angle_1 = 90
distance_advanced_1 = 3200

entry_point_rl_2 = 2000
entry_point_ap_2 = 3200
vertical_angle_2 = 80
horizontal_angle_2 = 0
distance_advanced_2 = 3900

probe_trajectory_1 = compute_probe_trajectory(entry_point_rl_1, entry_point_ap_1, vertical_angle_1, horizontal_angle_1, distance_advanced_1)
probe_trajectory_2 = compute_probe_trajectory(entry_point_rl_2, entry_point_ap_2, vertical_angle_2, horizontal_angle_2, distance_advanced_2)

scene.add(Line(probe_trajectory_1, c="blue", lw=3))
scene.add(Sphere(probe_trajectory_1[-1], r=50, c="blue"))
scene.add(Line(probe_trajectory_2, c="red", lw=3))
scene.add(Sphere(probe_trajectory_2[-1], r=50, c="red"))

# ==================== Midpoint of Probes ====================
probe_midpoint = np.mean([probe_trajectory_1[-1], probe_trajectory_2[-1]], axis=0)

# ==================== Load and Plot .pkl Spike Data ====================
def process_cluster_pkl(file_path, color="green", radius=80):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    if all(k in data for k in ("trained_x", "trained_y", "trained_z")):
        transformed_coord = [
            probe_midpoint[0] + data["trained_x"],
            probe_midpoint[1] + data["trained_y"],
            probe_midpoint[2] + data["trained_z"],
        ]
        scene.add(Sphere(transformed_coord, r=radius, c=color))
    else:
        print(f"Skipping {file_path} — missing trained_x/y/z")

def find_and_process_pkl_files(parent_dir):
    cluster_files = []

    for sub_folder in os.listdir(parent_dir):
        sub_folder_path = os.path.join(parent_dir, sub_folder)
        if os.path.isdir(sub_folder_path):
                # filtered_ids_PTP_xc_yc -> green, regular size
            filtered_path = os.path.join(sub_folder_path, "filtered_ids_PTP_xc_yc")
            if os.path.exists(filtered_path):
                for file in os.listdir(filtered_path):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(filtered_path, file)
                        cluster_files.append((full_path, "green", 50))

            # kernel -> purple, large size
            kernel_path7 = os.path.join(sub_folder_path, "kernel7")
            if os.path.exists(kernel_path7):
                for file in os.listdir(kernel_path7):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(kernel_path7, file)
                        cluster_files.append((full_path, "purple", 50))

            kernel_path11 = os.path.join(sub_folder_path, "kernel11")
            if os.path.exists(kernel_path11):
                for file in os.listdir(kernel_path11):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(kernel_path11, file)
                        cluster_files.append((full_path, "purple", 50))

            kernel_path19 = os.path.join(sub_folder_path, "kernel19")
            if os.path.exists(kernel_path19):
                for file in os.listdir(kernel_path19):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(kernel_path19, file)
                        cluster_files.append((full_path, "purple", 50))

            kernel_path131 = os.path.join(sub_folder_path, "kernel131")
            if os.path.exists(kernel_path131):
                for file in os.listdir(kernel_path131):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(kernel_path131, file)
                        cluster_files.append((full_path, "purple", 50))

            kernel_path137 = os.path.join(sub_folder_path, "kernel137")
            if os.path.exists(kernel_path137):
                for file in os.listdir(kernel_path137):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(kernel_path137, file)
                        cluster_files.append((full_path, "purple", 50))

            kernel_path214 = os.path.join(sub_folder_path, "kernel214")
            if os.path.exists(kernel_path214):
                for file in os.listdir(kernel_path214):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(kernel_path214, file)
                        cluster_files.append((full_path, "purple", 50))

            kernel_path253 = os.path.join(sub_folder_path, "kernel253")
            if os.path.exists(kernel_path253):
                for file in os.listdir(kernel_path253):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(kernel_path253, file)
                        cluster_files.append((full_path, "purple", 50))

            kernel_path358 = os.path.join(sub_folder_path, "kernel358")
            if os.path.exists(kernel_path358):
                for file in os.listdir(kernel_path358):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(kernel_path358, file)
                        cluster_files.append((full_path, "purple", 50))

            vision_path50 = os.path.join(sub_folder_path, "vision50")
            if os.path.exists(vision_path50):
                for file in os.listdir(vision_path50):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(vision_path50, file)
                        cluster_files.append((full_path, "black", 50))

            vision_path278 = os.path.join(sub_folder_path, "vision278")
            if os.path.exists(vision_path278):
                for file in os.listdir(vision_path278):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(vision_path278, file)
                        cluster_files.append((full_path, "black", 50))

            vision_path310 = os.path.join(sub_folder_path, "vision310")
            if os.path.exists(vision_path310):
                for file in os.listdir(vision_path310):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(vision_path310, file)
                        cluster_files.append((full_path, "black", 50))

            vision_path330 = os.path.join(sub_folder_path, "vision330")
            if os.path.exists(vision_path330):
                for file in os.listdir(vision_path330):
                    if file.startswith("cluster_") and file.endswith(".pkl"):
                        full_path = os.path.join(vision_path330, file)
                        cluster_files.append((full_path, "black", 50))

    for file_path, color, radius in tqdm(cluster_files, desc="Processing .pkl cluster files", unit="file"):
        process_cluster_pkl(file_path, color=color, radius=radius)

# ==================== Set Your Data Directory ====================
parent_dir = r"nonlinear"
find_and_process_pkl_files(parent_dir)

# ==================== Final Render ====================
scene.render(camera=cameras["sagittal"])
