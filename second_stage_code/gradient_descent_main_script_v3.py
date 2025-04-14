import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from partial_derivative_gradient_descent import compute_derivatives

def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_expression_for_channels(data, x, y, z, alpha):
    channel_ids = np.array(data['filtered_sorted_channel_ids'])
    ptp_values = np.array(data['filtered_sorted_PTP'])
    x_c_values = np.array(data['filtered_sorted_Xc_coordinates'])
    y_c_values = np.array(data['filtered_sorted_Yc_coordinates'])

    total_sum = 0
    for i in range(len(channel_ids)):
        ptp_c = ptp_values[i]
        x_c = x_c_values[i]
        y_c = y_c_values[i]
        term = ptp_c - (alpha / np.sqrt((x - x_c)**2 + (z - y_c)**2 + y**2))
        squared_term = term ** 2
        total_sum += squared_term

    return total_sum

def update_parameters(x, y, z, alpha, learning_rate, dS_dalpha, dS_dx, dS_dy, dS_dz):
    alpha_new = alpha - learning_rate * dS_dalpha
    x_new = x - learning_rate * dS_dx
    y_new = y - learning_rate * dS_dy
    z_new = z - learning_rate * dS_dz
    return x_new, y_new, z_new, alpha_new

# --- Settings ---
num_iterations = 500000
learning_rate = 10
initial_params = (100, 100, 100, 1)

# --- Paths ---
parent_folder = 
cluster_folder = os.path.join(parent_folder, "filtered_ids_PTP_xc_yc")
pkl_files = sorted(glob.glob(os.path.join(cluster_folder, "cluster_*.pkl")))

# --- Log file path ---
log_file_path = os.path.join(parent_folder, "processed_clusters.log")

# --- Load already processed files ---
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as f:
        processed_files = set(line.strip() for line in f.readlines())
else:
    processed_files = set()

# --- Filter out already processed files ---
unprocessed_files = [fp for fp in pkl_files if os.path.basename(fp) not in processed_files]

# --- Processing ---
all_loss_arrays = []

# --- Filter out already processed files ---
unprocessed_files = [fp for fp in pkl_files if os.path.basename(fp) not in processed_files]

# --- Fallback: No new clusters to process ---
if not unprocessed_files:
    print("No new clusters to process.")

    # --- Check for existing Excel file ---
    output_excel_path = os.path.join(parent_folder, "average_loss_per_iteration.xlsx")
    if os.path.exists(output_excel_path):
        print(f"Found existing average loss file: {output_excel_path}")
        df = pd.read_excel(output_excel_path)

        # --- Plot from Excel ---
        plt.figure(figsize=(10, 5), dpi=300)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 12

        plt.plot(df["Iteration"], df["Average_Loss"], label='Average Loss (from Excel)', linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Average Total Loss")
        plt.title("Average Loss Function Across All Clusters")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.ticklabel_format(useOffset=False, style='plain', axis='y')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.legend()
        plt.show()
    else:
        print("No Excel file found. Nothing to plot. Exiting.")
    exit()

# --- Proceed if there are unprocessed files ---
print(f"Processing {len(unprocessed_files)} unprocessed clusters...\n")
all_loss_arrays = []

for file_path in tqdm(unprocessed_files, desc="Processing clusters", unit="cluster"):
    file_name = os.path.basename(file_path)

    data = load_pkl_data(file_path)
    x, y, z, alpha = initial_params

    loss_values = []
    for _ in range(num_iterations):
        final_sum = compute_expression_for_channels(data, x, y, z, alpha)
        dS_dalpha, dS_dx, dS_dy, dS_dz = compute_derivatives(data, x, y, z, alpha)
        loss_values.append(final_sum)
        x, y, z, alpha = update_parameters(x, y, z, alpha, learning_rate, dS_dalpha, dS_dx, dS_dy, dS_dz)

    all_loss_arrays.append(loss_values)

    # --- Log the processed file ---
    with open(log_file_path, 'a') as log_f:
        log_f.write(f"{file_name}\n")

# --- Convert to numpy array for easy averaging ---
loss_matrix = np.array(all_loss_arrays)  # shape: (num_clusters, num_iterations)
average_loss = np.mean(loss_matrix, axis=0)

# --- Plot ---
plt.figure(figsize=(10, 5), dpi=300)
plt.rcParams["font.size"] = 12
plt.plot(range(1, num_iterations + 1), average_loss, label='Average Loss', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Average Total Loss")
plt.title("Average Loss Function Across All Clusters")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.legend()
plt.show()

# --- Save to Excel ---
df = pd.DataFrame({"Iteration": np.arange(1, num_iterations + 1), "Average_Loss": average_loss})
output_excel_path = os.path.join(parent_folder, "average_loss_per_iteration.xlsx")
df.to_excel(output_excel_path, index=False)

print(f"\nAverage loss saved to: {output_excel_path}")