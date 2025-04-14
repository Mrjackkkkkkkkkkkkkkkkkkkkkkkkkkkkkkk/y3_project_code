import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Define the parent directory
parent_dir = 

# Log file path
log_file_path = os.path.join(parent_dir, "nonlinear_processed_files.log")


def load_processed_files():
    """Load the list of already processed files from the log."""
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            return set(log_file.read().splitlines())
    return set()


def save_processed_file(file_name):
    """Append a processed file name to the log."""
    with open(log_file_path, "a") as log_file:
        log_file.write(file_name + "\n")


def find_and_process_pkl_files(parent_dir):
    """Process each cluster_i.pkl file in the specified directory once and plot average loss."""
    filtered_folder_path = os.path.join(parent_dir, "filtered_ids_PTP_xc_yc")
    image_save_folder = os.path.join(parent_dir, "nonlinear_algorithm_images")
    os.makedirs(image_save_folder, exist_ok=True)

    avg_loss_path = os.path.join(image_save_folder, "average_loss_per_iteration.xlsx")
    if os.path.exists(avg_loss_path):
        print(f"📄 Found existing average loss file: {avg_loss_path}")
        plot_average_loss_from_excel_if_exists(image_save_folder)
        return

    processed_files = load_processed_files()
    all_files = [f for f in os.listdir(filtered_folder_path) if f.startswith("cluster_") and f.endswith(".pkl")]
    all_loss_values = []

    for file in all_files:
        if file in processed_files:
            print(f"Skipping already processed file: {file}")
            continue

        file_path = os.path.join(filtered_folder_path, file)
        print(f"Processing file: {file_path}")

        success, loss_values = process_cluster_pkl(file_path, image_save_folder)
        if success:
            all_loss_values.append(loss_values)
        save_processed_file(file)

    # Save and plot average loss
    if all_loss_values:
        max_len = max(len(loss) for loss in all_loss_values)
        loss_matrix = np.zeros((len(all_loss_values), max_len))
        for i, loss in enumerate(all_loss_values):
            loss_matrix[i, :len(loss)] = loss

        valid_counts = (loss_matrix != 0).sum(axis=0)
        loss_matrix[loss_matrix == 0] = np.nan
        avg_loss = np.nanmean(loss_matrix, axis=0)

        df_avg = pd.DataFrame({
            "Iteration": np.arange(1, len(avg_loss) + 1),
            "Average Loss": avg_loss
        })
        df_avg.to_excel(avg_loss_path, index=False)
        print(f"💾 Saved average loss Excel: {avg_loss_path}")

        plot_average_loss_from_excel_if_exists(image_save_folder)
    else:
        print("❌ No loss values collected.")



def load_pkl_data(file_path):
    """Load .pkl data."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_trained_parameters(file_path, trained_x, trained_y, trained_z, trained_alpha):
    """Save trained parameters back to .pkl file."""
    data = load_pkl_data(file_path)
    data['trained_x'], data['trained_y'], data['trained_z'], data['trained_alpha'] = float(trained_x), float(
        trained_y), float(trained_z), float(trained_alpha)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved trained parameters in: {file_path}")


def compute_expression_for_channels(data, x, y, z, alpha):
    """Compute the loss function."""
    ptp_values = np.array(data['filtered_sorted_PTP'])
    x_c_values = np.array(data['filtered_sorted_Xc_coordinates'])
    y_c_values = np.array(data['filtered_sorted_Yc_coordinates'])
    z_adjusted = z - np.sqrt((x - x_c_values) ** 2 + (y - y_c_values) ** 2)
    term_sum = np.sum(ptp_values - (alpha / np.sqrt((x - x_c_values) ** 2 + (y - y_c_values) ** 2 + z_adjusted ** 2)))
    return term_sum ** 2.5


def compute_derivatives(data, x, y, z, alpha):
    """Compute gradients for optimization."""
    ptp_values = np.array(data['filtered_sorted_PTP'])
    x_c_values = np.array(data['filtered_sorted_Xc_coordinates'])
    y_c_values = np.array(data['filtered_sorted_Yc_coordinates'])
    z_adjusted = z - np.sqrt((x - x_c_values) ** 2 + (y - y_c_values) ** 2)
    denominator = np.sqrt((x - x_c_values) ** 2 + (y - y_c_values) ** 2 + z_adjusted ** 2)
    term_diff = ptp_values - (alpha / denominator)
    term_sum = np.sum(term_diff)
    exponent_factor = 2.5 * (term_sum ** 1.5)
    dS_dalpha = exponent_factor * np.sum(-1 / denominator)
    dS_dx = exponent_factor * np.sum(alpha * (x - x_c_values) / (denominator ** 3))
    dS_dy = exponent_factor * np.sum(alpha * (y - y_c_values) / (denominator ** 3))
    dS_dz = exponent_factor * np.sum(alpha * z_adjusted / (denominator ** 3))
    return dS_dalpha, dS_dx, dS_dy, dS_dz


def update_parameters_amsgrad(x, y, z, alpha, learning_rate, dS_dalpha, dS_dx, dS_dy, dS_dz, m, v, v_hat, t, beta1=0.95,
                              beta2=0.9995, epsilon=1e-8):
    """AMSGrad update rule for optimization."""
    grads = {'alpha': dS_dalpha, 'x': dS_dx, 'y': dS_dy, 'z': dS_dz}
    params = {'alpha': alpha, 'x': x, 'y': y, 'z': z}
    for param in params:
        m[param] = beta1 * m[param] + (1 - beta1) * grads[param]
        v[param] = beta2 * v[param] + (1 - beta2) * (grads[param] ** 2)
        v_hat[param] = np.maximum(v_hat[param], v[param])
        params[param] -= learning_rate * m[param] / (np.sqrt(v_hat[param]) + epsilon)
    return params['x'], params['y'], params['z'], params['alpha'], m, v, v_hat

def plot_average_loss_from_excel_if_exists(image_save_folder):
    """If average loss Excel exists, plot it."""
    avg_loss_path = os.path.join(image_save_folder, "average_loss_per_iteration.xlsx")
    if os.path.exists(avg_loss_path):
        df = pd.read_excel(avg_loss_path)
        plt.figure(figsize=(8, 5))
        plt.plot(df["Iteration"], df["Average Loss"], label='Average Loss (Euler)', color="black", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Average Loss")
        plt.title("Average Loss Across All Clusters")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(image_save_folder, "average_loss_curve.png"))
        plt.close()
        print(f"✅ Plotted average loss from: {avg_loss_path}")
    else:
        print("⚠️ No average loss Excel found to plot.")


def process_cluster_pkl(pkl_file_path, image_save_folder):
    """Process a single .pkl file for optimization."""
    data = load_pkl_data(pkl_file_path)

    if not data or 'filtered_sorted_PTP' not in data or len(data['filtered_sorted_PTP']) == 0:
        print(f"Skipping empty or invalid file: {pkl_file_path}")
        return False, []

    x, y, z, alpha = 100, 100, 100, 1
    initial_lr = 0.005
    num_iterations = 500000
    m, v, v_hat = ({'alpha': 0, 'x': 0, 'y': 0, 'z': 0} for _ in range(3))
    t = 0
    loss_values = []

    pbar = tqdm(range(1, num_iterations + 1), desc=f"Optimizing {os.path.basename(pkl_file_path)}", leave=False)

    for i in pbar:
        learning_rate = initial_lr / (1 + 0.00001 * i)
        final_sum = compute_expression_for_channels(data, x, y, z, alpha)
        dS_dalpha, dS_dx, dS_dy, dS_dz = compute_derivatives(data, x, y, z, alpha)
        t += 1
        x, y, z, alpha, m, v, v_hat = update_parameters_amsgrad(
            x, y, z, alpha, learning_rate, dS_dalpha, dS_dx, dS_dy, dS_dz, m, v, v_hat, t)
        loss_values.append(final_sum)

    save_trained_parameters(pkl_file_path, x, y, z, alpha)

    cluster_name = os.path.basename(pkl_file_path).replace(".pkl", "")

    # Optional: save individual loss Excel
    # df_loss = pd.DataFrame({'Iteration': np.arange(1, len(loss_values) + 1), 'Loss': loss_values})
    # df_loss.to_excel(os.path.join(image_save_folder, f"{cluster_name}_loss.xlsx"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, linestyle='-', color='b')
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Final Sum)")
    plt.title(f"Loss Curve: {cluster_name}")
    plt.grid(True)
    # plt.savefig(os.path.join(image_save_folder, f"loss_curve_{cluster_name}.png"))
    plt.close()
    return True, loss_values

# Run the processing pipeline
find_and_process_pkl_files(parent_dir)
