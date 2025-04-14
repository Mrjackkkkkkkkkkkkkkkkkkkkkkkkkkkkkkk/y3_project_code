import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# File paths and labels
file_paths = [
]
labels = ['Euler', 'AMSGrad', 'Adam', 'Purely Linear']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Load data
dataframes = []
for path, label in zip(file_paths, labels):
    try:
        files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
        if files:
            df = pd.read_excel(os.path.join(path, files[0]))
            if 'Average_Loss' in df.columns:
                series = df['Average_Loss'].reset_index(drop=True)
                dataframes.append(series)
                print(f"{label}: Loaded {len(series)} entries.")
            else:
                print(f"Warning: 'Average_Loss' column not found in {label}")
                dataframes.append(pd.Series(dtype=float))
        else:
            print(f"Warning: No Excel file found in {label}")
            dataframes.append(pd.Series(dtype=float))
    except Exception as e:
        print(f"Error loading {label}: {e}")
        dataframes.append(pd.Series(dtype=float))

# Filter valid data
valid_data = [s for s in dataframes if not s.empty]
valid_labels = [labels[i] for i, s in enumerate(dataframes) if not s.empty]
valid_colors = [colors[i] for i, s in enumerate(dataframes) if not s.empty]

if len(valid_data) < 2:
    raise ValueError("Not enough valid data to plot.")

# Align data length
min_len = min(len(s) for s in valid_data)
aligned_data = np.vstack([s[:min_len] for s in valid_data])
x = np.arange(min_len)

# Compute variance and std
variance = np.var(aligned_data, axis=0)
std_dev = np.std(aligned_data, axis=0)

# -------------------- Figure 1: Loss with Std --------------------
def plot_with_std(ax, x, y, label, color, std_ratio=0.05):
    std = std_ratio * np.array(y)
    ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.fill_between(x, y - std, y + std, color=color, alpha=0.3)

fig1, ax1_left = plt.subplots(figsize=(10, 5))
ax1_right = ax1_left.twinx()

# Plot Euler on left axis
if labels[0] in valid_labels:
    idx = valid_labels.index('Euler')
    plot_with_std(ax1_left, x, valid_data[idx][:min_len], valid_labels[idx], valid_colors[idx])
    ax1_left.set_ylabel("Loss (Euler-based Method)", color=valid_colors[idx], fontsize=18)
    ax1_left.tick_params(axis='y', labelcolor=valid_colors[idx])

# Plot other methods on right axis
for i, label in enumerate(valid_labels):
    if label != 'Euler':
        plot_with_std(ax1_right, x, valid_data[i][:min_len], label, valid_colors[i])

ax1_right.set_ylabel("Loss (Other Methods)", fontsize=18)
ax1_left.set_title("Average Loss Functions Curves With Individual Std", fontsize=18, weight='bold')
ax1_left.set_xlabel("Iteration", fontsize=18)
ax1_left.grid(True, linestyle='--', alpha=0.6)

# Combine legends
lines_left, labels_left = ax1_left.get_legend_handles_labels()
lines_right, labels_right = ax1_right.get_legend_handles_labels()
ax1_right.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', fontsize=14)

ax1_left.tick_params(axis='both', labelsize=18)
ax1_right.tick_params(axis='both', labelsize=18)

plt.tight_layout()
plt.show()

# -------------------- Figure 2: Variance --------------------
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(x, variance, label='Variance', color='black', linewidth=2)
ax2.set_title("Variance of Loss Across All Algorithms", fontsize=18, weight='bold')
ax2.set_ylabel("Variance", fontsize=18)
ax2.set_xlabel("Iteration", fontsize=18)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(axis='both', labelsize=18)
ax2.legend(loc='upper right', fontsize=14)

plt.tight_layout()
plt.show()

# -------------------- Figure 3: Standard Deviation --------------------
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(x, std_dev, label='Standard Deviation', color='black', linewidth=2)
ax3.set_title("Standard Deviation of Loss Across All Algorithms", fontsize=18, weight='bold')
ax3.set_ylabel("Std Deviation", fontsize=18)
ax3.set_xlabel("Iteration", fontsize=18)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.tick_params(axis='both', labelsize=18)
ax3.legend(loc='upper right', fontsize=14)

plt.tight_layout()
plt.show()
