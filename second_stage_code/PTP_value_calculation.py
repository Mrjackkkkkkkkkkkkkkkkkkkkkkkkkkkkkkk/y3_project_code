import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pickle

# --- Load data ---
template_waveforms_path = './major_paper_program/Alldata/Tatum_2017-12-09/clusters.templateWaveforms.npy'
template_waveforms_channels_path = './major_paper_program/Alldata/Tatum_2017-12-09/clusters.templateWaveformChans.npy'

template_waveforms = np.load(template_waveforms_path)
template_waveforms_channels = np.load(template_waveforms_channels_path)

# --- Choose cluster index ---
cluster_idx = 200
waveforms = template_waveforms[cluster_idx]
channel_ids = template_waveforms_channels[cluster_idx]

# --- Compute peak/trough/PTP ---
peak_amplitudes = np.max(waveforms, axis=0)
trough_amplitudes = np.min(waveforms, axis=0)
PTP = peak_amplitudes - trough_amplitudes

# --- Sort by channel ID for consistency ---
sorted_indices = np.argsort(channel_ids)
sorted_channel_ids = channel_ids[sorted_indices]
sorted_PTP = PTP[sorted_indices]

# --- Save PTP info ---
output_data = {
    'sorted_channel_ids': sorted_channel_ids,
    'sorted_PTP': sorted_PTP
}
output_file_path = './major_paper_checkpoints/Cori_2016-12-17/sorted_PTP_data.pkl'
with open(output_file_path, 'wb') as f:
    pickle.dump(output_data, f)

print(f"PTP data saved to {output_file_path}")
print(f"Number of channels: {len(channel_ids)}")

# --- Plot top N channels by PTP ---
N = 1
top_indices = np.argsort(sorted_PTP)[-N:]  # top N PTP channels
top_channel_ids = sorted_channel_ids[top_indices]
top_waveforms = waveforms[:, sorted_indices[top_indices]]
top_PTP = sorted_PTP[top_indices]

timepoints = np.arange(waveforms.shape[0])

# --- Colormap setup ---
norm = Normalize(vmin=np.min(top_PTP), vmax=np.max(top_PTP))
cmap = plt.get_cmap('viridis')

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(N):
    color = cmap(norm(top_PTP[i]))
    ax.plot(timepoints, top_waveforms[:, i], label=f'Ch {int(top_channel_ids[i])}', color=color, lw=2)

# Create the colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, ax=ax)

# Improve colorbar ticks and label
cbar.set_label('PTP Amplitude (μV)', fontsize=18, labelpad=10)
cbar.ax.tick_params(labelsize=16)  # ⬅️ This makes the tick numbers bigger and clearer

# 🔹 Make tick numbers larger
ax.tick_params(axis='both', labelsize=16)

# Labels and styling
ax.set_xlabel('Time', fontsize=18)
ax.set_ylabel('Amplitude (μV)', fontsize=18)
ax.set_title(f'Top {N} Template Waveforms for One Cluster', fontsize=18)
ax.grid(True)
ax.legend(fontsize=18, loc='upper right', frameon=False)
plt.tight_layout()
plt.show()


