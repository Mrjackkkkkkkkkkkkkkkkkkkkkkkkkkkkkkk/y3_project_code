import os
import glob
import numpy as np
import pickle

# Folder paths
input_folder = './major_paper_program/Alldata/Cori_2016-12-18'
output_folder = './major_paper_checkpoints/Cori_2016-12-18/clusters'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all files to process
template_waveform_files = sorted(glob.glob(os.path.join(input_folder, 'clusters.templateWaveforms*.npy')))
channel_files = sorted(glob.glob(os.path.join(input_folder, 'clusters.templateWaveformChans*.npy')))

# Ensure the number of files match
if len(template_waveform_files) != len(channel_files):
    raise ValueError("The number of template waveform files and channel files do not match!")

# Process each pair of files
for i, (waveform_path, channel_path) in enumerate(zip(template_waveform_files, channel_files)):
    print(f"Processing file pair {i+1}/{len(template_waveform_files)}:")
    print(f"  Waveform file: {waveform_path}")
    print(f"  Channel file: {channel_path}")

    # Load data
    template_waveforms = np.load(waveform_path)
    template_waveforms_channels = np.load(channel_path)

    # Process each cluster
    for cluster_idx in range(template_waveforms.shape[0]):
        waveforms = template_waveforms[cluster_idx]
        channel_ids = template_waveforms_channels[cluster_idx]

        # Calculate amplitudes
        peak_amplitudes = np.max(waveforms, axis=0)
        trough_amplitudes = np.min(waveforms, axis=0)
        PTP = peak_amplitudes - trough_amplitudes

        # Sort by channel ID
        sorted_indices = np.argsort(channel_ids)
        sorted_channel_ids = channel_ids[sorted_indices]
        sorted_PTP = PTP[sorted_indices]

        # Save data for the current cluster
        cluster_data = {
            'file_index': i + 1,
            'cluster_index': cluster_idx,
            'sorted_channel_ids': sorted_channel_ids,
            'sorted_PTP': sorted_PTP
        }
        cluster_output_file = os.path.join(output_folder, f'cluster_{cluster_idx}.pkl')
        with open(cluster_output_file, 'wb') as f:
            pickle.dump(cluster_data, f)

        # Print Peak-to-Peak Amplitudes
        print(f"    Cluster {cluster_idx}:")
        print(f"      PTP Amplitudes: {sorted_PTP}")

print("\nAll clusters saved to individual files.")
