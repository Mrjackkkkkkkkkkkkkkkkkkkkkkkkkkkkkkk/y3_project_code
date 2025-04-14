import os
import numpy as np
import pickle

# Input file paths
site_positions_file = './major_paper_program/Alldata/Cori_2016-12-18/channels.sitePositions.npy'
site_file = './major_paper_program/Alldata/Cori_2016-12-18/channels.site.npy'

# Output file path
output_file = './major_paper_checkpoints/Cori_2016-12-18/sorted_channel_positions.pkl'

# Load data from .npy files
site_positions = np.load(site_positions_file)  # Shape: (n, 2), where columns are Xc and Yc
site_channels = np.load(site_file)  # Shape: (n,)

# Ensure data shapes are consistent
if site_positions.shape[0] != site_channels.shape[0]:
    raise ValueError("Mismatch between the number of site positions and channel IDs!")

# Throw out replicated channel IDs and their positions
unique_channels, unique_indices = np.unique(site_channels, return_index=True)
unique_positions = site_positions[unique_indices]  # Keep only unique positions

# Extract Xc and Yc
Xc = unique_positions[:, 0]
Yc = unique_positions[:, 1]

# Print sorted values and their counts to the terminal
print("Sorted Channel IDs:", unique_channels)
print("Sorted_Xc_coordinates:", Xc)
print("Sorted_Yc_coordinates:", Yc)

# Print the number of elements in each
print("\nNumber of Channel IDs:", len(unique_channels))
print("Number of Sorted_Xc_coordinates:", len(Xc))
print("Number of Sorted_Yc_coordinates:", len(Yc))

# Save the results to a .pkl file
result_data = {
    'sorted_channel_ids': unique_channels,
    'sorted_Xc_coordinates': Xc,
    'sorted_Yc_coordinates': Yc
}

with open(output_file, 'wb') as f:
    pickle.dump(result_data, f)

print(f"\nChannel IDs and positions saved to {output_file}")
