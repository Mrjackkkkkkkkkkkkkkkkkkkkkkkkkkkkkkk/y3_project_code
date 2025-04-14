import os
import numpy as np
import pickle

# File paths (Update these paths as needed)
input_file_path = channels.site.npy
output_file_path = channels_site_labeled.txt

# Check if the input file exists
if not os.path.exists(input_file_path):
    print(f"Error: File not found at {input_file_path}")
else:
    # Load the .npy file
    data = np.load(input_file_path)

    # Extract x and y values, assuming x is the first column and y is the second column
    sorted_x = data[:, 0]
    sorted_y = data[:, 1]

    # Create labeled data
    labeled_data = {
        'sorted_channel_ids': np.arange(data.shape[0]),
        'sorted_x': sorted_x,
        'sorted_y': sorted_y
    }

    # Save the labeled data as a .pkl file
    with open(output_file_path, 'wb') as pkl_file:
        pickle.dump(labeled_data, pkl_file)

    print(f"Labeled data saved successfully to {output_file_path}")
