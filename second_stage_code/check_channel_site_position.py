import numpy as np
import os

# Define the input .npy file path
input_file_path = r'clusters.templateWaveforms.npy'

# Define the output directory and file name
output_dir = r'Cori_2016-12-17'
output_file_name = 'clusters_templateWaveforms.txt'
output_file_path = os.path.join(output_dir, output_file_name)

# Load the .npy file
data = np.load(input_file_path)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the data into a text file
np.savetxt(output_file_path, data, fmt='%s')

# Count the number of lines in the text file
with open(output_file_path, 'r') as file:
    line_count = sum(1 for _ in file)

# Confirmation message
print("Loaded Data:")
print(data)
print(f"\nData has been saved as a text file to: {output_file_path}")
print(f"Number of lines in the text file: {line_count}")