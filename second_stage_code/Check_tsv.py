import pandas as pd
import os

# Define the folder path
folder_path = 

# Iterate over all .tsv files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tsv'):
        file_path = os.path.join(folder_path, filename)

        # Load the data from the file
        data = pd.read_csv(file_path, sep='\t')

        # Print the filename as a title
        print(f'--- {filename} ---')

        # Print the data
        print(data)
        print('\n')  # Add a newline for better readability
