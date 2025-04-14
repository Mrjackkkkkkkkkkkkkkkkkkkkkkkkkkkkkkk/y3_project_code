import os
import pickle

def fetch_metadata_and_channel_ids(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

            # Ensure the required keys are in the data
            required_keys = ['file_index', 'cluster_index', 'filtered_sorted_channel_ids']
            if not all(key in data for key in required_keys):
                missing_keys = [key for key in required_keys if key not in data]
                raise KeyError(f"The required keys {missing_keys} are missing in the file: {file_path}")

            file_index = data['file_index']
            cluster_index = data['cluster_index']
            filtered_sorted_channel_ids = data['filtered_sorted_channel_ids']

            return file_index, cluster_index, filtered_sorted_channel_ids

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except KeyError as e:
        print(f"Error: {e}")
    except pickle.UnpicklingError:
        print(f"Error: The file at '{file_path}' could not be unpickled. Ensure it is a valid .pkl file.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{file_path}': {e}")
        return None, None, None

def save_metadata_and_channel_ids(output_file_path, file_index, cluster_index, filtered_sorted_channel_ids):
    try:
        # Create a dictionary to store the required fields
        filtered_data = {
            'file_index': file_index,
            'cluster_index': cluster_index,
            'filtered_sorted_channel_ids': filtered_sorted_channel_ids
        }
        with open(output_file_path, 'wb') as file:
            pickle.dump(filtered_data, file)
        print(f"Filtered channel IDs successfully saved to '{output_file_path}'.")
    except Exception as e:
        print(f"An error occurred while saving the data to '{output_file_path}': {e}")

# Paths
filtered_folder = './major_paper_checkpoints/Cori_2016-12-18/filtered_ids_xc_yc'
arrange_folder = './major_paper_checkpoints/Cori_2016-12-18/arrange_ids_isolation'

# Ensure the arrange folder exists
os.makedirs(arrange_folder, exist_ok=True)

# Process each cluster file in the filtered folder
for cluster_file in os.listdir(filtered_folder):
    if cluster_file.startswith("cluster_") and cluster_file.endswith(".pkl"):
        cluster_file_path = os.path.join(filtered_folder, cluster_file)

        # Fetch metadata and channel IDs from the filtered cluster file
        file_index, cluster_index, filtered_sorted_channel_ids = fetch_metadata_and_channel_ids(cluster_file_path)

        if filtered_sorted_channel_ids is not None:
            print(f"Processing {cluster_file}...")

            # Save the extracted data into the arrange folder
            arrange_output_path = os.path.join(arrange_folder, cluster_file)
            save_metadata_and_channel_ids(arrange_output_path, file_index, cluster_index, filtered_sorted_channel_ids)
