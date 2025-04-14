import os
import pickle
import numpy as np

def fetch_sorted_channel_ids(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

            # Ensure the required keys are in the data
            if not all(key in data for key in ('file_index', 'cluster_index', 'sorted_channel_ids')):
                raise KeyError(f"The required keys ('file_index', 'cluster_index', 'sorted_channel_ids') are not all present in the file: {file_path}")

            file_index = data['file_index']
            cluster_index = data['cluster_index']
            sorted_channel_ids = np.array(data['sorted_channel_ids'], dtype=np.float64)

            return file_index, cluster_index, sorted_channel_ids

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except KeyError as e:
        print(f"Error: {e}")
    except pickle.UnpicklingError:
        print(f"Error: The file at '{file_path}' could not be unpickled. Ensure it is a valid .pkl file.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{file_path}': {e}")
        return None, None, None

def fetch_data_from_pkl(file_path, sorted_channel_ids):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

            # Ensure required keys are in the data
            missing_keys = [key for key in ('sorted_channel_ids', 'sorted_Xc_coordinates', 'sorted_Yc_coordinates') if key not in data]
            if missing_keys:
                raise KeyError(f"The required keys {missing_keys} are missing in the file: {file_path}")

            channel_ids = np.array(data['sorted_channel_ids'])
            sorted_x = np.array(data['sorted_Xc_coordinates'])
            sorted_y = np.array(data['sorted_Yc_coordinates'])

            # Filter the data based on sorted_channel_ids
            mask = np.isin(channel_ids, sorted_channel_ids)
            filtered_channel_ids = channel_ids[mask]
            filtered_sorted_Xc = sorted_x[mask]
            filtered_sorted_Yc = sorted_y[mask]

            return filtered_channel_ids, filtered_sorted_Xc, filtered_sorted_Yc

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except KeyError as e:
        print(f"Error: {e}")
    except pickle.UnpicklingError:
        print(f"Error: The file at '{file_path}' could not be unpickled. Ensure it is a valid .pkl file.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{file_path}': {e}")
        return None

def save_filtered_data_to_pkl(output_file_path, file_index, cluster_index, filtered_channel_ids, filtered_sorted_Xc, filtered_sorted_Yc):
    try:
        # Ensure the format of filtered_channel_ids is consistent with decimal points
        filtered_channel_ids = np.array(filtered_channel_ids, dtype=np.float64)

        filtered_data = {
            'file_index': file_index,
            'cluster_index': cluster_index,
            'filtered_sorted_channel_ids': filtered_channel_ids,
            'filtered_sorted_Xc_coordinates': filtered_sorted_Xc,
            'filtered_sorted_Yc_coordinates': filtered_sorted_Yc
        }
        with open(output_file_path, 'wb') as file:
            pickle.dump(filtered_data, file)
        print(f"Filtered data successfully saved to '{output_file_path}'.")
    except Exception as e:
        print(f"An error occurred while saving the data to '{output_file_path}': {e}")

# Main Script
# Paths
clusters_folder = './major_paper_checkpoints/Cori_2016-12-18/clusters'
sorted_positions_file = './major_paper_checkpoints/Cori_2016-12-18/sorted_channel_positions.pkl'
output_folder = './major_paper_checkpoints/Cori_2016-12-18/filtered_ids_xc_yc'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each cluster file in the folder
for cluster_file in os.listdir(clusters_folder):
    if cluster_file.startswith("cluster_") and cluster_file.endswith(".pkl"):
        cluster_file_path = os.path.join(clusters_folder, cluster_file)

        # Fetch sorted_channel_ids and metadata (file_index, cluster_index) from the cluster file
        file_index, cluster_index, sorted_channel_ids = fetch_sorted_channel_ids(cluster_file_path)

        if sorted_channel_ids is not None:
            print(f"Processing {cluster_file}...")

            # Fetch and filter data from sorted positions file
            data = fetch_data_from_pkl(sorted_positions_file, sorted_channel_ids)

            if data:
                channel_ids, sorted_x, sorted_y = data
                print("Filtered Data:")
                print("  File Index:", file_index)
                print("  Cluster Index:", cluster_index)
                print("  Channel IDs:", channel_ids)
                print("  Xc Coordinates:", sorted_x)
                print("  Yc Coordinates:", sorted_y)

                # Save the filtered data with the same cluster_i.pkl naming format
                output_file_path = os.path.join(output_folder, cluster_file)
                save_filtered_data_to_pkl(output_file_path, file_index, cluster_index, channel_ids, sorted_x, sorted_y)
