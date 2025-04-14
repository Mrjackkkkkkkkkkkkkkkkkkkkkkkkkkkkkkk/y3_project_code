import os
import pickle

# Path to the folder containing cluster_i.pkl files
arrange_folder = './major_paper_checkpoints/Cori_2016-12-18/arrange_ids_PTP_v2'

# Flag to check if all counts are equal
all_equal = True

# Iterate through each cluster_i.pkl file in the folder
for cluster_file in os.listdir(arrange_folder):
    if cluster_file.startswith("cluster_") and cluster_file.endswith(".pkl"):
        cluster_file_path = os.path.join(arrange_folder, cluster_file)

        try:
            # Load the data from the pickle file
            with open(cluster_file_path, 'rb') as file:
                data = pickle.load(file)

            # Ensure the required keys are present in the file
            if 'filtered_sorted_channel_ids' not in data or 'filtered_sorted_PTP' not in data:
                missing_keys = [key for key in ['filtered_sorted_channel_ids', 'filtered_sorted_PTP'] if key not in data]
                raise KeyError(f"The keys {missing_keys} are missing in the file: {cluster_file_path}")

            # Count the number of filtered_sorted_channel_ids and filtered_sorted_PTP
            filtered_channel_ids = data['filtered_sorted_channel_ids']
            filtered_PTP = data['filtered_sorted_PTP']
            count_ids = len(filtered_channel_ids)
            count_PTP = len(filtered_PTP)

            # Print the counts for this cluster
            print(
                f"{cluster_file}: There are {count_ids} filtered_sorted_channel_ids and {count_PTP} filtered_sorted_PTP.")

            # Check for mismatched counts
            if count_ids != count_PTP:
                print(f"WARNING: Mismatch in counts for {cluster_file}! filtered_sorted_channel_ids: {count_ids}, filtered_sorted_PTP: {count_PTP}")
                all_equal = False  # Set the flag to False if a mismatch is found

        except Exception as e:
            print(f"Error processing {cluster_file_path}: {e}")

# Print the final result
if all_equal:
    print("All equal: All filtered_sorted_channel_ids and filtered_sorted_PTP have matching counts at each cluster.")
else:
    print("There were mismatched counts in some files.")
