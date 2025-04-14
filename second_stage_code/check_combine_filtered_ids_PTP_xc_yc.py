import os
import pickle

# Path to the folder containing cluster_i.pkl files
filtered_folder = './major_paper_checkpoints/Cori_2016-12-18/filtered_ids_PTP_xc_yc'

# Flag to track if all clusters have matching counts
all_match = True

# Iterate through each cluster_i.pkl file
for cluster_file in os.listdir(filtered_folder):
    if cluster_file.startswith("cluster_") and cluster_file.endswith(".pkl"):
        cluster_file_path = os.path.join(filtered_folder, cluster_file)

        try:
            # Load the data from the pickle file
            with open(cluster_file_path, 'rb') as file:
                data = pickle.load(file)

            # Fetch the required arrays
            filtered_channel_ids = data.get('filtered_sorted_channel_ids', [])
            filtered_PTP = data.get('filtered_sorted_PTP', [])
            filtered_Xc = data.get('filtered_sorted_Xc_coordinates', [])
            filtered_Yc = data.get('filtered_sorted_Yc_coordinates', [])

            # Count the elements in each array
            count_ids = len(filtered_channel_ids)
            count_PTP = len(filtered_PTP)
            count_Xc = len(filtered_Xc)
            count_Yc = len(filtered_Yc)


            # Check if all counts match
            if count_ids == count_PTP == count_Xc == count_Yc:
                print(f"{cluster_file}: Everything has the same number of terms.")
            else:
                print(
                    f"{cluster_file}: Mismatch in counts! idx: {count_ids}, PTP: {count_PTP}, Xc: {count_Xc}, Yc: {count_Yc}")
                all_match = False

        except Exception as e:
            print(f"Error processing {cluster_file_path}: {e}")
            all_match = False

# Final result message
if all_match:
    print("All clusters have the same number of terms in each array.")
else:
    print("Some clusters have mismatched counts.")
