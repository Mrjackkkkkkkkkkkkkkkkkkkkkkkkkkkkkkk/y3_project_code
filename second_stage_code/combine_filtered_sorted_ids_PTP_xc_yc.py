import os
import pickle

# Paths
arrange_folder = './major_paper_checkpoints/Cori_2016-12-18/arrange_ids_PTP_v2'
filtered_xc_yc_folder = './major_paper_checkpoints/Cori_2016-12-18/filtered_ids_xc_yc'
output_folder = './major_paper_checkpoints/Cori_2016-12-18/filtered_ids_PTP_xc_yc'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through each cluster_i.pkl in the arrange folder
for cluster_file in os.listdir(arrange_folder):
    if cluster_file.startswith("cluster_") and cluster_file.endswith(".pkl"):
        arrange_file_path = os.path.join(arrange_folder, cluster_file)
        filtered_xc_yc_file_path = os.path.join(filtered_xc_yc_folder, cluster_file)
        output_file_path = os.path.join(output_folder, cluster_file)

        try:
            # Load data from arrange_ids_PTP_v2
            with open(arrange_file_path, 'rb') as arrange_file:
                arrange_data = pickle.load(arrange_file)

            # Ensure 'filtered_sorted_PTP' exists in the arrange file
            if 'filtered_sorted_PTP' not in arrange_data:
                raise KeyError(f"'filtered_sorted_PTP' is missing in the file: {arrange_file_path}")

            filtered_sorted_PTP = arrange_data['filtered_sorted_PTP']

            # Load data from filtered_ids_xc_yc
            with open(filtered_xc_yc_file_path, 'rb') as filtered_xc_yc_file:
                filtered_xc_yc_data = pickle.load(filtered_xc_yc_file)

            # Ensure 'filtered_sorted_channel_ids' exists in the filtered_xc_yc file
            if 'filtered_sorted_channel_ids' not in filtered_xc_yc_data:
                raise KeyError(f"'filtered_sorted_channel_ids' is missing in the file: {filtered_xc_yc_file_path}")

            # Reorder dictionary to place 'filtered_sorted_PTP' in the desired position
            reordered_data = {
                'filtered_sorted_channel_ids': filtered_xc_yc_data['filtered_sorted_channel_ids'],
                'filtered_sorted_PTP': filtered_sorted_PTP,
                'filtered_sorted_Xc_coordinates': filtered_xc_yc_data['filtered_sorted_Xc_coordinates'],
                'filtered_sorted_Yc_coordinates': filtered_xc_yc_data['filtered_sorted_Yc_coordinates']
            }

            # Save the combined and reordered data to the output folder
            with open(output_file_path, 'wb') as output_file:
                pickle.dump(reordered_data, output_file)

            print(f"Combined data successfully saved to {output_file_path}.")

        except Exception as e:
            print(f"Error processing {cluster_file}: {e}")
