import os
import pickle


def find_and_check_pkl(main_folder):
    # Iterate through all subfolders
    for root, dirs, files in os.walk(main_folder):
        if 'filtered_ids_PTP_xc_yc' in dirs:
            filtered_folder = os.path.join(root, 'filtered_ids_PTP_xc_yc')
            for file in os.listdir(filtered_folder):
                if file.startswith('cluster_') and file.endswith('.pkl'):
                    file_path = os.path.join(filtered_folder, file)

                    # Open and check the content of the pickle file
                    try:
                        with open(file_path, 'rb') as f:
                            content = pickle.load(f)

                            if content == 0:  # Check if the content is exactly 0
                                print(f"0 content found in: {file_path}")
                            else:
                                print(f"Information is well stored at: {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")


# Define the main folder
main_folder = './major_paper_checkpoints/'

# Run the function
find_and_check_pkl(main_folder)
