import pickle

def fetch_filtered_data_from_pkl(file_path):
    try:
        # Open the pickle file in read mode
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Fetch the required keys
        filtered_sorted_x = data.get('filtered_sorted_x', None)
        filtered_sorted_y = data.get('filtered_sorted_y', None)

        # Check if the required keys are present
        if filtered_sorted_x is None or filtered_sorted_y is None:
            raise KeyError("One or both keys ('filtered_sorted_x', 'filtered_sorted_y') are missing in the file.")

        return filtered_sorted_x, filtered_sorted_y

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except pickle.UnpicklingError:
        print("Error: Failed to load the pickle file. The file may be corrupted or not a valid pickle file.")
    except Exception as e:
        print(f"An error occurred: {e}")

def add_data_to_pkl(file_path, filtered_sorted_x, filtered_sorted_y):
    try:
        # Load existing data from the target pickle file
        try:
            with open(file_path, 'rb') as file:
                existing_data = pickle.load(file)
        except (FileNotFoundError, EOFError):  # Handle file not found or empty file
            existing_data = {}

        # Add new data
        existing_data['filtered_sorted_x'] = filtered_sorted_x
        existing_data['filtered_sorted_y'] = filtered_sorted_y

        return existing_data

    except Exception as e:
        print(f"An error occurred while combining data: {e}")
        return None

def save_new_pkl(new_file_path, data):
    try:
        # Save the combined data into a new pickle file
        with open(new_file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully saved to {new_file_path}.")
    except Exception as e:
        print(f"An error occurred while saving the new pickle file: {e}")

if __name__ == "__main__":
    # Source pickle file path
    source_path =

    # Target pickle file path
    target_path =
    # New pickle file path
    new_file_path = 
    # Fetch data from source file
    result = fetch_filtered_data_from_pkl(source_path)

    if result:
        filtered_sorted_x, filtered_sorted_y = result

        # Combine data with existing data in the target file
        combined_data = add_data_to_pkl(target_path, filtered_sorted_x, filtered_sorted_y)

        if combined_data:
            # Save the combined data into a new pickle file
            save_new_pkl(new_file_path, combined_data)
