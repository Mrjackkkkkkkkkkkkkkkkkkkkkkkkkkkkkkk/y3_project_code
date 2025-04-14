import pickle

# Function to load a .pkl file
def load_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
file_path = './test_nonlinear/Hench_2017-06-15/matched_channel_355_data_clean.pkl'
data = load_pkl_file(file_path)

if data is not None:
    print("Data successfully loaded!")
    print(data)
else:
    print("Failed to load data.")
