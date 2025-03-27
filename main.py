import pickle


def read_pickle_file(file_path):
    """
    Read data from a pickle file
    
    Parameters:
    file_path (str): Path to the pickle file
    
    Returns:
    The unpickled data
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error occurred while reading the pickle file: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    pickle_file_path = "data.pkl"
    loaded_data = read_pickle_file(pickle_file_path)
    print(loaded_data)