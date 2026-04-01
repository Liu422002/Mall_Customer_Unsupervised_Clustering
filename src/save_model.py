import os
import pickle


def save_pickle(obj, file_path):
    folder = os.path.dirname(file_path)
    os.makedirs(folder, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)