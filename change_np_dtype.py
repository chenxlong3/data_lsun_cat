import numpy as np
from os import listdir, remove
from os.path import getsize
from tqdm import tqdm
def change_dtype(base_dir="./data/masks/"):
    dir_list = listdir(base_dir)
    print("Finish loading directory list.")
    for dir in tqdm(dir_list):
        np_file_path = f"./data/masks/{dir}/segments.npy"
        tmp_arr = np.load(np_file_path).astype(np.uint8)
        np.save(np_file_path, tmp_arr)

def remove_np_files(base_dir="./data/masks/"):
    dir_list = listdir(base_dir)
    print("Finish loading directory list.")
    for dir in tqdm(dir_list):
        try:
            np_file_path = f"./data/masks/{dir}/segments.npy"
            remove(np_file_path)
        except FileNotFoundError:
            continue

if __name__ == "__main__":
    # change_dtype()
    remove_np_files()