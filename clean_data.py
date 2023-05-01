from PIL import Image
from os import listdir, remove
from os.path import join
from tqdm.auto import tqdm
import json
def read_successfully(file):
    try:
        img = Image.open(file)
        del img
        return True
    except:
        return False

def get_clean_file_list(base_dir="./data/images"):
    file_list = listdir(base_dir)
    cleaned_file_list = []
    for i, file in tqdm(enumerate(file_list)):
        f_path = join(base_dir, file)
        if read_successfully(f_path):
            cleaned_file_list.append(file)
    del file_list
    with open("./data/valid_imgs.json", 'w') as f:
        json.dump(cleaned_file_list, f)
    return cleaned_file_list

# def remove_useless_files(base_dir="./data/images"):
#     file_list = listdir(base_dir)
#     cleaned_file_list = []
#     for i, file in tqdm(enumerate(file_list)):
#         f_path = join(base_dir, file)
#         if read_successfully(f_path):
#             cleaned_file_list.append(file)
#     del file_list
#     return cleaned_file_list

if __name__ == "__main__":
    get_clean_file_list()