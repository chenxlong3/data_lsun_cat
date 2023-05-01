from torch.utils.data import Dataset
from typing import Union
import os, json
from PIL import Image
from transformers import Mask2FormerImageProcessor
from clean_data import get_clean_file_list

class LSUN_CAT(Dataset):
    def __init__(self, img_dir:str, st_idx=0, file_list_json="./data/valid_imgs.json"):
        self.img_dir = img_dir
        if st_idx == 0:
            with open(file_list_json, 'r') as f:
                self.data = json.load(f)
            # self.data = os.listdir(img_dir)
        else:
            with open(file_list_json, 'r') as f:
                self.data = json.load(f)[st_idx:]
        
    def __getitem__(self, index):
        file_path = os.path.join(self.img_dir, self.data[index])
        proc = Mask2FormerImageProcessor(True, (384, 384))
        x = Image.open(file_path)
        return proc(x), self.data[index]
    
    def __len__(self):
        return len(self.data)
