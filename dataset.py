from torch.utils.data import Dataset
from typing import Union
import os
from PIL import Image
from transformers import Mask2FormerImageProcessor

class LSUN_CAT(Dataset):
    def __init__(self, img_dir:str):
        self.img_dir = img_dir
        self.data = os.listdir(img_dir)
        
    def __getitem__(self, index):
        file_path = os.path.join(self.img_dir, self.data[index])
        proc = Mask2FormerImageProcessor(True, (384, 384))
        x = Image.open(file_path)
        return proc(x), self.data[index]
    
    def __len__(self):
        return len(self.data)
