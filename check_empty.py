from tqdm import tqdm
import os, json

dir_list = os.listdir("./data/masks/")
empty_list = []
for dir in tqdm(dir_list):
    dir_path = f"./data/masks/{dir}"
    cnt = 0
    for file in os.listdir(dir_path):
        if file.endswith(".png"):
            cnt += 1
            break
    if cnt == 0:
        empty_list.append(dir)
with open("./data/hard_list.json", 'w') as f:
    json.dump(empty_list, f)