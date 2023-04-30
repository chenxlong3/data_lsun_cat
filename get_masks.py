from __future__ import print_function
import numpy as np
import os, time, json
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from dataset import LSUN_CAT
# from datetime import datetime
import logging
from parse import parse_args
from loggers import init_logs
args = parse_args()
MASK_INPUT_SIZE = (384, 384)
MASK_OUTPUT_SIZE = (384, 384)
processor = Mask2FormerImageProcessor(True, MASK_INPUT_SIZE)
device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu"
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic").to(device)


# Store data
os.makedirs("./data/masks/", exist_ok=True)

def create_dir(filename, mask_dir="./data/masks/"):
    # exclude the extension
    dir_name = filename[:-4]
    target_dir = os.path.join(mask_dir, dir_name)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def save_mask_img(mask_list:list, id_list, out_dir:str):
    for mask, id in zip(mask_list, id_list):
        out_file = os.path.join(out_dir, f"{id}.png")
        im = Image.fromarray(mask)
        im.save(out_file)
    return

def save_segments_np(seg_tensor:torch.Tensor, out_dir:str):
    seg_np = seg_tensor.cpu().numpy()
    out_path = os.path.join(out_dir, "segments.npy")
    np.save(out_path, seg_np)
    return

def save_segments_info(segments_info:list, out_dir:str):
    """
    Input:
        segments_info: (e.g. [{'id': 1, 'label_id': 15, 'was_fused': False, 'score': 0.999431}, {'id': 2, 'label_id': 132, 'was_fused': False, 'score': 0.930572}, {'id': 3, 'label_id': 122, 'was_fused': False, 'score': 0.910661}])
    """
    if type(segments_info) != list:
        raise SyntaxError
    out_path = os.path.join(out_dir, "segments_info.json")
    with open(out_path, 'w') as f:
        json.dump(segments_info, f)
    return

def get_vis_masks(results, labels=["dog", "cat", "kitten", "monkey"]) -> list:
    """
    Input:
        results (dict): {segmentation: tensor, segments_info: list}
        labels (list): the labels that we want to save
    """
    mask_list = []
    seg_id_list = []
    segment_to_label = {segment['id']: segment['label_id'] for segment in results["segments_info"]}
    
    for segment in results["segments_info"]:
        segment_id = segment['id']
        # if model.config.id2label[segment_to_label[segment_id]] == label:
        #   mask += (results['segmentation'].numpy() == segment_id)
        # visual_mask = np.clip((mask * 255).astype(np.uint8), None, 255)
        # visual_mask = Image.fromarray(visual_mask)
        # if model.config.id2label[segment_to_label[segment_id]] in labels:
        if segment["score"] >= 0.7 and model.config.id2label[segment_to_label[segment_id]] in labels:
            mask = (results['segmentation'].cpu().numpy() == segment_id)
            mask = (mask * 255).astype(np.uint8)
            seg_id_list.append(segment_id)
            mask_list.append(mask)
    return mask_list, seg_id_list


def saving_masks(outputs, files, target_size=(384, 384)):
    results = processor.post_process_panoptic_segmentation(outputs, target_sizes=[target_size for i in range(len(files))])
    for result, file in zip(results, files):
        out_dir = create_dir(file)
        mask_list, id_list = get_vis_masks(result)

        save_segments_np(result["segmentation"], out_dir)
        save_segments_info(result["segments_info"], out_dir)
        save_mask_img(mask_list, id_list, out_dir)
    return

def main():
    log_dir, log_file = init_logs()
    logging.info(f"Start Data Processing for LSUN-CAT on device: {device}")
    logging.info(str(args))
    ds_st_time = time.time()
    # Dataset and DataLoader
    ds = LSUN_CAT("./data/images/")
    data_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    logging.info(f"Loading data takes {time.time() - ds_st_time}s")
    for i, (x, files) in enumerate(data_loader):
        try:
            logging.info(f"--- Batch {i} ---")
            x["pixel_values"] = x["pixel_values"][0].to(device)
            x["pixel_mask"] = x["pixel_mask"][0].to(device)
            st_time = time.time()
            with torch.no_grad():
                outputs = model(**x)
            logging.info(f"Forward Pass with Batch {i} takes {time.time()-st_time}s")
            
            logging.info(f"Saving the masks")
            saving_masks(outputs, files, target_size=(384, 384))

            if i > 0 and i % 1000 == 0:
                logging.info("------------- Finish 1000 batches -------------")
        except Exception as e:
            logging.info(f"Stop at batch {i}")
            logging.info(e.args)
            break
    logging.info(f"The whole process takes {time.time() - ds_st_time}s")
    return

if __name__ == "__main__":
    main()