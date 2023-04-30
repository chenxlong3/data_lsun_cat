from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy as np
import os, time
from os.path import exists, join

from PIL import Image
import requests
__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


def view(db_path, idx=1):
    print('Viewing', db_path)
    print('Press ESC to exist or SPACE to advance.')
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for i, (key, val) in enumerate(cursor):
            print('Current key:', key)
            img = cv2.imdecode(
                np.frombuffer(val, dtype=np.uint8), 1)
            # cv2.imshow(window_name, img)
            if i == idx:
                break
        return img



def export_images(db_path, out_dir, flat=False, limit=-1):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            if not flat:
                image_out_dir = join(out_dir, '/'.join(key.decode('ascii')[:6]))
            else:
                image_out_dir = out_dir
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key.decode('ascii') + ".png")#'.webp')
            with open(image_out_path, 'wb') as fp:
                fp.write(val)
            count += 1
            if count == limit:
                break
            if count % 500 == 0:
                print('Finished', count, 'images')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--db_path', default="./data/cat/")
    parser.add_argument('-o', '--out_dir', default="./data/images/")
    parser.add_argument('--flat', type=int, default=1)
    parser.add_argument('--limit', type=int, default=-1)
    args = parser.parse_args()

    export_images(args.db_path, args.out_dir, True if args.flat==1 else 0, args.limit)

if __name__ == '__main__':
    main()