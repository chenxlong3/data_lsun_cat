from __future__ import print_function, division
import argparse
from os.path import join

import subprocess
from urllib.request import Request, urlopen
def download(out_dir, category):
    url = 'http://dl.yf.io/lsun/objects/' \
          '{category}.zip'.format(**locals())
    out_name = '{category}.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    cmd = ['curl', '-C', '-', url, '-o', out_path]
    print('Downloading', category, 'set')
    subprocess.call(cmd)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default="./data/")
    parser.add_argument('-c', '--category', default="cat")
    args = parser.parse_args()

    # categories = list_categories()
    
    download(args.out_dir, args.category)


if __name__ == '__main__':
    main()