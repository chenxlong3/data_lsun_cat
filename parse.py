import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go Mask2Former")
    parser.add_argument('--batch_size', type=int,default=32,
                        help="the batch size")
    parser.add_argument('--gpu_num', type=int,default=0,
                        help="gpu number")
    parser.add_argument('--start_idx', type=int, default=0)
    return parser.parse_args()