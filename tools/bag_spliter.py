import os
import argparse
import sys
from tqdm import tqdm
import json
sys.path.append("/home/mingquan/mhw_train")
from evaluation.utils.io_helper import read_files


def split_bags(bag_dir, output_dir):
    for file in tqdm(os.listdir(bag_dir)):
        if "_points.txt" not in file:
            continue
        bag_original_name = file.split("_points.txt")[0]
        file_data = read_files(bag_dir + "/" +file)
        file_data.pop(0)
        for line in file_data:
            with open(output_dir + "/" + bag_original_name + "_" + str(line['ts']['wm']) + ".txt", 'w') as f:
                f.write(json.dumps(line))
                
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--bag_dir', '-b', required=True, help='bag directory')
    parser.add_argument("--output_dir", "-o", required=False, default="/home/mingquan/splited_data")
    args = parser.parse_args()
    
    bag_dir = args.bag_dir
    output_dir = args.output_dir
    
    split_bags(bag_dir, output_dir)