import os 
import sys
import argparse
from mae_eval.mae_toolbox import get_lane_mae_report, test
from utils.io_helper import read_files
import json

def evaluate(test_data_dir, preds_dir):
    test_data = read_files(test_data_dir)
    for i, data_sample in enumerate(test_data):
        if i == 0:
            continue
        
        
    return 





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--test_data_dir', '-t', required=True, help='Training data directory')
    parser.add_argument('--pred', '-p', required=True, help="save folder")
    
    args = parser.parse_args()
    test_data_dir = args.test_data_dir
    pred_dir = args.pred
    
    evaluate(test_data_dir, pred_dir)