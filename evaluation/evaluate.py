import os 
import sys
import argparse
from mae_eval.mae_toolbox import get_lane_mae_report, test
from utils.io_helper import read_files
import json

def evaluate(test_data_dir, preds_dir):
    test_data = read_files(test_data_dir)
    step_width = test_data[0]['gt_scope']['step width']
    test_gt = []
    for i, data_sample in enumerate(test_data):
        if i == 0:
            continue
        test_gt.append(data_sample['gt'])
    print(len(test_gt))
    
    pred_data = read_files(preds_dir)
    
    pred_res = []
    pred_score = []
    
    for i, pred_sample in enumerate(pred_data):
        if i==0:
            continue
        pred_res.append(pred_sample['pred'])
        pred_score.append(pred_sample)
    
    assert(len(pred_score) == len(pred_res) == len(test_gt))
    
    get_lane_mae_report(test_gt, pred_res, step_width)
    return 





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--test_data_dir', '-t', required=True, help='Training data directory')
    parser.add_argument('--pred', '-p', required=True, help="save folder")
    
    args = parser.parse_args()
    test_data_dir = args.test_data_dir
    pred_dir = args.pred
    
    evaluate(test_data_dir, pred_dir)