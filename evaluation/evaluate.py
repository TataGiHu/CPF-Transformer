import os 
import sys
import argparse
import copy
from mae_eval.mae_toolbox import get_lane_mae_report
from class_eval.class_eval_tool import get_class_eval_report
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
    print("Number of gt: {}".format(len(test_gt)))
    
    pred_data = read_files(preds_dir)
    
    pred_res = []
    pred_score = []
    
    for i, pred_sample in enumerate(pred_data):
        if i==0:
            continue
        pred_res.append(pred_sample['pred'])
        pred_score.append(pred_sample['score'])
    print("Number of preds: {}".format(len(pred_res)))
    assert(len(pred_score) == len(pred_res) == len(test_gt))
    # Deep copy the original GT to prevent filling GT in MAE operation from affecting class evaluation
    test_gt_for_mae = copy.deepcopy(test_gt)
    print("======  Regression Evaluating  =====")
    result_mae_eval = get_lane_mae_report(test_gt_for_mae, pred_res, step_width)
    print("======  Classification Evaluating  =====")
    result_class_eval = get_class_eval_report(test_gt, pred_score)
    os.system('touch ../evaluation.txt')
    f = open('../evaluation.txt','w')
    f.write("Regression Evaluation:" + os.linesep)
    f.write(json.dumps(result_mae_eval) + os.linesep)
    f.write(os.linesep)
    f.write("Classification Evaluation:" + os.linesep)
    f.write(json.dumps(result_class_eval) + os.linesep)
    f.close()
    return 





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--test_data_dir', '-t', required=True, help='Training data directory')
    parser.add_argument('--pred', '-p', required=True, help="save folder")
    
    args = parser.parse_args()
    test_data_dir = args.test_data_dir
    pred_dir = args.pred
    
    evaluate(test_data_dir, pred_dir)