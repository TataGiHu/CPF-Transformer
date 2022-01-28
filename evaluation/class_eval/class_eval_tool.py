
import matplotlib 
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def get_class_eval_report(test_gt, pred_score):
    true_class = get_true_class(test_gt)
    threshold_list = np.arange(0, 1.01, 0.01).round(2)
    res = {}
    left_precision_list = []
    left_recall_list = []
    cur_precision_list = []
    cur_recall_list = []
    right_precision_list = []
    right_recall_list = []
    total_precision_list = []
    total_recall_list = []
    for threshold in threshold_list:
        cur_res = {}
        # get number of positive and negative
        left_TP, left_FN, left_FP, left_TN = get_num(true_class, pred_score, "left", threshold)
        cur_TP, cur_FN, cur_FP, cur_TN = get_num(true_class, pred_score, "cur", threshold)
        right_TP, right_FN, right_FP, right_TN = get_num(true_class, pred_score, "right", threshold)
        # calculation of precision and recall
        left_precision = get_precision(left_TP, left_FP)
        left_recall = get_recall(left_TP, left_FN)
        cur_precision = get_precision(cur_TP, cur_FP)
        cur_recall = get_recall(cur_TP, cur_FN)
        right_precision = get_precision(right_TP, right_FP)
        right_recall = get_recall(right_TP, right_FN)
        total_precision = get_precision(left_TP + cur_TP + right_TP, left_FP + cur_FP + right_FP)
        total_recall = get_recall(left_TP + cur_TP + right_TP, left_FN + cur_FN + right_FN)
        # save precision and recall
        left_precision_list.append(left_precision)
        left_recall_list.append(left_recall)
        cur_precision_list.append(cur_precision)
        cur_recall_list.append(cur_recall)
        right_precision_list.append(right_precision)
        right_recall_list.append(right_recall)
        total_precision_list.append(total_precision)
        total_recall_list.append(total_recall)
        # save 0.9 -- 1.0 threshold to txt file
        if threshold >= 0.9:
            cur_res["left_precision"]=left_precision
            cur_res["left_recall"] = left_recall
            cur_res["cur_precision"] = cur_precision
            cur_res["cur_recall"] = cur_recall
            cur_res["right_precision"] = right_precision
            cur_res["right_recall"] = right_recall
            cur_res["total_precision"] = total_precision
            cur_res["total_recall"] = total_recall
            res["threshold_"+str(threshold)] = cur_res
    # PR-curve
    plt.plot(left_recall_list,left_precision_list, label="left_PR_curve")
    plt.plot(cur_recall_list,cur_precision_list, label="cur_PR_curve")
    plt.plot(right_recall_list,right_precision_list, label="right_PR_curve")
    plt.plot(total_recall_list,total_precision_list, label="total_PR_curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R Curve")
    plt.legend()
    plt.show()
    plt.savefig("../PR_lines.png")
    print("---- Save PR Figures ----")
    return res  
        
        
# def plot_PR_curve(precision_list, recall_list, lane_mark):
#     plt.plot(recall_list, precision_list)
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(lane_mark+"P-R Curve")
#     plt.show()
#     plt.savefig('../'+lane_mark+'_PR_line.png')
#     print("---- Save {} PR Figures ----".format(lane_mark))
    
    
        
def get_precision(num_TP, num_FP):
    return num_TP/(num_TP+num_FP)

def get_recall(num_TP, num_FN):
    return num_TP/(num_TP+num_FN)

def get_true_class(test_gt):
    res = []
    # print("test_gt:{}".format(test_gt[0]))
    for frame in test_gt:
        cur_class = []
        for lane in frame:
            if len(lane)> 0:
                cur_class.append(1) 
            else:
                cur_class.append(0)
        # print("cur_class:{}".format(cur_class))
        res.append(cur_class)
    return res

def get_num(true_class, pred_score, lane_mark, threshold):
    assert(len(true_class)==len(pred_score))
    if lane_mark == "left":
        lane_idx = 0
    elif lane_mark == "cur":
        lane_idx = 1
    elif lane_mark == "right":
        lane_idx = 2
    num_TP = 0
    num_FN = 0
    num_FP = 0
    num_TN = 0
    frame_num = len(true_class)
    for i in range(frame_num):
        if true_class[i][lane_idx]==1 and pred_score[i][lane_idx]>=threshold:
            num_TP += 1
        if true_class[i][lane_idx]==1 and pred_score[i][lane_idx]<threshold:
            num_FN += 1
        if true_class[i][lane_idx]==0 and pred_score[i][lane_idx]>=threshold:
            num_FP += 1  
        if true_class[i][lane_idx]==0 and pred_score[i][lane_idx]<threshold:
            num_TN += 1  
    return num_TP, num_FN, num_FP, num_TN
