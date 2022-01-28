from matplotlib.pyplot import step
from sklearn.metrics import mean_absolute_error as MAE
import numpy as np
import math
BEGIN_X = -20
END_X = 101



def calculate_lane_mae(lane_true, lane_pred):
    return MAE(lane_true, lane_pred)


def calculate_lane_interval_mae(lane_true, lane_pred):
    res = {}
    temp_count = {}
    assert(len(lane_true)==len(lane_pred))
    for i in range(len(lane_true)):
        for j in range(len(lane_true[0])):
            for k in range(len(lane_true[0][0])):
                if lane_true[i][j][k][0] == float("inf") or lane_pred[i][j][k][0] == float("inf"):
                    continue
                if str(lane_true[i][j][k][0]) not in res.keys():
                    res[str(lane_true[i][j][k][0])] = 0
                    temp_count[str(lane_true[i][j][k][0])] = 0
                res[str(lane_true[i][j][k][0])] += abs(lane_true[i][j][k][1]-lane_pred[i][j][k][1])
                temp_count[str(lane_true[i][j][k][0])] += 1
    for key in res.keys():
        res[key] /= temp_count[key]
    return res

def effective_bit_reserve(arr):
    del_idx = []
    for i in range(len(arr)):
        if math.isinf(arr[i]):
            del_idx.append(i)
    arr = np.delete(arr, del_idx)
    return arr
            

def get_lane_mae_report(lane_true, lane_pred, step_width):
    for i in range(len(lane_true)):
        for j in range(3):
            lane_true_length = len(lane_true[i][j])
            lane_pred_length = len(lane_pred[i][j])
            place_holder = float('inf')
            lane_true[i][j] += [[place_holder,place_holder]] * (25 - lane_true_length)
            lane_pred[i][j] += [[place_holder,place_holder]] * (25 - lane_pred_length)
            if lane_true_length < lane_pred_length:
                lane_pred[i][j][lane_true_length:] = [[place_holder,place_holder]] * (25 - lane_true_length)
            elif lane_pred_length < lane_true_length:
                lane_true[i][j][lane_pred_length:] = [[place_holder,place_holder]] * (25 - lane_pred_length)              
    
    # shape all datas
    pred_array = np.array(lane_pred)[:,:,:,1]
    truth_array = np.array(lane_true)[:,:,:,1]
    # all lanes flatten
    pred_arr_flatten = pred_array.flatten()
    truth_arr_flatten = truth_array.flatten()
    pred_effective = effective_bit_reserve(pred_arr_flatten)
    truth_effective = effective_bit_reserve(truth_arr_flatten)
    # left lanes flatten
    pred_left_arr = pred_array[:,0].flatten()
    truth_left_arr = truth_array[:,0].flatten()
    pred_effective_left = effective_bit_reserve(pred_left_arr)
    truth_effective_left = effective_bit_reserve(truth_left_arr)
    # current lanes flatten
    pred_cur_arr = pred_array[:,1].flatten()
    truth_cur_arr = truth_array[:,1].flatten()
    pred_effective_cur = effective_bit_reserve(pred_cur_arr)
    truth_effective_cur = effective_bit_reserve(truth_cur_arr)
    # right lanes flatten
    pred_right_arr = pred_array[:,2].flatten()
    truth_right_arr = truth_array[:,2].flatten()
    pred_effective_right = effective_bit_reserve(pred_right_arr)
    truth_effective_right = effective_bit_reserve(truth_right_arr)
    # get MAE
    res_dict = {}
    lane_mae = {}
    res_dict["frame_num"] = np.shape(lane_pred)[0]
    lane_mae["total_MAE"] = calculate_lane_mae(pred_effective, truth_effective)
    lane_mae["left_MAE"] = calculate_lane_mae(pred_effective_left, truth_effective_left)
    lane_mae["cur_MAE"] = calculate_lane_mae(pred_effective_cur, truth_effective_cur)
    lane_mae["right_MAE"] = calculate_lane_mae(pred_effective_right, truth_effective_right)
    res_dict["lane_MAE"] = lane_mae
    res_dict["interval_mae"] = calculate_lane_interval_mae(lane_true, lane_pred)
    return res_dict


def test():
    sample_lane_pred = np.array([1, 2, 3, 4, 2, 3])
    sample_lane_true = np.array([1, 2, 3, 2, 2, 3])

    res_dict = get_lane_mae_report(sample_lane_true, sample_lane_pred, 5)

    print(res_dict)


if __name__ == "__main__":
    test()
