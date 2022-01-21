from matplotlib.pyplot import step
from sklearn.metrics import mean_absolute_error as MAE
import numpy as np
BEGIN_X = -20
END_X = 101


def combine_mae_results(mae_dict1, mae_dict_2):
    res_dict = {}
    for key in mae_dict1:


def calculate_lane_mae(lane_true, lane_pred):
    return MAE(lane_true, lane_pred)


def calculate_lane_interval_mae(lane_true, lane_pred, interval_begin_index):
    interval_end_index = interval_begin_index + 1
    if interval_end_index > lane_true.shape[0]:
        return -1

    return calculate_lane_mae(lane_true[interval_begin_index:interval_end_index],
                              lane_pred[interval_begin_index:interval_end_index])


def get_lane_mae_report(lane_true, lane_pred, step_width):

    res_dict = {}
    res_dict["total_mae"] = calculate_lane_mae(lane_true, lane_pred)
    res_dict["num"] = lane_pred.shape[0]
    for interval_begin_index in range(lane_pred.shape[0]):
        cur_interval_mae = calculate_lane_interval_mae(
            lane_true, lane_pred, interval_begin_index)
        if cur_interval_mae == -1:
            break
        res_dict[str(interval_begin_index * step_width) + "-" +
                 str((interval_begin_index + 1) * step_width)] = cur_interval_mae
    return res_dict


def test():
    sample_lane_pred = np.array([1, 2, 3, 4, 2, 3])
    sample_lane_true = np.array([1, 2, 3, 2, 2, 3])

    res_dict = get_lane_mae_report(sample_lane_true, sample_lane_pred, 5)

    print(res_dict)


if __name__ == "__main__":
    test()
