import numpy as np
import pandas as pa
from scipy.signal import hilbert
from scipy import signal


def train_test_split(use_ids, prop=0.8, seed=1):
    np.random.seed(seed)
    train_inds = np.random.choice(
        len(use_ids), int(prop * len(use_ids)), replace=False)
    train_use_ids = [int(use_ids[i]) for i in train_inds]
    test_use_ids = [int(j) for j in use_ids if j not in train_use_ids]
    return train_use_ids, test_use_ids


def get_framed_label(use_i, framed_timestamps, annotated_timestamps):
    res = [0] * len(framed_timestamps)
    for idx, framed_time in enumerate(framed_timestamps):
        for annotated_timestamp in annotated_timestamps:
            st, ed = annotated_timestamp
            framed_st, framed_ed = framed_time
            if framed_st >= st and framed_ed <= ed:
                res[idx] = 1
                break
    return res


def get_annotated_intervals(use_i, annotations, category='Defecation'):
    try:
        annotated_intervals = annotations[use_i]
        ans = []
        for interval in annotated_intervals:
            if interval[2] == category:
                ans.append(interval[:2])
        return ans
    except:
        print(f'use_i {use_i} is not annotated!')
        raise


def from_boolean_array_to_intervals(
        boolean_array, t0=0, left_extension=0.96, right_extension=1.04):
    intervals = []
    start = 0
    while start < len(boolean_array):
        while start < len(boolean_array) and boolean_array[start] == 0:
            start += 1
        end = start
        while end < len(boolean_array) and boolean_array[end] == 1:
            end += 1
        if end > start:
            intervals.append([t0+start-left_extension, t0+end+right_extension])
        start = end + 1

    return intervals


def classification_result(model, testX, testY, threshold=0.5):
    testYPredProb = model.predict_proba(testX)
    testYPred = (testYPredProb[:, 1] > threshold).astype(int)
    print(f"threshold = {threshold}", "\n")
    print(classification_report(testY, testYPred))


def variable_importance(trainX, model, top=10):
    top = min(top, trainX.shape[1])
    plt.figure(figsize=(20, 5))
    plt.bar(x=range(top), height=model.feature_importances_[:top])
    xticks_pos = np.arange(top)
    plt.xticks(xticks_pos, trainX.columns[:top], rotation=45, ha='right')
    pass


def get_framed_series(series, window_examples, hop_samples):
    nrows = ((series.size - window_examples) // hop_samples) + 1
    n = series.values.strides[0]
    framed_values = np.lib.stride_tricks.as_strided(
        series.values,
        shape=(nrows, window_examples),
        strides=(hop_samples*n, n)
    )
    framed_timestamps = [series.index[i*hop_samples] for i in range(nrows)]
    return pa.DataFrame(framed_values, index=framed_timestamps)


def apply_envelope(sz):
    analytic_signal = hilbert(sz)
    env_sz = pa.Series(np.abs(analytic_signal), index=sz.index)
    return env_sz


def apply_median_filter(sz, filter_window_size):
    filt_sz = pa.Series(signal.medfilt(
        sz, filter_window_size), index=sz.index)
    return filt_sz


def get_start_end_times_of_boolean_sz(sz):
    ts = sz.index
    start_end_times = []
    i = 0
    while i < len(sz):
        if sz.values[i] == True:
            j = i
            while (j < len(sz)-1) and (sz.values[j+1] == True):
                j += 1
            start_end_times.append([ts[i], ts[j]])
            i = j + 1
        else:
            i += 1
    return start_end_times


def get_area_list_of_intervals(a_list):
    return sum(interval[1]-interval[0] for interval in a_list)


def get_merged_list_of_intervals(a_list):
    merged_list = []
    for interval in a_list:
        if not merged_list or merged_list[-1][1] < interval[0]:
            merged_list.append(interval)
        else:
            merged_list[-1][1] = max(merged_list[-1][1], interval[1])
    return merged_list


def get_recall_two_list_of_intervals(true_list, pred_list):
    intersection = get_intersection_area_two_list_of_lists(
        true_list, pred_list)
    true_area = get_area_list_of_intervals(true_list)
    pred_area = get_area_list_of_intervals(pred_list)
    try:
        return intersection / true_area
    except:
        return float('inf')


def get_precision_two_list_of_intervals(true_list, pred_list):
    intersection = get_intersection_area_two_list_of_lists(
        true_list, pred_list)
    true_area = get_area_list_of_intervals(true_list)
    pred_area = get_area_list_of_intervals(pred_list)
    try:
        return intersection / pred_area
    except:
        return float('inf')


def get_intersection_area_two_intervals(a_interval, b_interval):
    a_st, a_ed = a_interval
    b_st, b_ed = b_interval
    if a_st >= b_ed or a_ed <= b_st:
        return 0

    if a_st >= b_st:
        return min(b_ed, a_ed) - a_st
    else:
        return min(a_ed, b_ed) - b_st


def get_intersection_area_two_list_of_lists(a_list, b_list):
    """
    Get the intersection area of two list of lists
    """
    ans = 0
    for a_interval in a_list:
        for b_interval in b_list:
            ans += get_intersection_area_two_intervals(
                a_interval, b_interval
            )
    return ans
