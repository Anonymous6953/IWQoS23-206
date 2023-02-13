# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import json
from sys import argv
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
import torch
import random
from .preprocess import two_seq_scale, minmax_scale


# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def label_evaluation_old(truth_file, result_file, delay=7):
    data = {'result': False, 'data': "", 'message': ""}

    if result_file[-4:] != '.csv':
        data['message'] = "提交的文件必须是csv格式"
        return json.dumps(data, ensure_ascii=False)
    else:
        result_df = pd.read_csv(result_file)

    if 'KPI ID' not in result_df.columns or 'timestamp' not in result_df.columns or \
            'predict' not in result_df.columns:
        data['message'] = "提交的文件必须包含KPI ID,timestamp,predict三列"
        return json.dumps(data, ensure_ascii=False)

    truth_df = pd.read_csv(truth_file)

    kpi_names = truth_df['KPI ID'].values
    kpi_names = np.unique(kpi_names)
    y_true_list = []
    y_pred_list = []

    for kpi_name in kpi_names:

        truth = truth_df[truth_df["KPI ID"] == kpi_name]
        y_true = reconstruct_label(truth["timestamp"], truth["label"])

        if kpi_name not in result_df["KPI ID"].values:
            data['message'] = "提交的文件缺少KPI %s 的结果" % kpi_name
            return json.dumps(data, ensure_ascii=False)

        result = result_df[result_df["KPI ID"] == kpi_name]

        if len(truth) != len(result):
            data['message'] = "文件长度错误"
            return json.dumps(data, ensure_ascii=False)

        y_pred = reconstruct_label(result["timestamp"], result["predict"])

        y_pred = get_range_proba(y_pred, y_true, delay)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    try:
        fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
    except:
        data['message'] = "predict列只能是0或1"
        return json.dumps(data, ensure_ascii=False)

    data['result'] = True
    data['data'] = fscore
    data['message'] = '计算成功'

    return json.dumps(data, ensure_ascii=False)


def label_evaluation(truth_file, result_file, delay=7):
    percent = 0.3

    data = {'result': False, 'data': "", 'message': ""}

    if result_file[-4:] != '.csv':
        data['message'] = "提交的文件必须是csv格式"
        return json.dumps(data, ensure_ascii=False)
    else:
        result_df = pd.read_csv(result_file)

    truth_df = pd.read_csv(truth_file)

    y_true_list = []
    y_pred_list = []

    eva_len = round(len(truth_df) * percent)

    truth = truth_df.iloc[eva_len:]
    y_true = reconstruct_label(truth["timestamp"], truth["label"])

    result = result_df[eva_len:]

    if len(truth) != len(result):
        data['message'] = "文件长度错误"
        return json.dumps(data, ensure_ascii=False)

    y_pred = reconstruct_label(result["timestamp"], result["label"])

    y_pred = get_range_proba(y_pred, y_true, delay)
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)

    try:
        fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
        p = precision_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
        r = recall_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
    except:
        data['message'] = "predict列只能是0或1"
        return json.dumps(data, ensure_ascii=False)

    data['result'] = True
    data['data'] = {'p': p, 'r': r, 'f': fscore}
    data['message'] = '计算成功'

    return json.dumps(data, ensure_ascii=False)


def evaluate_api(truth_df, result_df, delay=7, return_dic=True):
    if isinstance(truth_df, np.ndarray):
        truth_df = pd.DataFrame({'timestamp': list(range(len(truth_df))), 'label': truth_df})
    if isinstance(result_df, np.ndarray):
        result_df = pd.DataFrame({'timestamp': list(range(len(result_df))), 'predicted': result_df})

    percent = 0

    data = {'result': False, 'data': "", 'message': ""}

    y_true_list = []
    y_pred_list = []

    eva_len = round(len(truth_df) * percent)

    truth = truth_df.iloc[eva_len:]
    y_true = reconstruct_label(truth["timestamp"], truth["label"])

    result = result_df[eva_len:]

    if len(truth) != len(result):
        data['message'] = "文件长度错误"
        return json.dumps(data, ensure_ascii=False)

    y_pred = reconstruct_label(result["timestamp"], result["predicted"])

    y_pred = get_range_proba(y_pred, y_true, delay)
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)

    try:
        fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
        p = precision_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
        r = recall_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
    except:
        data['message'] = "predict列只能是0或1"
        return json.dumps(data, ensure_ascii=False)

    data['result'] = True
    data['data'] = {'p': p, 'r': r, 'f': fscore}
    data['message'] = '计算成功'

    if return_dic:
        return data
    else:
        return json.dumps(data, ensure_ascii=False)


def mse(seq1, seq2, scale=True):
    if scale:
        seq1, seq2 = two_seq_scale(seq1, seq2)

    return mean_squared_error(seq1, seq2)


def LOF(raw_seq, win_size=10, slide_step=1, k=5, scale=True):
    if scale:
        raw_seq = minmax_scale(raw_seq)

    begin = 0
    end = len(raw_seq) - win_size - k * slide_step
    time_delay_emb_idx = [i for i in range(begin, end, slide_step)]

    def rd(x_1, x_2):
        x_1, x_2 = np.array(x_1), np.array(x_2)
        return np.linalg.norm(x_1 - x_2)

    def neighbors(idx):
        return [raw_seq[idx + i*slide_step : idx + i*slide_step + win_size]
                for i in range(k) if idx + i*slide_step + win_size < len(raw_seq)]

    def neighbors_idx(idx):
        return [idx + i*slide_step for i in range(k) if idx + i*slide_step + win_size < len(raw_seq)]


    def idx_to_vec(idx):
        return raw_seq[idx : idx+win_size]

    def cal_LOF(idx):
        x = idx_to_vec(idx)
        x_neighbors_idx = neighbors_idx(idx)
        LOF_x = 0
        for x_skim_idx in x_neighbors_idx:
            frac_up, frac_down = 0, 0

            for x_hat_idx in x_neighbors_idx:
                x_hat = idx_to_vec(x_hat_idx)

                frac_up += rd(x, x_hat)

            x_skim_neighbors = neighbors(x_skim_idx)

            x_skim = idx_to_vec(x_skim_idx)
            for x_bar in x_skim_neighbors:
                frac_down += rd(x_skim, x_bar)

            LOF_x += frac_up / (frac_down + 1)
        return LOF_x / k

    LOF_ls = [cal_LOF(i) for i in time_delay_emb_idx]

    return np.average(LOF_ls)



if __name__ == '__main__':
    _, truth_file, result_file, delay = argv
    delay = (int)(delay)
    print(label_evaluation(truth_file, result_file, delay))
