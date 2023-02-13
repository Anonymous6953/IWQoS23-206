from hyperopt import fmin, tpe, hp, Trials, space_eval, rand, STATUS_OK, STATUS_FAIL
import pandas as pd
import numpy as np
import os
from time import time
import torch
import random
import json

from detector.HW import train_and_predict as hw_train_and_predict
from detector.LSTM import train_and_predict as lstm_train_and_predict
from detector.VAE import train_and_predict as donut_train_and_predict

from utils.preprocess import minmax_scale
from utils.postprocess import abs_anomaly_score
from utils.objfunc import best_f1_score, mse_factor, normal_factor

train_df = None
test_df = None

record = []


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_algorithms(params):
    algo_name = params['name']
    print(params)
    status = STATUS_OK

    train_and_predict = None
    if algo_name == 'holt_winter':
        train_and_predict = hw_train_and_predict
    elif algo_name == 'LSTM':
        train_and_predict = lstm_train_and_predict
    elif algo_name == 'Donut':
        train_and_predict = donut_train_and_predict


    try:
        predicted = train_and_predict(train_df['value'], test_df['value'], params['params'])
    except Exception as e:
        print("params conflict!")
        print(e)
        predicted = np.zeros(len(test_df))
        status = STATUS_FAIL
    # Make sure the prediction is one-dimension array
    predicted = predicted.reshape(-1)
    # replace nan with 0
    predicted[np.isnan(predicted)] = .0

    return predicted, status


def f1_score(params):
    predicted_value, status = run_algorithms(params)

    truth_value = test_df['value'].to_numpy()
    truth_labels = test_df['label'].to_numpy()

    anomaly_scores = abs_anomaly_score(truth_value, predicted_value)

    best_f1 = best_f1_score(truth_labels, anomaly_scores)


    return {'loss': -best_f1, 'status': status, 'f1-score': best_f1}


def mseNF(params):
    predicted_value, status = run_algorithms(params)

    truth_value = test_df['value'].to_numpy()
    truth_labels = test_df['label'].to_numpy()

    mse_score = mse_factor(truth_value, predicted_value)
    normal_score = normal_factor(predicted_value)
    score = mse_score + normal_score

    anomaly_scores = abs_anomaly_score(truth_value, predicted_value)
    best_f1 = best_f1_score(truth_labels, anomaly_scores)

    record.append((best_f1, score))

    return {'loss': score, 'status': status, 'f1-score': best_f1, "mse": mse_score, "normal factor": normal_score}


def mse(params):
    predicted_value, status = run_algorithms(params)

    truth_value = test_df['value'].to_numpy()
    truth_labels = test_df['label'].to_numpy()

    score = mse_factor(truth_value, predicted_value)

    anomaly_scores = abs_anomaly_score(truth_value, predicted_value)
    best_f1 = best_f1_score(truth_labels, anomaly_scores)

    record.append((best_f1, score))

    return {'loss': score, 'status': status, 'f1-score': best_f1}


def save_trials(trials: Trials, space, save_path):
    def format_trial_vals(trial_dic):
        trial_vals = trial_dic['misc']['vals']
        new_dic = {}
        for i in trial_vals:
            if len(trial_vals[i]) > 0:
                new_dic[i] = trial_vals[i][0]
        return space_eval(space, new_dic)

    class TrialEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(TrialEncoder, self).default(obj)

    try_ls = []
    for t in trials.trials:
        params = format_trial_vals(t)
        params['result'] = t['result']
        try_ls.append(params)
    json.dump(try_ls, open(save_path, 'w'), cls=TrialEncoder, sort_keys=True, indent=4)


def tune_kpi(kpi):
    global train_df, test_df
    train_df = pd.read_csv(f"data/tzs/train/{kpi}.csv")
    test_df = pd.read_csv(f"data/tzs/test/{kpi}.csv")

    train_df['value'] = minmax_scale(train_df['value'])
    test_df['value'] = minmax_scale(test_df['value'])


    t0 = time()
    trials = Trials()

    fn = mseNF
    algo = rand.suggest
    max_eval = 50

    dir_path = f"trials/{kpi}"

    os.makedirs(dir_path, exist_ok=True)

    if fn is mseNF:
        obj_str = "msenf"
    if fn is mse:
        obj_str = "mse"
    if fn is f1_score:
        obj_str = "f1"

    algo_str = "tpe" if algo is tpe.suggest else "rand"

    save_path = f"{dir_path}/{obj_str}_{algo_str}_{max_eval}.json"
    best = fmin(fn=fn, space=params_space, algo=algo, max_evals=max_eval, trials=trials)

    save_trials(trials, params_space, save_path)

    df = pd.DataFrame(record, columns=['F1', 'MSENF'])
    df.to_csv("f1_msenf.csv", index=False)


if __name__ == '__main__':
    setup_seed(329)

    params_space = hp.choice('detector', [
            {'name': 'holt_winter',
             'params': {
                 'trend': hp.choice('hw_trend', ['add', 'mul', 'additive', 'multiplicative']),
                 'seasonal': hp.choice('hw_seasonal', ['add', 'mul', 'additive', 'multiplicative']),
                 'damped_trend': hp.choice('hw_damped_trend', [True, False]),
                 'seasonal_periods': hp.randint('hw_seasonal_periods', 2, 14),
                 'initialization_method': hp.choice('hw_initialization_method',
                                                    ['estimated', 'heuristic', 'legacy-heuristic'])
                }
            },
            {'name': 'LSTM',
             'params': {
                 'seq_len': hp.choice('lstm_seq_len', [10, 20, 50]),
                 'batch_size': hp.choice('lstm_batch_size', [4, 32, 64, 256]),
                 'lr': hp.choice('lstm_lr', [0.1, 0.001, 0.0001]),
                 'epochs': hp.choice('lstm_epochs', [1, 5, 10, 20, 50])
                 # 'epochs': hp.choice('lstm_epochs', [1])
             }
            },
            {'name': 'Donut',
             'params': {
                 'seq_len': hp.choice('donut_seq_len', [10, 20, 50]),
                 'batch_size': hp.choice('donut_batch_size', [4, 32, 64, 256]),
                 'lr': hp.choice('donut_lr', [0.1, 0.001, 0.0001]),
                 'epochs': hp.choice('donut_epochs', [1, 5, 10, 20])
                 # 'epochs': hp.choice('donut_epochs', [1])
             }
             },
        ])

    kpis = [i[:-4] for i in os.listdir("data/tzs/train")]

    for i in kpis:
        tune_kpi(i)
