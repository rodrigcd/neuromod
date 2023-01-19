import pickle
import os
from datetime import datetime
import pandas as pd


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_var(variable, var_name, results_path="./results"):
    check_dir(results_path)
    file_path = os.path.join(results_path, var_name)
    pickle.dump(variable, open(file_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def get_date_time():
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S-%f")[:-3]
    now = str(dt_string)
    return now


def load_results(result_path, load_all=True):
    params = pd.read_pickle(os.path.join(result_path, "params.pkl"))
    if load_all:
        results = pd.read_pickle(os.path.join(result_path, "results.pkl"))
    else:
        results = {"weights_sim": [[], []]}
    return params, results


if __name__ == "__main__":
    print(get_date_time())