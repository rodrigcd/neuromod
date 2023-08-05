from metamod.utils import maml_routine, save_var, get_date_time
import numpy as np
import argparse
import sys, os
from tqdm import tqdm


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-name', type=str, default="MAML_sweep")
    parser.add_argument(
        '--save-path', type=str, default="../results/maml_sweep/")
    parser.add_argument(
        '--run-id', type=int, default=0)
    args = parser.parse_args(argv)
    args = vars(args)
    results_path = args["save_path"]
    run_id = args["run_id"]

    iter_control = 1000
    eval_steps = 8000
    skip_steps = 20
    n_steps = np.arange(0, 341, skip_steps)
    n_steps[0] = 2
    print(n_steps)

    for steps in tqdm(n_steps):
        results = maml_routine(n_steps=steps,
                               eval_steps=eval_steps,
                               iter_control=iter_control,
                               last_step=False)
        time_str = get_date_time()
        run_name = args["run_name"] + "_run_" + str(run_id) + "_n_steps_" + str(steps)
        saving_path = os.path.join(results_path, run_name + "_" + time_str)
        save_var(results, "results.pkl", results_path=saving_path)


if __name__ == "__main__":
    main(sys.argv[1:])
