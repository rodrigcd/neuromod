import numpy as np
from tqdm import tqdm
import os
from metamod.control import SingleLayerEq, SingleLayerControl


def main():
    args = vars(args)

    run_id = args["run_id"]
    save_path = args["save_path"]
    n_updates = args["n_updates"]
    random_transition = args["random_transition"]
    learning_rate = args["learning_rate"]

    ##########################################
    # parameters to run ######################
    ##########################################
    state_range = (2, 16)
    states_arange = np.arange(state_range[0], state_range[1])
    n_samples = 2**(states_arange+1)-2
    model_types = ["uniform", "oracle", "proportional", "rank-based", "uncertainty", "ensemble_prop",
                   "ensemble_rank", "DEUP"]
    noise_levels = [0, 0.01, 0.05, 0.1]
    noise_at_every_transition = [True, False]
    ##########################################
    # parameters to run ######################
    ##########################################

    number_of_combinations = len(states_arange) * len(model_types) * len(noise_levels)\
          * len(noise_at_every_transition)
    print("Total number of combinations", number_of_combinations)

    state_index = int(run_id % len(states_arange))
    index_leftover = int((run_id - state_index)/len(states_arange))
    model_index = int(index_leftover % len(model_types))
    index_leftover = int((index_leftover - model_index) / len(model_types))
    noise_index = int(index_leftover % len(noise_levels))
    index_leftover = int((index_leftover - noise_index) / len(noise_levels))
    noise_at_index = int(index_leftover % len(noise_at_every_transition))

    n_s = states_arange[state_index]
    model_id = model_types[model_index]
    reward_noise = noise_levels[noise_index]
    noise_every_transition = noise_at_every_transition[noise_at_index]
    print("n_s", n_s)
    print("model_id", model_id)
    print("reward_noise", reward_noise)
    print("noise_every_transition", noise_every_transition)

    time_str = get_date_time()
    run_name = "n_s_" + str(n_s)
    run_name += "_model_" + model_id
    run_name += "_reward-noise_" + str(reward_noise)
    run_name += "_noise-trans_"+str(noise_every_transition)
    saving_path = os.path.join(save_path, run_name + "_" + time_str)
    print(saving_path)

if __name__ == "__main__":
    main(sys.argv[1:])