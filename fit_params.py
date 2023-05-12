import numpy as np
import nengo
import pandas as pd
import nni
import pickle
from model import Inputs, build_network, run_once


def get_loss(simulated_cues, empirical_cues, trial):
    delta_cues = np.abs(simulated_cues - empirical_cues[trial])
    return delta_cues

def evaluate_fit(args):
    # each participant has a participant ID (a long string) and an associated "pid" (integer) in the database
    # load this data to compute the loss between simulated and model behavior
    with open('data/pids.pkl', 'rb') as f:
        pids = pickle.load(f)
    pid = args['pid']
    participant_ID = pids[pid]
    empirical = pd.read_pickle("data/empirical.pkl").query("maxSamples==12 & delta==0.1 & participant_id==@participant_ID")

    nTrials = empirical.shape[0]
    # nTrials = 3
    seed = args['pid']
    T = args["T"]
    m = args["m"]
    tau = args["tau"]
    delta = args["delta"]
    # print(seed, T, m, tau, delta)

    total_loss = 0
    for trial in range(nTrials):
        is_correct, cues_sampled = run_once(deltaP=0.1, empirical=empirical, trial=trial, T=T, m=m, tau=tau, delta=delta, seed=seed)
        loss = get_loss(cues_sampled, empirical['cues'].to_numpy(), trial)
        print(f"trial {trial}, loss={loss}")
        total_loss += loss
        nni.report_intermediate_result(total_loss)

    nni.report_final_result(total_loss)

if __name__ == '__main__':
    params = nni.get_next_parameter()
    # params = {"pid": 0, "T": 0.3, "m": 0.2, "tau": 0, "delta": 0}
    evaluate_fit(params)