import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector

def chi_squared_distance(a,b):
    distance = 0
    for i in range(len(a)):
        if a[i]+b[i]==0:
            continue
        else:
            distance += np.square(a[i] - b[i]) / (a[i]+b[i])
    return distance

def get_loss(simulated, empirical):
    total_loss = 0
    bins = np.arange(1, 7, 1)
    rts_sim = simulated['cues_sampled'].to_numpy()
    rts_emp = empirical['cues_sampled'].to_numpy()
    hist_rts_sim = np.histogram(rts_sim, bins=bins)[0]
    hist_rts_emp = np.histogram(rts_emp, bins=bins)[0]
    normed_hist_rts_sim = hist_rts_sim / len(rts_sim)
    normed_hist_rts_emp = hist_rts_emp / len(rts_emp)
    chi_loss = chi_squared_distance(normed_hist_rts_sim, normed_hist_rts_emp)
    # mean_loss = np.abs(np.mean(rts_sim) - np.mean(rts_emp))
    # median_loss = np.abs(np.median(rts_sim) - np.median(rts_emp))
    total_loss += chi_loss
    # total_loss += median_loss
    # total_loss += mean_loss
    return total_loss

def build_inputs(trial, weight_cond, on_time=0.8, off_time=0.2):
    # all input information as arrays
    values_A = pd.read_pickle("data/input_values.pkl").query("trial==@trial")['A'].to_numpy()
    values_B = pd.read_pickle("data/input_values.pkl").query("trial==@trial")['B'].to_numpy()
    weights = pd.read_pickle("data/input_weights.pkl").query("env_cond==@weight_cond")['weight'].to_numpy()
    # build nengo processes that feed these arrays over time
    dict_A = {}
    dict_B = {}
    dict_W = {}
    time = 0
    for cue in range(6):
        vA = values_A[cue]
        vB = values_B[cue]
        W = weights[cue]
        dict_A[time] = 0
        dict_B[time] = 0
        dict_W[time] = 0
        time += off_time
        dict_A[time] = vA
        dict_B[time] = vB
        dict_W[time] = W
        time += on_time
    value_A_input = nengo.processes.Piecewise(dict_A)
    value_B_input = nengo.processes.Piecewise(dict_B)
    weight_input = nengo.processes.Piecewise(dict_W)
    return {'A': value_A_input, 'B': value_B_input, 'W': weight_input}


def build_network(inputs, params):
    
    def ff_func(x):
        R = params['R']  #ramp rate
        A = x[0]  # value of A
        B = x[1]  # value of B
        W = x[2]  # recalled weight
        return [A*W*R, B*W*R]

    def eval_func(x):
        L = params['L']  # relative (L=1) vs absolute (L=0) evaluation
        A = x[0]  # value of A
        B = x[1]  # value of B
        return [A-B*L, B-A*L]

    def cog_load_func(x):
        A = x[0]  # value of A
        B = x[1]  # value of B
        return [A+B]

    def urg_func(x):
        U = params['U']  # urgency parameter scaling cognitive load
        S = params['S']  # emotional sensitivity
        cognitive_load = x[0]
        emotion = x[1]
        urgency = U*cognitive_load + S*emotion
        return [urgency, urgency]

    def pupil_func(x):
        U = params['U']  # urgency parameter scaling cognitive load
        S = params['S']  # emotional sensitivity
        cognitive_load = x[0]
        emotion = x[1]
        urgency = U*cognitive_load + S*emotion
        return [urgency]

    def inh_mult_func(x):
        U = params['U']  # urgency parameter scaling cognitive load
        S = params['S']  # emotional sensitivity
        I = params['I']  # scaling of inhibition from LC to multiply
        cognitive_load = x[0]
        emotion = x[1]
        urgency = U*cognitive_load + S*emotion
        return [I * urgency]
        
        
    with nengo.Network(seed=params['network_seed']) as network:
        # inputs
        A_input = nengo.Node(inputs['A'])
        B_input = nengo.Node(inputs['B'])
        W_input = nengo.Node(inputs['W'])
        E_input = nengo.Node(params['E'])
        pupil_input = nengo.Node(params['pupil_input'])
        motor_input = nengo.Node(params['motor_input'])
        # ensembles
        value = nengo.Ensemble(params['n_neurons'], 2, label='perception')
        weight = nengo.Ensemble(params['n_neurons'], 1, label='memory')
        multiply = nengo.Ensemble(params['n_neurons'], 3, label='orbitofrontal', radius=params['radius_multiply'])
        accumulate = nengo.Ensemble(params['n_accumulate'], 2, label='dorsolateral', radius=params['radius_accumulate'])
        evaluate = nengo.Ensemble(params['n_neurons'], 2, label='motor cortex', radius=params['radius_evaluate'])
        compete = nengo.networks.BasalGanglia(2, params['n_array'], label='basal ganglia', input_bias=params['T'])
        motor = nengo.networks.EnsembleArray(params['n_neurons'], 2, ens_dimensions=1, label='motor',
                                             intercepts=nengo.dists.Uniform(0, 1), encoders=nengo.dists.Choice([[1]]))
        control = nengo.Ensemble(params['n_control'], 2, label='locus ceruleus', radius=params['radius_control'])
        pupil = nengo.Ensemble(params['n_pupil'], 1, radius=params['radius_pupil'])
        differentiator = nengo.Ensemble(params['n_neurons'], 1,
                                        encoders=nengo.dists.Choice([[1]]), intercepts=nengo.dists.Uniform(0.01, 1))
        # connections
        nengo.Connection(A_input, value[0], synapse=None)
        nengo.Connection(B_input, value[1], synapse=None)
        nengo.Connection(W_input, weight, synapse=None)
        nengo.Connection(E_input, control[1], synapse=None)
        nengo.Connection(motor_input, motor.input, synapse=None, transform=[[1],[1]])
        nengo.Connection(W_input, differentiator, synapse=0.1, transform=1)
        nengo.Connection(W_input, differentiator, synapse=0.2, transform=-1)
        # nengo.Connection(differentiator, control.neurons, transform=10*np.ones((control.n_neurons, 1)))
        nengo.Connection(pupil_input, pupil)
        nengo.Connection(value, multiply[:2], synapse=params['syn_ff'])
        nengo.Connection(weight, multiply[2], synapse=params['syn_ff'])
        nengo.Connection(multiply, accumulate, synapse=params['syn_ff'], function=ff_func)
        nengo.Connection(accumulate, accumulate, synapse=params['syn_fb'])
        nengo.Connection(accumulate, evaluate, synapse=params['syn_ff'], function=eval_func)
        nengo.Connection(evaluate, compete.input, synapse=params['syn_ff'])
        nengo.Connection(compete.output, motor.input, synapse=params['syn_ff'])
        nengo.Connection(accumulate, control[0], synapse=params['syn_ff'], function=cog_load_func)
        nengo.Connection(control, evaluate, synapse=params['syn_ff'], function=urg_func)
        nengo.Connection(control, pupil, synapse=params['syn_ff'], function=pupil_func)
        nengo.Connection(control, multiply[2], synapse=params['syn_ff'], function=inh_mult_func)
        
        # probes
        network.p_A = nengo.Probe(A_input, synapse=None)
        network.p_B = nengo.Probe(B_input, synapse=None)
        network.p_W = nengo.Probe(W_input, synapse=None)
        network.p_value = nengo.Probe(value, synapse=params['syn_probe'])
        network.p_weight = nengo.Probe(weight, synapse=params['syn_probe'])
        network.p_accumulate = nengo.Probe(accumulate, synapse=params['syn_probe'])
        network.p_evaluate = nengo.Probe(evaluate, synapse=params['syn_probe'])
        network.p_compete = nengo.Probe(compete.output, synapse=params['syn_probe'])
        network.p_motor = nengo.Probe(motor.output, synapse=params['syn_probe'])
        network.p_control = nengo.Probe(control, synapse=params['syn_probe'])
        network.p_pupil = nengo.Probe(pupil, synapse=params['syn_probe'])

    return network


def objective(optuna_trial, ID, weights, emotion):
    # let optuna choose the next parameters
    params = {
        # individual parameters
        'T': -1.0,  # threshold in basal ganglia
        'L': 1.0,  # absolute vs relative evaluation
        'U': 0.5,  # urgency parameter, scales LC output
        'I': 0.0,  # scaling of inhibition from LC to multiply
        'R': 0.1,  # rate of evidence accumulation
        'S': 1.0,  # emotional sensitivity, scales LC output
        'E': -0.3,  # emotional input [0 for neutral, -1 for negative, +1 for positive]
        # neurons per population
        'n_neurons': 300,
        'n_accumulate': 1000,  # neurons in the accumulator population
        'n_control': 1000,  # neurons in the control population
        'n_pupil': 1000,  # neurons in the pupil population
        'n_array': 100,
        # radius
        'radius_multiply': 2.0,
        'radius_accumulate': 5.0,
        'radius_evaluate': 4.0,
        'radius_control': 5.0,  # radius of LC, to prevent saturation
        'radius_pupil': 4,  # radius of pupil ensemble, to prevent saturation
        # synapses
        'syn_ff': 0.05,  # feedforward synapse
        'syn_fb': 0.1,  # feedback synapse
        'syn_probe': 0.01,  # filter for decoding
        # simulation
        'trials': 36,
        'network_seed': 0,
        'dt': 0.001,  # simulation timestep
        'tmin': 0.03,  # discard the first "tmin" amount of data, to avoid startup effects
        'presentation_time': 0.8,  # time to present each value/weight (seconds)
        'intercue_interval': 0.2,  # time between cue presentations, used to generate P300 signal
        # inputs
        'motor_input': 0.1,  # threshold for motor actions
        'pupil_input': 0.5,  # baseline input to pupil population
    }

    params['T'] = optuna_trial.suggest_float("T", -3.0, 3.0, step=0.01)
    params['L'] = optuna_trial.suggest_float("L", 0.0, 1.0, step=0.01)
    params['U'] = optuna_trial.suggest_float("U", 0.0, 2.0, step=0.01)
    params['R'] = optuna_trial.suggest_categorical("R", [0.1])
    params['S'] = optuna_trial.suggest_categorical("S", [0.0])
    params['I'] = optuna_trial.suggest_categorical("I", [0.0])
    params['E'] = optuna_trial.suggest_categorical("E", [0.0])

    dfs = []
    columns = ['type', 'ID', 'weights', 'emotion', 'trial', 'choice', 'cues_sampled', 'correct', 'response_time']
    for trial in range(params['trials']-1):
        params['network_seed'] = trial  # build a unique network for every trial; this increases trial-to-trial variance
        inputs = build_inputs(trial+1, weights, params['presentation_time'], params['intercue_interval'])
        network = build_network(inputs, params)
        sim = nengo.Simulator(network, dt=params['dt'], progress_bar=False)
        # define what constitutes 'choice' for the model, and measure the number of cues samples
        tmax = 6*(params['presentation_time'] + params['intercue_interval'])
        time = 0
        while time<=tmax:
            sim.run(params['dt'])
            motor = sim.data[network.p_motor][-1]
            if np.any(motor>0.01) and time>params['tmin']:
                break
            time += params['dt']
        sumWA = np.sum(sim.data[network.p_A].ravel()*sim.data[network.p_W].ravel())
        sumWB = np.sum(sim.data[network.p_B].ravel()*sim.data[network.p_W].ravel())
        best_choice = 'A' if sumWA > sumWB else 'B'
        response_time = time
        if response_time>=tmax:
            evaluate = sim.data[network.p_evaluate][-1]
            choice = 'A' if np.argmax(evaluate)==0 else 'B'
            cues_sampled = 6
            correct = 1 if choice==best_choice else 0
        else:
            choice = 'A' if np.argmax(motor)==0 else 'B'
            cues_sampled = int(response_time/(params['presentation_time'] + params['intercue_interval']))+1
            correct = 1 if choice==best_choice else 0
        dfs.append(pd.DataFrame([['model', ID, weights, emotion, trial, choice, cues_sampled, correct, response_time]], columns=columns))
        print(f"ID {ID}, weights {weights}, trial {trial}, choice {choice}, best choice {best_choice}, cues sampled {cues_sampled}, correct {correct}, response time {response_time:.4}")

    simulated = pd.concat(dfs, ignore_index=True)
    empirical = pd.read_pickle("data/wichary.pkl").query("ID==@ID & weights==@weights & emotion==@emotion")
    loss = get_loss(simulated, empirical)
    return loss

if __name__ == '__main__':

    ID = int(sys.argv[1])
    weights = sys.argv[2]
    emotion = sys.argv[3]
    label = sys.argv[4]
    study_name = f"wichary_{ID}_{weights}_{emotion}_{label}"
    optuna_trials = 1000

    # objective(None, ID)
    # raise

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial, ID, weights, emotion), n_trials=optuna_trials)