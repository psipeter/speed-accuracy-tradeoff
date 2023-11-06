import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Inputs_TwoDots():
    def __init__(self, dt=0.001, dt_sample=0.02, seed=0):
        self.coherence = None  # motion strength: 0=equal in two directions, 1=entirely one direction
        self.L = None  # percent of dots moving left on this trial
        self.R = None  # percent of dots moving right on this trial
        self.correct = None  # predominant direction of motion for this trial
        self.dt = dt  # nengo timestep
        self.dt_sample = dt_sample  # periodically resample the environment using a noisy perceptual system
        self.dL = None  # currently sampled fraction of dots moving left
        self.dR = None  # currently sampled fraction of dots moving right
        self.rng = np.random.RandomState(seed=seed)
    def create(self, coherence, correct=None):
        assert 0 <= coherence <= 1
        self.coherence = coherence
        self.dL = 0
        self.dR = 0
        if correct is not None:
            assert (correct=="L" or correct=="R")
            self.correct = correct
        else:
            self.correct = "L" if self.rng.rand() < 0.5 else "R"
        if self.correct == "L":
            self.L = 0.5 + self.coherence/2
            self.R = 0.5 - self.coherence/2
        if self.correct == "R":
            self.L = 0.5 - self.coherence/2
            self.R = 0.5 + self.coherence/2
        assert self.L + self.R == 1
    def sample(self, t):
        if self.dt_sample > 0:  # noisy perceptual sampling process
            if t % self.dt_sample < self.dt:
                self.dL = 1 if self.rng.rand() < self.L else 0
                # self.dR = 1 if self.rng.rand() < self.R else 0
                self.dR = 1 - self.dL
            return [self.dL, self.dR]
        else:  # directly perceive coherence level
            return [self.L, self.R]

def build_network(inputs, nNeurons=1000, synapse=0.1, seed=0, ramp=1, threshold=0.5, relative=0, probe_spikes=False):
    
    net = nengo.Network(seed=seed)
    net.config[nengo.Connection].synapse = 0.03
    net.config[nengo.Probe].synapse = 0.03

    # references
    net.inputs = inputs
    net.seed = seed
    net.synapse = synapse
    net.nNeurons = nNeurons

    net.ramp = ramp
    net.threshold = threshold
    net.relative = relative

    func_input = lambda t: net.inputs.sample(t)
    func_threshold = lambda t: net.threshold
    func_ramp = lambda x: net.synapse * net.ramp * x
    func_value = lambda x: [x[0]-x[1]*net.relative, x[1]-x[0]*net.relative]  # raw evidence vs relative advantage
    
    ePos = nengo.dists.Choice([[1]])
    iPos = nengo.dists.Uniform(0, 1)

    with net:
        # Inputs
        environment = nengo.Node(func_input)
        threshold = nengo.Node(func_threshold)
        # Ensembles
        perception = nengo.Ensemble(nNeurons, 2)
        accumulator = nengo.Ensemble(nNeurons, 2)
        value = nengo.Ensemble(nNeurons, 2)
        gate = nengo.Ensemble(nNeurons, 1, encoders=ePos, intercepts=iPos)
        action = nengo.networks.EnsembleArray(nNeurons, 2, encoders=ePos, intercepts=iPos)
        # Connections
        nengo.Connection(environment, perception)  # external inputs
        nengo.Connection(perception, accumulator, synapse=net.synapse, function=func_ramp)  # send percepts to accumulator
        nengo.Connection(accumulator, accumulator, synapse=net.synapse) # recurrent connection for accumulation
        nengo.Connection(accumulator, value, function=func_value)  # compute value from evidence in accumulator
        nengo.Connection(value, action.input)
        nengo.Connection(threshold, gate)  # external inputs
        nengo.Connection(gate, action.input, transform=[[-1], [-1]])  # inhibition via decision criteria
        # Probes
        net.pInputs = nengo.Probe(environment)
        net.pPerception = nengo.Probe(perception)
        net.pAccumulator = nengo.Probe(accumulator)
        net.pValue = nengo.Probe(value)
        net.pGate = nengo.Probe(gate)
        net.pAction = nengo.Probe(action.output)
        if probe_spikes:
            net.pSpikes = nengo.Probe(value.neurons)
        net.value = value

    return net


def single_trial(net, dt=0.001, progress_bar=False, tmax=10):
    sim = nengo.Simulator(net, progress_bar=False)
    choice = None
    while choice==None:
        sim.run(dt)
        if sim.data[net.pAction][-1,0] > 0:
            choice = "L"
            RT = sim.trange()[-1]
        elif sim.data[net.pAction][-1,1] > 0:
            choice = "R"
            RT = sim.trange()[-1]
        if sim.trange()[-1] > tmax:
            break
    correct = 1 if choice==net.inputs.correct else 0
    return 100*correct, 1000*RT