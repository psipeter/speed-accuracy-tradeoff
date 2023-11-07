import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DotPerception():
    def __init__(self, nActions, dt=0.001, dt_sample=0.02, seed=0):
        self.nActions = nActions
        self.coherence = None  # motion strength: 0=equal in two directions, 1=entirely one direction
        self.motions = []  # percent of dots moving towards each choice on this trial
        self.correct = None  # predominant direction of motion for this trial
        self.dt = dt  # nengo timestep
        self.dt_sample = dt_sample  # periodically resample the environment using a noisy perceptual system
        self.sampled = []  # currently sampled fraction of dots moving towards each choice
        self.rng = np.random.RandomState(seed=seed)
    def create(self, coherence, correct=None):
        assert 0 <= coherence <= 1
        self.coherence = coherence
        self.motions = np.zeros((self.nActions))
        self.sampled = np.zeros((self.nActions))
        if correct is not None:
            self.correct = correct
        else:
            self.correct = self.rng.randint(0, self.nActions)
        self.motions[:] = 1.0 / self.nActions
        for a in range(self.nActions):
            if a==self.correct:
                self.motions[a] += (self.nActions-1)/self.nActions * coherence
            else:
                self.motions[a] += -1.0 / self.nActions * coherence
    def sample(self, t):
        if self.dt_sample > 0:  # noisy perceptual sampling process
            if t % self.dt_sample < self.dt:
                for a in range(self.nActions):
                    self.sampled[a] = 1 if self.rng.rand() < self.motions[a] else 0
            return self.sampled
        else:  # directly perceive coherence level
            return self.motions

def build_network(inputs, nActions=2, nNeurons=1000, synapse=0.1, seed=0, ramp=1, threshold=0.5, relative=0, probe_spikes=False):
    
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
    if nActions==2:
        func_value = lambda x: [x[0]-x[1]*net.relative, x[1]-x[0]*net.relative]  # raw evidence vs relative advantage
    else:
        func_value = lambda x: x  # todo: write relative version for nActions>2

    ePos = nengo.dists.Choice([[1]])
    iPos = nengo.dists.Uniform(0, 1)

    with net:
        # Inputs
        environment = nengo.Node(func_input)
        threshold = nengo.Node(func_threshold)
        # Ensembles
        perception = nengo.Ensemble(nNeurons, nActions)
        accumulator = nengo.Ensemble(nNeurons, nActions)
        value = nengo.Ensemble(nNeurons, nActions)
        gate = nengo.Ensemble(nNeurons, 1, encoders=ePos, intercepts=iPos)
        action = nengo.networks.EnsembleArray(nNeurons, nActions, encoders=ePos, intercepts=iPos)
        # Connections
        nengo.Connection(environment, perception)  # external inputs
        nengo.Connection(perception, accumulator, synapse=net.synapse, function=func_ramp)  # send percepts to accumulator
        nengo.Connection(accumulator, accumulator, synapse=net.synapse) # recurrent connection for accumulation
        nengo.Connection(accumulator, value, function=func_value)  # compute value from evidence in accumulator
        nengo.Connection(value, action.input)
        nengo.Connection(threshold, gate)  # external inputs
        nengo.Connection(gate, action.input, transform=-1*np.ones((nActions, 1)))  # inhibition via decision criteria
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


def single_trial(net, nActions, dt=0.001, progress_bar=False, tmax=10):
    sim = nengo.Simulator(net, progress_bar=False)
    choice = None
    RT = None
    while choice==None:
        sim.run(dt)
        if np.any(sim.data[net.pAction][-1,:] > 0):
            choice = np.argmax(sim.data[net.pAction][-1,:])
            RT = sim.trange()[-1]
        if sim.trange()[-1] > tmax:
            return None, None
    correct = 1 if choice==net.inputs.correct else 0
    return 100*correct, 1000*RT