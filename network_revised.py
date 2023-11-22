import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DotPerception():
    def __init__(self, nActions, dt=0.001, dt_sample=0.01, seed=0, sigma=0):
        self.nActions = nActions
        self.coherence = None  # motion strength: 0=equal in two directions, 1=entirely one direction
        self.motions = []  # percent of dots moving towards each choice on this trial
        self.correct = None  # predominant direction of motion for this trial
        self.dt = dt  # nengo timestep
        self.dt_sample = dt_sample  # periodically resample the environment using a noisy perceptual system
        self.sampled = []  # currently sampled fraction of dots moving towards each choice
        self.sigma = sigma
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
                    # self.sampled[a] = 1 if self.rng.rand() < self.motions[a] else 0
                    self.sampled[a] = self.rng.normal(self.motions[a], self.sigma)
                    # self.sampled[self.correct] += self.coherence
            return self.sampled
        else:  # directly perceive coherence level
            return self.motions

class SequentialPerception():
    def __init__(self, dt=0.001, dt_sample=0.5, seed=0, max_cues=18):
        self.dP = None  # difference in probability of generating positive cues (dP=abs(P0 - P1))
        self.Ps = []  # true probabilities of generating positive cues for the current trial
        self.correct = None  # greater probability on the current trial (correct=argmax(Ps))
        self.incorrect = None  # less probability on the current trial (correct=argmin(Ps))
        self.dt = dt  # nengo timestep
        self.dt_sample = dt_sample  # period for sampling one cue (time that cue appears on screen)
        self.sampled = []  # sampled cues for the current timestep; alternates between [X,0] and [0, Y] every dt_sample
        self.sampled_first = None  # which cue is sampled first on the current trial
        self.rng = np.random.RandomState(seed=seed)
    def create(self, dP):
        assert 0<=dP<=1
        self.dP = dP
        self.Ps = np.zeros((2))
        self.sampled = np.zeros((2))
        self.correct = 0 if self.rng.rand() < 0.5 else 1
        self.incorrect = 1 if self.correct==0 else 0
        self.Ps[self.correct] = self.rng.uniform(0.1+self.dP, 0.9)
        self.Ps[self.incorrect] = self.Ps[self.correct] - self.dP
        self.sampled_first = 0 if self.rng.rand() < 0.5 else 1
        if self.sampled_first == 0:
            self.sampled[0] = 1 if self.rng.rand()<self.Ps[0] else -1
            self.sampled[1] = 0
        elif self.sampled_first == 1:
            self.sampled[1] = 1 if self.rng.rand()<self.Ps[1] else -1
            self.sampled[0] = 0
        # print(self.Ps)
        # print(self.sampled)
    def sample(self, t):
        if t % self.dt_sample < self.dt/10 and t>self.dt:
            current_cue = np.where(self.sampled!=0)[0]
            next_cue = np.where(self.sampled==0)[0]
            self.sampled[current_cue] = 0
            self.sampled[next_cue] = 1 if self.rng.rand()<self.Ps[next_cue] else -1
            # print(t, self.sampled)    
        return self.sampled


def build_network(inputs, nActions=2, nNeurons=2000, synapse=0.1, seed=0, ramp=1, threshold=0.5, relative=0, rA=4, probe_spikes=False):
    
    net = nengo.Network(seed=seed)
    net.config[nengo.Connection].synapse = 0.03
    net.config[nengo.Probe].synapse = 0.03
    net.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(100, 200)

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
        func_value = lambda x: [
            x[0]-x[1]*net.relative + x[0]-x[2]*net.relative + x[0]-x[3]*net.relative,
            x[1]-x[0]*net.relative + x[1]-x[2]*net.relative + x[1]-x[3]*net.relative,
            x[2]-x[0]*net.relative + x[2]-x[1]*net.relative + x[2]-x[3]*net.relative,
            x[3]-x[0]*net.relative + x[3]-x[1]*net.relative + x[3]-x[2]*net.relative,
            ]

    ePos = nengo.dists.Choice([[1]])
    iPos = nengo.dists.Uniform(0, 1)

    with net:
        # Inputs
        environment = nengo.Node(func_input)
        threshold = nengo.Node(func_threshold)
        # Ensembles
        perception = nengo.Ensemble(nNeurons, nActions)
        accumulator = nengo.Ensemble(nNeurons, nActions, radius=rA)
        value = nengo.Ensemble(nNeurons, nActions, radius=net.threshold)
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
        net.pInputs = nengo.Probe(environment, synapse=None)
        net.pPerception = nengo.Probe(perception)
        net.pAccumulator = nengo.Probe(accumulator)
        net.pValue = nengo.Probe(value)
        net.pGate = nengo.Probe(gate)
        net.pAction = nengo.Probe(action.output)
        if probe_spikes:
            net.pSpikes = nengo.Probe(value.neurons)
        net.value = value

    return net