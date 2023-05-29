import numpy as np
import nengo

class Inputs():
    def __init__(self, deltaP, maxCues, seed=0, empirical=None):
        self.deltaP = deltaP  # task difficulty
        self.maxCues = maxCues  # maximum number of cues presented per trial
        self.correct = None  # correct choice for this trial
        self.empirical = empirical
        self.pA = None
        self.pB = None
        self.As = [0]
        self.Bs = [0]
        self.rng = np.random.RandomState(seed=seed)
    def set_AB(self, trial=None):
        self.As = np.zeros((self.maxCues))  # all cues for option A
        self.Bs = np.zeros((self.maxCues))  # all cues for option B
        if np.any(self.empirical):
            # populate the A and B arrays with the samples actually drawn in the empirical trial
            self.pA = self.empirical['pA'].to_numpy()[trial]
            self.pB = self.empirical['pB'].to_numpy()[trial]
            self.correct = "A" if self.pA > self.pB else "B"
            empAs = list(str(self.empirical['A'].to_numpy()[trial]))
            empAs = np.array([2*int(x)-1 for x in empAs])
            empBs = list(str(self.empirical['B'].to_numpy()[trial]))
            empBs = np.array([2*int(x)-1 for x in empBs])
            self.As[:len(empAs)] = empAs
            self.Bs[:len(empBs)] = empBs
            # fill out the remaining cues with randomly generated cues
            self.As[len(empAs):] = 2*self.rng.randint(2, size=self.maxCues-len(empAs))-1
            self.Bs[len(empBs):] = 2*self.rng.randint(2, size=self.maxCues-len(empBs))-1
        else:                
            self.correct = "A" if self.rng.rand()<0.5 else "B"
            highs = np.arange(0.1+self.deltaP, 0.9, 0.1)
            Pcorrect = highs[self.rng.randint(len(highs))]  # sample probability for the correct option
            Pincorrect = Pcorrect - self.deltaP  # sample probability for the incorrect option (P_incorrect = P_correct - deltaP)
            nUpCorrect = int(Pcorrect*self.maxCues)  # number of "up" cues for the correct option
            nUpIncorrect = int(Pincorrect*self.maxCues)  # number of "up" cues for the incorrect option
            # populate A/B cue lists with "up" and "down" cues
            if self.correct=="A":
                self.As[:nUpCorrect] = 1
                self.As[nUpCorrect:] = -1
                self.Bs[:nUpIncorrect] = 1
                self.Bs[nUpIncorrect:] = -1
                self.pA = nUpCorrect / self.maxCues  # true ration of "up" cues for A
                self.pB = nUpIncorrect / self.maxCues  # true ration of "up" cues for B
            else:
                self.Bs[:nUpCorrect] = 1
                self.Bs[nUpCorrect:] = -1
                self.As[:nUpIncorrect] = 1
                self.As[nUpIncorrect:] = -1
                self.pB = nUpCorrect / self.maxCues  # true ration of "up" cues for A
                self.pA = nUpIncorrect / self.maxCues  # true ration of "up" cues for B
            # randomize the order of A/B cue lists
            self.rng.shuffle(self.As)
            self.rng.shuffle(self.Bs)
    def get_AB(self, t):
        # A presented for 500ms, then B presented for 500ms:
        # diplay x=[A[t], 0] if in first 500ms, and display x=[0, B[t]] if in the second 500ms
        AB = [self.As[int(t)], self.Bs[int(t)]] if t<self.maxCues else [0,0] 
        AB = [AB[0], 0] if t%1.0<0.5 else [0, AB[1]]  
        return AB

def build_network(inputs, nNeurons=1000, synapse=0.1, seed=0, tau=0, m=0.2, delta=0, T=0.3):
    
    net = nengo.Network(seed=seed)
    net.config[nengo.Connection].synapse = 0.03
    net.config[nengo.Probe].synapse = 0.03

    # references
    net.tau = tau
    net.m = m
    net.delta = delta
    net.T = T
    net.inputs = inputs
    net.seed = seed
    net.synapse = synapse
    net.nNeurons = nNeurons
    
    func_input = lambda t: net.inputs.get_AB(t)
    func_threshold = lambda t: T
    func_urgency = lambda t: -tau * t
    func_ramp = lambda x: synapse * m * x
    func_confidence = lambda x: -delta * np.abs(x[0]-x[1])
    func_greater = lambda x: [x[0]-x[1], x[1]-x[0]] 
    
    ePos = nengo.dists.Choice([[1]])
    iPos = nengo.dists.Uniform(0.01, 1)

    with net:
        # Inputs
        environment = nengo.Node(func_input)
        time = nengo.Node(func_urgency)
        threshold = nengo.Node(func_threshold)
        # Ensembles
        value = nengo.networks.EnsembleArray(nNeurons, 2)
        accumulator = nengo.networks.EnsembleArray(nNeurons, 2)
        combined = nengo.Ensemble(nNeurons, 2)
        gate = nengo.Ensemble(nNeurons, 1, encoders=ePos, intercepts=iPos)
        action = nengo.networks.EnsembleArray(nNeurons, 2, encoders=ePos, intercepts=iPos)
        # Connections
        nengo.Connection(environment, value.input)  # external inputs
        nengo.Connection(value.ea_ensembles[0], accumulator.ea_ensembles[0], synapse=synapse, function=func_ramp)  # load input to WM
        nengo.Connection(value.ea_ensembles[1], accumulator.ea_ensembles[1], synapse=synapse, function=func_ramp)  # load input to WM
        nengo.Connection(accumulator.output, accumulator.input, synapse=synapse)  # recurrent WM connection
        nengo.Connection(accumulator.output, combined)  # send WM output to a single ensemble, to allow computation of func_confidence and func_greater
        nengo.Connection(combined, gate, function=func_confidence)
        nengo.Connection(combined, action.input, function=func_greater)
        nengo.Connection(time, gate)  # external inputs
        nengo.Connection(threshold, gate)  # external inputs
        nengo.Connection(gate, action.input, transform=[[-1], [-1]])  # inhibition via decision criteria
        # Probes
        net.pInputs = nengo.Probe(environment)
        net.pValue = nengo.Probe(value.output)
        net.pAccumulator = nengo.Probe(accumulator.output)
        net.pGate = nengo.Probe(gate)
        net.pAction = nengo.Probe(action.output)

    return net


def run_once(net, dt=0.001, progress_bar=False):
    sim = nengo.Simulator(net, seed=net.seed, progress_bar=progress_bar, optimize=False, dt=dt)
    chosen = False
    cues_sampled = 0
    with sim:
        while chosen==False and cues_sampled<=2*net.inputs.maxCues:
            sim.run(0.5, progress_bar=progress_bar)
            chooseA = np.argwhere(sim.data[net.pAction][:,0] > 0)  # indices of time points when model was choosing A as action output
            chooseB = np.argwhere(sim.data[net.pAction][:,1] > 0)  # indices of time points when model was choosing B as action output
            if not chosen: cues_sampled += 1
            chosen = True if (len(chooseA)>0 or len(chooseB)>0) else False

    if chosen:  # if the model made a choice before maxCues was reached
        chooseA = np.argwhere(sim.data[net.pAction][:,0] > 0)
        chooseB = np.argwhere(sim.data[net.pAction][:,1] > 0)
        firstA = chooseA[0][0] if len(chooseA)>0 else int(net.inputs.maxCues/dt)  # first time point when model chose A
        firstB = chooseB[0][0] if len(chooseB)>0 else int(net.inputs.maxCues/dt)  # first time point when model chose B
        choice = "A" if firstA < firstB else "B"
    else:  # if the model was forced to choose after sampling maxCues
        valueA = sim.data[net.pAccumulator][-1][0]
        valueB = sim.data[net.pAccumulator][-1][1]
        choice = "A" if valueA > valueB else "B"
    is_correct = True if choice==net.inputs.correct else False  
    return is_correct, cues_sampled