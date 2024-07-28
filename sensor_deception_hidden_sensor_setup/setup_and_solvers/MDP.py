
from scipy import stats
import numpy as np

from pydot import Dot, Edge, Node
import copy



class MDP:
    """A Markov Decision Process, defined by an initial state,
        transition model --- the probability transition matrix, np.array
        prob[a][0,1] -- the probability of going from 0 to 1 with action a.
        and reward function. We also keep track of a gamma value, for
        use by algorithms. The transition model is represented
        somewhat differently from the text.  Instead of T(s, a, s')
        being probability number for each state/action/state triplet,
        we instead have T(s, a) return a list of (p, s') pairs.  We
        also keep track of the possible states, terminal states, and
        actions for each state.  The input transitions is a
        dictionary: (state,action): list of next state and probability
        tuple.  AP: a set of atomic propositions. Each proposition is
        identified by an index between 0 -N.  L: the labeling
        function, implemented as a dictionary: state: a subset of AP."""

    def __init__(self, init=None, actlist=[], states=[], prob=dict([]), trans=dict([])):
        self.init = init
        self.actlist = actlist
        self.states = states
        self.prob = prob
        self.trans = trans
        self.suppDict = dict([])

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a row in the matrix for next-state probability."""
        i = self.states.index(state)
        return self.prob[action][i, :]

    def P(self, state, action, next_state):
        "Derived from the transition model. For a state, an action and the next_state, return the probability of this transition."
        i = self.states.index(state)
        j = self.states.index(next_state)
        return self.prob[action][i, j]

    def actions(self, state):
        N = len(self.states)
        S = set([])
        for a in self.actlist:
            if not np.array_equal(self.T(state, a), np.zeros(N)):
                S.add(a)
        return S

    def labeling(self, s, A):
        self.L[s] = A

    def get_supp(self):
        self.suppDict = dict([])
        for s in self.states:
            for a in self.actlist:
                self.suppDict[(s,a)] = self.supp(s,a)
        return

    def supp(self, state, action):
        supp = set([])
        for next_state in self.states:
            if self.P(state, action, next_state) !=0:
                supp.add(next_state)
        return supp

    def get_prec(self,state, act):
        # given a state and action, compute the set of states from which by taking that action, can reach that state with a nonzero probability.
        prec = set([])
        for pre_state in self.states:
            if self.P(pre_state, act, state) > 0:
                prec.add(pre_state)
        return prec

    def get_prec_anyact(self,state):
        # compute the set of states that can reach 'state' with some action.
        prec_all = set([])
        for act in self.actlist:
            prec_all = prec_all.union(self.get_prec(state, act))
        return prec_all


    def sample(self, state, action, num=1):
        """Sample the next state according to the current state, the action, and the transition probability. """
        if action not in self.actions(state):
            return None
        N = len(self.states)
        i = self.states.index(state)
        next_index = np.random.choice(N, num, p=self.prob[action][i, :])[
            0]  # Note that only one element is chosen from the array, which is the output by random.choice
        return self.states[next_index]



    def show_diagram(self, path=None):  # pragma: no cover
        """
            Creates the graph associated with this MDP
        """
        # Nodes are set of states

        graph = Dot(graph_type='digraph', rankdir='LR')
        nodes = {}
        for state in self.states:
            if state == self.init:
                # color start state with green
                initial_state_node = Node(
                       str(state),
                        style='filled',
                        peripheries=2,
                        fillcolor='#66cc33')
                nodes[str(state)] = initial_state_node
                graph.add_node(initial_state_node)
            else:
                state_node = Node(str(state))
                nodes[str(state)] = state_node
                graph.add_node(state_node)
        # adding edges
        for state in self.states:
            i = self.states.index(state)
            for act in self.actlist:
                for next_state in self.states:
                    j  = self.states.index(next_state)
                    if self.prob[act][i,j] != 0:
                        weight = self.prob[act][i,j]
                        graph.add_edge(Edge(
                            nodes[str(state)],
                            nodes[str(next_state)],
                            label = act + str(': ') + str(weight)
                        ))
        if path:
            graph.write_png(path)
        return graph


def get_NFA(mdp):
    """
    This function obtains the graph structure, which is essentially an non-deterministic finite state automaton from the original mdp.
    """
    nfa = NFA()
    nfa.initial_state = mdp.init
    nfa.states = mdp.states
    nfa.alphabet = mdp.actlist
    for a in mdp.actlist:
        for s in mdp.states:
            next_state_list = []
            for next_s in mdp.states:
                if mdp.prob[a][mdp.states.index(s), mdp.states.index(next_s)] != 0:
                    next_state_list.append(next_s)
            nfa.add_transition(a, s, next_state_list)
    nfa.final_states = mdp.terminals
    return nfa


def sub_MDP(mdp, H):
    """
    For a given MDP and a subset of the states H, construct a sub-mdp
    that only includes the set of states in H, and a sink states for
    all transitions to and from a state outside H.
    """
    if H == set(mdp.states):  # If H is the set of states in mdp, return mdp as it is.
        return mdp
    submdp = MDP()
    submdp.states = list(H)
    submdp.states.append(-1)  # -1 is the sink state.
    N = len(submdp.states)
    submdp.actlist = list(mdp.actlist)
    submdp.prob = {a: np.zeros((N, N)) for a in submdp.actlist}
    temp = np.zeros(len(mdp.states))
    for k in set(mdp.states) - H:
        temp[mdp.states.index(k)] = 1
    for a in submdp.actlist:
        for s in H:  # except the last sink state.
            i = submdp.states.index(s)
            for next_s in H:
                j = submdp.states.index(next_s)
                submdp.prob[a][i, j] = mdp.P(s, a, next_s)
            submdp.prob[a][i, -1] = np.inner(mdp.T(s, a), temp)
        submdp.prob[a][submdp.states.index(-1), submdp.states.index(-1)] = 1
    acc = []
    for (J, K) in mdp.acc:
        Jsub = set(H).intersection(J)
        Ksub = set(H).intersection(K)
        acc.append((Jsub, Ksub))
    acc.append(({}, {-1}))
    submdp.acc = acc
    return submdp


def read_from_file_MDP(fname):
    """
    This function takes the input file and construct an MDP based on the transition relations.
    The first line of the file is the list of states.
    The second line of the file is the list of actions.
    Starting from the second line, we have
    state, action, next_state, probability
    """
    f = open(fname, 'r')
    array = []
    for line in f:
        array.append(line.strip('\n'))
    f.close()
    mdp = MDP()
    state_str = array[0].split(",")
    mdp.states = [i for i in state_str]
    act_str = array[1].split(",")
    mdp.actlist = act_str
    mdp.prob = dict([])
    N = len(mdp.states)
    for a in mdp.actlist:
        mdp.prob[a] = np.zeros((N, N))
    for line in array[2: len(array)]:
        trans_str = line.split(",")
        state = trans_str[0]
        act = trans_str[1]
        next_state =  trans_str[2]
        p = float(trans_str[3])
        mdp.prob[act][mdp.states.index(state), mdp.states.index(next_state)] = p
    return mdp


def read_from_file_MDP_old(fname):
    """
    This function takes the input file and construct an MDP based on the transition relations.
    The first line of the file is the list of states.
    The second line of the file is the list of actions.
    Starting from the second line, we have
    state, action, next_state, probability
    """
    f=open(fname,'r')
    array = []
    for line in f:
        array.append( line.strip('\n') )
    f.close()
    mdp=MDP()
    state_str=array[0].split(",")
    mdp.states=[int(i) for i in state_str]
    act_str=array[1].split(",")
    mdp.actlist=act_str
    mdp.prob=dict([])
    N=len(mdp.states)
    for a in mdp.actlist:
        mdp.prob[a]=np.zeros((N,N))
    for line in array[2: len(array)]:
        trans_str=line.split(",")
        state=int(trans_str[0])
        act=trans_str[1]
        next_state=int(trans_str[2])
        p=float(trans_str[3])
        mdp.prob[act][mdp.states.index(state),mdp.states.index(next_state)]=p
    return mdp


