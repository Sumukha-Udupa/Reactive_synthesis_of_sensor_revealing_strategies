from MDP import *
import copy
def get_attractor(mdp, F):
    """

    :param mdp:
    :param F: a set of target states
    :return: the almost-sure winning region to reach F, and an almost-sure winning strategy that maps each state to a set of actions enabled from that state.
    """
    Y = copy.deepcopy(mdp.states)
    Ysets = []
    Xnew = set([])
    X = copy.deepcopy(F)
    Xsets = []
    flag = True
    while True:
        if Xnew == Y:
            break
        else:
            if flag:
                flag = False
                pass
            else:
                Y = Xnew
                Xsets = []
            X = F
            Xsets.append(X)
            Ysets.append(Y)
            while True:
                Xnew = get_Pre(mdp, X,Y)
                if Xnew == X:
                    break
                else:
                    X = Xnew
                    Xsets.append(X)
    # construct the policy
    pol = dict([])
    for s in Y:
        pol[s] = set([])
        for a in mdp.actlist:
            if mdp.suppDict[(s,a)].issubset(Y):
                pol[s].add(a)
    return Ysets, Xsets, pol

def get_safe(mdp, S):
    """
    :param mdp:
    :param S: A set of safe states.
    :return: the safe winning region and the policy.
    """

    X = set([]) # the safe winning region.
    Xnew = S
    safepol = dict([])
    while True:
        if X == Xnew:
            break
        else:
            X = Xnew
            Xnew = set([])
            for s in X:
                for a in mdp.actlist:
                    if mdp.suppDict[(s,a)].issubset(X):
                        Xnew.add(s)
                        break
    safewin  = copy.deepcopy(X)
    for s in safewin:
        safepol[s] = set([])
        for a in mdp.actlist:
            if mdp.suppDict[(s,a)].issubset(safewin):
                safepol[s].add(a)
    return safewin, safepol

def get_reach_avoid(mdp, unsafe, F):
    # compute a policy that ensures to reach the target with probability one while avoiding unsafe states.
    """

    :param mdp:
    :param unsafe: a set of states to be avoided
    :param target: a set of goal states to reach
    :return: asw region and policy
    """
    Y = copy.deepcopy(set(mdp.states)) -set(unsafe) # a set of safe states
    Ysets = []
    Xnew = set([])
    X = copy.deepcopy(F)
    Xsets = []
    flag = True
    while True:
        if Xnew == Y:
            break
        else:
            if flag:
                flag = False
                pass
            else:
                Y = Xnew
                Xsets = []
            X = F
            Xsets.append(X)
            Ysets.append(Y)
            while True:
                Xnew = get_Pre(mdp, X,Y)
                if Xnew == X:
                    break
                else:
                    X = Xnew
                    Xsets.append(X)
    # construct the policy
    pol = dict([])
    for s in Y:
        if s not in unsafe and s not in F:
            pol[s] = set([])
            for a in mdp.actlist:
                if mdp.suppDict[(s,a)].issubset(Y):
                    pol[s].add(a)
        else:
            pol[s] = set([]) # once the unsafe state or the target is reached, allowed actions are empty.
    return Ysets, Xsets, pol

def get_Pre(mdp, X, Y):
    Pre=set([])
    for s in mdp.states:
        for a in mdp.actlist:
            if len(mdp.suppDict[(s,a)].intersection(X)) != 0 and mdp.suppDict[(s,a)].issubset(Y):
                Pre.add(s)
                break
    return X.union(Pre)

def printPolicy(mdp, pol, polfile, state_labels =None):
    f = open(polfile, 'w')

    if state_labels == None:
        for s in pol.keys():
            f.write(str(s)+ str(": ")+str(pol[s])+'\n')
    else:
        for s in pol.keys():
            f.write(str(state_labels[s] )+ str(": ")+ str(pol[s])+'\n')
    f.close()
    return