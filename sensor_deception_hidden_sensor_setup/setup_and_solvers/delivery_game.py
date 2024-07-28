__author__ = 'Sumukha Udupa, sudupa@ufl.edu'

import sensor_deception_hidden_sensor_setup.setup_and_solvers.active_sensing_and_sensing_attacks_game as active_sensing_and_sensing_attacks_game
import numpy as np
# from active_sensing_and_sensing_attacks_game import *
import copy


class DeliveryGame(active_sensing_and_sensing_attacks_game.ActiveSensingGame):
    def __init__(self, init=None, actlist=[], states=[], prob=dict([]), trans=dict([]), stateObs=dict([]),
                 actionAtt=dict([]), agentmdp=None, sensor=None, target=None, non_init_states=[], unsafe_obs=[],
                 precise_sensors=[]):
        super().__init__()
        self.agentmdp = agentmdp
        self.sensor = sensor
        self.final = target
        self.non_init_states = non_init_states
        self.unsafe_obstacles = unsafe_obs
        self.precise_sensors = precise_sensors

    def set_agentmdp(self, agentmdp):
        self.agentmdp = agentmdp
        self.actlist = copy.deepcopy(self.agentmdp.actlist)

    def set_sensor(self, sensor):
        self.sensor = sensor

    def get_game(self):
        self.actlist = self.agentmdp.actlist

        init_states = set(self.agentmdp.states) - set(self.non_init_states)

        init_state = list(init_states)
        states = list(init_states)
        self.init = init_state

        # Uncomment the following three lines to have the  experiment run for single initial state.
        # init_state = self.agentmdp.init
        # states = [init_state]
        # self.init = [init_state]
        trans = dict([])
        suppDict = dict([])
        count = 0
        while len(states) > count:
            s = states[count]
            gs_idx = states.index(s)
            count = count + 1
            for a in self.actlist:
                suppDict[(s, a)] = set([])
                trans[(gs_idx, a)] = dict([])
                for ns in self.agentmdp.suppDict[(s, a)]:
                    if ns not in states:
                        states.append(ns)
                    ngs_idx = states.index(ns)
                    trans[(gs_idx, a)][ngs_idx] = self.agentmdp.P(s, a, ns)
                    suppDict[(s, a)].add(ns)
                    # for nt in self.sensor.suppDict[(t,'-')]:
                    #     if (ns, nt) not in states:
                    #         states.append((ns,nt))
                    #     ngs_idx = states.index((ns,nt))
                    #     trans[(gs_idx,a)][ngs_idx] = self.agentmdp.P(s,a,ns) * self.sensor.P(t,'-',nt)
                    #     suppDict[((s,t), a)].add((ns,nt))
        self.states = states
        N = len(self.states)
        prob = {a: np.zeros((N, N)) for a in self.actlist}
        for (state, act) in trans.keys():
            tempdict = trans[(state, act)]
            for nextstate in tempdict.keys():
                prob[act][state, nextstate] = tempdict[nextstate]
        self.prob = prob
        self.suppDict = suppDict
        return

    def get_stateObs(self):
        # compute the sensor information.
        # determine if the sensor is precise or not.
        self.stateObs = dict([])
        sensor_query_actions = self.sensor.query()

        for s in self.states:
            for ga1 in sensor_query_actions:
                for ga2 in self.actionAtt:
                    self.stateObs[(s, ga1, ga2)] = set([])
                    active_sensors = sensor_query_actions[ga1] - self.actionAtt[ga2]
                    observ = set(self.agentmdp.states)
                    for act_sense in active_sensors:
                        if act_sense not in self.precise_sensors:
                            if s in self.sensor.coverage[act_sense]:
                                observ = observ.intersection(self.sensor.coverage[act_sense])
                            else:
                                temp = set(self.agentmdp.states) - self.sensor.coverage[act_sense]
                                observ = observ.intersection(temp)
                        else:
                            if s in self.sensor.coverage[act_sense]:
                                observ = observ.intersection({s})
                            else:
                                temp = set(self.agentmdp.states) - self.sensor.coverage[act_sense]
                                observ = observ.intersection(temp)

                    self.stateObs[(s, ga1, ga2)] = observ

            #
            # self.stateObs[(s,t)] = set([])
            # if s in self.sensor.coverage[t]: # the current agent state is within the sensor range. return the exact state.
            #     #temp = set([s])
            #     temp = self.sensor.coverage[t]
            # else:
            #     temp = set(self.agentmdp.states) - self.sensor.coverage[t] # any states that is not in the current sensor coverge is possible.
            # for obs_s in temp:
            #     self.stateObs[(s,t)].add((obs_s,t)) # observation equilvalent state but the same sensor state t.
            # if (s,t) not in self.stateObs[(s,t)]:
            #     print("ERROR in constructing state observation")
        return
