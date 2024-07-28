# The older version of this class ONLY did NOT have the input query actions and the function query.
from sensor_deception_hidden_sensor_setup.setup_and_solvers.MDP import *


class sensor(MDP):
    # a stochastic transition system as finite memory sensor model.
    def __init__(self, init=None, actlist=['-'], states=[], prob=dict([]), trans=dict([]), coverage=dict([]),
                 query_actions=dict([])):
        # a sensor is modeled as a finite memory transition system with outputs.
        super().__init__()
        self.coverage = coverage
        self.get_supp()
        self.query_actions = query_actions

    def set_coverage(self, state, covered_set):
        # for each state of the sensor, a subset of the surveillance region
        self.coverage[state] = covered_set
        return

    def get_coverge(self, state):
        return self.coverage[state]

    def query(self):
        return self.query_actions
