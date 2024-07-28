__author__ = 'Sumukha Udupa, sudupa@ufl.edu'

import gc
import itertools
import numpy as np
from scipy.sparse import lil_matrix

from sensor_deception_hidden_sensor_setup.setup_and_solvers.MDP import *
import copy
import tqdm as tqdm
from sensor_deception_hidden_sensor_setup.setup_and_solvers.stochatic_two_player_game_solver import *


class ActiveSensingGame(MDP):
    def __init__(self, init=None, actlist=[], states=[], prob=dict([]), trans=dict([]), stateObs=dict([]),
                 nature_act='O', actionAtt=dict([]), hidden_sensor_queries=dict([]), hidden_sensor_attacks=dict([]),
                 sensor=None, target=None):
        super().__init__()
        self.stateObs = stateObs
        self.actionAtt = actionAtt
        self.suppDict = dict([])
        self.get_supp()
        self.sensor = sensor
        self.final = target  # This is a list of states in the grid that are the reachability goals.
        self.hidden_sensor_queries = hidden_sensor_queries
        self.hidden_sensor_attacks = hidden_sensor_attacks
        # self.unsafe_obstacles = unsafe_obstacles
        # self.allowed_action_states = self.get_allowed_actions()

    def get_allowed_actions(self):
        """

        :return: a dictionary for each state with allowed actions.
        """
        allowed_actions_states = dict([])

        for s in self.states:
            allowed_actions = set([])
            for a in self.actlist:
                post_states = self.suppDict[(s, a)]
                if len(post_states.intersection(set(self.unsafe_obstacles))) == 0:
                    allowed_actions.add(a)
            allowed_actions_states[s] = allowed_actions
        self.allowed_action_states = allowed_actions_states

    def get_allowed_combined_actions(self, B):
        """

        :param B: current belief
        :return: set of allowed actions
        """
        allow = set([])
        for s in B:
            allow = allow.union(self.allowed_action_states[s])
        return allow

    def get_post(self, B, a):
        """
        :param B: the belief, a subset of states
        :param a: the action.
        :return: the set of states that can be reached from the current belief/state set.
        """
        post = set([])
        for s in B:
            post = post.union(set(self.suppDict[(s, a)]))
        return post

    def belief_update(self, B, oa, os):
        """
        :param B: the current belief
        :param oa: the observation of actions.
        :param os: the observation of states.
        :return: the new belief give the observations.
        """
        new_belief = set([])
        for a in oa:  # a set of observational equivalent actions.
            temp = self.get_post(B, a)
            temp.intersection_update(os)  # find the next states consistent with the observation.
            new_belief = new_belief.union(temp)
        return new_belief

    # def consistentBelief(self, B, oa, allowed):
    #     Br = set([])
    #     for s in B:
    #         if s in allowed.keys():
    #             for a in allowed[s]:
    #                 if self.actionObs[(s,a)] == oa: # the action observation for s,a is consistent with the actual observation oa.
    #                     Br.add(s)
    #     return Br

    def get_aug_sensing_game_perceptual(self):
        """
        :return: augmented_mdp for Delayed-attack game.
        """
        augmdp = ActiveSensingGame()
        p1_states = list([])
        p1_states_idx = list([])
        p1_states_idx_dict = dict([])
        nature_states = list([])
        nature_states_idx = list([])
        nature_states_idx_dict = dict([])
        p2_states = list([])
        p2_states_idx = list([])
        p2_states_idx_dict = dict([])
        allowed_actions_list = dict([])
        allowed_attack_actions_list = dict([])

        augstates_membership_dict = dict([])

        sensor_queries = self.sensor.query()
        augstates = ['final']
        index_num = 0

        for init in self.init:
            init_state = (init, {init}, -1)
            augstates.append(init_state)
            p1_states.append(init_state)
            index_num = index_num + 1
            augstates_membership_dict[(init, frozenset({init}), -1)] = index_num

            p1_states_idx.append(index_num)
            p1_states_idx_dict[index_num] = 1

            # p1_states_idx.append(augstates.index(init_state))
            # p1_states_idx_dict[augstates.index(init_state)] = 1

        # init = (self.init, {self.init}, -1)
        # p1_states.append(init)

        keys_to_remove_attacks = set(self.actionAtt) - set(self.hidden_sensor_attacks)
        remaining_sensor_attacks = dict()
        remaining_sensor_attacks = {k: v for k, v in self.actionAtt.items() if k in keys_to_remove_attacks}

        keys_to_remove_query = set(self.sensor.query()) - set(self.hidden_sensor_queries)
        remaining_sensor_queries = dict()
        remaining_sensor_queries = {k: v for k, v in sensor_queries.items() if k in keys_to_remove_query}

        # if self.init not in asw:
        #     print("the initial state is not in the ASW region for P1.")
        #     return
        # augstates  = ['final', init]
        # p1_states_idx.append(augstates.index(init))

        # sink_idx = augstates.index('sink')
        # final_idx = augstates.index('final')

        final_idx = 0

        trans = dict([])
        suppDict = dict([])

        for a in self.actlist:
            for gaf in remaining_sensor_queries:
                suppDict[(final_idx, (a, gaf))] = {final_idx}

        count = 1
        player_flag = 1
        # pbar = tqdm(desc='Construction of P2s perceptual game', total=len(augstates))  #  Uncomment for progress bar
        while len(augstates) > count:
            state_info = augstates[count]
            if len(state_info) == 3:
                (s, B, kd) = state_info
                # s_idx = augstates.index((s, B, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), kd))
                player_flag = 1
            elif len(state_info) == 5:
                (s, B, a, ga, kd) = state_info
                # s_idx = augstates.index((s, B, a, ga, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), a, ga, kd))
                player_flag = 0
            else:
                (s, B, ga, kd) = state_info
                # s_idx = augstates.index((s, B, ga, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), ga, kd))
                player_flag = 2

            count = count + 1
            if player_flag == 1:
                # if len(B) != 0:
                act_list = list([])
                # for a in self.actlist:
                for a in self.get_allowed_combined_actions(B):
                    nextB = set(self.get_post(B, a))
                    for ga in remaining_sensor_queries:
                        trans[(s_idx, (a, ga))] = dict(
                            [])  # Have to consider the sensor query here... Is there a better way to do it?
                        suppDict[(s_idx, (a, ga))] = set([])
                        next_state = (s, nextB, a, ga, -1)
                        # if next_state not in augstates:
                        if (s, frozenset(nextB), a, ga, -1) not in augstates_membership_dict:
                            augstates.append(next_state)
                            # nature_states.append(next_state)
                            index_num = index_num + 1
                            # nature_states_idx.append(augstates.index(next_state))
                            # nature_states_idx_dict[augstates.index(next_state)] = 0
                            augstates_membership_dict[(s, frozenset(nextB), a, ga, -1)] = index_num
                            nature_states_idx.append(index_num)
                            nature_states_idx_dict[index_num] = 0

                        ns_idx = augstates_membership_dict.get((s, frozenset(nextB), a, ga, -1))

                        trans[(s_idx, (a, ga))][ns_idx] = 1
                        suppDict[(s_idx, (a, ga))].add(ns_idx)
                        act_list.append((a, ga))
                allowed_actions_list[s_idx] = act_list

                # else:
                #     for a in self.actlist:
                #         trans[(s_idx, a)] = {sink_idx: 1}

            elif player_flag == 0:
                trans[(s_idx, 'O')] = dict([])
                suppDict[(s_idx, 'O')] = set([])

                post_s = set(self.suppDict[(s, a)])
                if post_s.issubset(set(self.final)):
                    trans[(s_idx, 'O')][final_idx] = 1
                    suppDict[(s_idx, 'O')].add(final_idx)


                elif len(post_s.intersection(set(self.final))) == 0:

                    for ns in post_s:  # Rewrite CONSIDERING the winning states and add transitions to the winning state!!
                        next_state = (ns, B, ga, -1)
                        # if next_state not in augstates:
                        if (ns, frozenset(B), ga, -1) not in augstates_membership_dict:
                            augstates.append(next_state)
                            # p2_states.append(next_state)
                            index_num = index_num + 1
                            augstates_membership_dict[(ns, frozenset(B), ga, -1)] = index_num
                            p2_states_idx.append(index_num)
                            p2_states_idx_dict[index_num] = 2

                            # p2_states_idx.append(augstates.index(next_state))
                            # p2_states_idx_dict[augstates.index(next_state)] = 2
                        ns_idx = augstates_membership_dict.get((ns, frozenset(B), ga, -1))
                        trans[(s_idx, 'O')][ns_idx] = self.P(s, a, ns)
                        suppDict[(s_idx, 'O')].add(ns_idx)
                else:
                    trans[(s_idx, 'O')][final_idx] = 0.3
                    suppDict[(s_idx, 'O')].add(final_idx)

                    for ns in post_s:
                        next_state = (ns, B, ga, -1)
                        # if next_state not in augstates:
                        if (ns, frozenset(B), ga, -1) not in augstates_membership_dict:
                            augstates.append(next_state)
                            # p2_states.append(next_state)
                            index_num = index_num + 1
                            augstates_membership_dict[(ns, frozenset(B), ga, -1)] = index_num
                            p2_states_idx.append(index_num)
                            p2_states_idx_dict[index_num] = 2

                            # p2_states_idx.append(augstates.index(next_state))
                            # p2_states_idx_dict[augstates.index(next_state)] = 2
                        ns_idx = augstates_membership_dict.get((ns, frozenset(B), ga, -1))
                        trans[(s_idx, 'O')][ns_idx] = 0.7 * self.P(s, a, ns)
                        suppDict[(s_idx, 'O')].add(ns_idx)

            else:
                allowed_attack_actions = list([])
                for ga2 in remaining_sensor_attacks:
                    trans[(s_idx, ga2)] = dict([])
                    suppDict[(s_idx, ga2)] = set([])

                    next_obs = self.stateObs[(s, ga, ga2)]
                    nextB = B.intersection(next_obs)
                    next_state = (s, nextB, -1)
                    # if next_state not in augstates:
                    if (s, frozenset(nextB), -1) not in augstates_membership_dict:
                        augstates.append(next_state)
                        p1_states.append(next_state)

                        index_num = index_num + 1
                        augstates_membership_dict[(s, frozenset(nextB), -1)] = index_num
                        p1_states_idx.append(index_num)
                        p1_states_idx_dict[index_num] = 1

                        # p1_states_idx.append(augstates.index(next_state))
                        # p1_states_idx_dict[augstates.index(next_state)] = 1
                    ns_idx = augstates_membership_dict.get((s, frozenset(nextB), -1))
                    trans[(s_idx, ga2)][ns_idx] = 1
                    suppDict[(s_idx, ga2)].add(ns_idx)
                    allowed_attack_actions.append(ga2)
                allowed_attack_actions_list[s_idx] = allowed_attack_actions
            # pbar.update(len(augstates))   #  Uncomment for progress bar.

        # observation_equivalent_states = dict([])
        print(f"Done with the while loop and moving on to the observation equivalent states.")
        observation_equivalent_index = dict([])
        for (s, B, kd) in p1_states:

            # temp_state = list([])
            temp_idx = list([])
            # construction
            for x in B:

                # nstate = (x, B, kd)
                # if nstate in p1_states:
                if (x, frozenset(B), kd) in augstates_membership_dict:
                    # temp_state.append(nstate)
                    # temp_idx.append(augstates.index(nstate))

                    # temp_state.append(nstate)
                    temp_idx.append(augstates_membership_dict.get((x, frozenset(B), kd)))
                # else:
                #     print("Error - Obs equivalent state not in reachable states.")

            observation_equivalent_index[augstates_membership_dict.get((s, frozenset(B), kd))] = set(temp_idx)
            # observation_equivalent_states[(s, B, kd)] = set(temp_state)

        print(f"Done with the observation equivalent states.")
        del augstates_membership_dict

        augmdp.states = range(len(augstates))
        augmdp.sensor = copy.deepcopy(self.sensor)
        augmdp.hidden_sensor_queries = copy.deepcopy(self.hidden_sensor_queries)
        # augmdp.actionAtt = copy.deepcopy(remaining_sensor_attacks)
        augmdp.actionAtt = allowed_attack_actions_list
        # augmdp.final = ['final']
        augmdp.final_idx = [final_idx]
        N = len(augmdp.states)
        nature_actions = {'O'}
        player_1_actions = list(itertools.product(self.actlist, remaining_sensor_queries))
        allActions = set(player_1_actions).union(set(remaining_sensor_attacks)).union(nature_actions)
        # augmdp.actlist = copy.deepcopy(player_1_actions)  # Use this for single init state case.
        augmdp.actlist = allowed_actions_list
        augmdp.nature_act = nature_actions
        # augmdp.p1_states = p1_states
        augmdp.p1_states_idx = p1_states_idx
        augmdp.p1_states_idx_dict = p1_states_idx_dict
        # augmdp.nature_states = nature_states
        augmdp.nature_states_idx = nature_states_idx
        augmdp.nature_states_idx_dict = nature_states_idx_dict
        # augmdp.p2_states = p2_states
        augmdp.p2_states_idx = p2_states_idx
        augmdp.p2_states_idx_dict = p2_states_idx_dict
        augmdp.observation_equivalent_idx = observation_equivalent_index
        # augmdp.observation_equivalent_st  = observation_equivalent_states

        # prob = {a: np.zeros((N, N)) for a in allActions}
        # # prob =  {a: np.zeros((N, N)) for a in augmdp.actlist}
        # for (state, act) in trans.keys():
        #     tempdict = trans[(state, act)]
        #     for nextstate in tempdict.keys():
        #         prob[act][state, nextstate] = tempdict[nextstate]
        # augmdp.prob = prob

        # # Create a dictionary of sparse matrices using LIL format
        # prob_sparse = {a: lil_matrix((N, N)) for a in allActions}
        # # prob_sparse = {a: lil_matrix((N, N)) for a in augmdp.actlist}
        #
        # for (state, act) in trans.keys():
        #     tempdict = trans[(state, act)]
        #     for nextstate in tempdict.keys():
        #         prob_sparse[act][state, nextstate] = tempdict[nextstate]
        #
        # # Convert the sparse matrices to CSR format (Compressed Sparse Row)
        # for a in prob_sparse.keys():
        #     prob_sparse[a] = prob_sparse[a].tocsr()
        #
        # # Assign the sparse matrices to augmdp.prob
        # augmdp.prob = prob_sparse

        augmdp.suppDict = suppDict
        return augmdp, augstates, p1_states

    def get_aug_delayed_attack_game(self, aug_p1_states_percep, tau_r):
        """
        Augmented delayed attack game.
        aug_p1_states_percep: P1's states from the augmented P2's perceptual games.
        tau_r: the max time delay.
        :return: Augmented delayed attack game & augmented states.
        """
        augmdp = ActiveSensingGame()
        p1_states = list([])
        p1_states_idx = list([])
        p1_states_idx_dict = dict([])

        # nature_states = list([])
        nature_states_idx = list([])
        nature_states_idx_dict = dict([])

        # p2_states = list([])
        p2_states_idx = list([])
        p2_states_idx_dict = dict([])

        allowed_actions_list = dict([])
        allowed_attack_actions_list = dict([])

        augstates_membership_dict = dict([])

        sensor_queries = self.sensor.query()
        # init = (self.init, {self.init}, -1)
        # p1_states.append(init)

        augstates = ['final']
        index_num = 0

        # for (s, B, kd) in tqdm(aug_p1_states_percep): #  Uncomment for progress bar.
        for (s, B, kd) in aug_p1_states_percep:
            if (s not in self.final) or (s in self.final and len(B) != 1):
                # for a in self.actlist:
                for a in self.get_allowed_combined_actions(B):
                    for ga in self.hidden_sensor_queries:
                        post_states = self.get_post(B, a)  # See if you can cache this.
                        init_state = (s, post_states, a, ga, 0)
                        # if init_state not in augstates:
                        if (s, frozenset(post_states), a, ga,
                            0) not in augstates_membership_dict:
                            augstates.append(init_state)
                            index_num = index_num + 1
                            augstates_membership_dict[(s, frozenset(post_states), a, ga, 0)] = index_num

                            # nature_states.append(init_state)
                            # nature_states_idx.append(augstates.index(init_state))
                            # nature_states_idx_dict[augstates.index(init_state)] = 0
                            nature_states_idx.append(index_num)
                            nature_states_idx_dict[index_num] = 0

        keys_to_remove_attacks = set(self.actionAtt) - set(self.hidden_sensor_attacks)
        remaining_sensor_attacks = dict()
        remaining_sensor_attacks = {k: v for k, v in self.actionAtt.items() if k in keys_to_remove_attacks}

        # keys_to_remove_query = set(self.sensor.query()) - set(self.hidden_sensor_queries)
        # remaining_sensor_queries = dict()
        # remaining_sensor_queries = {k: v for k, v in sensor_queries.items() if k in keys_to_remove_query}

        # if self.init not in asw:
        #     print("the initial state is not in the ASW region for P1.")
        #     return
        # augstates = ['final', init]
        # p1_states_idx.append(augstates.index(init))

        # sink_idx = augstates.index('sink')

        # final_idx = augstates.index('final')
        final_idx = 0
        augstates_membership_dict["final"] = 0
        trans = dict([])
        suppDict = dict([])

        for a in self.actlist:
            for gaf in sensor_queries:
                suppDict[(final_idx, (a, gaf))] = {final_idx}

        count = 1
        player_flag = 1
        # pbar = tqdm(desc='Construction of delayed attack game', total=len(augstates))  #  Uncomment for progress bar.
        while len(augstates) > count:
            state_info = augstates[count]
            if len(state_info) == 3:
                (s, B, kd) = state_info
                # s_idx = augstates.index((s, B, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), kd))
                player_flag = 1
            elif len(state_info) == 5:
                (s, B, a, ga, kd) = state_info
                # s_idx = augstates.index((s, B, a, ga, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), a, ga, kd))
                player_flag = 0
            else:
                (s, B, ga, kd) = state_info
                # s_idx = augstates.index((s, B, ga, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), ga, kd))
                player_flag = 2

            count = count + 1
            if player_flag == 1:
                # if len(B) != 0:
                act_list = list([])
                # for a in self.actlist:
                for a in self.get_allowed_combined_actions(B):
                    nextB = set(self.get_post(B, a))
                    if kd < tau_r:
                        new_kd = kd + 1
                    else:
                        new_kd = kd

                    for ga in sensor_queries:
                        trans[(s_idx, (a, ga))] = dict([])
                        suppDict[(s_idx, (a, ga))] = set([])
                        next_state = (s, nextB, a, ga, new_kd)
                        # if next_state not in augstates: # This is the original line. Trying to change this to a dict check from list.
                        if (s, frozenset(nextB), a, ga, new_kd) not in augstates_membership_dict:
                            augstates.append(next_state)
                            index_num = index_num + 1
                            augstates_membership_dict[(s, frozenset(nextB), a, ga, new_kd)] = index_num
                            # nature_states.append(next_state)
                            # nature_states_idx.append(augstates.index(next_state))
                            # nature_states_idx_dict[augstates.index(next_state)] = 0
                            nature_states_idx.append(index_num)
                            nature_states_idx_dict[index_num] = 0

                        # ns_idx = augstates.index(next_state)
                        ns_idx = augstates_membership_dict.get((s, frozenset(nextB), a, ga, new_kd))

                        trans[(s_idx, (a, ga))][ns_idx] = 1
                        suppDict[(s_idx, (a, ga))].add(ns_idx)
                        act_list.append((a, ga))
                allowed_actions_list[s_idx] = act_list

                # else:
                #     for a in self.actlist:
                #         trans[(s_idx, a)] = {sink_idx: 1}

            elif player_flag == 0:
                trans[(s_idx, 'O')] = dict([])
                suppDict[(s_idx, 'O')] = set([])

                post_s = set(self.suppDict[(s, a)])
                if post_s.issubset(set(self.final)):
                    trans[(s_idx, 'O')][final_idx] = 1
                    suppDict[(s_idx, 'O')].add(final_idx)


                elif len(post_s.intersection(set(self.final))) == 0:

                    for ns in post_s:
                        next_state = (ns, B, ga, kd)
                        # if next_state not in augstates:
                        if (ns, frozenset(B), ga, kd) not in augstates_membership_dict:
                            augstates.append(next_state)
                            index_num = index_num + 1
                            augstates_membership_dict[(ns, frozenset(B), ga, kd)] = index_num
                            # p2_states.append(next_state)
                            # p2_states_idx.append(augstates.index(next_state))
                            # p2_states_idx_dict[augstates.index(next_state)] = 2

                            p2_states_idx.append(index_num)
                            p2_states_idx_dict[index_num] = 2

                        # ns_idx = augstates.index(next_state)
                        ns_idx = augstates_membership_dict.get((ns, frozenset(B), ga, kd))
                        trans[(s_idx, 'O')][ns_idx] = self.P(s, a, ns)
                        suppDict[(s_idx, 'O')].add(ns_idx)
                else:
                    trans[(s_idx, 'O')][final_idx] = 0.3
                    suppDict[(s_idx, 'O')].add(final_idx)

                    for ns in post_s:
                        next_state = (ns, B, ga, kd)
                        # if next_state not in augstates:
                        if (ns, frozenset(B), ga, kd) not in augstates_membership_dict:
                            augstates.append(next_state)
                            index_num = index_num + 1
                            augstates_membership_dict[(ns, frozenset(B), ga, kd)] = index_num
                            # p2_states.append(next_state)
                            #     p2_states_idx.append(augstates.index(next_state))
                            #     p2_states_idx_dict[augstates.index(next_state)] = 2
                            # ns_idx = augstates.index(next_state)

                            p2_states_idx.append(index_num)
                            p2_states_idx_dict[index_num] = 2
                        ns_idx = augstates_membership_dict.get((ns, frozenset(B), ga, kd))

                        trans[(s_idx, 'O')][ns_idx] = 0.7 * self.P(s, a, ns)
                        suppDict[(s_idx, 'O')].add(ns_idx)

            else:
                allowed_attack_actions = list([])
                if kd < tau_r:
                    for ga2 in remaining_sensor_attacks:
                        trans[(s_idx, ga2)] = dict([])
                        suppDict[(s_idx, ga2)] = set([])

                        next_obs = self.stateObs[(s, ga, ga2)]
                        nextB = B.intersection(next_obs)
                        next_state = (s, nextB, kd)
                        # if next_state not in augstates:
                        if (s, frozenset(nextB), kd) not in augstates_membership_dict:
                            augstates.append(next_state)
                            index_num = index_num + 1
                            augstates_membership_dict[(s, frozenset(nextB), kd)] = index_num
                            p1_states.append(next_state)
                            #     p1_states_idx.append(augstates.index(next_state))
                            #     p1_states_idx_dict[augstates.index(next_state)] = 1
                            # ns_idx = augstates.index(next_state)

                            p1_states_idx.append(index_num)
                            p1_states_idx_dict[index_num] = 1
                        ns_idx = augstates_membership_dict.get((s, frozenset(nextB), kd))

                        trans[(s_idx, ga2)][ns_idx] = 1
                        suppDict[(s_idx, ga2)].add(ns_idx)
                        allowed_attack_actions.append(ga2)
                    allowed_attack_actions_list[s_idx] = allowed_attack_actions
                else:
                    for ga2 in self.actionAtt:
                        trans[(s_idx, ga2)] = dict([])
                        suppDict[(s_idx, ga2)] = set([])

                        next_obs = self.stateObs[(s, ga, ga2)]
                        nextB = B.intersection(next_obs)
                        next_state = (s, nextB, kd)
                        # if next_state not in augstates:
                        if (s, frozenset(nextB), kd) not in augstates_membership_dict:
                            augstates.append(next_state)
                            index_num = index_num + 1
                            augstates_membership_dict[(s, frozenset(nextB), kd)] = index_num
                            p1_states.append(next_state)

                            # p1_states_idx.append(augstates.index(next_state))
                            # p1_states_idx_dict[augstates.index(next_state)] = 1

                            p1_states_idx.append(index_num)
                            p1_states_idx_dict[index_num] = 1

                        # ns_idx = augstates.index(next_state)
                        ns_idx = augstates_membership_dict.get((s, frozenset(nextB), kd))

                        trans[(s_idx, ga2)][ns_idx] = 1
                        suppDict[(s_idx, ga2)].add(ns_idx)
                        allowed_attack_actions.append(ga2)
                    allowed_attack_actions_list[s_idx] = allowed_attack_actions
            # pbar.update(len(augstates))  #  Uncomment for progress bar

        # observation_equivalent_states = dict([])
        print(f"Done with the while loop and moving on to the observation equivalent states.")
        observation_equivalent_index = dict([])
        for (s, B, kd) in p1_states:

            # temp_state = list([])
            temp_idx = list([])
            # construction
            for x in B:

                # nstate = (x, B, kd)
                # if nstate in p1_states:
                if (x, frozenset(B), kd) in augstates_membership_dict:
                    # temp_state.append(nstate)
                    # temp_idx.append(augstates.index(nstate))
                    temp_idx.append(augstates_membership_dict.get((x, frozenset(B), kd)))

                # else:
                #     print("Error - Obs equivalent state not in reachable states.")

            # observation_equivalent_index[augstates.index((s, B, kd))] = set(temp_idx)

            observation_equivalent_index[augstates_membership_dict.get((s, frozenset(B), kd))] = set(temp_idx)

        print(f"Done with the observation equivalent states.")
        # del augstates_membership_dict

        augmdp.states = range(len(augstates))
        augmdp.sensor = copy.deepcopy(self.sensor)
        augmdp.hidden_sensor_queries = copy.deepcopy(self.hidden_sensor_queries)
        # augmdp.actionAtt = copy.deepcopy(self.actionAtt)
        augmdp.actionAtt = allowed_attack_actions_list
        # augmdp.final = ['final']
        augmdp.final_idx = [final_idx]
        # N = len(augmdp.states)
        nature_actions = {'O'}
        player_1_actions = list(itertools.product(self.actlist, sensor_queries))
        allActions = set(player_1_actions).union(set(self.actionAtt)).union(nature_actions)
        # augmdp.actlist = copy.deepcopy(player_1_actions) # Use this for single init state case.
        augmdp.actlist = allowed_actions_list
        augmdp.nature_act = nature_actions
        augmdp.p1_states = p1_states
        augmdp.p1_states_idx = p1_states_idx
        augmdp.p1_states_idx_dict = p1_states_idx_dict
        # augmdp.nature_states = nature_states
        augmdp.nature_states_idx = nature_states_idx
        augmdp.nature_states_idx_dict = nature_states_idx_dict
        # augmdp.p2_states = p2_states
        augmdp.p2_states_idx = p2_states_idx
        augmdp.p2_states_idx_dict = p2_states_idx_dict
        augmdp.observation_equivalent_idx = observation_equivalent_index
        # augmdp.observation_equivalent_st  = observation_equivalent_states

        # prob = {a: np.zeros((N, N)) for a in allActions}
        # # prob =  {a: np.zeros((N, N)) for a in augmdp.actlist}
        # for (state, act) in trans.keys():
        #     tempdict = trans[(state, act)]
        #     for nextstate in tempdict.keys():
        #         prob[act][state, nextstate] = tempdict[nextstate]
        # augmdp.prob = prob

        augmdp.suppDict = suppDict
        return augmdp, augstates, p1_states, augstates_membership_dict

    def get_aug_initial_game(self, asw_states_delayed, asw_states_percep):
        """
        :return: augmented_mdp for Initial game.
        """
        augmdp = ActiveSensingGame()
        p1_states = list([])
        p1_states_idx = list([])
        p1_states_idx_dict = dict([])
        nature_states = list([])
        nature_states_idx = list([])
        nature_states_idx_dict = dict([])
        p2_states = list([])
        p2_states_idx = list([])
        p2_states_idx_dict = dict([])
        allowed_actions_list = dict([])
        allowed_attack_actions_list = dict([])

        augstates_membership_dict = dict([])

        sensor_queries = self.sensor.query()
        augstates = ['final']
        index_num = 0

        for init in self.init:
            init_state = (init, {init}, -1)
            augstates.append(init_state)
            p1_states.append(init_state)

            index_num = index_num + 1
            augstates_membership_dict[(init, frozenset({init}), -1)] = index_num

            p1_states_idx.append(index_num)
            p1_states_idx_dict[index_num] = 1

            # p1_states_idx.append(augstates.index(init_state))
            # p1_states_idx_dict[augstates.index(init_state)] = 1

        # init = (self.init, {self.init}, -1)
        # p1_states.append(init)


        keys_to_remove_attacks = set(self.actionAtt) - set(self.hidden_sensor_attacks)
        remaining_sensor_attacks = dict()
        remaining_sensor_attacks = {k: v for k, v in self.actionAtt.items() if k in keys_to_remove_attacks}

        keys_to_remove_query = set(self.sensor.query()) - set(self.hidden_sensor_queries)
        remaining_sensor_queries = dict()
        remaining_sensor_queries = {k: v for k, v in sensor_queries.items() if k in keys_to_remove_query}

        # if self.init not in asw:
        #     print("the initial state is not in the ASW region for P1.")
        #     return
        # augstates  = ['final', init]
        # p1_states_idx.append(augstates.index(init))

        # sink_idx = augstates.index('sink')
        # final_idx = augstates.index('final')

        final_idx = 0

        trans = dict([])
        suppDict = dict([])

        for a in self.actlist:
            for gaf in remaining_sensor_queries:
                suppDict[(final_idx, (a, gaf))] = {final_idx}
        count = 1
        player_flag = 1
        # pbar = tqdm(desc='Construction of initial game', total=len(augstates))  # Uncomment for progress bar.
        while len(augstates) > count:
            state_info = augstates[count]

            if len(state_info) == 3:
                (s, B, kd) = state_info
                # s_idx = augstates.index((s, B, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), kd))
                player_flag = 1
            elif len(state_info) == 5:
                (s, B, a, ga, kd) = state_info
                # s_idx = augstates.index((s, B, a, ga, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), a, ga, kd))
                player_flag = 0
            else:
                (s, B, ga, kd) = state_info
                # s_idx = augstates.index((s, B, ga, kd))
                s_idx = augstates_membership_dict.get((s, frozenset(B), ga, kd))
                player_flag = 2

            count = count + 1
            if player_flag == 1:
                # if (s, B, kd) in asw_states_percep: # Use this when the input asw states are in a list.
                if (s, frozenset(B), kd) in asw_states_percep:  # Use this when the input asw states are in a dict.
                    trans[(s_idx, ('M', 1))] = dict(
                        [])  # Have to consider the sensor query here... Is there a better way to do it?
                    suppDict[(s_idx, ('M', 1))] = set([])

                    trans[(s_idx, ('M', 1))][final_idx] = 1
                    suppDict[(s_idx, ('M', 1))].add(final_idx)
                    allowed_actions_list[s_idx] = [('M', 1)]
                else:
                    # if len(B) != 0:
                    act_list = list([])
                    # for a in self.actlist:
                    for a in self.get_allowed_combined_actions(B):
                        nextB = set(self.get_post(B, a))
                        for ga in sensor_queries:
                            if ga in remaining_sensor_queries:
                                trans[(s_idx, (a, ga))] = dict(
                                    [])  # Have to consider the sensor query here... Is there a better way to do it?
                                suppDict[(s_idx, (a, ga))] = set([])
                                next_state = (s, nextB, a, ga, -1)
                                # if next_state not in augstates:
                                if (s, frozenset(nextB), a, ga, -1) not in augstates_membership_dict:
                                    augstates.append(next_state)
                                    # nature_states.append(next_state)
                                    index_num = index_num + 1
                                    augstates_membership_dict[(s, frozenset(nextB), a, ga, -1)] = index_num
                                    nature_states_idx.append(index_num)
                                    nature_states_idx_dict[index_num] = 0

                                    # nature_states_idx.append(augstates.index(next_state))
                                    # nature_states_idx_dict[augstates.index(next_state)] = 0
                                ns_idx = augstates_membership_dict.get((s, frozenset(nextB), a, ga, -1))
                                trans[(s_idx, (a, ga))][ns_idx] = 1
                                suppDict[(s_idx, (a, ga))].add(ns_idx)
                                act_list.append((a, ga))
                            else:
                                trans[(s_idx, (a, ga))] = dict(
                                    [])  # Have to consider the sensor query here... Is there a better way to do it?
                                suppDict[(s_idx, (a, ga))] = set([])
                                next_state = (s, nextB, a, ga, 0)
                                # if next_state not in augstates:
                                if (s, frozenset(nextB), a, ga, 0) not in augstates_membership_dict:
                                    augstates.append(next_state)
                                    # nature_states.append(next_state)
                                    # nature_states_idx.append(augstates.index(next_state))
                                    # nature_states_idx_dict[augstates.index(next_state)] = 0

                                    index_num = index_num + 1
                                    augstates_membership_dict[(s, frozenset(nextB), a, ga, 0)] = index_num
                                    nature_states_idx.append(index_num)
                                    nature_states_idx_dict[index_num] = 0

                                ns_idx = augstates_membership_dict.get((s, frozenset(nextB), a, ga, 0))
                                trans[(s_idx, (a, ga))][ns_idx] = 1
                                suppDict[(s_idx, (a, ga))].add(ns_idx)
                                act_list.append((a, ga))
                    allowed_actions_list[s_idx] = act_list
                # else:
                #     for a in self.actlist:
                #         trans[(s_idx, a)] = {sink_idx: 1}

            elif player_flag == 0:
                # if (s, B, a, ga, kd) in asw_states_percep or (s, B, a, ga, kd) in asw_states_delayed:
                if (s, frozenset(B), a, ga, kd) in asw_states_percep or (
                        s, frozenset(B), a, ga, kd) in asw_states_delayed:
                    trans[(s_idx, 'O')] = dict([])
                    suppDict[(s_idx, 'O')] = set([])

                    trans[(s_idx, 'O')][final_idx] = 1
                    suppDict[(s_idx, 'O')].add(final_idx)

                else:
                    if kd == -1:

                        trans[(s_idx, 'O')] = dict([])
                        suppDict[(s_idx, 'O')] = set([])

                        post_s = set(self.suppDict[(s, a)])
                        if post_s.issubset(set(self.final)):
                            trans[(s_idx, 'O')][final_idx] = 1
                            suppDict[(s_idx, 'O')].add(final_idx)


                        elif len(post_s.intersection(set(self.final))) == 0:

                            for ns in post_s:  # Rewrite CONSIDERING the winning states and add transitions to the winning state!!
                                next_state = (ns, B, ga, -1)
                                # if next_state not in augstates:
                                if (ns, frozenset(B), ga, -1) not in augstates_membership_dict:
                                    augstates.append(next_state)
                                    # p2_states.append(next_state)
                                    index_num = index_num + 1
                                    augstates_membership_dict[(ns, frozenset(B), ga, -1)] = index_num
                                    p2_states_idx.append(index_num)
                                    p2_states_idx_dict[index_num] = 2

                                    # p2_states_idx.append(augstates.index(next_state))
                                    # p2_states_idx_dict[augstates.index(next_state)] = 2
                                # ns_idx = augstates.index(next_state)

                                ns_idx = augstates_membership_dict.get((ns, frozenset(B), ga, -1))
                                trans[(s_idx, 'O')][ns_idx] = self.P(s, a, ns)
                                suppDict[(s_idx, 'O')].add(ns_idx)
                        else:
                            trans[(s_idx, 'O')][final_idx] = 0.3
                            suppDict[(s_idx, 'O')].add(final_idx)

                            for ns in post_s:
                                next_state = (ns, B, ga, -1)
                                # if next_state not in augstates:
                                if (ns, frozenset(B), ga, -1) not in augstates_membership_dict:
                                    augstates.append(next_state)
                                    # p2_states.append(next_state)
                                    # p2_states_idx.append(augstates.index(next_state))
                                    # p2_states_idx_dict[augstates.index(next_state)] = 2

                                    index_num = index_num + 1
                                    augstates_membership_dict[(ns, frozenset(B), ga, -1)] = index_num
                                    p2_states_idx.append(index_num)
                                    p2_states_idx_dict[index_num] = 2

                                ns_idx = augstates_membership_dict.get((ns, frozenset(B), ga, -1))
                                trans[(s_idx, 'O')][ns_idx] = 0.7 * self.P(s, a, ns)
                                suppDict[(s_idx, 'O')].add(ns_idx)
                    else:
                        trans[(s_idx, 'O')] = dict([])
                        suppDict[(s_idx, 'O')] = set([])

                        trans[(s_idx, 'O')][s_idx] = 1
                        suppDict[(s_idx, 'O')].add(s_idx)

            else:
                allowed_attack_actions = list([])
                for ga2 in remaining_sensor_attacks:
                    trans[(s_idx, ga2)] = dict([])
                    suppDict[(s_idx, ga2)] = set([])

                    next_obs = self.stateObs[(s, ga, ga2)]
                    nextB = B.intersection(next_obs)
                    next_state = (s, nextB, -1)
                    # if next_state not in augstates:
                    if (s, frozenset(nextB), -1) not in augstates_membership_dict:
                        augstates.append(next_state)
                        p1_states.append(next_state)
                        # p1_states_idx.append(augstates.index(next_state))
                        # p1_states_idx_dict[augstates.index(next_state)] = 1
                        index_num = index_num + 1
                        augstates_membership_dict[(s, frozenset(nextB), -1)] = index_num
                        p1_states_idx.append(index_num)
                        p1_states_idx_dict[index_num] = 1

                    ns_idx = augstates_membership_dict.get((s, frozenset(B), -1))
                    trans[(s_idx, ga2)][ns_idx] = 1
                    suppDict[(s_idx, ga2)].add(ns_idx)
                    allowed_attack_actions.append(ga2)
                allowed_attack_actions_list[s_idx] = allowed_attack_actions
                # pbar.update(len(augstates))  #  Uncomment for progress bar.

        # observation_equivalent_states = dict([])
        print(f"Done with the while loop and starting with observation equivalent states.")
        observation_equivalent_index = dict([])
        for (s, B, kd) in p1_states:

            # temp_state = list([])
            temp_idx = list([])
            # construction
            for x in B:

                # nstate = (x, B, kd)
                # if nstate in p1_states:
                if (x, frozenset(B), kd) in augstates_membership_dict:
                    #     temp_state.append(nstate)
                    # temp_idx.append(augstates.index(nstate))
                    temp_idx.append(augstates_membership_dict.get((x, frozenset(B), kd)))

                # else:
                #     print("Error - Obs equivalent state not in reachable states.")

            observation_equivalent_index[augstates_membership_dict.get((s, frozenset(B), kd))] = set(temp_idx)
            # observation_equivalent_states[(s, B, kd)] = set(temp_state)

        print(f"Done with the observation equivalent states.")
        del augstates_membership_dict

        augmdp.states = range(len(augstates))
        augmdp.sensor = copy.deepcopy(self.sensor)
        augmdp.hidden_sensor_queries = copy.deepcopy(self.hidden_sensor_queries)
        # augmdp.actionAtt = copy.deepcopy(remaining_sensor_attacks)
        augmdp.actionAtt = allowed_attack_actions_list
        # augmdp.final = ['final']
        augmdp.final_idx = [final_idx]
        N = len(augmdp.states)
        nature_actions = {'O'}
        player_1_actions = list(itertools.product(self.actlist, remaining_sensor_queries))
        allActions = set(player_1_actions).union(set(remaining_sensor_attacks)).union(nature_actions)
        # augmdp.actlist = copy.deepcopy(player_1_actions)  # Use this for single init state case.
        augmdp.actlist = allowed_actions_list
        augmdp.nature_act = nature_actions
        # augmdp.p1_states = p1_states
        augmdp.p1_states_idx = p1_states_idx
        augmdp.p1_states_idx_dict = p1_states_idx_dict
        # augmdp.nature_states = nature_states
        augmdp.nature_states_idx = nature_states_idx
        augmdp.nature_states_idx_dict = nature_states_idx_dict
        # augmdp.p2_states = p2_states
        augmdp.p2_states_idx = p2_states_idx
        augmdp.p2_states_idx_dict = p2_states_idx_dict
        augmdp.observation_equivalent_idx = observation_equivalent_index
        # augmdp.observation_equivalent_st  = observation_equivalent_states

        # prob = {a: np.zeros((N, N)) for a in allActions}
        # # prob =  {a: np.zeros((N, N)) for a in augmdp.actlist}
        # for (state, act) in trans.keys():
        #     tempdict = trans[(state, act)]
        #     for nextstate in tempdict.keys():
        #         prob[act][state, nextstate] = tempdict[nextstate]
        # augmdp.prob = prob

        # # Create a dictionary of sparse matrices using LIL format
        # prob_sparse = {a: lil_matrix((N, N)) for a in allActions}
        # # prob_sparse = {a: lil_matrix((N, N)) for a in augmdp.actlist}
        #
        # for (state, act) in trans.keys():
        #     tempdict = trans[(state, act)]
        #     for nextstate in tempdict.keys():
        #         prob_sparse[act][state, nextstate] = tempdict[nextstate]
        #
        # # Convert the sparse matrices to CSR format (Compressed Sparse Row)
        # for a in prob_sparse.keys():
        #     prob_sparse[a] = prob_sparse[a].tocsr()
        #
        # # Assign the sparse matrices to augmdp.prob
        # augmdp.prob = prob_sparse

        augmdp.suppDict = suppDict
        return augmdp, augstates, p1_states


class AttackAwareP1SolverHiddenSensorSecondCB:
    def __init__(self, game, final):
        self.game = game
        self.final = set(final)

        #  self.losing = losing
        self.win = list()
        self.attractor = list()

    # # Below is the working version of the allowed actions function.
    # def allow(self, q, y):
    #     allowed_actions = set()
    #     # for act in self.game.actlist:  # Use this for a single init state case.
    #     for act in self.game.actlist[q]:
    #         if len(self.game.suppDict[q, act]) != 0:
    #             if self.game.suppDict[q, act].issubset(y):
    #                 allowed_actions.add(act)
    #
    #
    #     return allowed_actions

    # Below is the optimized version of the above allowed actions function.
    def allow(self, q, y):
        allowed_actions = set()

        for act in self.game.actlist[q]:
            supp_states = self.game.suppDict[q, act]
            if supp_states and supp_states.intersection(y) == supp_states:
                allowed_actions.add(act)

        return allowed_actions

    def allow_equivalent(self, q, y):
        allowed_actions = set()
        m = 1
        obs_eqv_states = self.game.observation_equivalent_idx[q]
        for q_eqv in obs_eqv_states:
            allowed = self.allow(q_eqv, y)
            if m == 1:
                allowed_actions = allowed
                m = m + 1
            else:
                # allowed_actions = allowed_actions.intersection(allowed)
                allowed_actions.intersection_update(allowed)

        return allowed_actions

    def progressive_transition_from_1(self, set_r, set_y):
        # # Below is the working solver code.
        # progressive_transition_states_1 = set()
        # print("Obtaining progressive transition states 1")
        # for q in tqdm(set(self.game.p1_states_idx).intersection(set_y)):
        #     allowed_action = self.allow_equivalent(q, set_y)
        #     if len(allowed_action) != 0:
        #         for m in allowed_action:
        #             post_states = self.game.suppDict[q, m]
        #             if len(post_states.intersection(set_r)) != 0:
        #                 progressive_transition_states_1.add(q)
        #                 break

        # Below is the attempt at the optimized code of the above version.
        progressive_transition_states_1 = set()
        p1_states_set = set(self.game.p1_states_idx)
        set_y_indx = set_y.intersection(p1_states_set)

        print("Obtaining progressive transition states 1")
        for q in tqdm(set_y_indx):
            allowed_action = self.allow_equivalent(q, set_y)
            if allowed_action:
                for m in allowed_action:
                    post_states = self.game.suppDict[q, m]
                    if post_states.intersection(set_r):
                        progressive_transition_states_1.add(q)
                        break  # Break out of the inner loop once a suitable result is found

        return progressive_transition_states_1

    def progressive_transition_from_2(self, set_r, set_y):
        progressive_transition_states_2 = set()
        all_beta_flag = 0

        p2_states_set = set(self.game.p2_states_idx)
        set_y = set_y.intersection(p2_states_set)

        print("Obtaining progressive transition states 2")
        # for q in tqdm(set(self.game.p2_states_idx).intersection(set_y)):
        for q in tqdm(set_y):
            # for beta in self.game.actionAtt:
            for beta in self.game.actionAtt[q]:
                post_p2_states = set(self.game.suppDict[q, beta])
                if post_p2_states.issubset(set_r):
                    continue
                else:
                    all_beta_flag = 1

            if all_beta_flag == 0:
                progressive_transition_states_2.add(q)
            else:
                all_beta_flag = 0

        return progressive_transition_states_2

    def progressive_transition_from_n(self, set_r, set_y):
        progressive_transition_states_n = set()
        print("Obtaining progressive transition states 0")

        nature_states_idx_set = set(self.game.nature_states_idx)
        set_y_indx = set_y.intersection(nature_states_idx_set)

        # for q in tqdm(set(self.game.nature_states_idx).intersection(set_y)):
        for q in tqdm(set_y_indx):
            # Check if the post_n_states is correct.
            post_n_states = self.game.suppDict[q, 'O']
            if post_n_states.issubset(set_y) and (len(post_n_states.intersection(set_r)) != 0):
                progressive_transition_states_n.add(q)

        return progressive_transition_states_n

    def safe_pos_reach(self, level_y):
        """
        Implementation of Alg. 3.
        :param
        """
        level_r = list()

        final_states_qf = copy.deepcopy(self.final)
        level_r.append(set(final_states_qf))
        while True:
            top_level_r = level_r[-1]
            new_level_r = set()

            p1_states = self.progressive_transition_from_1(top_level_r, level_y)
            p2_states = self.progressive_transition_from_2(top_level_r, level_y)
            pn_states = self.progressive_transition_from_n(top_level_r, level_y)

            new_level_r = set.union(top_level_r, p1_states, p2_states, pn_states)

            print("Executing SafePosReach.")

            if top_level_r == new_level_r:
                break
            else:
                level_r.append(new_level_r)

        return new_level_r

    def solve(self):
        """
        Implementation of Alg. 2.
        :return: ASW states, strategy_1
        """
        level_y = list()

        positive_winning = PositiveWReach(self.game, final=self.final)
        positive_winning.solve()
        unreachable = positive_winning.positive_win
        perfect_win_p1 = positive_winning.asw1
        level_y.append(set(self.game.states) - set(unreachable))
        # strategy_1 = {u: {(a, s) for _, a, s in self.game.post_sig(u)} for u in
        #               self.q1}
        # strategy_p1 = {u: {(a, s) for _, a, s in self.game.post_sig(u)} for u in
        #                self.q1}
        while True:
            top_level_y = level_y[-1]
            new_level_y = self.safe_pos_reach(top_level_y)

            print("Solving the two-player stochastic game.")

            if top_level_y == new_level_y:
                break
            else:
                level_y.append(new_level_y)

        strategy_1 = self.strategy_construction(new_level_y)
        state_in_perfect_not_in_asw = perfect_win_p1 - new_level_y
        unreachable_for_p2 = list(set(self.game.states) - set(new_level_y))

        return new_level_y, strategy_1, unreachable_for_p2, perfect_win_p1, state_in_perfect_not_in_asw

    def strategy_construction(self, level_y):
        # strategy_p1 = {u: {(a, s) for _, a, s in self.game.post_sig(u)} for u in
        #                self.q1}

        pol = dict([])
        for q in level_y:
            # if q in self.game.p1_states_idx:
            if q in self.game.p1_states_idx_dict:
                allowed_actions = self.allow_equivalent(q, level_y)
                pol[q] = allowed_actions
            else:
                pol[q] = set([])

        #
        # for q in self.q1:
        #     allowed_actions = self.allow_equivalent(q, level_y)
        #     strategy_p1[q] = strategy_p1[q].intersection(allowed_actions)

        return pol

    def verify(self):
        return print("Verification code not yet written")

    def get_win_initials(num_user, asw, augstates):
        win_init = list([])
        win_init_idx = list([])
        for k in asw:
            state = augstates[k]
            flag = True
            if len(state) != 3 or len(state[1]) != 1:
                flag = False

            if flag == True:  # for all users, the initial belief is the same as the current state:
                if state not in win_init:
                    win_init.append(state)
                    win_init_idx.append(k)
        return win_init
