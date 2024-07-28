__author__ = 'Sumukha Udupa, sudupa@ufl.edu'

import copy
from tqdm import tqdm

class PositiveWReach:
    def __init__(self, game, final):
        # Game parameters
        self.game = game
        self.final = final
        #  self.losing = losing  # losing states for Player1 or winning states for Player2

        # Solution parameters
        self.attractor = list()
        self.positive_win = set()
        self.asw1 = set()

    def pre_x_y(self, x_0, y):
        """
        Obtains the pre of the states in the stochastic game.
        """
        pre_set = set()

        y_diff = y - x_0

        # for q in tqdm(y):
        for q in tqdm(y_diff):
            # for player 1 state
#

            if q in self.game.p1_states_idx_dict:  # Use this when the input is a dictionary of P1 state indices.
                # find the post states of q
                post_states = set([])
                # for act in self.game.actlist:
                for act in self.game.actlist[q]:
                    temp = self.game.suppDict[(q, act)]
                    post_states = post_states.union(temp)

                if not (not post_states):
                    if len(post_states.intersection(x_0)) > 0:
                        pre_set.add(q)


            # for nature's state
            # elif q in self.game.nature_states_idx:
            elif q in self.game.nature_states_idx_dict:
                # find the post states of q
                post_states = self.game.suppDict[(q, 'O')]

                if not (not post_states):

                    if len(post_states.intersection(x_0)) > 0:
                        #  union_states = post_states.intersection(y).union(post_states.intersection(x_0))
                        if post_states.issubset(y):
                            pre_set.add(q)


            # for player 2's state
            # elif q in self.game.p2_states_idx:
            elif q in self.game.p2_states_idx_dict:

                # find the post states of q
                post_states = set([])
                # for act in self.game.actionAtt:
                for act in self.game.actionAtt[q]:
                    temp = self.game.suppDict[(q, act)]
                    post_states = post_states.union(temp)

                if not (not post_states):
                    if post_states.issubset(x_0):
                        pre_set.add(q)

            # # for the final state
            # elif q in self.game.final_idx:
            #     pre_set.add(q)
        return pre_set

    def win_1(self):
        # level_set_x and level_set_y are going to be the final and safety lists

        level_set_x = list()
        # safety set starting with all states other than the final states
        level_set_y = list()
        final_states = copy.deepcopy(self.final)
        level_set_x.append(set(final_states))
        level_set_y.append(set(self.game.states))  # .difference(set(final_states)))
        print("Starting the solver")
        i = 1
        j = 1

        while True:
            top_level_y = level_set_y[-1]
            new_level_y = set()
            while True:
                top_level_x = level_set_x[-1]
                new_level_x = set()
                pre_states = self.pre_x_y(top_level_x, top_level_y)

                new_level_x = top_level_x.union(pre_states)

                level_set_x.append(new_level_x)

                if top_level_x == new_level_x:
                    if i == 1:
                        print("i =", i)
                    elif i == 1000:
                        print("i =", i)
                    elif i == 100000:
                        print("i =", i)

                    i += 1
                    break

            new_level_y = new_level_x
            level_set_y.append(new_level_y)
            level_set_x = list()
            level_set_x.append(set(final_states))
            if top_level_y == new_level_y:
                if j == 1:
                    print("j =", j)
                elif j == 1000:
                    print("j =", j)
                elif j == 100000:
                    print("j =", j)
                break

        return new_level_y

    def solve(self):
        self.asw1 = self.win_1()

        self.positive_win = set(self.game.states) - set(self.asw1)
