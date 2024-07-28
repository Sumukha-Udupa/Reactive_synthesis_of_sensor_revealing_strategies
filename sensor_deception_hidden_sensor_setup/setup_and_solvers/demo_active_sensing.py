__author__ = 'Sumukha Udupa, sudupa@ufl.edu'


import os, sys, getopt, pdb, string

import time

import pygame
# import pygame.locals as pgl

# import random as pr
from random import choice
import random
import matplotlib.pyplot as plt
from loguru import logger


def max_bel_att(state, augmented_game, augmented_game_states, sense_query_set):
    belief = 0
    possible_attacks = sense_query_set.intersection(set(augmented_game.actionAtt[state]))
    for a in possible_attacks:
        nxt_st_id = list(augmented_game.suppDict[(state, a)])[0]
        nxt_st = augmented_game_states[nxt_st_id]

        bel_size = len(nxt_st[1])
        if bel_size > belief:
            attack = a
            belief = bel_size

    return attack


def follow(gwg, augmdp_percep, augmdp_delayed, augmdp_initial, state_id, percep_limit, delayed_limit,
           combined_augstates, augstates_percep, augstates_delayed, augstates_initial, policy, final_state,
           query_actions, quite):
    state = combined_augstates[state_id]
    print(f"The current state is: {state}")
    logger.debug(f"The current state is: {state}")

    gwg.current = state[0]
    action = policy[state_id]

    if action is not None:
        action = random.choice(list(action))
        print(f"The action taken: {action}")
        logger.debug(f"The action taken: {action}")
        sense_query_set = query_actions[action[1]]

    # Obtaining the next nature's state, P2's state and next P1 state.
    if action is not None:

        if state_id <= percep_limit:
            state_percep_id = augstates_percep.index(state)
            nature_state_id = list(augmdp_percep.suppDict[(state_percep_id, action)])[0]

            p2_state_id = augmdp_percep.suppDict[(nature_state_id, 'O')]
            p2_state_id = random.choice(list(p2_state_id))

            if p2_state_id != 0:
                # attack = random.choice(list(augmdp_percep.actionAtt[p2_state_id]))
                attack = max_bel_att(p2_state_id, augmdp_percep, augstates_percep, sense_query_set)
                print(f"The attack chosen: {attack}")
                logger.debug(f"The attack chosen: {attack}")
                next_state_percep_id = list(augmdp_percep.suppDict[(p2_state_id, attack)])[0]
                next_state = augstates_percep[next_state_percep_id]
                print(f"The next P1 state reached: {next_state}")
                logger.debug(f"The next P1 state reached: {next_state}")

            else:
                next_state = augstates_percep[p2_state_id]
                print(f"The nature state reached before the next state: {augstates_percep[nature_state_id]}")
                print(f"The next P1 state reached: {next_state}")
                logger.debug(f"The next P1 state reached: {next_state}")

            # for a in augmdp_percep.actionAtt[p2_state_id]:


        elif state_id > percep_limit and state_id <= delayed_limit:
            state_delayed_id = augstates_delayed.index(state)
            nature_state_id = list(augmdp_delayed.suppDict[(state_delayed_id, action)])[0]

            p2_state_id = augmdp_delayed.suppDict[(nature_state_id, 'O')]
            p2_state_id = random.choice(list(p2_state_id))

            if p2_state_id != 0:
                # attack = random.choice(list(augmdp_delayed.actionAtt[p2_state_id]))
                attack = max_bel_att(p2_state_id, augmdp_delayed, augstates_delayed, sense_query_set)
                print(f"The attack chosen: {attack}")
                logger.debug(f"The attack chosen: {attack}")
                next_state_delayed_id = list(augmdp_delayed.suppDict[(p2_state_id, attack)])[0]
                next_state = augstates_delayed[next_state_delayed_id]
                print(f"The next P1 state reached: {next_state}")
                logger.debug(f"The next P1 state reached: {next_state}")


            else:
                next_state = augstates_delayed[p2_state_id]
                print(f"The nature state reached before the next state: {augstates_delayed[nature_state_id]}")
                print(f"The next P1 state reached: {next_state}")
                logger.debug(f"The next P1 state reached: {next_state}")



        else:
            if action[1] in augmdp_percep.hidden_sensor_queries:
                hidden_action_flag = True
            else:
                hidden_action_flag = False

            state_initial_id = augstates_initial.index(state)
            nature_state_id = list(augmdp_initial.suppDict[(state_initial_id, action)])[0]

            if hidden_action_flag:
                nature_state = augstates_initial[nature_state_id]
                nature_state_id = augstates_delayed.index(nature_state)
                p2_state_id = augmdp_delayed.suppDict[(nature_state_id, 'O')]
                p2_state_id = random.choice(list(p2_state_id))
            else:
                p2_state_id = augmdp_initial.suppDict[(nature_state_id, 'O')]
                p2_state_id = random.choice(list(p2_state_id))

            if p2_state_id != 0:
                if hidden_action_flag:
                    # attack = random.choice(list(augmdp_delayed.actionAtt[p2_state_id]))
                    attack = max_bel_att(p2_state_id, augmdp_delayed, augstates_delayed, sense_query_set)
                    print(f"The attack chosen: {attack}")
                    logger.debug(f"The attack chosen: {attack}")
                    next_state_initial_id = list(augmdp_delayed.suppDict[(p2_state_id, attack)])[0]
                    next_state = augstates_delayed[next_state_initial_id]
                    print(f"The next P1 state reached: {next_state}")
                    logger.debug(f"The next P1 state reached: {next_state}")

                else:
                    # attack = random.choice(list(augmdp_initial.actionAtt[p2_state_id]))
                    attack = max_bel_att(p2_state_id, augmdp_initial, augstates_initial, sense_query_set)
                    next_state_initial_id = list(augmdp_initial.suppDict[(p2_state_id, attack)])[0]
                    next_state = augstates_initial[next_state_initial_id]
                    print(f"The next P1 state reached: {next_state}")
                    logger.debug(f"The next P1 state reached: {next_state}")


            else:
                next_state = augstates_initial[p2_state_id]
                print(f"The nature state reached before the next state: {augstates_initial[nature_state_id]}")
                print(f"The next P1 state reached: {next_state}")
                logger.debug(f"The next P1 state reached: {next_state}")


    else:
        next_state = state

    if next_state != 'final':
        gwg.current = next_state[0]
        final_flag = False
    else:
        gwg.current = final_state[0]
        next_state = final_state
        final_flag = True

    if gwg.updategui:
        gwg.state2circle(gwg.current)  # Check what is happening here and if it is working properly.

    if not quite:
        time.sleep(0.05)
    # if type(action) == set:
    #     action = random.choice(list(action))
    #     new_state_id = augmdp.sample(state_id, action)
    #     gwg.current = augstates[new_state_id][0][0]
    # else:
    #     new_state_id = augmdp.sample(state_id, action)
    #     gwg.current = augstates[new_state_id][0][0]
    # if action == None:  # reach the final state.
    #     new_state_id = state_id
    # if gwg.updategui:
    #     gwg.state2circle(gwg.current)
    # if not quite:
    #     time.sleep(0.05)
    return next_state, final_flag


def draw_defender_belief_single(gwg, belief):
    gwg.draw_region(belief)
    return


# def run_policy(gwg, augmdp, augstates, policy, targets):
#     gwg.screen.blit(gwg.surface, (0, 0))
#     pygame.display.flip()
#     state_id = augmdp.init
#     while True:
#         new_state_id = follow(gwg, augmdp, state_id, augstates, policy)
#         gwg.current = augstates[new_state_id][0][0]
#         state_id = new_state_id
#         if gwg.current in gwg.obstacles:
#             # hitting the obstacles
#             print("Hitting the obstacles, restarting ...")
#             # raw_input('Press Enter to restart ...')
#             state_id = augmdp.init  # restart the game
#         belief = set([])
#         for b in augstates[state_id][1][0]:
#             belief.add(b[0])
#         print("the current state is {}".format(gwg.current))
#         print("the defender's belief is {}".format(belief))
#         gwg.draw_state_region(gwg.current,belief)
#         time.sleep(0.05)
#         gwg.redraw()
#         if state_id in targets:
#             input('Press Enter to continue ...')
#         gwg.screen.blit(gwg.surface, (0, 0))
#         pygame.display.flip()

def run_policy_plot(gwg, initial_state, augmdp_percep, augmdp_delayed, augmdp_initial, augstates, augstates_percep,
                    augstates_delayed, augstates_initial, policy, targets, percep_limit, delayed_limit, final_idx,
                    query_actions, quite=False):
    # the following code works only for single user case.
    gwg.screen.blit(gwg.surface, (0, 0))
    pygame.display.flip()
    state_id = augstates.index(initial_state)
    belief_history = []
    state_in_belief = []

    while True:
        new_state, final_flag = follow(gwg, augmdp_percep, augmdp_delayed, augmdp_initial, state_id, percep_limit,
                                       delayed_limit,
                                       augstates, augstates_percep, augstates_delayed, augstates_initial, policy,
                                       targets, query_actions,
                                       quite)
        gwg.current = new_state[0]

        if final_flag:
            state_id = 0
        else:
            state_id = augstates.index(new_state)
        # if gwg.current in gwg.obstacles:
        #     # hitting the obstacles
        #     print("Hitting the obstacles, restarting ...")
        #     # raw_input('Press Enter to restart ...')
        #     state_id = augmdp.init  # restart the game

        if final_flag is False:
            belief = new_state[1]
            belief_history.append(belief)
        else:
            belief = {gwg.current}
            belief_history.append(belief)

        gwg.draw_state_region(gwg.current, belief)
        gwg.redraw()
        if state_id == final_idx:
            print("reached the target.")
            logger.debug("Reached the target.")
            gwg.screen.blit(gwg.surface, (0, 0))
            pygame.display.flip()
            break
        gwg.screen.blit(gwg.surface, (0, 0))
        pygame.display.flip()
    return belief_history


def plot_beliefs(belief_history):
    # Calculate the length of each belief in the history
    size_belief = [len(b) for b in belief_history]

    # Create a time step range based on the number of beliefs
    time_steps = range(len(belief_history))

    # Plot the size of beliefs against time steps
    plt.plot(time_steps, size_belief, 'b-')

    # Labeling
    plt.xlabel('Time Step')
    plt.ylabel('Size of Belief')

    # Display the plot
    plt.show()
