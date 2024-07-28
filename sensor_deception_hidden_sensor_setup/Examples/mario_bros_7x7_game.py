__author__ = 'Sumukha Udupa, sudupa@ufl.edu'

# import demo_active_sensing
# from gridworld import *
# from delivery_game import *
# from sensor import *
# from stochatic_two_player_game_solver import *

from sensor_deception_hidden_sensor_setup.setup_and_solvers.demo_active_sensing import *
from sensor_deception_hidden_sensor_setup.setup_and_solvers.gridworld import *
from sensor_deception_hidden_sensor_setup.setup_and_solvers.delivery_game import *
from sensor_deception_hidden_sensor_setup.setup_and_solvers.sensor import *
from sensor_deception_hidden_sensor_setup.setup_and_solvers.stochatic_two_player_game_solver import *

from miscellaneous.policy import *
from loguru import logger

import pickle

import time

logger.remove()
logger.add("../../log_files/log_of_mario_game_exp_40_6X6_s2_s4_hidden_tau_varying_for_time_try_1.log")

# # The following is the setup for the exp  with fewer obstacles example-25 (7X7).
# target = [6]
# obstacles = [0, 11, 17, 16, 26, 39, 42, 45, 46, 48, 35]
# unsafe_u = [4, 10, 33, 27]
# non_init_states = [0, 6, 11, 17, 16, 26, 39, 42, 45, 46, 48, 4, 10, 33, 27, 35]
# initial = 36
# ncols = 7
# nrows = 7

# The following is the setup for the exp with fewer obstacles that is in the paper. (6X6)
# The following is the setup for the exp  with fewer obstacles example-25 (6X6).
target = [5]
obstacles = [9, 13, 14, 22, 33]
unsafe_u = [3, 8, 23, 28]
non_init_states = [9, 13, 14, 22, 33, 3, 8, 23, 28]
initial = 25
ncols = 6
nrows = 6

robot_ts = read_from_file_MDP_old('robotmdp.txt')

###test simple, deterministic sensor ex_sensor_deter.txt
sensor_ts = read_from_file_MDP_old('ex_sensor_deter.txt')

# The below are for experiment 15. (5X5)
query_actions = dict([])
query_actions[0] = {0, 1}  # Hidden
query_actions[1] = {0, 2}
query_actions[2] = {3, 2}  # Hidden
# query_actions[3] = {1, 3}
query_actions[4] = {0, 1, 3}  # Hidden

attack_actions = dict([])
attack_actions[0] = {0}
attack_actions[1] = {1}  # Hidden
attack_actions[2] = {2}
attack_actions[3] = {3}  # Hidden
# attack_actions[4] = {4}

hidden_sensor_queries = dict([])
hidden_sensor_queries[0] = {0, 1}
hidden_sensor_queries[2] = {2, 3}
hidden_sensor_queries[4] = {0, 1, 3}
# hidden_sensor_queries[3] = {1, 3}


hidden_sensor_attacks = dict([])
hidden_sensor_attacks[1] = {1}
hidden_sensor_attacks[3] = {3}

# setting the precise sensor if there is any in the environment.
precise_sensors = []

# Log the sensor queries and attacks.

sensor = sensor()
sensor.states = sensor_ts.states
sensor.actlist = sensor_ts.actlist
sensor.prob = sensor_ts.prob
sensor.init = 1  # this is specific to the example.
sensor.get_supp()
sensor.query_actions = query_actions
# sensor.set_coverage(1, set([]))
# sensor.set_coverage(1, set([]))
# this is to randomize between two sensors.

gwg_user1 = GridworldGui(initial, nrows, ncols, robot_ts, target, obstacles, unsafe_u)
gwg_user1.mdp.get_supp()
gwg_user1.draw_state_labels()

complete_states = gwg_user1.mdp.states

# Coverage for 6X6
set0 = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
set1 = {6, 7, 8, 9, 10, 11, 3, 15, 21, 27, 33}
set2 = {0, 1, 2, 3, 4, 5, 6, 12, 18, 24, 30}
set3 = {18, 19, 20, 21, 22, 23}

# Coverage for 7X7
# set0 = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
# set1 = {28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41}
# set2 = {7, 8, 9, 10, 11, 12, 13, 4, 18, 25, 32, 39, 46}
# set3 = {0, 1, 2, 3, 4, 5, 6, 8, 15, 22, 29, 36, 43}
# set4 = {21, 22, 23, 24, 25, 26, 27}

sensor.set_coverage(0, set0)
sensor.set_coverage(1, set1)
sensor.set_coverage(2, set2)
sensor.set_coverage(3, set3)
# sensor.set_coverage(4, set4)

tic = time.perf_counter()
deliveryGame = DeliveryGame()
deliveryGame.agentmdp = gwg_user1.mdp
deliveryGame.sensor = sensor
deliveryGame.actionAtt = attack_actions
deliveryGame.final = target
deliveryGame.hidden_sensor_queries = hidden_sensor_queries
deliveryGame.hidden_sensor_attacks = hidden_sensor_attacks
deliveryGame.non_init_states = non_init_states
deliveryGame.unsafe_obstacles = unsafe_u
deliveryGame.precise_sensors = precise_sensors
deliveryGame.get_game()
deliveryGame.get_stateObs()
deliveryGame.get_allowed_actions()
# toc = time.perf_counter()
# logger.debug(f"Constructed the delivery game in {toc - tic:0.4f} seconds")
# print(f"Constructed the delivery game in {toc - tic:0.4f} seconds")

# Defining variables for the combined winning region.
combined_winning_P1_states = list([])
combined_winning_P1_state_pol = dict([])
combined_winning_states = list([])

# Constructing P2's perceptual game
# tic = time.perf_counter()
augmented_perceptual_game, augmented_percep_games_states, aug_p1_states_percep = deliveryGame.get_aug_sensing_game_perceptual()
# toc = time.perf_counter()
# logger.debug(f"Constructed the P2's perceptual game in {toc - tic:0.4f} seconds")
# print(f"Constructed the P2's perceptual game in {toc - tic:0.4f} seconds")

# Solving the P2's perceptual game
attack_aware_solver_P1_perceptual = active_sensing_and_sensing_attacks_game.AttackAwareP1SolverHiddenSensorSecondCB(
    augmented_perceptual_game, augmented_perceptual_game.final_idx)

# tic = time.perf_counter()
asw_states_percep, P1_strategy_in_percep, p1_unreachable_percep, perfect_win_P1_percep, states_in_perfect_not_in_asw = attack_aware_solver_P1_perceptual.solve()
# toc = time.perf_counter()
# logger.debug(f"Solved the policy for P2's perceptual game in {toc - tic:0.4f} seconds")
# print(f"Solved the policy in {toc - tic:0.4f} seconds")

for sindx in tqdm(asw_states_percep):
    state = augmented_percep_games_states[sindx]  # Change this list to a dictionary to make the for loop faster.
    if state not in combined_winning_states:
        combined_winning_states.append(augmented_percep_games_states[sindx])

    if sindx in augmented_perceptual_game.p1_states_idx_dict:
        if state not in combined_winning_P1_states:
            combined_winning_P1_states.append(state)
            P1_strategy = P1_strategy_in_percep[sindx]
            combined_winning_P1_state_pol[combined_winning_states.index(state)] = P1_strategy

percep_limit = len(combined_winning_states) - 1

# Constructing Delayed attack game
tau_r = 0  # The minimum delay before P2 can attack the hidden sensor.
p1_state_indices_in_asw = asw_states_percep.intersection(augmented_perceptual_game.p1_states_idx)
p1_states_in_asw = list([])
for index in tqdm(p1_state_indices_in_asw):
    aug_p1_states_percep.remove(augmented_percep_games_states[index])

# tic = time.perf_counter()
augmented_delayed_attack_game, augmented_delayed_att_game_states, p1_states_delayed_att, augmented_delayed_att_game_states_dict = deliveryGame.get_aug_delayed_attack_game(
    aug_p1_states_percep, tau_r)
# toc = time.perf_counter()
# logger.debug(f"Constructed the delayed attack game in {toc - tic:0.4f} seconds")
# print(f"Constructed the delayed attack game in {toc - tic:0.4f} seconds")

# Solving the Delayed attack game.
attack_aware_solver_P1_delayed_attack = active_sensing_and_sensing_attacks_game.AttackAwareP1SolverHiddenSensorSecondCB(
    augmented_delayed_attack_game, augmented_delayed_attack_game.final_idx)

# tic = time.perf_counter()
asw_states_delayed, P1_strategy_in_delayed, p1_unreachable_delayed, perfect_win_P1_delayed, states_in_perfect_not_in_asw_delayed = attack_aware_solver_P1_delayed_attack.solve()
# toc = time.perf_counter()
# logger.debug(f"Solved the policy for delivery game in {toc - tic:0.4f} seconds")
# print(f"Solved the policy in {toc - tic:0.4f} seconds")

print(f"Started reversing.")
reverse_augmented_delay_att_gm_st_dict = {value: key for key, value in augmented_delayed_att_game_states_dict.items()}
print(f"Finished reversing.")

asw_states_delayed_calc = set(
    asw_states_delayed)  # The following loop is taking too long. Optimize it or see if the loop can be avoided altogether????
# for sindx in tqdm(set(asw_states_delayed)):
for sindx in tqdm(asw_states_delayed_calc):
    # state = augmented_delayed_att_game_states[sindx]
    state_tmp = reverse_augmented_delay_att_gm_st_dict[sindx]
    if (sindx != 0) and len(state_tmp) == 3:
        state = (state_tmp[0], set(state_tmp[1]), state_tmp[2])
    elif (sindx != 0) and len(state_tmp) == 5:
        state = (state_tmp[0], set(state_tmp[1]), state_tmp[2], state_tmp[3], state_tmp[4])
    elif (sindx != 0) and len(state_tmp) == 4:
        state = (state_tmp[0], set(state_tmp[1]), state_tmp[2], state_tmp[3])
    else:
        state = state_tmp

    if state not in combined_winning_states: # Change combined winning states to a dict rather than a list
        combined_winning_states.append(augmented_delayed_att_game_states[sindx])

    if sindx in augmented_delayed_attack_game.p1_states_idx_dict:
        if state not in combined_winning_P1_states:  # Change the list here!! time consuming!!
            combined_winning_P1_states.append(state)
            P1_strategy = P1_strategy_in_delayed[sindx]
            combined_winning_P1_state_pol[combined_winning_states.index(state)] = P1_strategy

delayed_limit = len(combined_winning_states) - 1

# Constructing P1's initial game
# asw_states_delayed_actual = list([])  # Modify this.. we only need the winning nature's states.
# for indx in asw_states_delayed:
#     asw_states_delayed_actual.append(augmented_delayed_att_game_states[indx])

asw_states_delayed_actual = dict([])
for indx in tqdm(asw_states_delayed):
    # winning_delayed_state = augmented_delayed_att_game_states[
    #     indx]  # Change the augmented delayed att game states to dict as well.
    winning_delayed_state = reverse_augmented_delay_att_gm_st_dict[indx]
    if len(winning_delayed_state) == 5:
        (s, B, a, ga, kd) = winning_delayed_state
        asw_states_delayed_actual[(s, frozenset(B), a, ga, kd)] = 0

# asw_states_percep_actual = list([])
# for indx in asw_states_percep:
#     asw_states_percep_actual.append((augmented_percep_games_states[indx]))

asw_states_percep_actual = dict([])
for indx in tqdm(asw_states_percep):
    winning_percep_actual = augmented_percep_games_states[indx]
    if len(winning_percep_actual) == 3:
        (s, B, kd) = winning_percep_actual
        asw_states_percep_actual[(s, frozenset(B), kd)] = 1
    elif len(winning_percep_actual) == 5:
        (s, B, a, ga, kd) = winning_percep_actual
        asw_states_percep_actual[(s, frozenset(B), a, ga, kd)] = 0
    else:
        (s, B, ga, kd) = winning_percep_actual
        asw_states_percep_actual[(s, frozenset(B), ga, kd)] = 2

# tic = time.perf_counter()
augmented_initial_game, augmented_initial_game_states, aug_p1_states_initial = deliveryGame.get_aug_initial_game(
    asw_states_delayed_actual, asw_states_percep_actual)
# toc = time.perf_counter()
# logger.debug(f"Constructed the P1's initial game in {toc - tic:0.4f} seconds")
# print(f"Constructed the P1's initial game in {toc - tic:0.4f} seconds")

# Solving P1's initial game
attack_aware_solver_P1_initial = active_sensing_and_sensing_attacks_game.AttackAwareP1SolverHiddenSensorSecondCB(
    augmented_initial_game, augmented_initial_game.final_idx)

# tic = time.perf_counter()
asw_states_initial, P1_strategy_in_initial, p1_unreachable_initial, perfect_win_P1_initial, states_in_perfect_not_in_asw_initial = attack_aware_solver_P1_initial.solve()
toc = time.perf_counter()
# logger.debug(f"Solved the policy for P1's initial game in {toc - tic:0.4f} seconds")
# print(f"Solved the policy in {toc - tic:0.4f} seconds")

for sindx in tqdm(asw_states_initial):
    state = augmented_initial_game_states[sindx]

    if state not in combined_winning_states:
        combined_winning_states.append(augmented_initial_game_states[sindx])

    if sindx in augmented_initial_game.p1_states_idx_dict:
        if state not in combined_winning_P1_states:
            combined_winning_P1_states.append(state)
            P1_strategy = P1_strategy_in_initial[sindx]
            combined_winning_P1_state_pol[combined_winning_states.index(state)] = P1_strategy

# Obtaining the winning initial states in the initial game! (as that is the game that is sufficient to compute the VoD)
win_initial_state = attack_aware_solver_P1_initial.get_win_initials(asw_states_initial, augmented_initial_game_states)
print(f"Obtained the winning initial states With the hidden sensor.")
toc = time.perf_counter()
print(f"Solved all the games in {toc - tic:0.4f} seconds.")

# Compare the difference in the winning initial states with and without the Hidden sensor.
win_initial_state_without_hidden = attack_aware_solver_P1_perceptual.get_win_initials(asw_states_percep,
                                                                                      augmented_percep_games_states)
print(f"Obtained the winning initial states Without the hidden sensor.")

diff_win_init = list(filter(lambda item: item not in win_initial_state_without_hidden, win_initial_state))

logger.debug(f"Computed the difference in the winning states with and without the hidden sensor")
logger.debug(f"Number of new winning initial states: {len(diff_win_init)}")
logger.debug(f"ASW initial states without hidden sensor: {win_initial_state_without_hidden}")
logger.debug(f"ASW initial states with hidden sensor: {win_initial_state}")
logger.debug(f"New ASW initial states added with hidden sensor: {diff_win_init}")
logger.debug(f"------------- Policy for the combined winning region ----------------------")

print(f"Computed the difference in the winning states with and without the hidden sensor.")
print(f"Number of new winning initial states : {len(diff_win_init)}")
print(f"ASW initial states without hidden sensor: {win_initial_state_without_hidden}")
print(f"ASW initial states with hidden sensor: {win_initial_state}")
print(f"New ASW initial states added with hidden sensor: {diff_win_init}")
print(f"------------- Policy for the combined winning region ----------------------")

for item in combined_winning_P1_states:
    logger.debug(f"State: {item}")
    logger.debug(f"Policy: {combined_winning_P1_state_pol[combined_winning_states.index(item)]}")

    print(f"State: {item}")
    print(f"Policy: {combined_winning_P1_state_pol[combined_winning_states.index(item)]}")

logger.debug(
    f"***************************************************************************************************************")
print(
    f"***************************************************************************************************************")

print(f"Combined winning states and Strategy being saved.")
f = open('../../results/ex39_tau_2_try_1_results.p', 'wb')
pickle.dump(combined_winning_states, f)
pickle.dump(combined_winning_P1_state_pol, f)

# Compute the value of deception.


# Run policy with the combined strategy. (May be also have a run for without hidden sensor)
quite = False

winning_initial_state_without_final = list([])
for ini in win_initial_state:
    if ini[0] not in target:
        winning_initial_state_without_final.append(ini)

final_idx = combined_winning_states.index('final')

# while True:
#     initial_state = random.choice(winning_initial_state_without_final)
#     belief_history = demo_active_sensing.run_policy_plot(gwg_user1, initial_state,
#                                                          augmented_perceptual_game,
#                                                          augmented_delayed_attack_game,
#                                                          augmented_initial_game, combined_winning_states,
#                                                          augmented_percep_games_states,
#                                                          augmented_delayed_att_game_states,
#                                                          augmented_initial_game_states,
#                                                          combined_winning_P1_state_pol, target,
#                                                          percep_limit, delayed_limit, final_idx, query_actions, quite)
#     winning_initial_state_without_final.remove(initial_state)
#     print(belief_history)
#     logger.debug(f"Belief History on this run: {belief_history}")
#     demo_active_sensing.plot_beliefs(belief_history)
#
#     print(
#         "************************************************************************************************************")
#     logger.debug(f"***************************************************************************************************")
#     if len(winning_initial_state_without_final) == 0:
#         break

print("completed.")
