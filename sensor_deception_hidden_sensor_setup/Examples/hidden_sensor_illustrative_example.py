__author__ = 'Sumukha Udupa, sudupa@ufl.edu'


# from delivery_game import *
# from sensor import *
# from stochatic_two_player_game_solver import *
from loguru import logger
# from MDP import *
from sensor_deception_hidden_sensor_setup.setup_and_solvers.MDP import *
from sensor_deception_hidden_sensor_setup.setup_and_solvers.delivery_game import *
from sensor_deception_hidden_sensor_setup.setup_and_solvers.sensor import *
from sensor_deception_hidden_sensor_setup.setup_and_solvers.stochatic_two_player_game_solver import *
# import tqdm as tqdm
import pickle

import time

logger.remove()
logger.add("../../log_files/log_of_illustrative_example_B_hidden_tau_1_try_2.log")

# The following is the setup for the illustrative mdp example.
target = [11, 12]
obstacles = []
unsafe_u = [3, 9, 16]
non_init_states = [11, 12, 9, 3, 16]
initial = 0
states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
init_states = list(set(states) - set(non_init_states))

sensor_ts = read_from_file_MDP_old('ex_sensor_deter.txt')


game_mdp = MDP()
game_mdp.init = init_states
game_mdp.states = states
game_mdp.actlist = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

N = len(game_mdp.states)
for a in game_mdp.actlist:
    game_mdp.prob[a] = np.zeros((N, N))

game_mdp.prob['D1'][0][1] = 0.5
game_mdp.prob['D1'][0][7] = 0.7
game_mdp.prob['D1'][1][0] = 0.5
game_mdp.prob['D1'][1][6] = 0.3
game_mdp.prob['D1'][1][2] = 0.2
game_mdp.prob['D1'][3][3] = 1
game_mdp.prob['D1'][6][9] = 0.5
game_mdp.prob['D1'][6][7] = 0.5
game_mdp.prob['D1'][7][0] = 0.3
game_mdp.prob['D1'][7][8] = 0.3
game_mdp.prob['D1'][7][6] = 0.4
game_mdp.prob['D1'][8][7] = 1
game_mdp.prob['D1'][9][9] = 1
game_mdp.prob['D1'][11][11] = 1
game_mdp.prob['D1'][12][12] = 1
game_mdp.prob['D1'][16][16] = 1
game_mdp.prob['D1'][18][12] = 1
game_mdp.prob['D1'][19][18] = 0.5
game_mdp.prob['D1'][19][20] = 0.5
game_mdp.prob['D1'][20][9] = 1
game_mdp.prob['D1'][23][21] = 1
game_mdp.prob['D1'][21][23] = 1

game_mdp.prob['D2'][5][4] = 0.7
game_mdp.prob['D2'][5][10] = 0.3
game_mdp.prob['D2'][3][3] = 1
game_mdp.prob['D2'][9][9] = 1
game_mdp.prob['D2'][11][11] = 1
game_mdp.prob['D2'][12][12] = 1
game_mdp.prob['D2'][16][16] = 1
game_mdp.prob['D2'][18][21] = 0.5
game_mdp.prob['D2'][18][16] = 0.5
game_mdp.prob['D2'][20][12] = 1
game_mdp.prob['D2'][21][18] = 0.3
game_mdp.prob['D2'][21][19] = 0.5
game_mdp.prob['D2'][21][22] = 0.2
game_mdp.prob['D2'][22][21] = 0.5
game_mdp.prob['D2'][22][24] = 0.3
game_mdp.prob['D2'][22][20] = 0.2
game_mdp.prob['D2'][23][22] = 1
game_mdp.prob['D2'][24][23] = 0.5
game_mdp.prob['D2'][24][22] = 0.5

game_mdp.prob['D3'][2][3] = 1
game_mdp.prob['D3'][3][3] = 1
game_mdp.prob['D3'][5][6] = 1
game_mdp.prob['D3'][6][5] = 1
game_mdp.prob['D3'][8][9] = 1
game_mdp.prob['D3'][9][9] = 1
game_mdp.prob['D3'][11][11] = 1
game_mdp.prob['D3'][12][12] = 1
game_mdp.prob['D3'][14][17] = 1
game_mdp.prob['D3'][16][16] = 1
game_mdp.prob['D3'][17][14] = 1


game_mdp.prob['D4'][1][6] = 0.5
game_mdp.prob['D4'][1][2] = 0.5
game_mdp.prob['D4'][2][5] = 1
game_mdp.prob['D4'][3][3] = 1
game_mdp.prob['D4'][5][2] = 1
game_mdp.prob['D4'][6][9] = 0.5
game_mdp.prob['D4'][6][7] = 0.5
game_mdp.prob['D4'][7][6] = 0.5
game_mdp.prob['D4'][7][8] = 0.5
game_mdp.prob['D4'][8][7] = 1
game_mdp.prob['D4'][9][9] = 1
game_mdp.prob['D4'][11][11] = 1
game_mdp.prob['D4'][12][12] = 1
game_mdp.prob['D4'][16][16] = 1

game_mdp.prob['D5'][3][3] = 1
game_mdp.prob['D5'][4][11] = 1
game_mdp.prob['D5'][8][13] = 1
game_mdp.prob['D5'][9][9] = 1
game_mdp.prob['D5'][10][9] = 1
game_mdp.prob['D5'][11][11] = 1
game_mdp.prob['D5'][12][12] = 1
game_mdp.prob['D5'][13][12] = 1
game_mdp.prob['D5'][15][16] = 1
game_mdp.prob['D5'][16][16] = 1


game_mdp.prob['D6'][3][3] = 1
game_mdp.prob['D6'][4][3] = 1
game_mdp.prob['D6'][8][13] = 1
game_mdp.prob['D6'][9][9] = 1
game_mdp.prob['D6'][10][11] = 1
game_mdp.prob['D6'][11][11] = 1
game_mdp.prob['D6'][12][12] = 1
game_mdp.prob['D6'][13][8] = 0.3
game_mdp.prob['D6'][13][14] = 0.4
game_mdp.prob['D6'][13][9] = 0.3
game_mdp.prob['D6'][14][13] = 0.5
game_mdp.prob['D6'][14][15] = 0.5
game_mdp.prob['D6'][15][14] = 0.5
game_mdp.prob['D6'][15][12] = 0.5
game_mdp.prob['D6'][16][16] = 1


# The below are illustrative example.
query_actions = dict([])
query_actions[0] = {0, 3, 4}
query_actions[1] = {0, 1, 2}  # Hidden
query_actions[2] = {0, 3, 2}
query_actions[3] = {1, 3, 2}  # Hidden
query_actions[4] = {1, 3, 4}  # Hidden
query_actions[5] = {2, 3, 4}


attack_actions = dict([])
attack_actions[0] = {0}
attack_actions[1] = {1}  # Hidden
attack_actions[2] = {2}
attack_actions[3] = {3}
attack_actions[4] = {4}

hidden_sensor_queries = dict([])
hidden_sensor_queries[1] = {0, 1, 2}
hidden_sensor_queries[3] = {1, 2, 3}
hidden_sensor_queries[4] = {3, 4, 1}
# hidden_sensor_queries[3] = {1, 3}


hidden_sensor_attacks = dict([])
hidden_sensor_attacks[1] = {1}

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

game_mdp.get_supp()
complete_states = game_mdp.states

# Coverage for illustrative example
set0 = {4, 10, 13, 15, 17, 19}
set1 = {4, 5, 6, 15, 18, 20, 23, 24}
set2 = {3, 9, 11, 12, 16, 18, 20, 21, 22, 23, 24}
set3 = {3, 4, 7, 9, 10, 17, 19, 21, 22, 23, 24}
set4 = {1, 2, 5, 6, 8, 14}

sensor.set_coverage(0, set0)
sensor.set_coverage(1, set1)
sensor.set_coverage(2, set2)
sensor.set_coverage(3, set3)
sensor.set_coverage(4, set4)

tic = time.perf_counter()
deliveryGame = DeliveryGame()
deliveryGame.agentmdp = game_mdp
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
    state = augmented_percep_games_states[sindx]
    if state not in combined_winning_states:
        combined_winning_states.append(augmented_percep_games_states[sindx])

    if sindx in augmented_perceptual_game.p1_states_idx_dict:
        if state not in combined_winning_P1_states:
            combined_winning_P1_states.append(state)
            P1_strategy = P1_strategy_in_percep[sindx]
            combined_winning_P1_state_pol[combined_winning_states.index(state)] = P1_strategy

percep_limit = len(combined_winning_states) - 1

# Constructing Delayed attack game
tau_r = 3  # The minimum delay before P2 can attack the hidden sensor.
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
    asw_states_delayed)
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

    if state not in combined_winning_states:
        combined_winning_states.append(augmented_delayed_att_game_states[sindx])

    if sindx in augmented_delayed_attack_game.p1_states_idx_dict:
        if state not in combined_winning_P1_states:
            combined_winning_P1_states.append(state)
            P1_strategy = P1_strategy_in_delayed[sindx]
            combined_winning_P1_state_pol[combined_winning_states.index(state)] = P1_strategy

delayed_limit = len(combined_winning_states) - 1

# Constructing P1's initial game
# asw_states_delayed_actual = list([])
# for indx in asw_states_delayed:
#     asw_states_delayed_actual.append(augmented_delayed_att_game_states[indx])

asw_states_delayed_actual = dict([])
for indx in tqdm(asw_states_delayed):
    # winning_delayed_state = augmented_delayed_att_game_states[
    #     indx]
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
print(f"Solved the complete policy in {toc - tic: 0.4f} seconds")

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
f = open('../../results/illustrative_example_tau_1_results.p', 'wb')
pickle.dump(combined_winning_states, f)
pickle.dump(combined_winning_P1_state_pol, f)





print("completed.")
