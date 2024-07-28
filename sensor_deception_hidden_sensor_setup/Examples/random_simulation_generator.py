from gridworld import *
from delivery_game import *
from sensor import *
from active_sensing_and_sensing_attacks_game import *
from stochatic_two_player_game_solver import *
from miscellaneous.policy import *
from loguru import logger

import pickle

import time

# Define a function to run simulations for a given configuration.
def run_simulation(ncols, nrows, target, obstacle_positions, unsafe_positions, all_positions, sensor, sensor_coverages, config_num):

    #  The following is the working code for experimentation.
    obstacles = obstacle_positions
    unsafe_u = unsafe_positions
    non_init_states = list(set(target).union(set(obstacles).union(set(unsafe_u))))
    initial = random.choice(list(set(all_positions)-set(non_init_states)))

    robot_ts = read_from_file_MDP_old('robotmdp.txt')

    ###test simple, deterministic sensor ex_sensor_deter.txt
    sensor_ts = read_from_file_MDP_old('ex_sensor_deter.txt')

    # The below are for experiment 15. (5X5)
    query_actions = dict([])
    query_actions[0] = {0, 3}
    query_actions[1] = {0, 2}  # Hidden
    query_actions[2] = {3, 2}  # Hidden
    query_actions[3] = {1, 3}
    # query_actions[4] = {2, 4, 0}  # Hidden

    attack_actions = dict([])
    attack_actions[0] = {0}
    attack_actions[1] = {1}
    attack_actions[2] = {2}  # Hidden
    attack_actions[3] = {3}
    # attack_actions[4] = {4}  # Hidden

    hidden_sensor_queries = dict([])
    hidden_sensor_queries[1] = {0, 2}
    hidden_sensor_queries[2] = {2, 3}
    # hidden_sensor_queries[4] = {2, 4, 0}
    # hidden_sensor_queries[3] = {1, 3}

    hidden_sensor_attacks = dict([])
    hidden_sensor_attacks[2] = {2}
    # hidden_sensor_attacks[4] = {4}

    # setting the precise sensor if there is any in the environment.
    precise_sensors = []

    # Log the sensor queries and attacks.
    # sensor = sensor()
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

    set0 = sensor_coverages[0]
    set1 = sensor_coverages[1]
    set2 = sensor_coverages[2]
    set3 = sensor_coverages[3]
    # set4 = {18, 19, 20, 21, 22, 23}

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
    toc = time.perf_counter()
    # logger.debug(f"Constructed the delivery game in {toc - tic:0.4f} seconds")
    print(f"Constructed the delivery game in {toc - tic:0.4f} seconds")

    # Defining variables for the combined winning region.
    combined_winning_P1_states = list([])
    combined_winning_P1_state_pol = dict([])
    combined_winning_states = list([])

    # Constructing P2's perceptual game
    tic = time.perf_counter()
    augmented_perceptual_game, augmented_percep_games_states, aug_p1_states_percep = deliveryGame.get_aug_sensing_game_perceptual()
    toc = time.perf_counter()
    # logger.debug(f"Constructed the P2's perceptual game in {toc - tic:0.4f} seconds")
    print(f"Constructed the P2's perceptual game in {toc - tic:0.4f} seconds")

    # Solving the P2's perceptual game
    attack_aware_solver_P1_perceptual = active_sensing_and_sensing_attacks_game.AttackAwareP1SolverHiddenSensorSecondCB(
        augmented_perceptual_game, augmented_perceptual_game.final_idx)

    tic = time.perf_counter()
    asw_states_percep, P1_strategy_in_percep, p1_unreachable_percep, perfect_win_P1_percep, states_in_perfect_not_in_asw = attack_aware_solver_P1_perceptual.solve()
    toc = time.perf_counter()
    # logger.debug(f"Solved the policy for P2's perceptual game in {toc - tic:0.4f} seconds")
    print(f"Solved the policy in {toc - tic:0.4f} seconds")

    for sindx in asw_states_percep:
        state = augmented_percep_games_states[
            sindx]
        if state not in combined_winning_states:
            combined_winning_states.append(augmented_percep_games_states[sindx])

        if sindx in augmented_perceptual_game.p1_states_idx_dict:
            if state not in combined_winning_P1_states:
                combined_winning_P1_states.append(state)
                P1_strategy = P1_strategy_in_percep[sindx]
                combined_winning_P1_state_pol[combined_winning_states.index(state)] = P1_strategy

    percep_limit = len(combined_winning_states) - 1

    # Constructing Delayed attack game
    tau_r = 1  # The minimum delay before P2 can attack the hidden sensor.
    p1_state_indices_in_asw = asw_states_percep.intersection(augmented_perceptual_game.p1_states_idx)
    p1_states_in_asw = list([])
    for index in p1_state_indices_in_asw:
        aug_p1_states_percep.remove(augmented_percep_games_states[index])

    tic = time.perf_counter()
    augmented_delayed_attack_game, augmented_delayed_att_game_states, p1_states_delayed_att, augmented_delayed_att_game_states_dict = deliveryGame.get_aug_delayed_attack_game(
        aug_p1_states_percep, tau_r)
    toc = time.perf_counter()
    # logger.debug(f"Constructed the delayed attack game in {toc - tic:0.4f} seconds")
    print(f"Constructed the delayed attack game in {toc - tic:0.4f} seconds")

    # Solving the Delayed attack game.
    attack_aware_solver_P1_delayed_attack = active_sensing_and_sensing_attacks_game.AttackAwareP1SolverHiddenSensorSecondCB(
        augmented_delayed_attack_game, augmented_delayed_attack_game.final_idx)

    tic = time.perf_counter()
    asw_states_delayed, P1_strategy_in_delayed, p1_unreachable_delayed, perfect_win_P1_delayed, states_in_perfect_not_in_asw_delayed = attack_aware_solver_P1_delayed_attack.solve()
    toc = time.perf_counter()
    # logger.debug(f"Solved the policy for delivery game in {toc - tic:0.4f} seconds")
    print(f"Solved the policy in {toc - tic:0.4f} seconds")

    print(f"Started reversing.")
    reverse_augmented_delay_att_gm_st_dict = {value: key for key, value in
                                              augmented_delayed_att_game_states_dict.items()}
    print(f"Finished reversing.")

    asw_states_delayed_calc = set(
        asw_states_delayed)
    # for sindx in tqdm(set(asw_states_delayed)):
    for sindx in asw_states_delayed_calc:
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
    for indx in asw_states_delayed:
        # winning_delayed_state = augmented_delayed_att_game_states[
        #     indx]
        winning_delayed_state = reverse_augmented_delay_att_gm_st_dict[indx]
        if len(winning_delayed_state) == 5:
            (s, B, a, ga, kd) = winning_delayed_state
            asw_states_delayed_actual[(s, frozenset(B), a, ga, kd)] = 0


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

    tic = time.perf_counter()
    augmented_initial_game, augmented_initial_game_states, aug_p1_states_initial = deliveryGame.get_aug_initial_game(
        asw_states_delayed_actual, asw_states_percep_actual)
    toc = time.perf_counter()
    # logger.debug(f"Constructed the P1's initial game in {toc - tic:0.4f} seconds")
    print(f"Constructed the P1's initial game in {toc - tic:0.4f} seconds")

    # Solving P1's initial game
    attack_aware_solver_P1_initial = active_sensing_and_sensing_attacks_game.AttackAwareP1SolverHiddenSensorSecondCB(
        augmented_initial_game, augmented_initial_game.final_idx)

    tic = time.perf_counter()
    asw_states_initial, P1_strategy_in_initial, p1_unreachable_initial, perfect_win_P1_initial, states_in_perfect_not_in_asw_initial = attack_aware_solver_P1_initial.solve()
    toc = time.perf_counter()
    # logger.debug(f"Solved the policy for P1's initial game in {toc - tic:0.4f} seconds")
    print(f"Solved the policy in {toc - tic:0.4f} seconds")

    for sindx in asw_states_initial:
        state = augmented_initial_game_states[sindx]

        if state not in combined_winning_states:
            combined_winning_states.append(augmented_initial_game_states[sindx])

        if sindx in augmented_initial_game.p1_states_idx_dict:
            if state not in combined_winning_P1_states:
                combined_winning_P1_states.append(state)
                P1_strategy = P1_strategy_in_initial[sindx]
                combined_winning_P1_state_pol[combined_winning_states.index(state)] = P1_strategy

    # Obtaining the winning initial states in the initial game! (as that is the game that is sufficient to compute the VoD)
    win_initial_state = attack_aware_solver_P1_initial.get_win_initials(asw_states_initial,
                                                                        augmented_initial_game_states)
    print(f"Obtained the winning initial states With the hidden sensor.")

    # Compare the difference in the winning initial states with and without the Hidden sensor.
    win_initial_state_without_hidden = attack_aware_solver_P1_perceptual.get_win_initials(asw_states_percep,
                                                                                          augmented_percep_games_states)
    print(f"Obtained the winning initial states Without the hidden sensor.")

    diff_win_init = list(filter(lambda item: item not in win_initial_state_without_hidden, win_initial_state))

    if len(diff_win_init) > 0:
        logger.debug(f"This is a configuration with advantage on using ")
        logger.debug(f"The obstacles in this config are: {obstacles}")
        logger.debug(f"The unsafe states are: {unsafe_u}")
        logger.debug(f"The sensor coverage is: {sensor.get_coverge(0)}")
        logger.debug(sensor.get_coverge(1))
        logger.debug(sensor.get_coverge(2))
        logger.debug(sensor.get_coverge(3))

        logger.debug(f"Computed the difference in the winning states with and without the hidden sensor")
        logger.debug(f"Number of new winning initial states: {len(diff_win_init)}")
        logger.debug(f"ASW initial states without hidden sensor: {win_initial_state_without_hidden}")
        logger.debug(f"ASW initial states with hidden sensor: {win_initial_state}")
        logger.debug(f"New ASW initial states added with hidden sensor: {diff_win_init}")
        logger.debug(f"------------- Policy for the combined winning region ----------------------")

        print(f"Combined winning states, strategy and configuration being saved.")
        f = open(f'results/configuration_{config_num}_{tau_r}_results.p', 'wb')
        pickle.dump(combined_winning_states, f)
        pickle.dump(combined_winning_P1_state_pol, f)

        # print(f"Computed the difference in the winning states with and without the hidden sensor.")
        # print(f"Number of new winning initial states : {len(diff_win_init)}")
        # print(f"ASW initial states without hidden sensor: {win_initial_state_without_hidden}")
        # print(f"ASW initial states with hidden sensor: {win_initial_state}")
        # print(f"New ASW initial states added with hidden sensor: {diff_win_init}")
        # print(f"------------- Policy for the combined winning region ----------------------")
        # for item in combined_winning_P1_states:
        #     logger.debug(f"State: {item}")
        #     logger.debug(f"Policy: {combined_winning_P1_state_pol[combined_winning_states.index(item)]}")
        #
        #     print(f"State: {item}")
        #     print(f"Policy: {combined_winning_P1_state_pol[combined_winning_states.index(item)]}")

        logger.debug(
            f"***************************************************************************************************************")
        logger.debug(f"************** Running for Tau = 2 *******************************************************************")
        print(
            f"***************************************************************************************************************")

        # Constructing Delayed attack game
        tau_r = 2  # The minimum delay before P2 can attack the hidden sensor.
        tic = time.perf_counter()
        augmented_delayed_attack_game, augmented_delayed_att_game_states, p1_states_delayed_att, augmented_delayed_att_game_states_dict = deliveryGame.get_aug_delayed_attack_game(
            aug_p1_states_percep, tau_r)
        toc = time.perf_counter()
        # logger.debug(f"Constructed the delayed attack game in {toc - tic:0.4f} seconds")
        print(f"Constructed the delayed attack game in {toc - tic:0.4f} seconds")

        # Solving the Delayed attack game.
        attack_aware_solver_P1_delayed_attack = active_sensing_and_sensing_attacks_game.AttackAwareP1SolverHiddenSensorSecondCB(
            augmented_delayed_attack_game, augmented_delayed_attack_game.final_idx)

        tic = time.perf_counter()
        asw_states_delayed, P1_strategy_in_delayed, p1_unreachable_delayed, perfect_win_P1_delayed, states_in_perfect_not_in_asw_delayed = attack_aware_solver_P1_delayed_attack.solve()
        toc = time.perf_counter()
        # logger.debug(f"Solved the policy for delivery game in {toc - tic:0.4f} seconds")
        print(f"Solved the policy in {toc - tic:0.4f} seconds")

        print(f"Started reversing.")
        reverse_augmented_delay_att_gm_st_dict = {value: key for key, value in
                                                  augmented_delayed_att_game_states_dict.items()}
        print(f"Finished reversing.")

        asw_states_delayed_calc = set(
            asw_states_delayed)
        # for sindx in tqdm(set(asw_states_delayed)):
        for sindx in asw_states_delayed_calc:
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
        for indx in asw_states_delayed:
            # winning_delayed_state = augmented_delayed_att_game_states[
            #     indx]
            winning_delayed_state = reverse_augmented_delay_att_gm_st_dict[indx]
            if len(winning_delayed_state) == 5:
                (s, B, a, ga, kd) = winning_delayed_state
                asw_states_delayed_actual[(s, frozenset(B), a, ga, kd)] = 0

        tic = time.perf_counter()
        augmented_initial_game, augmented_initial_game_states, aug_p1_states_initial = deliveryGame.get_aug_initial_game(
            asw_states_delayed_actual, asw_states_percep_actual)
        toc = time.perf_counter()
        # logger.debug(f"Constructed the P1's initial game in {toc - tic:0.4f} seconds")
        print(f"Constructed the P1's initial game in {toc - tic:0.4f} seconds")

        # Solving P1's initial game
        attack_aware_solver_P1_initial = active_sensing_and_sensing_attacks_game.AttackAwareP1SolverHiddenSensorSecondCB(
            augmented_initial_game, augmented_initial_game.final_idx)

        tic = time.perf_counter()
        asw_states_initial, P1_strategy_in_initial, p1_unreachable_initial, perfect_win_P1_initial, states_in_perfect_not_in_asw_initial = attack_aware_solver_P1_initial.solve()
        toc = time.perf_counter()
        # logger.debug(f"Solved the policy for P1's initial game in {toc - tic:0.4f} seconds")
        print(f"Solved the policy in {toc - tic:0.4f} seconds")

        for sindx in asw_states_initial:
            state = augmented_initial_game_states[sindx]

            if state not in combined_winning_states:
                combined_winning_states.append(augmented_initial_game_states[sindx])

            if sindx in augmented_initial_game.p1_states_idx_dict:
                if state not in combined_winning_P1_states:
                    combined_winning_P1_states.append(state)
                    P1_strategy = P1_strategy_in_initial[sindx]
                    combined_winning_P1_state_pol[combined_winning_states.index(state)] = P1_strategy

        # Obtaining the winning initial states in the initial game! (as that is the game that is sufficient to compute the VoD)
        win_initial_state_2 = attack_aware_solver_P1_initial.get_win_initials(asw_states_initial,
                                                                            augmented_initial_game_states)
        print(f"Obtained the winning initial states With the hidden sensor.")

        diff_win_init_2 = list(filter(lambda item: item not in win_initial_state, win_initial_state_2))

        if len(diff_win_init_2) > 0:
            logger.debug(f"Advantage with tau 2: {diff_win_init_2}")
            print(f"Combined winning states, strategy and configuration being saved.")
            f = open(f'results/configuration_{config_num}_{tau_r}_results.p', 'wb')
            pickle.dump(combined_winning_states, f)
            pickle.dump(combined_winning_P1_state_pol, f)
        else:
            logger.debug(f"No additional advantage with tau 2.")






#  Initialize the logger
logger.remove()
logger.add("../../log_files/log_of_simulations_5.log")

ncols = 5
nrows = 5
target = [9]
all_positions = list(range(25))
sensor = sensor()

# Define the rows and columns for sensors
rows = {
    'row0': {0, 1, 2, 3, 4},
    'row1': {5, 6, 7, 8, 9},
    'row2': {10, 11, 12, 13, 14},
    'row3': {15, 16, 17, 18, 19},
    'row4': {20, 21, 22, 23, 24}
    # Define other rows similarly
}

cols = {
    'col0': {0, 5, 10, 15, 20},
    'col1': {1, 6, 11, 16, 21},
    'col2': {2, 7, 12, 17, 22},
    'col3': {3, 8, 13, 18, 23},
    'col4': {4, 9, 14, 19, 24}
    # Define other columns similarly
}

# Generate all possible combinations of rows and columns
available_rows_and_cols = list(rows.items()) + list(cols.items())

# Specify the number of obstacles you want to explore
num_obstacles = 3  # Change this value as needed
num_unsafe = 4

# Generate all possible combinations of obstacles, unsafe states, and sensor combinations
all_obstacle_combinations = itertools.combinations(range(25), num_obstacles)
all_unsafe_combinations = itertools.combinations(range(25), num_unsafe)

# Uncomment the below stuff to ensure that you are iterating through all possible sensor configurations.
# # Generate all possible combinations of sensor combinations with values
# row_combinations = [(rows[key[0]].union(rows[key[1]])) for key in combinations(rows.keys(), 2)]
# col_combinations = [(cols[key[0]].union(cols[key[1]])) for key in combinations(cols.keys(), 2)]
# row_col_combinations = [(rows[row_key].union(cols[col_key])) for row_key, col_key in itertools.product(rows.keys(), cols.keys())]
#
# all_sensor_combinations = list(combinations(row_combinations + col_combinations + row_col_combinations, 4))

sensor_combination = [{5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, {20, 21, 22, 23, 24}, {5, 6, 7, 8, 9, 1, 11, 16, 21}, {0, 1, 2, 3, 4, 5, 10, 15, 20}]

# The following set-up is to iterate through all possible sensor combinations as well.
# n = 1
# for sensor_combination in tqdm(all_sensor_combinations):
#     for obstacle_positions in tqdm(all_obstacle_combinations):
#         for unsafe_positions in tqdm(all_unsafe_combinations):
#             if len(set(obstacle_positions).intersection(set(unsafe_positions))) == 0:
#                 run_simulation(ncols, nrows, target, obstacle_positions, unsafe_positions, all_positions, sensor, sensor_combination, n)
#                 logger.debug(f"Completed {n} configurations.")
#                 n = n+1

n = 1
for obstacle_positions in tqdm(all_obstacle_combinations):
    for unsafe_positions in tqdm(all_unsafe_combinations):
        if len(set(obstacle_positions).intersection(set(unsafe_positions))) == 0:
            run_simulation(ncols, nrows, target, obstacle_positions, unsafe_positions, all_positions, sensor, sensor_combination, n)
            logger.debug(f"Completed {n} configurations.")
            n = n+1

print(f"Finished all the simulations.")
logger.debug(f"Finished all the simulations.")

