#!/usr/bin/env python3
import numpy as np
import time
import random

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):
    

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find the best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        # max depth to reach, should be a specified by user, don`t forget it to pu
        # todo put this in a function and depth_max as parameter default 3

        t0 = time.time()
        time_threshold = 0.05
        current_depth = 0
        action = 0

        while time.time() - t0 < time_threshold:
            current_depth += 1
            children = initial_tree_node.compute_and_get_children()
            values = [-np.inf] * len(children)
            for i, child in enumerate(children):
                values[i] = minimax(t0, child, 1, -np.inf, np.inf, current_depth)
            if time.time() - t0 < time_threshold:
                best_score = max(values)
                best_score_indexes = get_index_condition(values, best_score)
                if len(best_score_indexes) > 1:
                    index = random.choice(best_score_indexes)
                else:
                    index = values.index(best_score)
                action = children[index].move
            else:
                # take the previous depth because of timeout
                current_depth -= 1

        return ACTION_TO_STR[action]


def minimax(t0, node, player, alpha, beta, dept_max=4):
    curr_state = node.state
    remaining_fishes = len(list(curr_state.fish_positions.keys()))
    time_threshold = 0.05

    # If timeout occurs return stupid values :
    if time.time() - t0 > time_threshold:
        return -np.inf

    # if all fishes have been caught :
    elif remaining_fishes == 0 or node.depth >= dept_max:
        value = calculate_heuristic(node)
        return value

    else:
        children = node.compute_and_get_children()
        children_score = [-np.inf] * len(children)
        for i, child in enumerate(children):
            children_score[i] = calculate_heuristic(child)

        if player == 0:
            v = - np.inf
            children = reorder_branches(children, arg_sort(children_score)[::-1])
            for child in children:
                v = max(v, minimax(t0, child, 1, alpha, beta, dept_max))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
        else:
            v = np.inf
            children = reorder_branches(children, arg_sort(children_score))
            for child in children:
                v = min(v, minimax(t0, child, 0, alpha, beta, dept_max))
                beta = min(beta, v)
                if beta <= alpha:
                    break
        return v


def calculate_heuristic(node):
    """
    Calculate the heuristic function for a player at a given state
    """
    curr_state = node.state
    curr_score = curr_state.player_scores[0] - curr_state.player_scores[1]
    alpha = 0.3
    # 19 down and 9 to the middle
    worst_case_distance = 19 + 9

    # remaining fishes
    if len(list(curr_state.fish_positions.keys())) == 0:
        return curr_score

    # get the index if the caught for each player, -1 if no fish is currently being caught
    index_of_fish_currently_caught_by_max = curr_state.player_caught[0]
    index_of_fish_currently_caught_by_min = curr_state.player_caught[1]

    index_of_lowest_score_fish, smallest_distance_to_max, smallest_distance_to_min = get_fish_info_to_min_max(curr_state)
    if index_of_fish_currently_caught_by_max != -1:
        curr_score += curr_state.fish_scores[index_of_fish_currently_caught_by_max]
    else:
        curr_score += (alpha * curr_state.fish_scores[index_of_lowest_score_fish]) * (worst_case_distance - smallest_distance_to_max) / (
                    worst_case_distance)
    if index_of_fish_currently_caught_by_min != -1:
        curr_score -= curr_state.fish_scores[index_of_fish_currently_caught_by_min]
    else:
        curr_score -= alpha * curr_state.fish_scores[index_of_lowest_score_fish] * (worst_case_distance - smallest_distance_to_min) / (
                    worst_case_distance)

    return curr_score


def get_index_condition(array, condition):
    indexes = []
    for i in range(len(array)):
        if array[i] == condition:
            indexes.append(i)
    return indexes


def arg_sort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def reorder_branches(array, order_seq):
    return [array[i] for i in order_seq]


def manhattan_distance_of_fishes_to_player(position_list, p):
    distance = []
    for position in position_list:
        distance.append(abs(position[0] - p[0]) + abs(position[1] - p[1]))
    return distance


def get_positions_for_positive_fish(state):
    # consider only fish with positive score
    fish_scores = state.fish_scores
    fish_positions = state.fish_positions
    positions_list = []
    real_indexes_list = []
    for ind in list(fish_positions.keys()):
        if fish_scores[ind] > 0:
            positions_list.append([fish_positions[ind][0], fish_positions[ind][1]])
            real_indexes_list.append(ind)
    if len(positions_list) > 0:
        return real_indexes_list, positions_list
    else:
        return list(state.fish_positions.keys()), [[p[0], p[1]] for p in list(state.fish_positions.values())]


def norm_distance_for_all_fishes_for_loop(state, player):
    fish_real_indexes, fish_positions = get_positions_for_positive_fish(state)
    player_position = state.hook_positions[player]
    opponent_position = state.hook_positions[abs(player - 1)]
    if player_position[0] < opponent_position[0]:
        for i in range(len(fish_positions)):
            if fish_positions[i][0] >= opponent_position[0]:
                fish_positions[i][0] -= 20
    elif player_position[0] > opponent_position[0]:
        for i in range(len(fish_positions)):
            if fish_positions[i][0] <= opponent_position[0]:
                fish_positions[i][0] += 20
    return fish_real_indexes, manhattan_distance_of_fishes_to_player(fish_positions, player_position)


def get_fish_info_to_min_max(state):
    current_fish_indexes, array_of_fish_distances_to_max = norm_distance_for_all_fishes_for_loop(state, 0)
    current_fish_indexes, array_of_fish_distances_to_min = norm_distance_for_all_fishes_for_loop(state, 1)
    smallest_distance_to_max = min(array_of_fish_distances_to_max)  # we take the first one if several fishes are equidistant
    smallest_distance_to_min = min(array_of_fish_distances_to_min)  # we take the first one if several fishes are equidistant
    score_fishes = [state.fish_scores[i] for i in current_fish_indexes]
    index_of_lowest_score_fish = current_fish_indexes[score_fishes.index(min(score_fishes))]
    return index_of_lowest_score_fish, smallest_distance_to_max, smallest_distance_to_min
