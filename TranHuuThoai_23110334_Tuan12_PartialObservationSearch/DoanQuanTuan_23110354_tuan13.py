import pygame
import numpy as np
from collections import deque
import heapq
import itertools
import time 
import random


LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (135, 206, 235)
LIGHT_GREEN = (204, 255, 204)  
DARK_GREEN = (153, 204, 153)
LIGHT_YELLOW = (255, 255, 153)
DARK_YELLOW = (230, 230, 50)


move = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
move_cost = {
    (-1, 0): 1,  # Lên
    (1, 0): 1,   # Xuống
    (0, -1): 2,  # Trái
    (0, 1): 2    # Phải
}


def get_successors(state):
    successors = []
    x, y = np.where(state == 0)[0][0], np.where(state == 0)[1][0]
    for dx, dy in move:
        if 0 <= x + dx < 3 and 0 <= y + dy < 3:  
            new_state = state.copy()           
            new_state[x, y], new_state[x + dx, y + dy] = new_state[x + dx, y + dy], new_state[x, y]
            successors.append(new_state)
    return successors

def get_successors_with_cost(state):
    successors = []
    x, y = np.where(state == 0)[0][0], np.where(state == 0)[1][0]
    
    for (dx, dy), cost in move_cost.items():
        if 0 <= x + dx < 3 and 0 <= y + dy < 3:  
            new_state = state.copy()
            new_state[x, y], new_state[x + dx, y + dy] = new_state[x + dx, y + dy], new_state[x, y]
            successors.append((new_state, cost))  
    return successors

def get_successors_with_path(state):
    successors = []
    x, y = np.where(state == 0)[0][0], np.where(state == 0)[1][0]  
    for dx, dy in move:
        if 0 <= x + dx < 3 and 0 <= y + dy < 3:  
            new_state = state.copy()           
            new_state[x, y], new_state[x + dx, y + dy] = new_state[x + dx, y + dy], new_state[x, y]
            successors.append((new_state, (dx, dy)))  
    return successors


def heuristic(state, goal_state):
    return np.sum((state != goal_state) & (state != 0))

def check(start_state, goal_state):
    a = start_state.copy().ravel()
    b = goal_state.copy().ravel()
    count_a, count_b = 0, 0
    for i in range(8):
        for j in range(i + 1, 9):
            if a[i] > a[j] and a[i] != 0 and a[j] != 0: count_a += 1
            if b[i] > b[j] and b[i] != 0 and b[j] != 0: count_b += 1
            
    if (count_a % 2) == (count_b % 2):
        return True
    return False
    
def bfs(start_state, goal_state):
    if np.array_equal(start_state, goal_state):
        return [], 0

    open = deque([(start_state, [])])
    closed = set()
    nodes_visited = 0  

    while open:
        node, path = open.popleft()
        node_tuple = tuple(map(tuple, node))  
        if node_tuple in closed:
            continue

        closed.add(node_tuple)
        nodes_visited += 1  

        for successor in get_successors(node):
            successor_tuple = tuple(map(tuple, successor))
            if np.array_equal(successor, goal_state):
                return path + [successor], nodes_visited
            if successor_tuple not in closed:
                open.append((successor, path + [successor]))

    return None, nodes_visited

def dfs(start_state, goal_state): 
    if np.array_equal(start_state, goal_state):
        return [], 0  

    open = deque([(start_state, [])]) 
    closed = set()
    nodes_visited = 0

    while open:
        node, path = open.pop()  
        node_tuple = tuple(map(tuple, node))  

        if node_tuple in closed:
            continue
        closed.add(node_tuple)
        nodes_visited += 1  

        for successor in get_successors(node):
            successor_tuple = tuple(map(tuple, successor))
            if np.array_equal(successor, goal_state):
                return path + [successor], nodes_visited
            if successor_tuple not in closed:
                open.append((successor, path + [successor]))

    return None, nodes_visited  
 
def depth_limited_search(start_state, goal_state, depth_limit):
    open = [(start_state, [], 0)]
    closed = set()
    nodes_visited = 0  # đếm số node đã duyệt

    while open:
        node, path, depth = open.pop()
        node_tuple = tuple(map(tuple, node))
        
        if node_tuple in closed:
            continue
        if depth > depth_limit:
            continue

        closed.add(node_tuple)
        nodes_visited += 1

        if np.array_equal(node, goal_state):
            return path, nodes_visited
        
        for successor in get_successors(node):
            successor_tuple = tuple(map(tuple, successor))
            if successor_tuple not in closed:
                open.append((successor, path + [successor], depth + 1))

    return None, nodes_visited


def ids(start_state, goal_state):
    if np.array_equal(start_state, goal_state):
        return [], 0
    
    depth_limit = 0
    total_nodes_visited = 0

    while True:
        result, nodes_visited = depth_limited_search(start_state, goal_state, depth_limit)
        total_nodes_visited += nodes_visited
        if result is not None:
            return result, total_nodes_visited
        depth_limit += 1

def ucs(start_state, goal_state):    
    if np.array_equal(start_state, goal_state):
        return [start_state], 0, 0 

    open = []
    counter = itertools.count()
    heapq.heappush(open, (0, next(counter), start_state, [start_state], 0))  
    closed = {}
    nodes_visited = 0

    while open:
        cost, _, node, path, total_cost = heapq.heappop(open)
        node_tuple = tuple(map(tuple, node))

        if node_tuple in closed and closed[node_tuple] <= cost:
            continue

        closed[node_tuple] = cost
        nodes_visited += 1

        if np.array_equal(node, goal_state):
            return path, total_cost, nodes_visited
        
        for successor, action_cost in get_successors_with_cost(node):
            successor_tuple = tuple(map(tuple, successor))
            new_cost = cost + action_cost
            if successor_tuple not in closed or new_cost < closed[successor_tuple]:
                new_path = path + [successor] 
                new_total_cost = total_cost + action_cost
                heapq.heappush(open, (new_cost, next(counter), successor, new_path, new_total_cost))

    return None, float('inf'), nodes_visited
def manhattan_distance(state, goal_state):
    goal_positions = {}
    for i in range(3):
        for j in range(3):
            goal_positions[goal_state[i][j]] = (i, j)
    
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                goal_x, goal_y = goal_positions[state[i][j]]
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

import heapq
import numpy as np

def greedy(start_state, goal_state):  
    if np.array_equal(start_state, goal_state):  
        return [start_state], 0, 0  
    
    open_heap = []  
    initial_cost = manhattan_distance(start_state, goal_state)  
    heapq.heappush(open_heap, (initial_cost, start_state.tobytes(), 0))  
    
    came_from = {start_state.tobytes(): None}    
    closed = set()  
    nodes_visited = 0

    while open_heap:  
        heuristic_cost, current_bytes, cost = heapq.heappop(open_heap) 
        current = np.frombuffer(current_bytes, dtype=start_state.dtype).reshape(start_state.shape)  

        if current_bytes in closed:
            continue
        closed.add(current_bytes)
        nodes_visited += 1

        if np.array_equal(current, goal_state):  
            path = []  
            while current_bytes is not None:  
                path.append(np.frombuffer(current_bytes, dtype=start_state.dtype).reshape(start_state.shape))  
                current_bytes = came_from[current_bytes]  
            return path[::-1], cost, nodes_visited

        for next_state, move in get_successors_with_path(current):  
            next_bytes = next_state.tobytes()  
            if next_bytes not in closed:
                move_cost_value = move_cost.get(move, 1)  
                new_heuristic = manhattan_distance(next_state, goal_state)  
                heapq.heappush(open_heap, (new_heuristic, next_bytes, cost + move_cost_value))  
                came_from[next_bytes] = current.tobytes()  

    return [], float('inf'), nodes_visited

def a_star(start_state, goal_state):  
    if np.array_equal(start_state, goal_state):  
        return [start_state], 0, 0

    open_heap = []  
    initial_cost = manhattan_distance(start_state, goal_state)  
    heapq.heappush(open_heap, (initial_cost, start_state.tobytes(), 0)) 

    came_from = {start_state.tobytes(): None}  
    g_score = {start_state.tobytes(): 0} 
    closed = set()  
    nodes_visited = 0  

    while open_heap:  
        _, current_bytes, cost = heapq.heappop(open_heap)  
        current = np.frombuffer(current_bytes, dtype=start_state.dtype).reshape(start_state.shape)  

        if current_bytes in closed:
            continue

        closed.add(current_bytes)
        nodes_visited += 1

        if np.array_equal(current, goal_state):  
            path = []  
            while current_bytes is not None:  
                path.append(np.frombuffer(current_bytes, dtype=start_state.dtype).reshape(start_state.shape))  
                current_bytes = came_from[current_bytes]  
            return path[::-1], g_score[goal_state.tobytes()], nodes_visited

        for next_state, move in get_successors_with_path(current):  
            next_bytes = next_state.tobytes()  
            move_cost_value = 1  

            tentative_g_score = g_score[current_bytes] + move_cost_value  

            if next_bytes in closed and tentative_g_score >= g_score.get(next_bytes, float('inf')):
                continue

            if next_bytes not in g_score or tentative_g_score < g_score[next_bytes]:  
                g_score[next_bytes] = tentative_g_score
                f_score = tentative_g_score + manhattan_distance(next_state, goal_state)  
                heapq.heappush(open_heap, (f_score, next_bytes, tentative_g_score))  
                came_from[next_bytes] = current_bytes  

    return [], float('inf'), nodes_visited


def ida_star(start_state, goal_state):  
    def search(node, g_cost, threshold, path, visited, nodes_visited):  
        f_cost = g_cost + manhattan_distance(node, goal_state)  
        if f_cost > threshold:  
            return None, f_cost, nodes_visited 

        if np.array_equal(node, goal_state):  
            return path + [node], g_cost, nodes_visited 

        min_cost = float('inf')  
        for next_state, move_cost in get_successors_with_cost(node):  
            state_tuple = tuple(map(tuple, next_state))  
            if state_tuple in visited:  
                continue  

            if not isinstance(move_cost, (int, float)):  
                continue  

            visited.add(state_tuple)  

            nodes_visited += 1  
            result, temp_cost, nodes_visited = search(next_state, g_cost + move_cost, threshold, path + [node], visited, nodes_visited)  
            visited.remove(state_tuple)  

            if result is not None:  
                return result, temp_cost, nodes_visited 
            if isinstance(temp_cost, (int, float)):  
                min_cost = min(min_cost, temp_cost)  

        return None, min_cost, nodes_visited  

    threshold = manhattan_distance(start_state, goal_state)  
    path = []  
    visited = set()
    visited.add(tuple(map(tuple, start_state)))

    nodes_visited = 1 

    while True:  
        result, temp_cost, nodes_visited = search(start_state, 0, threshold, path, visited, nodes_visited) 
        if result is not None:  
            return result, temp_cost, nodes_visited 
        if temp_cost == float('inf'):  
            return None, None, nodes_visited  
        threshold = temp_cost  




        
def simple_hill_climbing(start_state, goal_state):
    solution = [start_state]
    current_state = start_state
    current_score = manhattan_distance(current_state, goal_state)

    nodes_visited = 1  

    while True:
        if np.array_equal(current_state, goal_state):
            return solution, current_score, nodes_visited  

        neighbors = get_successors(current_state)
        best_move = None
        best_score = current_score

        for neighbor in neighbors:
            nodes_visited += 1  
            score = manhattan_distance(neighbor, goal_state)
            if score < best_score:
                best_score = score
                best_move = neighbor

        if best_move is None or best_score >= current_score:
            break  

        current_state = best_move
        current_score = best_score
        solution.append(current_state)

    return solution, current_score, nodes_visited



def steepest_ascent_hill_climbing(start_state, goal_state):
    solution = [start_state]
    current_state = start_state
    current_score = manhattan_distance(current_state, goal_state)
    nodes_visited = 1  

    while True:
        if np.array_equal(current_state, goal_state):
            return solution, current_score, nodes_visited

        neighbors = get_successors(current_state)
        best_move = None
        best_score = float('inf')

        for neighbor in neighbors:
            nodes_visited += 1  
            score = manhattan_distance(neighbor, goal_state)
            if score < best_score:
                best_score = score
                best_move = neighbor

   
        if best_move is None or best_score >= current_score:
            break

        current_state = best_move
        current_score = best_score
        solution.append(current_state)

    return solution, current_score, nodes_visited

def stochastic_hill_climbing(start_state, goal_state):
    path = [start_state]
    current_state = start_state
    current_score = manhattan_distance(current_state, goal_state)
    nodes_visited = 1 

    for _ in range(100): 
        if np.array_equal(current_state, goal_state):
            return path, current_score, nodes_visited

        neighbors = get_successors(current_state)
        better_neighbors = []

        for neighbor in neighbors:
            nodes_visited += 1
            score = manhattan_distance(neighbor, goal_state)
            if score < current_score:
                better_neighbors.append((neighbor, score))

        if not better_neighbors:
            break 

        chosen_neighbor, chosen_score = random.choice(better_neighbors)
        current_state = chosen_neighbor
        current_score = chosen_score
        path.append(current_state)

    return path, current_score, nodes_visited

import numpy as np
import random

def simulated_annealing(start_state, goal_state):
    T = 1000              
    alpha = 0.99          
    T_min = 1e-3           
    max_iterations = 10000

    current_state = start_state
    current_cost = manhattan_distance(current_state, goal_state)
    path = [current_state]

    best_state = current_state
    best_cost = current_cost
    best_path = list(path)

    nodes_visited = 1
    iteration = 0

    while T > T_min and iteration < max_iterations:
        neighbors = get_successors(current_state)
        if not neighbors:
            break

        new_state = random.choice(neighbors)
        new_cost = manhattan_distance(new_state, goal_state)
        nodes_visited += 1

        delta = new_cost - current_cost
        
        if delta < 0 or np.exp(-delta / T) > np.random.rand():
            current_state = new_state
            current_cost = new_cost
            path.append(current_state)

            if current_cost < best_cost:
                best_state = current_state
                best_cost = current_cost
                best_path = list(path)

            if current_cost == 0:
                return path, 0, nodes_visited

        T *= alpha
        iteration += 1

    return best_path, best_cost, nodes_visited

def beam(start_state, goal_state):
    k=2
    current_states = [(start_state, heuristic(start_state, goal_state), [start_state])]
    nodes_visited = 1  
    best_path = None
    best_cost = float('inf')

    visited = set()  
    visited.add(tuple(map(tuple, start_state)))

    while current_states:
        successors_with_heuristic = []

        for state, _, path in current_states:
            if np.array_equal(state, goal_state):
                return path, 0, nodes_visited  

            successors = get_successors(state)
            for successor in successors:
                successor_tuple = tuple(map(tuple, successor))
                if successor_tuple in visited:
                    continue 

                h = heuristic(successor, goal_state)
                new_path = path + [successor]
                successors_with_heuristic.append((successor, h, new_path))
                visited.add(successor_tuple)
                nodes_visited += 1

                if h < best_cost:
                    best_cost = h
                    best_path = new_path

        current_states = sorted(successors_with_heuristic, key=lambda x: x[1])[:k]

        if not current_states:
            break

    return best_path, best_cost, nodes_visited 

##############################################################################################
random.seed(time.time())
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def apply_move(state, move):
    x, y = np.where(state == 0)
    x, y = int(x[0]), int(y[0])
    dx, dy = move
    new_x, new_y = x + dx, y + dy
    if 0 <= new_x < 3 and 0 <= new_y < 3:
        new_state = state.copy()
        new_state[x, y], new_state[new_x, new_y] = new_state[new_x, new_y], new_state[x, y]
        return new_state
    return state

def fitness(individual, start_state, goal_state):
    state = start_state.copy()
    for move in individual:
        state = apply_move(state, move)
    return 1 / (manhattan_distance(state, goal_state) + 1)

def mutate(individual, mutation_rate=0.05):
    return [random.choice(moves) if random.random() < mutation_rate else m for m in individual]

def reproduce(parent1, parent2):
    n = len(parent1)
    c = random.randint(1, n - 1)
    return parent1[:c] + parent2[c:]

def weighted_random_choices(population, weights, k=2):
    return random.choices(population, weights=weights, k=k)

def genetic_algorithm(start_state, goal_state, pop_size=100, seq_length=20, generations=1000):
    population = [[random.choice(moves) for _ in range(seq_length)] for _ in range(pop_size)]

    for _ in range(generations):
        fitness_values = [fitness(ind, start_state, goal_state) for ind in population]
        weights = fitness_values
        population2 = []

        for _ in range(pop_size):
            parent1, parent2 = weighted_random_choices(population, weights)
            child = reproduce(parent1, parent2)

            if random.random() < 0.1:  
                child = mutate(child)

            population2.append(child)

        population = population2
        for individual in population:
            state = start_state.copy()
            path = [state.copy()]
            for move in individual:
                state = apply_move(state, move)
                path.append(state.copy())
                if np.array_equal(state, goal_state):
                    return path, len(path)
                
    best_individual = max(population, key=lambda ind: fitness(ind, start_state, goal_state))
    state = start_state.copy()
    path = [state.copy()]
    for move in best_individual:
        state = apply_move(state, move)
        path.append(state.copy())
    return path, len(path)

#################################################################################
def and_or_search(start_state, goal_state):
    problem = {
        "start_state": np.array(start_state, copy=True),
        "goal_state": np.array(goal_state, copy=True)
    }

    def is_cycle(state, path):
        states = [p[0] for p in path]
        return len(states) != len(set(map(str, states)))

    def actions(state):
        i, j = np.where(state == 0)
        i, j = i[0], j[0]
        possible_actions = []
        if i > 0: possible_actions.append("up")
        if i < 2: possible_actions.append("down")
        if j > 0: possible_actions.append("left")
        if j < 2: possible_actions.append("right")
        return possible_actions

    def result(state, action):
        i, j = np.where(state == 0)
        i, j = i[0], j[0]
        new_state = np.array(state, copy=True)
        if action == "up":
            new_state[i][j], new_state[i-1][j] = new_state[i-1][j], new_state[i][j]
        elif action == "down":
            new_state[i][j], new_state[i+1][j] = new_state[i+1][j], new_state[i][j]
        elif action == "left":
            new_state[i][j], new_state[i][j-1] = new_state[i][j-1], new_state[i][j]
        elif action == "right":
            new_state[i][j], new_state[i][j+1] = new_state[i][j+1], new_state[i][j]
        return new_state

    def or_search(problem, state, path):
        if np.array_equal(state, problem["goal_state"]):
            return [state]
        if is_cycle(state, path):
            return []
        for action in actions(state):
            next_state = result(state, action)
            plan = and_search(problem, next_state, path + [state])
            if plan:
                return [state] + plan
        return []

    def and_search(problem, state, path):
        return or_search(problem, state, path)

    return or_search(problem, problem["start_state"], [])

####################################################################################
def search_with_no_observation():
   run_no_observation()
   
def search_with_partial_observation(partial_goal):
   run_partial_observation(partial_goal)
########################################################################################
# Backtracking
np.random.seed(42)

def is_complete(state):
    return np.count_nonzero(state) == 8

def is_consistent(state, row, col, value):
    if value != 0:
        if value in state:
            return False
        if col > 0 and state[row, col - 1] != 0 and value != state[row, col - 1] + 1:
            return False
        if row > 0 and state[row - 1, col] != 0 and value != state[row - 1, col] + 3:
            return False
    else:  # value == 0
        return np.count_nonzero(state == 0) <= 1
    return True

def find_unassigned(state):
    zeros = np.argwhere(state == 0)
    return zeros[0] if zeros.size > 0 else (None, None)

def recursive_backtracking(state, all_states):
    if is_complete(state):
        all_states.append(state.copy())  # Lưu trạng thái lời giải
        return True

    row, col = find_unassigned(state)
    if row is None:
        return False

    values = np.arange(1, 9).tolist() + [0]
    np.random.shuffle(values)

    for value in values:
        if is_consistent(state, row, col, value):
            state[row, col] = value
            all_states.append(state.copy())  

            if recursive_backtracking(state, all_states):
                return True

            state[row, col] = 0
            all_states.append(state.copy())  # Lưu trạng thái sau backtrack

    return False
def get_all_states_backtracking(start_state, goal_state):
    all_states = []
    recursive_backtracking(start_state, all_states)
    return all_states

#######################################################################################################
# kiểm thử
def check_constraints(state, row, col, value):
    # Kiểm tra giá trị đã được sử dụng
    values = [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None and (i, j) != (row, col)]
    if value in values:
        return False
    
    # Kiểm tra ràng buộc ngang (hàng 0 và 1)
    if row < 2:
        if col == 0:
            if state[row][1] is not None and state[row][1] != value + 1:
                return False
            if state[row][2] is not None and state[row][2] != value + 2:
                return False
        elif col == 1:
            if state[row][0] is not None and state[row][0] != value - 1:
                return False
            if state[row][2] is not None and state[row][2] != value + 1:
                return False
        elif col == 2:
            if state[row][1] is not None and state[row][1] != value - 1:
                return False
            if state[row][0] is not None and state[row][0] != value - 2:
                return False
    
    # Kiểm tra ràng buộc dọc (cột 0 và 1)
    if col < 2:
        if row == 0:
            if state[1][col] is not None and state[1][col] != value + 3:
                return False
            if state[2][col] is not None and state[2][col] != value + 6:
                return False
        elif row == 1:
            if state[0][col] is not None and state[0][col] != value - 3:
                return False
            if state[2][col] is not None and state[2][col] != value + 3:
                return False
        elif row == 2:
            if state[1][col] is not None and state[1][col] != value - 3:
                return False
            if state[0][col] is not None and state[0][col] != value - 6:
                return False
    
    return True

def calculate_constraints():
    constraints_count = {
        (0, 0): 2, (0, 1): 3, (0, 2): 2,
        (1, 0): 3, (1, 1): 4, (1, 2): 3,
        (2, 0): 2, (2, 1): 3, (2, 2): 0
    }
    return sorted(constraints_count.keys(), key=lambda k: -constraints_count[k])

def generate_and_test(start_state, goal_state):
    state = [[None for _ in range(3)] for _ in range(3)]
    all_states = []
    state_count = [0]  
    order = calculate_constraints()
    other_positions = [pos for pos in order if pos != (1, 1) and pos != (2, 2)]
    
    center_values = random.sample(range(1, 9), 8)
    
    def backtrack(index):
        all_states.append(np.array([[0 if x is None else x for x in row] for row in state]))
        state_count[0] += 1
        
        if index == len(other_positions) + 1:  
            state[2][2] = 0  
            all_states.append(np.array([[0 if x is None else x for x in row] for row in state]))
            state_count[0] += 1
            return True
        
        if index == 0:  
            row, col = (1, 1)
            values = center_values
        else:  
            row, col = other_positions[index - 1]
            values = [v for v in range(1, 9) if v not in [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None]]
        
        for value in values:
            if check_constraints(state, row, col, value):
                state[row][col] = value
                if backtrack(index + 1):
                    return True
                state[row][col] = None
        
        return False
    backtrack(0)
    return all_states


####################################################################################
# ac3
np.random.seed(42)

DOMAIN = list(range(1, 9)) + [0]

def is_complete(state):
    return np.count_nonzero(state) == 8

def is_consistent(state, row, col, value):
    if value != 0:
        if value in state:
            return False
        if col > 0 and state[row, col - 1] != 0 and value != state[row, col - 1] + 1:
            return False
        if row > 0 and state[row - 1, col] != 0 and value != state[row - 1, col] + 3:
            return False
    else:  
        return np.count_nonzero(state == 0) <= 1
    return True

def find_unassigned(state):
    zeros = np.argwhere(state == 0)
    return zeros[0] if zeros.size > 0 else (None, None)

def ac3(domains, variables, neighbors):
    queue = deque((xi, xj) for xi in variables for xj in neighbors[xi])
    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj):
            if not domains[xi]:
                return False
            for xk in neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True

def revise(domains, xi, xj):
    revised = False
    for x in domains[xi][:]:
        if all(x == y for y in domains[xj]):
            domains[xi].remove(x)
            revised = True
    return revised

def backtracking_with_ac3(state, all_states):
    if is_complete(state):
        all_states.append(state.copy())
        return True

    row, col = find_unassigned(state)
    if row is None:
        return False

    values = DOMAIN.copy()
    np.random.shuffle(values)

    for value in values:
        if is_consistent(state, row, col, value):
            state[row, col] = value
            all_states.append(state.copy()) 

            if backtracking_with_ac3(state, all_states):
                return True

            state[row, col] = 0
            all_states.append(state.copy()) 
    return False

def get_all_states_csp_ac3(start_state, goal_stateSS):
    all_states = []
    backtracking_with_ac3(start_state, all_states)
    return all_states

###############################################################################################################

def draw_state(screen, state, pos_x, pos_y, cell_size, color_cell, color_number, size, title=""):
    font = pygame.font.SysFont("Arial", size, bold=True)
    for i in range(3):
        for j in range(3):
            value = state[i, j]
            x = j * cell_size + pos_x
            y = i * cell_size + pos_y
            rect = pygame.Rect(x, y, cell_size, cell_size)

            pygame.draw.rect(screen, color_cell, rect)
            pygame.draw.rect(screen, "white", rect, 3)

            if value != 0:
                text = font.render(str(value), True, color_number)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

    # Vẽ tiêu đề dưới ma trận
    text = font.render(title, True, "black")
    title_rect = text.get_rect(center=(pos_x + cell_size * 1.5, pos_y + cell_size * 3 + 15))
    screen.blit(text, title_rect)


def draw_start_state(screen, state):
    draw_state(screen, state, 0, 0, 50, LIGHT_BLUE, "black", 20, "START STATE")
    
def draw_goal_state(screen, state):
    draw_state(screen, state, 200, 0, 50, LIGHT_BLUE, "black", 20, "GOAL STATE")
    
def draw_operating_state(screen, state):
    draw_state(screen, state, 50, 220, 80, (222, 184, 135), (181, 101, 29), 30, "ARRANGING STATE")
    
def draw_choices(screen, mouse):
    font = pygame.font.SysFont("Arial", 16)

    unformed_search = ["BFS", "DFS", "IDS", "UCS"]
    informed_search = ["GREEDY", "A*", "IDA*"]
    local_search = ["SIMPLE HC", "STEEPEST ASCENT HC", "STOCHASTIC HC", "SIMULATED ANNEALING", "BEAM", "GENETIC"]
    search_in_complex_env = ["AND OR", "NO OBSERVATION", "PARTIAL OBSERVATION"]
    CSPs = ["BACKTRACKING", "GENERATE & TEST", "AC-3"]
    Reinforcement_learning = ["Q-learning"]

    algorithm_groups = [
        ("Uninformed Search", unformed_search),
        ("Informed Search", informed_search),
        ("Local Search", local_search),
        ("Search in Complex Env", search_in_complex_env),
        ("CSPs", CSPs),
        ("RL", Reinforcement_learning)
    ]

    list_width, item_height = 180, 30
    highlight = (230, 230, 230)
    shadow = (150, 150, 150)

    for idx, (title, group) in enumerate(algorithm_groups):
        list_x = 400 + idx * (list_width + 20)
        list_y = 40
        total_height = 180
        listbox_rect = pygame.Rect(list_x, list_y, list_width, total_height)
        
        font_title = pygame.font.SysFont("Arial", 20, bold=True)
        font_title.set_underline(True)
        title_surf = font_title.render(title, True, "black")
        title_rect = title_surf.get_rect(center=(list_x + list_width // 2, list_y - 20))
        screen.blit(title_surf, title_rect)
        
        pygame.draw.rect(screen, (220, 220, 220), listbox_rect)

        for j, text in enumerate(group):
            y = list_y + j * item_height
            item_rect = pygame.Rect(list_x, y, list_width, item_height)
            hovered = item_rect.collidepoint(mouse)
            color = (240, 240, 240) if hovered else (220, 220, 220)

            pygame.draw.rect(screen, color, item_rect)

            txt = font.render(text, True, "black")
            screen.blit(txt, txt.get_rect(center=item_rect.center))
        
        pygame.draw.line(screen, shadow, listbox_rect.topleft, listbox_rect.topright, 2)
        pygame.draw.line(screen, shadow, listbox_rect.topleft, listbox_rect.bottomleft, 2)
        pygame.draw.line(screen, highlight, listbox_rect.bottomleft, listbox_rect.bottomright, 2)
        pygame.draw.line(screen, highlight, listbox_rect.topright, listbox_rect.bottomright, 2)


def draw_all_states(screen, solution, page):
    if not solution:
        font = pygame.font.SysFont("Arial", 30)
        return

    font = pygame.font.SysFont("Arial", 16)
    cell_size = 40
    spacing_x = 150
    spacing_y = 160
    start_x = 430
    start_y = 270


    start_idx = page * 21
    end_idx = min(start_idx + 21, len(solution))

    for idx, i in enumerate(range(start_idx, end_idx)):
        state = solution[i]
        row = idx // 7
        col = idx % 7
        pos_x = start_x + col * spacing_x
        pos_y = start_y + row * spacing_y

        for r in range(3):
            for c in range(3):
                value = state[r][c]
                x = pos_x + c * cell_size
                y = pos_y + r * cell_size
                rect = pygame.Rect(x, y, cell_size, cell_size)

                pygame.draw.rect(screen, "black", rect, 1)

                if value != 0:
                    text = font.render(str(value), True, "black")
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

        step_text = font.render(f"Step {i + 1}", True, "black")
        screen.blit(step_text, (pos_x + cell_size, pos_y + cell_size * 3 + 5))

def pagination(screen, mouse):
    previous_btn = pygame.Rect(400, 755, 100, 30)
    next_btn = pygame.Rect(1400, 755, 100, 30)
    buttons = [(previous_btn, "\u2190 PREVIOUS"), (next_btn, "NEXT \u2192")]
    font = pygame.font.SysFont("Arial", 15)
    for btn, text in buttons:
        color = LIGHT_YELLOW if btn.collidepoint(mouse) else DARK_YELLOW
        pygame.draw.rect(screen, color, btn)
        txt = font.render(text, True, "black")
        txt_rect = txt.get_rect(center=btn.center)
        screen.blit(txt, txt_rect)
        
def restart(screen, mouse):
    restart_btn = [(pygame.Rect(50, 500, 100, 40), "RESTART")]
    font = pygame.font.SysFont("Arial", 15)
    for btn, text in restart_btn:
        color = LIGHT_GREEN if btn.collidepoint(mouse) else DARK_GREEN 
        pygame.draw.rect(screen, color, btn)
        txt = font.render(text, True, "black")
        txt_rect = txt.get_rect(center=btn.center)
        screen.blit(txt, txt_rect)
        
def show_path(screen, mouse):
    show_btn = [(pygame.Rect(190, 500, 100, 40), "SHOW PATH")]
    font = pygame.font.SysFont("Arial", 15)
    for btn, text in show_btn:
        color = LIGHT_GREEN if btn.collidepoint(mouse) else DARK_GREEN 
        pygame.draw.rect(screen, color, btn)
        txt = font.render(text, True, "black")
        txt_rect = txt.get_rect(center=btn.center)
        screen.blit(txt, txt_rect)
        
def info(screen, time, cost, node_visited, steps):
    
    time = round(time, 5)
    font = pygame.font.SysFont("Arial", 20, italic = True)
    screen.blit(font.render(f"Time: {time}s", True, "black"), (30, 560))
    screen.blit(font.render(f"Cost: {cost}", True, "black"), (30, 600))
    screen.blit(font.render(f"Node visited: {node_visited}", True, "black"), (30, 640))
    screen.blit(font.render(f"Steps: {steps}", True, "black"), (30, 680))
    
        
def run_algorithm(algorithm, *args):
    if check(*args):
        start_time = time.time()
        result = algorithm(*args)
        end_time = time.time()
        exe_time = end_time - start_time

        # Nếu thuật toán trả về 2 hoặc 3 giá trị
        if isinstance(result, tuple):
            if len(result) == 3:
                path, cost, node_visited = result
                return path, cost, node_visited, exe_time
            elif len(result) == 2:
                path, node_visited = result
                return path, 0, node_visited, exe_time
        return result, 0, 0, exe_time
    else:
        return [], 0, 0, 0

def main():
    pygame.init()
    screen = pygame.display.set_mode((700, 400), pygame.RESIZABLE)
    pygame.display.set_caption("23110354 - Đoàn Quân Tuấn")

    goal_state = np.zeros((3, 3), dtype=int)
    start_state = np.zeros((3, 3), dtype=int)
    
  
    node_visited = 0
    start_value = goal_value = cost = exe_time = i = page = 0
    state = start_state.copy()
    solution = []
    solution1 = []
    path = False
    running = True

    algorithms = {
    (400, 580): [  # Uninformed Search
        (40, 70, bfs),
        (70, 100, dfs),
        (100, 130, ids),
        (130, 160, ucs),
    ],
    (600, 780): [  # Informed Search
        (40, 70, greedy),
        (70, 100, a_star),
        (100, 130, ida_star),
    ],
    (800, 980): [  # Local Search
        (40, 70, simple_hill_climbing),
        (70, 100, steepest_ascent_hill_climbing),
        (100, 130, stochastic_hill_climbing),
        (130, 160, simulated_annealing),
        (160, 190, beam),
        (190, 220, genetic_algorithm),
    ],
    (1000, 1180): [  # Search in Complex Env
        (40, 70, and_or_search),
     #   (100, 130, search_with_partial_observation)
   
    ],
    (1200, 1380): [  # CSPs
        (40, 70, get_all_states_backtracking),
        (70, 100, generate_and_test),
        (100, 130, get_all_states_csp_ac3)
    ],
    }
    
    algorithm_search_no_observation = {(1000, 1180): [
        (70, 100, search_with_no_observation),
        
    ]}

    algorithm_search_partial_observation = {(1000, 1180): [
        (100, 130, search_with_partial_observation),
     ]}

    while running:
        screen.fill("white")
        mouse = pygame.mouse.get_pos()
        draw_start_state(screen, start_state)
        draw_goal_state(screen, goal_state)
        draw_operating_state(screen, state)
        pygame.draw.rect(screen, (181, 101, 29), (400, 250, 1100, 500), 3)
        pygame.draw.rect(screen, "black", (20, 550, 350, 200), 1)
        if solution and len(solution) > i:
            state = solution[i]
           
            i += 1
            time.sleep(0.1)
        else:
            info(screen, exe_time, cost, node_visited, len(solution))

        if path:
            draw_all_states(screen, solution, page)
            
        # if solution1:
        #     draw_belief_state(screen, solution1, initial_belief, page)

        draw_choices(screen, mouse)
        pagination(screen, mouse)
        restart(screen, mouse)
        show_path(screen, mouse)
        

        # if not check(start_state, goal_state):
        #     font = pygame.font.SysFont("Arial", 30)
            

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if x < 150 and y < 150:
                    row, col = y // 50, x // 50
                    start_value += 1
                    if start_value < 9: start_state[row, col] = start_value
                elif 200 < x < 350 and y < 150:
                    row, col = y // 50, (x - 200) // 50
                    goal_value += 1
                    if goal_value < 9: goal_state[row, col] = goal_value
                elif 500 < y < 540:
                    if 50 < x < 150:
                        start_state = np.zeros((3, 3), dtype=int)
                        goal_state = np.zeros((3, 3), dtype=int)
                        state = start_state.copy()
                        start_value = goal_value = cost = exe_time = node_visited = page = i = 0
                        solution = []
                        path = False
                    elif 190 < x < 290:
                        path = True
                elif 755 < y < 785:
                    if 400 < x < 500:
                        page -= 1
                    elif 1400 < x < 1500:
                        page += 1
                else:
                    for x_range, algos in algorithms.items():
                        if x_range[0] < x < x_range[1]:
                            for y_min, y_max, algo in algos:
                                if y_min < y < y_max:
                                    args = (start_state.copy(), goal_state.copy())
                                    solution, cost, node_visited, exe_time = run_algorithm(algo, *args)
                                    break
                    for x_range, algos in algorithm_search_no_observation.items():
                        if x_range[0] < x < x_range[1]:
                            for y_min, y_max, algo in algos:
                                if y_min < y < y_max:
                                    search_with_no_observation()
                    for x_range, algos in algorithm_search_partial_observation.items():
                        if x_range[0] < x < x_range[1]:
                            for y_min, y_max, algo in algos:
                                if y_min < y < y_max:
                                    search_with_partial_observation(tuple(goal_state.flatten()))   
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    