# solver.py
from collections import deque
import heapq
import random
import math
import numpy as np

class PuzzleSolver:
    def __init__(self, desState=None):
        if desState is not None:
            if not self.is_valid_state(desState):
                raise ValueError("desState không hợp lệ: Phải là ma trận 3x3 chứa số từ 0-8, không trùng lặp, và có đúng 1 ô trống.")
            self.desState = desState
        else:
            self.desState = None
        self.visited_count = 0  # Đếm số trạng thái đã thăm
        self.max_memory = 0  # Kích thước tối đa của open list/quần thể

    def reset_metrics(self):
        self.visited_count = 0
        self.max_memory = 0

    def solve(self, startState, algorithm="DFS", observed_positions=None):
        self.reset_metrics()
        if algorithm not in ["Backtracking", "GenerateAndTest", "AC3"] and not self.is_valid_state(startState):
            raise ValueError("startState không hợp lệ: Phải là ma trận 3x3 chứa số từ 0-8, không trùng lặp, và có đúng 1 ô trống.")
        if algorithm == "BFS":
            return self.bfs(startState)
        elif algorithm == "UCS":
            return self.ucs(startState)
        elif algorithm == "DFS":
            return self.dfs(startState)
        elif algorithm == "IDDFS":
            return self.iddfs(startState)
        elif algorithm == "GREEDY":
            return self.GREEDY(startState)
        elif algorithm == "A*":
            return self.astar(startState)
        elif algorithm == "IDA*":
            return self.ida_star(startState)
        elif algorithm == "SimpleHillClimbing":
            return self.simpleHillClimbing(startState)
        elif algorithm == "SteepestHillClimbing":
            return self.steepestHillClimbing(startState)
        elif algorithm == "StochasticHillClimbing":
            return self.stochasticHillClimbing(startState)
        elif algorithm == "SimulatedAnnealing":
            return self.simulatedAnnealing(startState)
        elif algorithm == "BeamSearch":
            return self.beamSearch(startState, beam_width=3)
        elif algorithm == "GeneticAlgorithm":
            return self.genetic_algorithm(startState, population_size=100, generations=500, mutation_rate=0.1, individual_length=50)
        elif algorithm == "And-Or":
            return self.and_or_search(startState)
        elif algorithm == "NoObservationSearch":
            return self.no_observation_search(startState)
        elif algorithm == "Backtracking":
            return self.backtracking()
        elif algorithm == "GenerateAndTest":
            return self.generate_and_test()
        elif algorithm == "AC3":
            return self.ac3()
        elif algorithm == "Q-Learning":
            return self.q_learning(startState)
        elif algorithm == "PartialObservationSearch":
            return self.no_observation_search(startState)  # Giả sử tương tự NoObservationSearch cho đơn giản
        else:
            raise ValueError(f"Thuật toán {algorithm} không được hỗ trợ.")

    def bfs(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        open = deque([(start, [])])
        close = set([tuple(self.flatten(start))])
        self.visited_count += 1
        self.max_memory = max(self.max_memory, len(open) + len(close))
        while open:
            state, path = open.popleft()
            if not self.is_valid_state(state):
                continue
            if self.are_states_equal(state, self.desState):
                return path + [self.copy_state(state)]
            for newState in self.newStates(state):
                newState_tuple = tuple(self.flatten(newState))
                if newState_tuple not in close:
                    close.add(newState_tuple)
                    open.append((self.copy_state(newState), path + [self.copy_state(newState)]))
                    self.visited_count += 1
                    self.max_memory = max(self.max_memory, len(open) + len(close))
        return []

    def dfs(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        open = [(start, [])]
        close = set([tuple(self.flatten(start))])
        self.visited_count += 1
        self.max_memory = max(self.max_memory, len(open) + len(close))
        while open:
            state, path = open.pop()
            if not self.is_valid_state(state):
                continue
            if self.are_states_equal(state, self.desState):
                return path + [self.copy_state(state)]
            for newState in self.newStates(state):
                newState_tuple = tuple(self.flatten(newState))
                if newState_tuple not in close:
                    close.add(newState_tuple)
                    open.append((self.copy_state(newState), path + [self.copy_state(newState)]))
                    self.visited_count += 1
                    self.max_memory = max(self.max_memory, len(open) + len(close))
        return []

    def ucs(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        open = []
        heapq.heappush(open, (0, start, []))
        close = set([tuple(self.flatten(start))])
        self.visited_count += 1
        self.max_memory = max(self.max_memory, len(open) + len(close))
        while open:
            cost, state, path = heapq.heappop(open)
            if not self.is_valid_state(state):
                continue
            if self.are_states_equal(state, self.desState):
                return path + [self.copy_state(state)]
            for newState in self.newStates(state):
                newState_tuple = tuple(self.flatten(newState))
                if newState_tuple not in close:
                    close.add(newState_tuple)
                    heapq.heappush(open, (cost + 1, self.copy_state(newState), path + [self.copy_state(newState)]))
                    self.visited_count += 1
                    self.max_memory = max(self.max_memory, len(open) + len(close))
        return []

    def iddfs(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        depth = 0
        while True:
            result = self.dls(start, depth)
            if result is not None:
                return result
            depth += 1

    def dls(self, start, depth_limit):
        open = [(start, [], 0)]
        close = set([tuple(self.flatten(start))])
        self.visited_count += 1
        self.max_memory = max(self.max_memory, len(open) + len(close))
        while open:
            state, path, depth = open.pop()
            if not self.is_valid_state(state):
                continue
            if self.are_states_equal(state, self.desState):
                return path + [self.copy_state(state)]
            if depth >= depth_limit:
                continue
            for newState in self.newStates(state):
                newState_tuple = tuple(self.flatten(newState))
                if newState_tuple not in close:
                    close.add(newState_tuple)
                    open.append((self.copy_state(newState), path + [self.copy_state(newState)], depth + 1))
                    self.visited_count += 1
                    self.max_memory = max(self.max_memory, len(open) + len(close))
        return []

    def GREEDY(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        open = []
        heapq.heappush(open, (self.manhattan_distance(start), start, []))
        close = set([tuple(self.flatten(start))])
        self.visited_count += 1
        self.max_memory = max(self.max_memory, len(open) + len(close))
        while open:
            _, state, path = heapq.heappop(open)
            if not self.is_valid_state(state):
                continue
            if self.are_states_equal(state, self.desState):
                return path + [self.copy_state(state)]
            for newState in self.newStates(state):
                newState_tuple = tuple(self.flatten(newState))
                if newState_tuple not in close:
                    close.add(newState_tuple)
                    heuristic = self.manhattan_distance(newState)
                    heapq.heappush(open, (heuristic, self.copy_state(newState), path + [self.copy_state(newState)]))
                    self.visited_count += 1
                    self.max_memory = max(self.max_memory, len(open) + len(close))
        return []

    def astar(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        open = []
        heapq.heappush(open, (self.manhattan_distance(start), 0, start, []))
        close = set([tuple(self.flatten(start))])
        self.visited_count += 1
        self.max_memory = max(self.max_memory, len(open) + len(close))
        while open:
            h, g, state, path = heapq.heappop(open)
            if not self.is_valid_state(state):
                continue
            if self.are_states_equal(state, self.desState):
                return path + [self.copy_state(state)]
            for newState in self.newStates(state):
                newState_tuple = tuple(self.flatten(newState))
                if newState_tuple not in close:
                    close.add(newState_tuple)
                    new_g = g + 1
                    new_h = self.manhattan_distance(newState)
                    new_f = new_g + new_h
                    heapq.heappush(open, (new_f, new_g, self.copy_state(newState), path + [self.copy_state(newState)]))
                    self.visited_count += 1
                    self.max_memory = max(self.max_memory, len(open) + len(close))
        return []

    def ida_star(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        def search(state, g, threshold, path, visited):
            h = self.manhattan_distance(state)
            f = g + h
            if f > threshold:
                return None, f
            if self.are_states_equal(state, self.desState):
                return path + [self.copy_state(state)], f
            state_tuple = tuple(self.flatten(state))
            if state_tuple in visited:
                return None, float('inf')
            visited.add(state_tuple)
            self.visited_count += 1
            self.max_memory = max(self.max_memory, len(visited))
            min_exceeded = float('inf')
            for newState in self.newStates(state):
                if not self.is_valid_state(newState):
                    continue
                result, new_f = search(newState, g + 1, threshold, path + [self.copy_state(newState)], visited.copy())
                if result is not None:
                    return result, new_f
                min_exceeded = min(min_exceeded, new_f)
            return None, min_exceeded
        threshold = self.manhattan_distance(start)
        while True:
            result, new_threshold = search(start, 0, threshold, [], set())
            if result is not None:
                return result
            if new_threshold == float('inf'):
                return []
            threshold = new_threshold

    def simpleHillClimbing(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        current_state = self.copy_state(start)
        path = []
        visited = set()
        while True:
            if self.are_states_equal(current_state, self.desState):
                return path
            current_tuple = tuple(self.flatten(current_state))
            if current_tuple in visited:
                return []
            visited.add(current_tuple)
            self.visited_count += 1
            neighbors = self.newStates(current_state)
            self.max_memory = max(self.max_memory, len(neighbors))
            if not neighbors:
                return []
            best_neighbor = None
            best_heuristic = self.manhattan_distance(current_state)
            for neighbor in neighbors:
                neighbor_heuristic = self.manhattan_distance(neighbor)
                if neighbor_heuristic < best_heuristic:
                    best_heuristic = neighbor_heuristic
                    best_neighbor = neighbor
                    break
            if best_neighbor is None:
                return []
            current_state = self.copy_state(best_neighbor)
            path.append(self.copy_state(current_state))

    def steepestHillClimbing(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        current_state = self.copy_state(start)
        path = []
        visited = set()
        while True:
            if self.are_states_equal(current_state, self.desState):
                return path
            current_tuple = tuple(self.flatten(current_state))
            if current_tuple in visited:
                return []
            visited.add(current_tuple)
            self.visited_count += 1
            neighbors = self.newStates(current_state)
            self.max_memory = max(self.max_memory, len(neighbors))
            if not neighbors:
                return []
            best_neighbor = None
            best_heuristic = self.manhattan_distance(current_state)
            for neighbor in neighbors:
                neighbor_heuristic = self.manhattan_distance(neighbor)
                if neighbor_heuristic < best_heuristic:
                    best_heuristic = neighbor_heuristic
                    best_neighbor = neighbor
            if best_neighbor is None:
                return []
            current_state = self.copy_state(best_neighbor)
            path.append(self.copy_state(current_state))

    def stochasticHillClimbing(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        current_state = self.copy_state(start)
        path = []
        visited = set()
        while True:
            if self.are_states_equal(current_state, self.desState):
                return path
            current_tuple = tuple(self.flatten(current_state))
            if current_tuple in visited:
                return []
            visited.add(current_tuple)
            self.visited_count += 1
            neighbors = self.newStates(current_state)
            self.max_memory = max(self.max_memory, len(neighbors))
            if not neighbors:
                return []
            random.shuffle(neighbors)
            found_better = False
            for neighbor in neighbors:
                neighbor_heuristic = self.manhattan_distance(neighbor)
                current_heuristic = self.manhattan_distance(current_state)
                if neighbor_heuristic < current_heuristic:
                    current_state = self.copy_state(neighbor)
                    path.append(self.copy_state(current_state))
                    found_better = True
                    break
            if not found_better:
                return []

    def simulatedAnnealing(self, start, max_restarts=100):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        best_path = None
        best_distance = float('inf')
        actions = ["up", "down", "left", "right"]
        for _ in range(max_restarts):
            current_state = self.copy_state(start)
            path = [self.copy_state(current_state)]
            temperature = 100000
            cooling_rate = 0.99
            min_temperature = 0.01
            max_iterations = 100000
            visited_states = {tuple(self.flatten(current_state))}
            self.visited_count += 1
            iteration = 0
            while temperature > min_temperature and iteration < max_iterations:
                if self.are_states_equal(current_state, self.desState):
                    for i in range(len(path) - 1):
                        valid = False
                        for action in actions:
                            applied_state = self.apply_action(path[i], action)
                            if applied_state is not None and self.are_states_equal(applied_state, path[i + 1]):
                                valid = True
                                break
                        if not valid:
                            break
                    else:
                        return path
                    break
                valid_neighbors = [(self.apply_action(current_state, action), action)
                                  for action in actions
                                  if self.apply_action(current_state, action) is not None]
                valid_neighbors = [(state, action) for state, action in valid_neighbors
                                  if state is not None and tuple(self.flatten(state)) not in visited_states]
                self.visited_count += len(valid_neighbors)
                self.max_memory = max(self.max_memory, len(valid_neighbors))
                if not valid_neighbors:
                    break
                valid_neighbors.sort(key=lambda x: self.manhattan_distance(x[0]))
                top_neighbors = valid_neighbors[:min(3, len(valid_neighbors))]
                next_state, selected_action = random.choice(top_neighbors)
                current_value = self.manhattan_distance(current_state)
                next_value = self.manhattan_distance(next_state)
                delta_e = current_value - next_value
                if delta_e > 0 or random.uniform(0, 1) < math.exp(delta_e / temperature):
                    current_state = self.copy_state(next_state)
                    path.append(self.copy_state(current_state))
                    visited_states.add(tuple(self.flatten(current_state)))
                temperature *= cooling_rate
                iteration += 1
            final_distance = self.manhattan_distance(current_state)
            if final_distance < best_distance:
                best_distance = final_distance
                best_path = path[:]
            if self.are_states_equal(current_state, self.desState):
                for i in range(len(path) - 1):
                    valid = False
                    for action in actions:
                        applied_state = self.apply_action(path[i], action)
                        if applied_state is not None and self.are_states_equal(applied_state, path[i + 1]):
                            valid = True
                            break
                    if not valid:
                        break
                else:
                    return path
        if best_path:
            for i in range(len(best_path) - 1):
                valid = False
                for action in actions:
                    applied_state = self.apply_action(best_path[i], action)
                    if applied_state is not None and self.are_states_equal(applied_state, best_path[i + 1]):
                        valid = True
                        break
                if not valid:
                    return []
            return best_path
        return []

    def beamSearch(self, start, beam_width=2):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        beam = [(self.manhattan_distance(start), self.copy_state(start), [])]
        self.visited_count += 1
        self.max_memory = max(self.max_memory, len(beam))
        while beam:
            current_beam = []
            for _, current_state, current_path in beam:
                if self.are_states_equal(current_state, self.desState):
                    return current_path
                neighbors = self.newStates(current_state)
                self.visited_count += len(neighbors)
                for next_state in neighbors:
                    heuristic = self.manhattan_distance(next_state)
                    new_path = current_path + [self.copy_state(next_state)]
                    current_beam.append((heuristic, self.copy_state(next_state), new_path))
            if not current_beam:
                break
            current_beam.sort()
            beam = current_beam[:beam_width]
            self.max_memory = max(self.max_memory, len(beam))
        return []

    def genetic_algorithm(self, start, population_size=100, generations=500, mutation_rate=0.1, individual_length=50):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        moves = ["up", "down", "left", "right"]

        def generate_individual(length=individual_length):
            return [random.choice(moves) for _ in range(length)]

        def apply_moves(state, moves):
            current_state = self.copy_state(state)
            for move in moves:
                zero_x, zero_y = self.find_zero(current_state)
                if zero_x is None or zero_y is None:
                    break
                new_x, new_y = zero_x, zero_y
                if move == "up" and zero_x > 0:
                    new_x = zero_x - 1
                elif move == "down" and zero_x < 2:
                    new_x = zero_x + 1
                elif move == "left" and zero_y > 0:
                    new_y = zero_y - 1
                elif move == "right" and zero_y < 2:
                    new_y = zero_y + 1
                else:
                    continue
                current_state[zero_x][zero_y], current_state[new_x][new_y] = current_state[new_x][new_y], current_state[zero_x][zero_y]
                self.visited_count += 1
            return current_state

        def fitness(individual):
            state = apply_moves(start, individual)
            manhattan = self.manhattan_distance(state)
            correct_tiles = sum(1 for i in range(3) for j in range(3) 
                               if state[i][j] == self.desState[i][j])
            return manhattan - (correct_tiles * 0.1)

        def crossover(parent1, parent2):
            if len(parent1) != len(parent2):
                return parent1[:], parent2[:]
            point = random.randint(1, len(parent1) - 2)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2

        def mutate(individual, mutation_rate):
            for i in range(len(individual)):
                if random.random() < mutation_rate:
                    individual[i] = random.choice(moves)
            return individual

        def optimize_path(path, start_state):
            current_state = self.copy_state(start_state)
            optimized_moves = []
            for move in path:
                zero_x, zero_y = self.find_zero(current_state)
                if zero_x is None or zero_y is None:
                    break
                new_x, new_y = zero_x, zero_y
                if move == "up" and zero_x > 0:
                    new_x = zero_x - 1
                elif move == "down" and zero_x < 2:
                    new_x = zero_x + 1
                elif move == "left" and zero_y > 0:
                    new_y = zero_y - 1
                elif move == "right" and zero_y < 2:
                    new_y = zero_y + 1
                else:
                    continue
                current_state[zero_x][zero_y], current_state[new_x][new_y] = current_state[new_x][new_y], current_state[zero_x][zero_y]
                optimized_moves.append(move)
                if self.are_states_equal(current_state, self.desState):
                    break
            i = 0
            while i < len(optimized_moves) - 1:
                if ((optimized_moves[i] == "up" and optimized_moves[i+1] == "down") or
                    (optimized_moves[i] == "down" and optimized_moves[i+1] == "up") or
                    (optimized_moves[i] == "left" and optimized_moves[i+1] == "right") or
                    (optimized_moves[i] == "right" and optimized_moves[i+1] == "left")):
                    del optimized_moves[i:i+2]
                    i = max(0, i-1)
                else:
                    i += 1
            return optimized_moves

        population = [generate_individual() for _ in range(population_size)]
        self.max_memory = max(self.max_memory, population_size)
        best_individual = None
        best_fitness = float('inf')
        for generation in range(generations):
            fitness_scores = [(fitness(ind), ind) for ind in population]
            fitness_scores.sort()
            current_best_fitness, current_best_individual = fitness_scores[0]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()
                best_state = apply_moves(start, best_individual)
                if self.are_states_equal(best_state, self.desState):
                    break
            if generation % 50 == 0:
                best_state = apply_moves(start, best_individual)
                if self.are_states_equal(best_state, self.desState):
                    break
            elite_size = population_size // 5
            elites = [ind for _, ind in fitness_scores[:elite_size]]
            new_population = elites.copy()
            while len(new_population) < population_size:
                tournament_size = 5
                parent1 = random.choice([ind for _, ind in random.sample(fitness_scores[:elite_size*2], tournament_size)])
                parent2 = random.choice([ind for _, ind in random.sample(fitness_scores[:elite_size*2], tournament_size)])
                offspring1, offspring2 = crossover(parent1[:], parent2[:])
                offspring1 = mutate(offspring1, mutation_rate)
                offspring2 = mutate(offspring2, mutation_rate)
                new_population.append(offspring1)
                if len(new_population) < population_size:
                    new_population.append(offspring2)
            population = new_population
        if best_individual is None and fitness_scores:
            _, best_individual = fitness_scores[0]
        if best_individual is None:
            return []
        optimized_moves = optimize_path(best_individual, start)
        path = []
        current_state = self.copy_state(start)
        path.append(self.copy_state(current_state))
        for move in optimized_moves:
            zero_x, zero_y = self.find_zero(current_state)
            if zero_x is None or zero_y is None:
                break
            new_x, new_y = zero_x, zero_y
            if move == "up" and zero_x > 0:
                new_x = zero_x - 1
            elif move == "down" and zero_x < 2:
                new_x = zero_x + 1
            elif move == "left" and zero_y > 0:
                new_y = zero_y - 1
            elif move == "right" and zero_y < 2:
                new_y = zero_y + 1
            else:
                continue
            current_state[zero_x][zero_y], current_state[new_x][new_y] = current_state[new_x][new_y], current_state[zero_x][zero_y]
            path.append(self.copy_state(current_state))
            if self.are_states_equal(current_state, self.desState):
                break
        return path

    def no_observation_search(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        belief_set = [self.copy_state(start)]
        path = [[self.copy_state(start)]]
        max_steps = 100
        max_belief_size = 10
        self.visited_count += 1
        self.max_memory = max(self.max_memory, len(belief_set))
        step = 0
        while step < max_steps:
            for state in belief_set:
                if self.are_states_equal(state, self.desState):
                    return path
            belief_set.sort(key=lambda s: self.manhattan_distance(s))
            best_state = belief_set[0]
            astar_path = self.astar(best_state)
            if astar_path and len(astar_path) > 1:
                next_state = astar_path[1]
                zero_x, zero_y = self.find_zero(best_state)
                new_zero_x, new_zero_y = self.find_zero(next_state)
                dx, dy = new_zero_x - zero_x, new_zero_y - zero_y
                actions = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
                best_action = actions.get((dx, dy))
            else:
                best_action = None
                best_avg_distance = float('inf')
                for action in ['up', 'down', 'left', 'right']:
                    total_distance = 0
                    valid_moves = 0
                    for state in belief_set:
                        new_state = self.apply_action(state, action)
                        if new_state and self.is_valid_state(new_state) and self.check(new_state):
                            total_distance += self.manhattan_distance(new_state)
                            valid_moves += 1
                    if valid_moves > 0:
                        avg_distance = total_distance / valid_moves
                        if avg_distance < best_avg_distance:
                            best_avg_distance = avg_distance
                            best_action = action
                if not best_action:
                    return []
            new_belief_set = []
            seen = set()
            for state in belief_set:
                new_state = self.apply_action(state, best_action)
                if new_state and self.is_valid_state(new_state) and self.check(new_state):
                    state_tuple = tuple(self.flatten(new_state))
                    if state_tuple not in seen:
                        new_belief_set.append(new_state)
                        seen.add(state_tuple)
                        self.visited_count += 1
            belief_set = new_belief_set[:max_belief_size]
            self.max_memory = max(self.max_memory, len(belief_set))
            if not belief_set:
                return []
            path.append([self.copy_state(s) for s in belief_set])
            step += 1
        return []

    def backtracking(self):
        self.reset_metrics()
        np.random.seed(0)
        state = np.zeros((3, 3), dtype=int)
        all_states = []
        
        def recursive_backtracking(state, all_states):
            if self.is_complete(state):
                all_states.append(state.tolist())
                self.visited_count += 1
                return True
            row, col = self.find_unassigned(state)
            if row is None:
                return False
            values = list(range(1, 9)) + [0]
            np.random.shuffle(values)
            for value in values:
                if self.is_consistent(state, row, col, value):
                    state[row][col] = value
                    all_states.append(state.tolist())
                    self.visited_count += 1
                    self.max_memory = max(self.max_memory, len(all_states))
                    if recursive_backtracking(state.copy(), all_states):
                        return True
                    state[row][col] = 0
                    all_states.append(state.tolist())
                    self.visited_count += 1
            return False

        recursive_backtracking(state, all_states)
        return all_states

    def generate_and_test(self):
        self.reset_metrics()
        random.seed(0)
        state = [[None for _ in range(3)] for _ in range(3)]
        all_states = []
        state_count = [0]
        order = self.calculate_constraints()
        other_positions = [pos for pos in order if pos != (1, 1) and pos != (2, 2)]
        center_values = random.sample(range(1, 9), 8)

        def backtrack(index):
            all_states.append([[0 if x is None else x for x in row] for row in state])
            state_count[0] += 1
            self.visited_count += 1
            self.max_memory = max(self.max_memory, state_count[0])
            if index == len(other_positions) + 1:
                state[2][2] = 0
                all_states.append([[0 if x is None else x for x in row] for row in state])
                state_count[0] += 1
                self.visited_count += 1
                return True
            if index == 0:
                row, col = (1, 1)
                values = center_values
            else:
                row, col = other_positions[index - 1]
                values = [v for v in range(1, 9) if v not in [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None]]
            for value in values:
                if self.check_constraints(state, row, col, value):
                    state[row][col] = value
                    if backtrack(index + 1):
                        return True
                    state[row][col] = None
            return False

        backtrack(0)
        return all_states

    def ac3(self):
        self.reset_metrics()
        DOMAIN = list(range(1, 9)) + [0]
        np.random.seed(0)
        state = np.zeros((3, 3), dtype=int)
        all_states = []

        def revise(domains, xi, xj):
            revised = False
            for x in domains[xi][:]:
                if all(x == y for y in domains[xj]):
                    domains[xi].remove(x)
                    revised = True
            return revised

        def ac3_algorithm(domains, variables, neighbors):
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

        def backtracking_with_ac3(state, all_states):
            if self.is_complete(state):
                all_states.append(state.tolist())
                self.visited_count += 1
                return True
            row, col = self.find_unassigned(state)
            if row is None:
                return False
            values = DOMAIN.copy()
            np.random.shuffle(values)
            for value in values:
                if self.is_consistent(state, row, col, value):
                    state[row][col] = value
                    all_states.append(state.tolist())
                    self.visited_count += 1
                    self.max_memory = max(self.max_memory, len(all_states))
                    if backtracking_with_ac3(state.copy(), all_states):
                        return True
                    state[row][col] = 0
                    all_states.append(state.tolist())
                    self.visited_count += 1
            return False

        backtracking_with_ac3(state, all_states)
        return all_states

    def q_learning(self, start, episodes=300000, alpha=0.2, gamma=0.95, epsilon=0.2):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        actions = ["up", "down", "left", "right"]
        q_table = {}

        def get_q_value(state, action):
            state_tuple = tuple(self.flatten(state))
            return q_table.get((state_tuple, action), 0.0)

        def choose_action(state, epsilon):
            if random.random() < epsilon:
                return random.choice(actions)
            q_values = [get_q_value(state, a) for a in actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            return random.choice(best_actions) if best_actions else random.choice(actions)

        for episode in range(episodes):
            current_state = self.copy_state(start)
            steps = 0
            max_steps = 300
            while steps < max_steps:
                action = choose_action(current_state, epsilon)
                next_state = self.apply_action(current_state, action)
                state_tuple = tuple(self.flatten(current_state))
                self.visited_count += 1
                self.max_memory = max(self.max_memory, len(q_table))

                if next_state is None:
                    reward = -10
                    q_table[(state_tuple, action)] = get_q_value(current_state, action) + alpha * (reward - get_q_value(current_state, action))
                    break

                next_state_tuple = tuple(self.flatten(next_state))
                reward = 100 if self.are_states_equal(next_state, self.desState) else -1
                max_future_q = max([get_q_value(next_state, a) for a in actions], default=0.0)
                q_table[(state_tuple, action)] = get_q_value(current_state, action) + alpha * (reward + gamma * max_future_q - get_q_value(current_state, action))

                current_state = next_state
                steps += 1

                if self.are_states_equal(current_state, self.desState):
                    break

        path = []
        current_state = self.copy_state(start)
        path.append(self.copy_state(current_state))
        visited = set()
        max_steps = 300
        steps = 0

        while steps < max_steps and not self.are_states_equal(current_state, self.desState):
            state_tuple = tuple(self.flatten(current_state))
            if state_tuple in visited:
                return []
            visited.add(state_tuple)
            self.visited_count += 1
            q_values = [(get_q_value(current_state, a), a) for a in actions]
            _, best_action = max(q_values, key=lambda x: x[0], default=(0, random.choice(actions)))
            next_state = self.apply_action(current_state, best_action)
            if next_state is None:
                return []
            current_state = next_state
            path.append(self.copy_state(current_state))
            steps += 1

        if not self.are_states_equal(current_state, self.desState):
            return []
        return path

    # Các phương thức khác giữ nguyên nhưng cần thêm visited_count và max_memory
    def or_search(self, state, path, depth=0, max_depth=50):
        self.visited_count += 1
        if depth > max_depth:
            return None
        if self.are_states_equal(state, self.desState):
            return []
        state_tuple = tuple(self.flatten(state))
        if state_tuple in path:
            return None
        new_path = path | {state_tuple}
        next_states = self.newStates(state)
        self.max_memory = max(self.max_memory, len(next_states))
        if not next_states:
            return None
        next_states.sort(key=lambda s: self.manhattan_distance(s))
        for next_state in next_states:
            result = self.and_search([next_state], new_path, depth + 1, max_depth)
            if result is not None:
                return [self.copy_state(next_state)] + result
        return None

    def and_search(self, states, path, depth, max_depth):
        if not states:
            return []
        all_results = []
        for state in states:
            result = self.or_search(state, path, depth, max_depth)
            if result is None:
                return None
            all_results.extend(result)
        return all_results

    def and_or_search(self, start):
        self.reset_metrics()
        if not self.check(start) or self.desState is None:
            return []
        max_depth = 50
        initial_path = set()
        result = self.or_search(start, initial_path, 0, max_depth)
        if result is None:
            return []
        return [self.copy_state(start)] + result

    # Các phương thức tiện ích giữ nguyên
    def manhattan_distance(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                value = state[i][j]
                if value != 0:
                    target_i, target_j = self.find_position(value, self.desState)
                    distance += abs(i - target_i) + abs(j - target_j)
        return distance

    def find_position(self, value, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == value:
                    return i, j
        return None, None

    def newStates(self, state):
        newStates = []
        if not self.is_valid_state(state):
            return newStates
        zeroX, zeroY = self.find_zero(state)
        if zeroX is None or zeroY is None:
            return newStates
        moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        for dx, dy in moves:
            x, y = zeroX + dx, zeroY + dy
            if 0 <= x < 3 and 0 <= y < 3:
                newState = self.copy_state(state)
                newState[zeroX][zeroY], newState[x][y] = newState[x][y], newState[zeroX][zeroY]
                newStates.append(newState)
        return newStates

    def check(self, state):
        if not self.is_valid_state(state):
            return False
        tmp = self.flatten(state)
        if sorted(tmp) != list(range(9)):
            return False
        count = sum(
            tmp[i] > tmp[j] and tmp[i] != 0 and tmp[j] != 0
            for i in range(9)
            for j in range(i + 1, 9)
        )
        return count % 2 == 0

    def find_zero(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None, None

    def are_states_equal(self, state1, state2):
        for i in range(3):
            for j in range(3):
                if state1[i][j] != state2[i][j]:
                    return False
        return True

    def copy_state(self, state):
        return [row[:] for row in state]

    def flatten(self, state):
        return [state[i][j] for i in range(3) for j in range(3)]

    def is_valid_state(self, state):
        if not isinstance(state, list) or len(state) != 3:
            return False
        for row in state:
            if not isinstance(row, list) or len(row) != 3:
                return False
        flat_state = self.flatten(state)
        return (all(0 <= num <= 8 for num in flat_state) and
                sorted(flat_state) == list(range(9)) and
                flat_state.count(0) == 1)

    def apply_action(self, state, action):
        zero_x, zero_y = self.find_zero(state)
        if zero_x is None or zero_y is None:
            return None
        new_x, new_y = zero_x, zero_y
        if action == "up" and zero_x > 0:
            new_x = zero_x - 1
        elif action == "down" and zero_x < 2:
            new_x = zero_x + 1
        elif action == "left" and zero_y > 0:
            new_y = zero_y - 1
        elif action == "right" and zero_y < 2:
            new_y = zero_y + 1
        else:
            return None
        new_state = self.copy_state(state)
        new_state[zero_x][zero_y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[zero_x][zero_y]
        return new_state

    def is_complete(self, state):
        return np.count_nonzero(state) == 8

    def is_consistent(self, state, row, col, value):
        if value != 0:
            if value in state:
                return False
            if col > 0 and state[row][col - 1] != 0 and value != state[row][col - 1] + 1:
                return False
            if row > 0 and state[row - 1][col] != 0 and value != state[row - 1][col] + 3:
                return False
        else:
            return np.count_nonzero(state == 0) <= 1
        return True

    def find_unassigned(self, state):
        zeros = np.argwhere(state == 0)
        return zeros[0] if zeros.size > 0 else (None, None)

    def check_constraints(self, state, row, col, value):
        values = [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None and (i, j) != (row, col)]
        if value in values:
            return False
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

    def calculate_constraints(self):
        constraints_count = {
            (0, 0): 2, (0, 1): 3, (0, 2): 2,
            (1, 0): 3, (1, 1): 4, (1, 2): 3,
            (2, 0): 2, (2, 1): 3, (2, 2): 0
        }
        return sorted(constraints_count.keys(), key=lambda k: -constraints_count[k])