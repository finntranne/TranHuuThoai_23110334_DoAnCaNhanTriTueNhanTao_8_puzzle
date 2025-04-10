from collections import deque
import heapq
import random
import math

class PuzzleSolver:
    def __init__(self, desState=None):
        if desState is not None:
            if not self.is_valid_state(desState):
                raise ValueError("desState không hợp lệ: Phải là ma trận 3x3 chứa số từ 0-8, không trùng lặp, và có đúng 1 ô trống.")
            self.desState = desState
        else:
            self.desState = None

    def solve(self, startState, algorithm="DFS"):
        if not self.is_valid_state(startState):
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

    def bfs(self, start):
        if not self.check(start) or self.desState is None:
            return None
        open = deque([(start, [])])
        close = set([tuple(self.flatten(start))])
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
        return None

    def dfs(self, start):
        if not self.check(start) or self.desState is None:
            return None
        open = [(start, [])]
        close = set([tuple(self.flatten(start))])
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
        return None

    def ucs(self, start):
        if not self.check(start) or self.desState is None:
            return None
        open = []
        heapq.heappush(open, (0, start, []))
        close = set([tuple(self.flatten(start))])
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
        return None

    def iddfs(self, start):
        if not self.check(start) or self.desState is None:
            return None
        depth = 0
        while True:
            result = self.dls(start, depth)
            if result is not None:
                return result
            depth += 1

    def dls(self, start, depth_limit):
        open = [(start, [], 0)]
        close = set([tuple(self.flatten(start))])
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
        return None

    def GREEDY(self, start):
        if not self.check(start) or self.desState is None:
            return None
        open = []
        heapq.heappush(open, (self.manhattan_distance(start), start, []))
        close = set([tuple(self.flatten(start))])
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
        return None

    def astar(self, start):
        if not self.check(start) or self.desState is None:
            return None
        open = []
        heapq.heappush(open, (self.manhattan_distance(start), 0, start, []))
        close = set([tuple(self.flatten(start))])
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
        return None

    def ida_star(self, start):
        if not self.check(start) or self.desState is None:
            return None
        def search(state, g, threshold, path, visited):
            h = self.manhattan_distance(state)
            f = g + h
            if f > threshold:
                return None, f 
            if self.are_states_equal(state, self.desState):
                return path + [self.copy_state(state)], f  
            min_exceeded = float('inf') 
            state_tuple = tuple(self.flatten(state))
            if state_tuple in visited:
                return None, min_exceeded
            visited.add(state_tuple)
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
                return None  
            threshold = new_threshold  

    def simpleHillClimbing(self, start):
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
            neighbors = self.newStates(current_state)
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
            neighbors = self.newStates(current_state)
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
            neighbors = self.newStates(current_state)
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


    def simulatedAnnealing(self, start, max_restarts=10):
        if not self.check(start) or self.desState is None:
            return []

        for attempt in range(max_restarts):
            current_state = self.copy_state(start)
            path = [self.copy_state(current_state)]  
            temperature = 1000                 
            cooling_rate = 0.99               
            min_temperature = 0.01
            while temperature > min_temperature:
                if self.are_states_equal(current_state, self.desState):
                    return path
                neighbors = self.newStates(current_state)
                if not neighbors:
                    break  
                neighbors.sort(key=lambda s: self.manhattan_distance(s))
                next_state = random.choice(neighbors[:3]) if len(neighbors) >= 3 else random.choice(neighbors)

                current_value = self.manhattan_distance(current_state)
                next_value = self.manhattan_distance(next_state)
                delta_e = current_value - next_value 
                if delta_e > 0 or random.uniform(0, 1) < math.exp(delta_e / temperature):
                    current_state = self.copy_state(next_state)
                    path.append(self.copy_state(current_state))
                temperature *= cooling_rate
        return []


    def beamSearch(self, start, beam_width=2):
        if not self.check(start) or self.desState is None:
            return []
        beam = [(self.manhattan_distance(start), self.copy_state(start), [])]
        while beam:
            current_beam = []
            for _, current_state, current_path in beam:
                if self.are_states_equal(current_state, self.desState):
                    return current_path
                neighbors = self.newStates(current_state)
                if not neighbors:
                    continue
                for next_state in neighbors:
                    heuristic = self.manhattan_distance(next_state)
                    new_path = current_path + [self.copy_state(next_state)]
                    current_beam.append((heuristic, self.copy_state(next_state), new_path))
            if not current_beam:
                break
            current_beam.sort()
            beam = current_beam[:beam_width]
        return []

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
        current_state_tuple = tuple(self.flatten(state))
        for dx, dy in moves:
            x, y = zeroX + dx, zeroY + dy
            if 0 <= x < 3 and 0 <= y < 3:
                newState = self.copy_state(state)
                newState[zeroX][zeroY], newState[x][y] = newState[x][y], newState[zeroX][zeroY]
                newState_tuple = tuple(self.flatten(newState))
                if newState_tuple != current_state_tuple:
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