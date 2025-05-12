import pygame
import numpy as np
import random
import logging
from solver import PuzzleSolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FPS = 60

# Display settings
PUZZLES_PER_ROW = 8
PUZZLES_PER_COL = 4
TILE_SIZE = 50
PUZZLE_WIDTH = TILE_SIZE * 3
PUZZLE_HEIGHT = TILE_SIZE * 3
PADDING = 10
WIDTH = PUZZLES_PER_ROW * (PUZZLE_WIDTH + PADDING) + PADDING
HEIGHT = PUZZLES_PER_COL * (PUZZLE_HEIGHT + PADDING) + PADDING + 150

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)  # Màu cam cho trạng thái đích
DARK_GREEN = (0, 200, 0)
LIGHT_GRAY = (200, 200, 200)
FADED_GRAY = (200, 200, 200, 255)  # Màu xám cho trạng thái không hợp lệ

class BeliefSearchUI:
    def __init__(self, root, solver, switch_callback, algorithm, step_delay, observed_positions=None):
        self.root = root
        self.solver = solver
        self.switch_callback = switch_callback
        self.algorithm = algorithm
        self.step_delay = step_delay  # Nhận từ ui.py
        self.observed_positions = observed_positions if observed_positions else []  # Nhận danh sách ô quan sát
        self.des_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.solver.desState = self.des_state
        self.belief_set = []
        self.actual_state = None  # Trạng thái thực tế để mô phỏng quan sát
        self.current_step = 0
        self.is_running = False
        self.immovable_states = set()
        self.guaranteed_idx = 0
        self.last_step_time = 0
        self.current_action = None
        self.prev_action = None
        self.setup_pygame()
        self.initialize_belief_set()

    def setup_pygame(self):
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("8-Puzzle Belief States Visualization")
            self.font = pygame.font.SysFont('arial', 24)
            self.title_font = pygame.font.SysFont('arial', 30, bold=True)
            self.action_font = pygame.font.SysFont('arial', 20)
            self.clock = pygame.time.Clock()
            logger.info("Pygame initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Pygame: {e}")
            raise

    def initialize_belief_set(self):
        self.belief_set = []
        self.immovable_states = set()
        num_states = 32
        max_attempts = 1000
        attempts = 0
        seen_states = set()

        des_state_tuple = tuple(tuple(row) for row in self.des_state)

        if self.algorithm == "PartialObservationSearch":
            initial_state = self.solver.copy_state(self.actual_state) if self.actual_state else self.solver.copy_state(self.des_state)
            state_tuple = tuple(tuple(row) for row in initial_state)

            if state_tuple == des_state_tuple:
                initial_state = self.generate_valid_state(seen_states, des_state_tuple)
                state_tuple = tuple(tuple(row) for row in initial_state)
                logger.info("initial_state là trạng thái đích, đã thay thế bằng trạng thái mới.")

            if (self.solver.check(initial_state) and 
                self.solver.is_valid_state(initial_state) and 
                state_tuple != des_state_tuple):
                self.belief_set.append(initial_state)
                seen_states.add(state_tuple)

            while len(self.belief_set) < num_states and attempts < max_attempts:
                current_state = self.solver.copy_state(self.des_state)
                for _ in range(20):
                    zero_x, zero_y = self.solver.find_zero(current_state)
                    if zero_x is None or zero_y is None:
                        break
                    possible_moves = []
                    if zero_x > 0:
                        possible_moves.append(("up", zero_x - 1, zero_y))
                    if zero_x < 2:
                        possible_moves.append(("down", zero_x + 1, zero_y))
                    if zero_y > 0:
                        possible_moves.append(("left", zero_x, zero_y - 1))
                    if zero_y < 2:
                        possible_moves.append(("right", zero_x, zero_y + 1))
                    if not possible_moves:
                        break
                    move = random.choice(possible_moves)
                    new_x, new_y = move[1], move[2]
                    current_state[zero_x][zero_y], current_state[new_x][new_y] = current_state[new_x][new_y], current_state[zero_x][zero_y]

                state_tuple = tuple(tuple(row) for row in current_state)
                if state_tuple in seen_states or state_tuple == des_state_tuple:
                    attempts += 1
                    continue

                if (self.solver.check(current_state) and 
                    self.solver.is_valid_state(current_state) and
                    all(current_state[i][j] == initial_state[i][j] for i, j in self.observed_positions)):
                    self.belief_set.append(current_state)
                    seen_states.add(state_tuple)
                    logger.debug(f"Generated state {len(self.belief_set)}: {current_state}")
                else:
                    logger.debug(f"Generated state invalid, unsolvable, or mismatched observations: {current_state}")
                attempts += 1

            if state_tuple == des_state_tuple:
                logger.error("actual_state không được là trạng thái đích, tạo trạng thái mới.")
                initial_state = self.generate_valid_state(seen_states, des_state_tuple)
            self.actual_state = initial_state


            if not any(self.solver.are_states_equal(initial_state, s) for s in self.belief_set):
                self.belief_set[0] = initial_state
                seen_states.add(tuple(tuple(row) for row in initial_state))

        else:
            # Khởi tạo cho NoObservationSearch
            while len(self.belief_set) < num_states and attempts < max_attempts:
                current_state = self.solver.copy_state(self.des_state)
                for _ in range(20):
                    zero_x, zero_y = self.solver.find_zero(current_state)
                    if zero_x is None or zero_y is None:
                        break
                    possible_moves = []
                    if zero_x > 0:
                        possible_moves.append(("up", zero_x - 1, zero_y))
                    if zero_x < 2:
                        possible_moves.append(("down", zero_x + 1, zero_y))
                    if zero_y > 0:
                        possible_moves.append(("left", zero_x, zero_y - 1))
                    if zero_y < 2:
                        possible_moves.append(("right", zero_x, zero_y + 1))
                    if not possible_moves:
                        break
                    move = random.choice(possible_moves)
                    new_x, new_y = move[1], move[2]
                    current_state[zero_x][zero_y], current_state[new_x][new_y] = current_state[new_x][new_y], current_state[zero_x][zero_y]

                state_tuple = tuple(tuple(row) for row in current_state)
                if state_tuple in seen_states or state_tuple == des_state_tuple:
                    attempts += 1
                    continue

                if self.solver.check(current_state) and self.solver.is_valid_state(current_state):
                    self.belief_set.append(current_state)
                    seen_states.add(state_tuple)
                    logger.debug(f"Generated state {len(self.belief_set)}: {current_state}")
                else:
                    logger.debug(f"Generated state invalid or unsolvable: {current_state}")
                attempts += 1

            if len(self.belief_set) < num_states:
                logger.warning(f"Could only generate {len(self.belief_set)} states after {max_attempts} attempts")
            self.actual_state = random.choice(self.belief_set) if self.belief_set else self.generate_valid_state(seen_states, des_state_tuple)

        logger.info(f"Actual state initialized: {self.actual_state}")
        self.current_step = 0
        self.is_running = True
        self.last_step_time = pygame.time.get_ticks()
        logger.info(f"Initialized belief set with {len(self.belief_set)} states.")

    def generate_valid_state(self, seen_states, des_state_tuple):
        """Tạo một trạng thái hợp lệ không trùng với trạng thái đích."""
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            current_state = self.solver.copy_state(self.des_state)
            for _ in range(20):
                zero_x, zero_y = self.solver.find_zero(current_state)
                if zero_x is None or zero_y is None:
                    break
                possible_moves = []
                if zero_x > 0:
                    possible_moves.append(("up", zero_x - 1, zero_y))
                if zero_x < 2:
                    possible_moves.append(("down", zero_x + 1, zero_y))
                if zero_y > 0:
                    possible_moves.append(("left", zero_x, zero_y - 1))
                if zero_y < 2:
                    possible_moves.append(("right", zero_x, zero_y + 1))
                if not possible_moves:
                    break
                move = random.choice(possible_moves)
                new_x, new_y = move[1], move[2]
                current_state[zero_x][zero_y], current_state[new_x][new_y] = current_state[new_x][new_y], current_state[zero_x][zero_y]

            state_tuple = tuple(tuple(row) for row in current_state)
            if (state_tuple not in seen_states and 
                state_tuple != des_state_tuple and 
                self.solver.check(current_state) and 
                self.solver.is_valid_state(current_state)):
                seen_states.add(state_tuple)
                return current_state
            attempts += 1
        
        logger.warning("Could not generate a valid state, using fallback state.")
        return [[1, 2, 3], [4, 0, 5], [7, 8, 6]]

    def draw_belief_states(self, title_text):
        self.screen.fill(WHITE)

        title = self.title_font.render(title_text, True, BLACK)
        title_rect = title.get_rect(center=(WIDTH // 2, 25))
        self.screen.blit(title, title_rect)

        if self.current_action:
            action_text = self.action_font.render(f"Action: {self.current_action}", True, BLACK)
            action_rect = action_text.get_rect(center=(WIDTH // 2, 60))
            self.screen.blit(action_text, action_rect)

        for idx, state in enumerate(self.belief_set):
            row = idx // PUZZLES_PER_ROW
            col = idx % PUZZLES_PER_ROW
            offset_x = col * (PUZZLE_WIDTH + PADDING) + PADDING
            offset_y = row * (PUZZLE_HEIGHT + PADDING) + PADDING + 100

            is_goal = self.solver.are_states_equal(state, self.des_state)
            is_immovable = idx in self.immovable_states

            border_color = ORANGE if is_goal else (255, 0, 0)
            pygame.draw.rect(self.screen, border_color, (offset_x, offset_y, PUZZLE_WIDTH, PUZZLE_HEIGHT), 2)

            for i in range(3):
                for j in range(3):
                    x = offset_x + j * TILE_SIZE
                    y = offset_y + i * TILE_SIZE
                    value = state[i][j]
                    if is_goal:
                        color = ORANGE
                    else:
                        color = FADED_GRAY if is_immovable else (GREEN if value == 0 else WHITE)
                    pygame.draw.rect(self.screen, color, (x, y, TILE_SIZE, TILE_SIZE))
                    pygame.draw.rect(self.screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE), 2)
                    # Chỉ vẽ viền đỏ cho ô quan sát trong PartialObservationSearch
                    if self.algorithm == "PartialObservationSearch" and (i, j) in self.observed_positions:
                        pygame.draw.rect(self.screen, (255, 0, 0), (x, y, TILE_SIZE, TILE_SIZE), 4)
                    if value != 0:
                        text = self.font.render(str(value), True, BLACK)
                        text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
                        self.screen.blit(text, text_rect)

        pygame.display.flip()
        return None, None

    def update_belief_set(self, action):
        """
        Cập nhật belief_set dựa trên hành động và quan sát (nếu có).
        """
        new_belief_set = []
        new_immovable_states = set()

        if self.algorithm == "PartialObservationSearch":
            # Áp dụng hành động cho actual_state
            new_actual_state = self.solver.apply_action(self.actual_state, action)
            if new_actual_state and self.solver.is_valid_state(new_actual_state) and self.solver.check(new_actual_state):
                self.actual_state = new_actual_state
            else:
                logger.debug("Actual state became invalid after action, keeping old state.")

            for idx, state in enumerate(self.belief_set):
                if idx in self.immovable_states:
                    new_belief_set.append(state)
                    new_immovable_states.add(idx)
                    continue

                new_state = self.solver.apply_action(state, action)
                if not new_state or not self.solver.is_valid_state(new_state) or not self.solver.check(new_state):
                    new_immovable_states.add(idx)
                    new_belief_set.append(state)
                    logger.debug(f"State {idx} marked as immovable.")
                    continue

                matches_observation = True
                for i, j in self.observed_positions:
                    if new_state[i][j] != self.actual_state[i][j]:
                        matches_observation = False
                        break

                if matches_observation:
                    new_belief_set.append(new_state)
                else:
                    new_immovable_states.add(idx)
                    new_belief_set.append(state)
                    logger.debug(f"State {idx} discarded due to mismatch with observed positions.")

        else:
            # Logic gốc cho NoObservationSearch
            for idx, state in enumerate(self.belief_set):
                if idx in self.immovable_states:
                    new_belief_set.append(state)
                    new_immovable_states.add(idx)
                    continue

                new_state = self.solver.apply_action(state, action)
                if new_state and self.solver.is_valid_state(new_state) and self.solver.check(new_state):
                    new_belief_set.append(new_state)
                else:
                    new_immovable_states.add(idx)
                    new_belief_set.append(state)
                    logger.debug(f"State {idx} marked as immovable.")

        self.belief_set = new_belief_set
        self.immovable_states = new_immovable_states
        logger.info(f"Belief set updated. Size: {len(self.belief_set)}, Immovable states: {len(self.immovable_states)}")
        
    def next_step(self):
        if not self.is_running or not self.belief_set:
            logger.error("Belief set not initialized or invalid.")
            return False, False

        logger.info(f"Starting step {self.current_step + 1} with algorithm {self.algorithm}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.is_running = False
                return False, True

        best_action = None
        best_avg_distance = float('inf')
        movable_states = [s for idx, s in enumerate(self.belief_set) if idx not in self.immovable_states]
        if not movable_states:
            logger.info("No movable states, stopping.")
            self.is_running = False
            return False, True

        if self.algorithm == "PartialObservationSearch":
            # Ưu tiên trạng thái khớp với ô quan sát
            scored_states = []
            for state in movable_states:
                score = sum(1 for i, j in self.observed_positions if state[i][j] == self.actual_state[i][j])
                heuristic = self.solver.manhattan_distance(state)
                scored_states.append((score, heuristic, state))
            scored_states.sort(key=lambda x: (-x[0], x[1]))
            best_state = scored_states[0][2]
        else:
            movable_states.sort(key=lambda s: self.solver.manhattan_distance(s))
            best_state = movable_states[0]

        astar_path = self.solver.astar(best_state)
        if astar_path and len(astar_path) > 1:
            next_state = astar_path[1]
            zero_x, zero_y = self.solver.find_zero(best_state)
            new_zero_x, new_zero_y = self.solver.find_zero(next_state)
            dx, dy = new_zero_x - zero_x, new_zero_y - zero_y
            actions = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
            best_action = actions.get((dx, dy))
            logger.debug(f"A* selected action: {best_action}")

        if not best_action:
            sample_size = min(2, len(movable_states))
            sampled_states = random.sample(movable_states, sample_size) if len(movable_states) > sample_size else movable_states
            for action in ['up', 'down', 'left', 'right']:
                total_distance = 0
                valid_moves = 0
                for state in sampled_states:
                    new_state = self.solver.apply_action(state, action)
                    if new_state and self.solver.is_valid_state(new_state) and self.solver.check(new_state):
                        if self.algorithm == "PartialObservationSearch":
                            matches = sum(1 for i, j in self.observed_positions if new_state[i][j] == self.actual_state[i][j])
                            total_distance += self.solver.manhattan_distance(new_state) - matches
                        else:
                            total_distance += self.solver.manhattan_distance(new_state)
                        valid_moves += 1
                if valid_moves > 0:
                    avg_distance = total_distance / valid_moves
                    if avg_distance < best_avg_distance:
                        best_avg_distance = avg_distance
                        best_action = action
            if not best_action:
                logger.info("No valid action found, stopping.")
                self.is_running = False
                return False, True
            logger.debug(f"Heuristic selected action: {best_action}")

        self.current_action = best_action
        logger.info(f"Applying action: {self.current_action}")

        self.update_belief_set(best_action)

        goal_reached = False
        for idx, state in enumerate(self.belief_set):
            if self.solver.are_states_equal(state, self.des_state):
                self.is_running = False
                goal_reached = True
                logger.info(f"Goal state reached at step {self.current_step}.")
                break

        self.current_step += 1
        return goal_reached, len(self.immovable_states) == len(self.belief_set)

    def run(self):
        running = True
        self.draw_belief_states(
            f"Initial Belief States (ESC to quit, R to restart)"
        )
        self.is_running = True

        while running:
            current_time = pygame.time.get_ticks()
            if self.is_running and (current_time - self.last_step_time >= self.step_delay):
                goal_reached, all_immovable = self.next_step()
                self.last_step_time = current_time
                if goal_reached:
                    self.draw_belief_states(
                        f"Goal State Reached for Belief Set (Press ESC to quit, R to restart)"
                    )
                elif all_immovable:
                    self.draw_belief_states(
                        f"All States Immovable - Stopped (Press ESC to quit, R to restart)"
                    )
                else:
                    self.draw_belief_states(
                        f"Step {self.current_step} (Algorithm: {self.algorithm}, ESC to quit, R to restart)"
                    )

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset()

            self.clock.tick(FPS)

        pygame.quit()
        self.switch_callback()

    def reset(self):
        self.initialize_belief_set()
        self.is_running = True
        self.current_action = None
        self.draw_belief_states(
            f"Initial Belief States (ESC to quit, R to restart)"
        )