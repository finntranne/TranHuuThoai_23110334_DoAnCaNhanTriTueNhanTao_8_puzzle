import tkinter as tk
from tkinter import ttk, messagebox
from solver import PuzzleSolver
import copy
import time
import random
import threading
from belief_search_ui import BeliefSearchUI

class PuzzleApp:
    def __init__(self, root, solver):
        self.root = root
        self.root.title("8-Puzzle Solver - TranHuuThoai_23110334")
        self.root.geometry("1280x720")
        self.root.resizable(False, False)
        self.solver = solver
        self.state = None
        self.des_state = None
        self.solution_steps = []
        self.current_step = -1
        self.is_running = False
        self.speed = 2000
        self.execution_time = 0
        self.visited_states = set()
        self.cost = 0
        self.observed_positions = []
        self.initial_label = None
        self.belief_ui = None
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Sidebar.TFrame", background="#2C3E50")
        style.configure("Content.TFrame", background="#ECF0F1")
        style.configure("Grid.TFrame", background="#ECF0F1")
        style.configure("StepLog.TFrame", background="#ECF0F1")
        style.configure("Step.TFrame", background="#ECF0F1")
        style.configure("Log.TFrame", background="#ECF0F1")
        style.configure("TButton", font=("Helvetica", 12), padding=10)
        style.configure("Large.TButton", font=("Helvetica", 14, "bold"), padding=12, background="#3498DB", foreground="white")
        style.map("Large.TButton", background=[("active", "#2980B9")])
        style.configure("TLabel", font=("Helvetica", 14), background="#ECF0F1")
        style.configure("TScale", background="#ECF0F1")

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
        self.setup_sidebar()
        self.setup_content_frame()

    def setup_sidebar(self):
        self.sidebar = ttk.Frame(self.main_frame, width=280, style="Sidebar.TFrame")
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        ttk.Label(self.sidebar, text="Menu", font=("Helvetica", 20, "bold"), background="#2C3E50", foreground="white").pack(pady=(20, 30))
        ttk.Label(self.sidebar, text="Algorithm", font=("Helvetica", 16, "bold"), background="#2C3E50", foreground="white").pack(pady=10)
        self.algorithm_var = tk.StringVar(value="BFS")
        self.algorithm_combobox = ttk.Combobox(self.sidebar, textvariable=self.algorithm_var, values=[
            "BFS", "DFS", "UCS", "IDDFS", "GREEDY", "A*", "IDA*",
            "SimpleHillClimbing", "SteepestHillClimbing", "StochasticHillClimbing",
            "SimulatedAnnealing", "BeamSearch", "GeneticAlgorithm",
            "And-Or", "NoObservationSearch", "PartialObservationSearch",
            "Backtracking", "GenerateAndTest", "AC3", "Q-Learning"
        ], state="readonly")
        self.algorithm_combobox.pack(pady=10, padx=20, fill="x")
        self.algorithm_combobox.bind("<<ComboboxSelected>>", self.on_algorithm_select)
        ttk.Label(self.sidebar, text="Speed", font=("Helvetica", 16, "bold"), background="#2C3E50", foreground="white").pack(pady=10)
        self.speed_scale = ttk.Scale(self.sidebar, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_speed_change, length=200)
        self.speed_scale.pack(pady=10, padx=20)
        self.speed_scale.set(0)
        ttk.Button(self.sidebar, text="Solve", command=self.loadState, style="Large.TButton").pack(pady=15, padx=20, fill="x")
        ttk.Button(self.sidebar, text="Stop", command=self.stop, style="Large.TButton").pack(pady=15, padx=20, fill="x")
        ttk.Button(self.sidebar, text="Reset", command=self.reset, style="Large.TButton").pack(pady=15, padx=20, fill="x")
        ttk.Button(self.sidebar, text="Random State", command=self.random_state, style="Large.TButton").pack(pady=15, padx=20, fill="x")

    def setup_content_frame(self):
        if hasattr(self, 'content_frame'):
            self.content_frame.destroy()
        self.content_frame = ttk.Frame(self.main_frame, style="Content.TFrame")
        self.content_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)
        ttk.Label(self.content_frame, text="8-Puzzle Solver", font=("Helvetica", 28, "bold"), foreground="#2C3E50").pack(pady=10)
        self.setup_puzzle_frame()
        self.setup_step_log_frame()
        self.info_frame = ttk.Frame(self.content_frame, style="Content.TFrame")
        self.info_frame.pack(pady=2)
        self.column1_frame = ttk.Frame(self.info_frame, style="Content.TFrame")
        self.column1_frame.pack(side="left", padx=40)
        self.step_label = ttk.Label(self.column1_frame, text="Step: 0", font=("Helvetica", 14, "bold"), foreground="#34495E")
        self.step_label.pack(pady=2)
        self.total_steps_label = ttk.Label(self.column1_frame, text="Total step: 0", font=("Helvetica", 14, "bold"), foreground="#34495E")
        self.total_steps_label.pack(pady=2)
        self.column2_frame = ttk.Frame(self.info_frame, style="Content.TFrame")
        self.column2_frame.pack(side="left", padx=40)
        self.execution_time_label = ttk.Label(self.column2_frame, text="Time: 0s", font=("Helvetica", 14, "bold"), foreground="#34495E")
        self.execution_time_label.pack(pady=2)
        self.cost_label = ttk.Label(self.column2_frame, text="Cost: 0", font=("Helvetica", 14, "bold"), foreground="#34495E")
        self.cost_label.pack(pady=2)

    def setup_puzzle_frame(self):
        self.puzzle_frame = ttk.Frame(self.content_frame, style="Content.TFrame")
        self.puzzle_frame.pack(pady=20)

        # Initial state
        self.start_frame = ttk.Frame(self.puzzle_frame, style="Content.TFrame")
        self.start_frame.pack(side="left", padx=100)
        self.initial_label = ttk.Label(self.start_frame, text="Initial state", font=("Helvetica", 16, "bold"), foreground="#34495E")
        self.initial_label.pack()
        self.start_grid_wrapper = tk.Frame(self.start_frame, bd=2, relief="solid", bg="#ECF0F1")
        self.start_grid_wrapper.pack(pady=5)
        self.start_grid = ttk.Frame(self.start_grid_wrapper, style="Grid.TFrame")
        self.start_grid.pack(padx=2, pady=2)
        self.start_entries = []
        for i in range(3):
            row = []
            for j in range(3):
                entry = tk.Entry(self.start_grid, width=3, font=("Helvetica", 20), justify="center", relief="flat", bg="#FFFFFF", borderwidth=2)
                entry.grid(row=i, column=j, padx=2, pady=2)
                entry.config(highlightthickness=2, highlightbackground="#000000")
                row.append(entry)
            self.start_entries.append(row)
        self.update_initial_state_frame()

        # Target state
        self.end_frame = ttk.Frame(self.puzzle_frame, style="Content.TFrame")
        self.end_frame.pack(side="left", padx=20)
        ttk.Label(self.end_frame, text="Target state", font=("Helvetica", 16, "bold"), foreground="#34495E").pack()
        self.end_grid_wrapper = tk.Frame(self.end_frame, bd=2, relief="solid", bg="#ECF0F1")
        self.end_grid_wrapper.pack(pady=5)
        self.end_grid = ttk.Frame(self.end_grid_wrapper, style="Grid.TFrame")
        self.end_grid.pack(padx=2, pady=2)
        self.end_entries = []
        default_goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        for i in range(3):
            row = []
            for j in range(3):
                entry = tk.Entry(self.end_grid, width=3, font=("Helvetica", 20), justify="center", relief="flat", bg="#FFFFFF", borderwidth=2)
                entry.grid(row=i, column=j, padx=2, pady=2)
                entry.config(highlightthickness=2, highlightbackground="#000000")
                if default_goal[i][j] != 0:
                    entry.insert(0, str(default_goal[i][j]))
                row.append(entry)
            self.end_entries.append(row)
        self.des_state = copy.deepcopy(default_goal)
        self.solver.desState = copy.deepcopy(default_goal)

    def update_initial_state_frame(self):
        algorithm = self.algorithm_var.get()
        csp_algorithms = ["Backtracking", "GenerateAndTest", "AC3"]
        if algorithm == "PartialObservationSearch":
            self.initial_label.config(text="Initial state (Ctrl+Click to mark observed)")
            for i in range(3):
                for j in range(3):
                    self.start_entries[i][j].unbind("<Control-Button-1>")
                    self.start_entries[i][j].bind("<Control-Button-1>", lambda event, x=i, y=j: self.toggle_observed(x, y))
                    if (i, j) in self.observed_positions:
                        self.start_entries[i][j].config(highlightbackground="#FF0000")
                    else:
                        self.start_entries[i][j].config(highlightbackground="#000000")
            self.set_start_entries_state("normal")
        elif algorithm in csp_algorithms:
            self.initial_label.config(text="Initial state (Ignored for CSP)")
            for i in range(3):
                for j in range(3):
                    self.start_entries[i][j].unbind("<Control-Button-1>")
                    self.start_entries[i][j].config(highlightbackground="#000000")
            self.set_start_entries_state("disabled")
            self.observed_positions = []
        else:
            self.initial_label.config(text="Initial state")
            for i in range(3):
                for j in range(3):
                    self.start_entries[i][j].unbind("<Control-Button-1>")
                    self.start_entries[i][j].config(highlightbackground="#000000")
            self.set_start_entries_state("normal")
            self.observed_positions = []

    def toggle_observed(self, i, j):
        pos = (i, j)
        if pos in self.observed_positions:
            self.observed_positions.remove(pos)
            self.start_entries[i][j].config(highlightbackground="#000000")
            self.log_text.insert(tk.END, f"Ô ({i},{j}) đã được bỏ chọn khỏi ô quan sát.\n")
        else:
            self.observed_positions.append(pos)
            self.start_entries[i][j].config(highlightbackground="#FF0000")
            self.log_text.insert(tk.END, f"Ô ({i},{j}) đã được chọn làm ô quan sát.\n")
        self.log_text.insert(tk.END, f"Danh sách ô quan sát hiện tại: {self.observed_positions}\n")
        self.log_text.see(tk.END)

    def setup_step_log_frame(self):
        self.step_log_frame = ttk.Frame(self.content_frame, style="StepLog.TFrame")
        self.step_log_frame.pack(pady=10, fill="both", expand=True)
        self.step_frame = ttk.Frame(self.step_log_frame, style="Step.TFrame")
        self.step_frame.pack(side="left", padx=10)
        ttk.Label(self.step_frame, text="Current state", font=("Helvetica", 16, "bold"), foreground="#34495E").pack(pady=10)
        self.step_grid_wrapper = tk.Frame(self.step_frame, bd=3, relief="solid", bg="#ECF0F1", highlightbackground="#E74C3C", highlightthickness=2)
        self.step_grid_wrapper.pack(pady=2)
        self.step_grid = ttk.Frame(self.step_grid_wrapper)
        self.step_grid.pack(padx=2, pady=2)
        self.step_cells = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                cell = tk.Label(self.step_grid, text="", width=3, font=("Helvetica", 28, "bold"), justify="center", relief="flat", bg="#FFFFFF", highlightthickness=3, highlightbackground="#000000")
                cell.grid(row=i, column=j, padx=5, pady=5)
                self.step_cells[i][j] = cell
        self.log_frame = ttk.Frame(self.step_log_frame, style="Log.TFrame")
        self.log_frame.pack(side="left", fill="both", expand=True, padx=10)
        ttk.Label(self.log_frame, text="Process log", font=("Helvetica", 16, "bold"), foreground="#34495E").pack(pady=10)
        self.log_scrollbar = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL)
        self.log_text = tk.Text(self.log_frame, height=8, width=30, font=("Helvetica", 14), yscrollcommand=self.log_scrollbar.set, bg="#F9E79F", relief="flat", borderwidth=2)
        self.log_scrollbar.config(command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill="both", expand=True)

    def format_matrix(self, matrix):
        return '\n'.join([' '.join(map(str, row)) for row in matrix])

    def find_zero(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None, None

    def get_move_description(self, prev_state, curr_state):
        if prev_state is None:
            return "Trạng thái ban đầu"
        if not self.solver.is_valid_state(prev_state) or not self.solver.is_valid_state(curr_state):
            return "Không di chuyển (Trạng thái không hợp lệ)"
        zeroX_prev, zeroY_prev = self.find_zero(prev_state)
        zeroX_curr, zeroY_curr = self.find_zero(curr_state)
        if zeroX_prev is None or zeroY_prev is None or zeroX_curr is None or zeroY_curr is None:
            return "Không di chuyển (Không tìm thấy ô trống)"
        dx, dy = zeroX_curr - zeroX_prev, zeroY_curr - zeroY_prev
        if dx == -1 and dy == 0:
            return "Di chuyển lên"
        elif dx == 1 and dy == 0:
            return "Di chuyển xuống"
        elif dx == 0 and dy == -1:
            return "Di chuyển trái"
        elif dx == 0 and dy == 1:
            return "Di chuyển phải"
        return "Không di chuyển"

    def on_algorithm_select(self, event):
        algorithm = self.algorithm_var.get()
        self.log_text.insert(tk.END, f"Đã chọn thuật toán: {algorithm}\n")
        
        if algorithm == "PartialObservationSearch":
            if not self.observed_positions:
                self.log_text.insert(tk.END, "Vui lòng chọn ít nhất một ô quan sát bằng Ctrl+Click trước khi giải.\n")
            if self.belief_ui:
                self.switch_to_default()
            self.update_initial_state_frame()
        elif algorithm == "NoObservationSearch":
            if self.belief_ui is None:
                self.content_frame.pack_forget()
                self.belief_ui = BeliefSearchUI(
                    self.root,
                    self.solver,
                    self.switch_to_default,
                    algorithm,
                    step_delay=self.speed,
                    observed_positions=None
                )
                threading.Thread(target=self.belief_ui.run, daemon=True).start()
            else:
                self.belief_ui.algorithm = algorithm
                self.belief_ui.step_delay = self.speed
                self.belief_ui.observed_positions = []
                self.belief_ui.reset()
        else:
            if self.belief_ui:
                self.switch_to_default()
            self.update_initial_state_frame()

    def on_speed_change(self, value):
        value = float(value)
        if value == 100:
            self.speed = 0
            final_speed = "Skip"
        else:
            speed_factor = 1 + (value / 100) * 9
            self.speed = int(2000 / speed_factor)
            final_speed = f"{speed_factor:.1f}x"
        self.root.after_cancel(self.after_id) if hasattr(self, 'after_id') else None
        self.after_id = self.root.after(500, lambda: self.log_text.insert(tk.END, f"Speed: {final_speed}\n"))
        if self.belief_ui:
            self.belief_ui.step_delay = self.speed

    def loadState(self):
        if self.belief_ui:
            return
        algorithm = self.algorithm_var.get()
        csp_algorithms = ["Backtracking", "GenerateAndTest", "AC3"]

        try:
            # Kiểm tra trạng thái đích
            end_numbers = []
            end_empty_count = 0
            for i in range(3):
                row = []
                for j in range(3):
                    value = self.end_entries[i][j].get().strip()
                    if value == "":
                        row.append(0)
                        end_empty_count += 1
                    else:
                        if not value.isdigit():
                            raise ValueError("Chỉ được nhập số từ 1-8 cho Trạng Thái Kết Thúc.")
                        num = int(value)
                        if num < 1 or num > 8:
                            raise ValueError("Số phải từ 1 đến 8 cho Trạng Thái Kết Thúc.")
                        row.append(num)
                end_numbers.append(row)
            if end_empty_count != 1:
                raise ValueError("Trạng Thái Kết Thúc phải có đúng 1 ô trống (không nhập số).")
            flat_end = [num for row in end_numbers for num in row if num != 0]
            if sorted(flat_end) != list(range(1, 9)):
                raise ValueError("Các số trong Trạng Thái Kết Thúc phải từ 1 đến 8 và không trùng nhau, ô trống để trống.")
            self.des_state = copy.deepcopy(end_numbers)
            self.solver.desState = copy.deepcopy(self.des_state)

            # Kiểm tra trạng thái khởi tạo (bỏ qua nếu là CSP)
            if algorithm not in csp_algorithms:
                start_numbers = []
                start_empty_count = 0
                all_empty = True
                for i in range(3):
                    row = []
                    for j in range(3):
                        value = self.start_entries[i][j].get().strip()
                        if value == "":
                            row.append(0)
                            start_empty_count += 1
                        else:
                            all_empty = False
                            if not value.isdigit():
                                raise ValueError("Chỉ được nhập số từ 0-8 hoặc để trống (0) cho Trạng Thái Ban Đầu.")
                            num = int(value)
                            if num < 0 or num > 8:
                                raise ValueError("Số phải từ 0 đến 8 cho Trạng Thái Ban Đầu.")
                            row.append(num)
                    start_numbers.append(row)
                
                if all_empty or start_empty_count == 0:
                    default_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
                    start_numbers = default_state
                    self.log_text.insert(tk.END, "Chưa nhập trạng thái ban đầu, sử dụng trạng thái mặc định: [1, 2, 3, 4, 5, 6, 7, 8, 0]\n")
                else:
                    if start_empty_count != 1:
                        raise ValueError("Trạng Thái Ban Đầu phải có đúng 1 ô trống (0).")
                    flat_start = [num for row in start_numbers for num in row]
                    if sorted(flat_start) != list(range(9)):
                        raise ValueError("Các số trong Trạng Thái Ban Đầu phải từ 0 đến 8 và không trùng nhau.")
                self.state = copy.deepcopy(start_numbers)

                if not self.solver.is_valid_state(self.state):
                    raise ValueError("Trạng thái ban đầu không hợp lệ.")
                if not self.solver.check(self.state):
                    raise ValueError("Trạng thái ban đầu không thể giải được.")

            if not self.solver.is_valid_state(self.des_state):
                raise ValueError("Trạng thái đích không hợp lệ.")
            if not self.solver.check(self.des_state):
                raise ValueError("Trạng thái đích không hợp lệ.")

            observed_positions = self.observed_positions if algorithm == "PartialObservationSearch" else None
            if algorithm == "PartialObservationSearch" and not observed_positions:
                raise ValueError("Phải chọn ít nhất một ô quan sát cho PartialObservationSearch (Ctrl+Click trên ô nhập liệu).")

            self.log_text.insert(tk.END, "Đã tải trạng thái hợp lệ. Bắt đầu giải...\n")
            if algorithm in ["NoObservationSearch", "PartialObservationSearch"]:
                self.content_frame.pack_forget()
                self.belief_ui = BeliefSearchUI(
                    self.root,
                    self.solver,
                    self.switch_to_default,
                    algorithm,
                    step_delay=self.speed,
                    observed_positions=observed_positions
                )
                threading.Thread(target=self.belief_ui.run, daemon=True).start()
            else:
                fresh_solver = PuzzleSolver(desState=copy.deepcopy(self.solver.desState))
                steps = fresh_solver.solve(
                    copy.deepcopy(self.state) if algorithm not in csp_algorithms else None,
                    algorithm,
                    copy.deepcopy(observed_positions) if observed_positions else None
                )

                self.solution_steps = steps
                seen_states = set()
                unique_steps = []
                for step in self.solution_steps:
                    step_tuple = tuple(self.solver.flatten(step))
                    if step_tuple not in seen_states:
                        unique_steps.append(step)
                        seen_states.add(step_tuple)
                self.solution_steps = unique_steps
                self.current_step = -1
                self.is_running = True
                self.set_entries_state("disabled")
                self.solvePuzzle()
                self.auto_run_steps()
        except ValueError as e:
            messagebox.showerror("Lỗi", f"Lỗi nhập dữ liệu: {e}")
            self.log_text.insert(tk.END, f"Lỗi: {e}\n")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi không xác định: {str(e)}")
            self.log_text.insert(tk.END, f"Lỗi không xác định: {str(e)}")

    def solvePuzzle(self):
        self.visited_states.clear()
        algorithm = self.algorithm_var.get()
        csp_algorithms = ["Backtracking", "GenerateAndTest", "AC3"]
        if not algorithm:
            messagebox.showerror("Lỗi", "Vui lòng chọn thuật toán!")
            return
        try:
            start_time = time.perf_counter()
            observed_positions = self.observed_positions if algorithm == "PartialObservationSearch" else None
            if algorithm == "PartialObservationSearch" and not observed_positions:
                raise ValueError("Phải chọn ít nhất một ô quan sát cho PartialObservationSearch (Ctrl+Click trên ô nhập liệu).")
            fresh_solver = PuzzleSolver(desState=copy.deepcopy(self.solver.desState))
            steps = fresh_solver.solve(
                copy.deepcopy(self.state) if algorithm not in csp_algorithms else None,
                algorithm,
                copy.deepcopy(observed_positions) if observed_positions else None
            )

            end_time = time.perf_counter()
            self.execution_time = end_time - start_time
            self.execution_time_label.config(text=f"Time: {self.execution_time:}s")
            if steps:
                self.solution_steps = [copy.deepcopy(self.state)] + steps
                seen_states = set()
                unique_steps = []
                for step in self.solution_steps:
                    step_tuple = tuple(self.solver.flatten(step))
                    if step_tuple not in seen_states:
                        unique_steps.append(step)
                        seen_states.add(step_tuple)
                self.solution_steps = unique_steps
                self.log_text.insert(tk.END, f"Đã tìm thấy lời giải bằng {algorithm}!\n")
                self.log_text.insert(tk.END, f"Thời gian thực thi: {self.execution_time:} giây\n")
                total_steps = len(self.solution_steps)
                self.total_steps_label.config(text=f"Total step: {total_steps}")
                cost_algorithms = ["UCS", "GREEDY", "A*", "IDA*", "SimulatedAnnealing"]
                if algorithm in cost_algorithms:
                    self.cost = total_steps
                else:
                    self.cost = 0
                self.cost_label.config(text=f"Cost: {self.cost}")
            else:
                messagebox.showerror("Lỗi", "Không tìm thấy lời giải.")
                self.solution_steps = []
                self.cost = 0
                self.cost_label.config(text="Cost: 0")
        except ValueError as e:
            messagebox.showerror("Lỗi", f"Lỗi khi giải: {str(e)}")
            self.solution_steps = []
            self.cost = 0
            self.cost_label.config(text="Cost: 0")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi giải: {str(e)}")
            self.solution_steps = []
            self.cost = 0
            self.cost_label.config(text="Cost: 0")

    def get_delay(self):
        return self.speed

    def get_unseen_child_states(self, curr_state, child_states):
        algorithm = self.algorithm_var.get()
        hill_climbing_algorithms = ["SimpleHillClimbing", "SteepestHillClimbing", "StochasticHillClimbing", "SimulatedAnnealing"]
        belief_based_algorithms = ["NoObservationSearch", "PartialObservationSearch"]
        csp_algorithms = ["Backtracking", "GenerateAndTest", "AC3"]
        unseen_child_states = []
        if algorithm in csp_algorithms:
            return []
        current_heuristic = self.solver.manhattan_distance(curr_state)
        for child_state in child_states:
            child_tuple = tuple(self.solver.flatten(child_state))
            if algorithm in belief_based_algorithms:
                unseen_child_states.append(child_state)
            else:
                if child_tuple not in self.visited_states:
                    if algorithm in hill_climbing_algorithms:
                        child_heuristic = self.solver.manhattan_distance(child_state)
                        if child_heuristic < current_heuristic:
                            unseen_child_states.append(child_state)
                    else:
                        unseen_child_states.append(child_state)
        return unseen_child_states

    def log_child_states(self, curr_state, unseen_child_states):
        algorithm = self.algorithm_var.get()
        csp_algorithms = ["Backtracking", "GenerateAndTest", "AC3"]
        if algorithm in csp_algorithms:
            return
        if unseen_child_states:
            self.log_text.insert(tk.END, "Các trạng thái con hợp lệ chưa được thăm:\n")
            unseen_child_states.sort(key=lambda s: self.solver.manhattan_distance(s))
            for i, child_state in enumerate(unseen_child_states, 1):
                child_matrix = self.format_matrix(child_state)
                is_chosen = (i == 1 and self.algorithm_var.get() in ["StochasticHillClimbing", "SimpleHillClimbing", "SteepestHillClimbing"])
                self.log_text.insert(tk.END, f"Trạng thái con {i}{'*' if is_chosen else ''}:\n{child_matrix}\n")
        else:
            self.log_text.insert(tk.END, "Không có trạng thái con chưa được thăm.\n")

    def updateStepGrid(self):
        algorithm = self.algorithm_var.get()
        curr_state = self.solution_steps[self.current_step]
        for i in range(3):
            for j in range(3):
                value = curr_state[i][j]
                self.step_cells[i][j].config(text="" if value == 0 else str(value))
                self.step_cells[i][j].config(bg="#2ECC71" if value == 0 else "#FFFFFF")

    def auto_run_steps(self):
        if not self.is_running:
            return
        if not self.solution_steps:
            messagebox.showerror("Lỗi", "Chưa có giải pháp, vui lòng nhập trạng thái hợp lệ.")
            return
        algorithm = self.algorithm_var.get()
        csp_algorithms = ["Backtracking", "GenerateAndTest", "AC3"]
        if self.speed == 0:
            while self.current_step < len(self.solution_steps) - 1 and self.is_running:
                self.current_step += 1
                curr_state = self.solution_steps[self.current_step]
                self.updateStepGrid()
                self.step_label.config(text=f"Step: {self.current_step + 1}")
                prev_state = self.solution_steps[self.current_step - 1] if self.current_step > 0 else None
                curr_state_tuple = tuple(self.solver.flatten(curr_state))
                self.visited_states.add(curr_state_tuple)
                matrix_str = self.format_matrix(curr_state)
                log_entry = f"---\nStep {self.current_step + 1}: Trạng thái hiện tại:\n{matrix_str}\n"
                self.log_text.insert(tk.END, log_entry)
                if algorithm not in csp_algorithms and self.solver.are_states_equal(curr_state, self.solver.desState):
                    self.log_text.insert(tk.END, "---\nĐã giải xong bài toán!\n")
                    messagebox.showinfo("Hoàn thành", "Đã đạt trạng thái đích.")
                    self.set_entries_state("normal")
                    return
                if algorithm not in csp_algorithms:
                    child_states = self.solver.newStates(curr_state)
                    valid_child_states = [state for state in child_states if self.solver.is_valid_state(state)]
                    self.log_child_states(curr_state, valid_child_states)
                self.log_text.see(tk.END)
            if self.is_running:
                self.log_text.insert(tk.END, "---\nĐã hoàn thành quá trình!\n")
                messagebox.showinfo("Hoàn thành", "Đã hoàn thành quá trình tìm kiếm.")
            self.set_entries_state("normal")
        else:
            if self.current_step < len(self.solution_steps) - 1 and self.is_running:
                self.current_step += 1
                curr_state = self.solution_steps[self.current_step]
                self.updateStepGrid()
                self.step_label.config(text=f"Step: {self.current_step + 1}")
                prev_state = self.solution_steps[self.current_step - 1] if self.current_step > 0 else None
                curr_state_tuple = tuple(self.solver.flatten(curr_state))
                self.visited_states.add(curr_state_tuple)
                matrix_str = self.format_matrix(curr_state)
                log_entry = f"---\nStep {self.current_step + 1}: Trạng thái hiện tại:\n{matrix_str}\n"
                self.log_text.insert(tk.END, log_entry)
                if algorithm not in csp_algorithms and self.solver.are_states_equal(curr_state, self.solver.desState):
                    self.log_text.insert(tk.END, "---\nĐã giải xong bài toán!\n")
                    messagebox.showinfo("Hoàn thành", "Đã đạt trạng thái đích.")
                    self.set_entries_state("normal")
                    return
                if algorithm not in csp_algorithms:
                    child_states = self.solver.newStates(curr_state)
                    valid_child_states = [state for state in child_states if self.solver.is_valid_state(state)]
                    self.log_child_states(curr_state, valid_child_states)
                self.log_text.see(tk.END)
                delay = self.get_delay()
                self.root.after(delay, self.auto_run_steps)
            else:
                if self.is_running:
                    self.log_text.insert(tk.END, "---\nĐã hoàn thành quá trình!\n")
                    messagebox.showinfo("Hoàn thành", "Đã hoàn thành quá trình tìm kiếm.")
                self.set_entries_state("normal")

    def reset(self):
        self.is_running = False
        if self.belief_ui:
            self.belief_ui.reset()
            return
        for i in range(3):
            for j in range(3):
                # self.start_entries[i][j].delete(0, tk.END)
                # self.start_entries[i][j].config(highlightbackground="#000000")
                self.step_cells[i][j].config(text="")
                self.step_cells[i][j].config(bg="white")
        self.observed_positions = []
        self.log_text.delete(1.0, tk.END)
        self.state = None
        self.solution_steps = []
        self.current_step = -1
        self.execution_time = 0
        self.visited_states.clear()
        self.cost = 0
        self.step_label.config(text="Step: 0")
        self.total_steps_label.config(text="Total step: 0")
        self.execution_time_label.config(text="Time: 0s")
        self.cost_label.config(text="Cost: 0")
        self.speed_scale.set(0)
        self.set_entries_state("normal")
        self.update_initial_state_frame()

    def random_state(self):
        if self.belief_ui:
            self.belief_ui.reset()
            return
        algorithm = self.algorithm_var.get()
        csp_algorithms = ["Backtracking", "GenerateAndTest", "AC3"]
        if algorithm in csp_algorithms:
            self.log_text.insert(tk.END, "Tạo trạng thái ngẫu nhiên bị bỏ qua cho thuật toán CSP.\n")
            return
        numbers = list(range(9))
        while True:
            random.shuffle(numbers)
            state = [numbers[i:i+3] for i in range(0, 9, 3)]
            if self.solver.check(state):
                break
        for i in range(3):
            for j in range(3):
                self.start_entries[i][j].delete(0, tk.END)
                self.start_entries[i][j].config(highlightbackground="#000000")
        self.observed_positions = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    self.start_entries[i][j].insert(0, "")
                else:
                    self.start_entries[i][j].insert(0, str(state[i][j]))
        self.state = copy.deepcopy(state)
        self.log_text.insert(tk.END, "Đã tạo trạng thái ban đầu ngẫu nhiên có thể giải được.\n")
        self.update_initial_state_frame()

    def stop(self):
        self.is_running = False
        if self.belief_ui:
            self.switch_to_default()
        else:
            self.log_text.insert(tk.END, "---\nĐã dừng bởi người dùng.\n")
            self.set_entries_state("normal")

    def set_entries_state(self, state):
        algorithm = self.algorithm_var.get()
        csp_algorithms = ["Backtracking", "GenerateAndTest", "AC3"]
        for i in range(3):
            for j in range(3):
                self.end_entries[i][j].config(state=state)
                if algorithm not in csp_algorithms:
                    self.start_entries[i][j].config(state=state)
                else:
                    self.start_entries[i][j].config(state="disabled")

    def set_start_entries_state(self, state):
        for i in range(3):
            for j in range(3):
                self.start_entries[i][j].config(state=state)

    def switch_to_default(self):
        if self.belief_ui:
            self.belief_ui = None
        self.content_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)
        self.setup_content_frame()
        self.update_initial_state_frame()