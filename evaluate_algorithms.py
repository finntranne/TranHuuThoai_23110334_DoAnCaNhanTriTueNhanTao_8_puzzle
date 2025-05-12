# evaluate_algorithms.py
from solver import PuzzleSolver
import copy
from time import perf_counter
import csv
import matplotlib.pyplot as plt
import numpy as np

def get_fixed_test_case():
    return [[1, 8, 2], [0, 4, 3], [7, 6, 5]]

def evaluate_algorithms(test_state, algorithms, num_runs=10):
    results = {alg: {"time": [], "visited": [], "memory": []} for alg in algorithms}
    
    for _ in range(num_runs):
        solver = PuzzleSolver(desState=[[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        for alg in algorithms:
            start_time = perf_counter()
            try:
                if alg in ["Backtracking", "GenerateAndTest", "AC3"]:
                    steps = solver.solve(None, algorithm=alg)
                else:
                    steps = solver.solve(copy.deepcopy(test_state), algorithm=alg)
            except Exception as e:
                print(f"Lỗi với {alg}: {e}")
                steps = []
            end_time = perf_counter()
            
            time_taken = end_time - start_time
            visited_states = solver.visited_count
            max_memory = solver.max_memory
            
            results[alg]["time"].append(time_taken)
            results[alg]["visited"].append(visited_states)
            results[alg]["memory"].append(max_memory)
    
    # Tính trung bình
    for alg in algorithms:
        results[alg]["avg_time"] = sum(results[alg]["time"]) / len(results[alg]["time"])
        results[alg]["avg_visited"] = sum(results[alg]["visited"]) / len(results[alg]["visited"])
        results[alg]["avg_memory"] = sum(results[alg]["memory"]) / len(results[alg]["memory"])
    
    return results

def save_to_csv(results, filename="algorithm_performance_fixed.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Group", "Avg Time (s)", "Avg Visited States", "Avg Memory Usage"])
        for group, algs in algorithm_groups.items():
            for alg in algs:
                if alg in results:
                    writer.writerow([
                        alg,
                        group,
                        results[alg]["avg_time"],
                        results[alg]["avg_visited"],
                        results[alg]["avg_memory"]
                    ])

def plot_performance_group(group, algorithms, results, metric, title, ylabel, filename):
    values = [results[alg][f"avg_{metric}"] for alg in algorithms if alg in results]
    if not values:
        return
    
    plt.figure(figsize=(8, 6))
    x = np.arange(len(algorithms))
    plt.bar(x, values, color='#1f77b4')
    
    plt.xticks(ticks=x, labels=algorithms, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    algorithm_groups = {
        "Uninformed Search": ["BFS", "DFS", "UCS", "IDDFS"],
        "Informed Search": ["GREEDY", "A*", "IDA*"],
        "Local Search": ["SimpleHillClimbing", "SteepestHillClimbing", "StochasticHillClimbing", 
                         "SimulatedAnnealing", "BeamSearch", "GeneticAlgorithm"],
        "CSP": ["Backtracking", "GenerateAndTest", "AC3"],
        "Optimization-Based": ["Q-Learning"]
    }
    algorithms = [alg for group, algs in algorithm_groups.items() for alg in algs]
    
    test_state = get_fixed_test_case()
    results = evaluate_algorithms(test_state, algorithms, num_runs=10)
    

    save_to_csv(results)
    
    for group, algs in algorithm_groups.items():
        plot_performance_group(
            group, algs, results, 
            "time", 
            f"Average Execution Time ({group})", 
            "Time (seconds)", 
            f"execution_time_{group.lower().replace(' ', '_')}.png"
        )
        plot_performance_group(
            group, algs, results, 
            "visited", 
            f"Average Visited States ({group})", 
            "Visited States", 
            f"visited_states_{group.lower().replace(' ', '_')}.png"
        )
        plot_performance_group(
            group, algs, results, 
            "memory", 
            f"Average Memory Usage ({group})", 
            "Memory Usage (states)", 
            f"memory_usage_{group.lower().replace(' ', '_')}.png"
        )
    
    for group, algs in algorithm_groups.items():
        print(f"\nNhóm: {group}")
        for alg in algs:
            if alg in results:
                print(f"  Thuật toán: {alg}")
                print(f"    Thời gian trung bình: {results[alg]['avg_time']:.4f} giây")
                print(f"    Số trạng thái đã thăm trung bình: {results[alg]['avg_visited']:.2f}")
                print(f"    Bộ nhớ sử dụng trung bình: {results[alg]['avg_memory']:.2f}")