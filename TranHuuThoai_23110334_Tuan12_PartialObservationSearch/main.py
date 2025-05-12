import tkinter as tk
from solver import PuzzleSolver
from ui import PuzzleApp

if __name__ == "__main__":
    root = tk.Tk()
    solver = PuzzleSolver()
    app = PuzzleApp(root, solver)
    root.mainloop()