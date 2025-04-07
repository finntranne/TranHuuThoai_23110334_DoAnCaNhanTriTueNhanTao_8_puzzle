
# 🔢 Đồ án cá nhân: 8-Puzzle Solver

## 🎯 Mục tiêu
Xây dựng một chương trình giải bài toán **8-Puzzle** sử dụng nhiều thuật toán tìm kiếm khác nhau trong lĩnh vực Trí tuệ nhân tạo.

---

## 🧠 Các thuật toán được triển khai

| Thuật Toán               | Mô Tả                                                                 | Minh Hóa GIF                              |
|--------------------------|----------------------------------------------------------------------|-------------------------------------------|
| **Breadth-First Search (BFS)** | Tìm kiếm theo chiều rộng, đảm bảo đường đi ngắn nhất.             | ![BFS](images/bfs.gif)                   |
| **Depth-First Search (DFS)**   | Tìm kiếm theo chiều sâu, có thể không tìm được đường ngắn nhất.    | ![DFS](images/dfs.gif)                   |
| **Uniform Cost Search (UCS)**  | Tìm kiếm dựa trên chi phí, tương tự BFS nhưng với trọng số.        | ![UCS](images/ucs.gif)                   |
| **Iterative Deepening DFS (IDDFS)** | Kết hợp DFS và giới hạn độ sâu, hiệu quả hơn DFS.                 | ![IDDFS](images/iddfs.gif)               |
| **Greedy Best-First Search**   | Sử dụng heuristic để ưu tiên trạng thái hứa hẹn nhất.             | ![GREEDY](images/greedy.gif)             |
| **A* Search**                 | Kết hợp chi phí và heuristic, tìm đường ngắn nhất hiệu quả.        | ![A*](images/astar.gif)                  |
| **IDA* Search**               | Phiên bản tối ưu của A* với giới hạn chi phí.                     | ![IDA*](images/ida_star.gif)             |
| **Simple Hill Climbing**       | Leo dốc đơn giản, dễ kẹt ở cực trị cục bộ.                       | ![Simple HC](images/simple_hc.gif)       |
| **Steepest Hill Climbing**     | Kiểm tra tất cả lân cận, chọn tốt nhất, nhưng vẫn có thể kẹt.     | ![Steepest HC](images/steepest_hc.gif)   |
| **Stochastic Hill Climbing**   | Leo dốc ngẫu nhiên, tránh cực trị cục bộ tốt hơn.                | ![Stochastic HC](images/stochastic_hc.gif) |
| **Simulated Annealing**        | Sử dụng nhiệt độ để chấp nhận giải pháp xấu, tìm giải toàn cục.    | ![Simulated Annealing](images/sa.gif)    |
| **Beam Search**                | Tìm kiếm chùm, giữ lại một số lượng cố định trạng thái tốt nhất.   | ![Beam Search](images/beam_search.gif)   |

## 👨‍💻 Tác giả

**Tran Huu Thoai**  
MSSV: `23110334`  
Course: `Artificial Intelligence`  

---
