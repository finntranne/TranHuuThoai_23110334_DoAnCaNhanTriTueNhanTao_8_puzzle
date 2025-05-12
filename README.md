# 🔢 Đồ án cá nhân: 8-Puzzle Solver

## 1. Mục tiêu
Xây dựng một chương trình giải bài toán **8-Puzzle** sử dụng nhiều thuật toán tìm kiếm trong lĩnh vực Trí tuệ nhân tạo, bao gồm các nhóm:
- Tìm kiếm không có thông tin (Uninformed Search).
- Tìm kiếm có thông tin (Informed Search).
- Tìm kiếm cục bộ (Local Search).
- Tìm kiếm trong môi trường phức tạp, không xác định (Belief-State Search).
- Tìm kiếm ràng buộc (Constraint Satisfaction Problem - CSP).
- Học tăng cường (Reinforcement Learning).

Chương trình cung cấp giao diện người dùng để nhập trạng thái ban đầu và đích, hiển thị quá trình giải, và đánh giá hiệu suất của các thuật toán dựa trên thời gian thực thi, số trạng thái đã thăm, và bộ nhớ sử dụng.

---

## 2. Nội dung

### 2.1. Các thuật toán Tìm kiếm không có thông tin: BFS, DFS, UCS, IDDFS

#### Thành phần chính của bài toán tìm kiếm
- **Không gian trạng thái**: Các trạng thái của bảng 3x3, với các số từ 0 đến 8, trong đó 0 là ô trống. Mỗi trạng thái là một cách sắp xếp khác nhau của các ô.
- **Trạng thái ban đầu**: Một trạng thái hợp lệ, ví dụ: `[[1, 8, 2], [0, 4, 3], [7, 6, 5]]`.
- **Trạng thái đích**: Thường là `[[1, 2, 3], [4, 5, 6], [7, 8, 0]]`.
- **Hành động**: Di chuyển ô trống lên, xuống, trái, phải (nếu hợp lệ).
- **Chi phí**: Mỗi bước di chuyển có chi phí là 1 (trừ UCS, chi phí có thể tùy chỉnh).
- **Solution**: Một chuỗi các trạng thái từ trạng thái ban đầu đến trạng thái đích, thể hiện các bước di chuyển hợp lệ.

#### Hình ảnh GIF minh họa
| Thuật toán | GIF |
|------------|-----|
| **BFS** | <img src="images/bfs.gif" width="500" alt="BFS"> |
| **DFS** | <img src="images/dfs.gif" width="500" alt="DFS"> |
| **UCS** | <img src="images/ucs.gif" width="500" alt="UCS"> |
| **IDDFS** | <img src="images/iddfs.gif" width="500" alt="IDDFS"> |

#### Hình ảnh so sánh hiệu suất
| Thời gian | Số trạng thái đã thăm | Bộ nhớ sử dụng |
|-----------|-----------------------|----------------|
| <img src="images/execution_time_uninformed_search.png" width="300" alt="Time"> | <img src="images/visited_states_uninformed_search.png" width="300" alt="Visited States"> | <img src="images/memory_usage_uninformed_search.png" width="300" alt="Memory"> |

#### Nhận xét về hiệu suất
- **BFS**: Đảm bảo tìm được đường ngắn nhất nhưng tốn nhiều bộ nhớ do lưu trữ tất cả trạng thái ở mỗi mức độ sâu. Phù hợp khi không gian trạng thái không quá lớn.
- **DFS**: Nhanh trong việc tìm giải pháp nhưng không đảm bảo đường đi ngắn nhất, dễ bị kẹt ở các nhánh sâu vô hạn nếu không kiểm soát.
- **UCS**: Tương tự BFS nhưng linh hoạt hơn với chi phí tùy chỉnh. Hiệu quả khi cần tối ưu chi phí nhưng vẫn tốn bộ nhớ.
- **IDDFS**: Kết hợp ưu điểm của BFS (đảm bảo đường ngắn nhất) và DFS (tiết kiệm bộ nhớ), hiệu quả cho bài toán 8-Puzzle với không gian trạng thái vừa phải.

---

### 2.2. Các thuật toán Tìm kiếm có thông tin: GREEDY, A*, IDA*

#### Thành phần chính của bài toán tìm kiếm
- Sử dụng hàm heuristic (khoảng cách Manhattan) để ưu tiên các trạng thái gần trạng thái đích hơn.
- **Solution**: Một chuỗi các trạng thái tối ưu hóa chi phí dựa trên heuristic và chi phí thực tế (trong A* và IDA*).

#### Hình ảnh GIF minh họa
| Thuật toán | GIF |
|------------|-----|
| **GREEDY** | <img src="images/greedy.gif" width="500" alt="GREEDY"> |
| **A*** | <img src="images/astar.gif" width="500" alt="A*"> |
| **IDA*** | <img src="images/idastar.gif" width="500" alt="IDA*"> |

#### Hình ảnh so sánh hiệu suất
| Thời gian | Số trạng thái đã thăm | Bộ nhớ sử dụng |
|-----------|-----------------------|----------------|
| <img src="images/execution_time_informed_search.png" width="300" alt="Time"> | <img src="images/visited_states_informed_search.png" width="300" alt="Visited States"> | <img src="images/memory_usage_informed_search.png" width="300" alt="Memory"> |

#### Nhận xét về hiệu suất
- **GREEDY**: Nhanh nhưng không đảm bảo đường đi ngắn nhất, dễ bị kẹt ở các trạng thái không tối ưu do chỉ dựa vào heuristic.
- **A***: Hiệu quả cao, đảm bảo đường đi ngắn nhất với chi phí tối thiểu, nhưng tốn bộ nhớ để lưu trữ danh sách mở.
- **IDA***: Tiết kiệm bộ nhớ hơn A* do sử dụng giới hạn chi phí, nhưng có thể chậm hơn trong một số trường hợp do tìm kiếm lặp lại.

---

### 2.3. Các thuật toán Tìm kiếm cục bộ: SimpleHillClimbing, SteepestHillClimbing, StochasticHillClimbing, SimulatedAnnealing, BeamSearch, GeneticAlgorithm

#### Thành phần chính của bài toán tìm kiếm
- Tìm kiếm dựa trên việc cải thiện dần trạng thái hiện tại, sử dụng heuristic để đánh giá chất lượng trạng thái.
- **Solution**: Một chuỗi các trạng thái cải thiện dần đến trạng thái đích hoặc trạng thái gần tối ưu.

#### Hình ảnh GIF minh họa
| Thuật toán | GIF |
|------------|-----|
| **SimpleHillClimbing** | <img src="images/simplehillclimbing.gif" width="500" alt="Simple HC"> |
| **SteepestHillClimbing** | <img src="images/steepesthillclimbing.gif" width="500" alt="Steepest HC"> |
| **StochasticHillClimbing** | <img src="images/stochastichillclimbing.gif" width="500" alt="Stochastic HC"> |
| **SimulatedAnnealing** | <img src="images/simulatedannealing.gif" width="500" alt="Simulated Annealing"> |
| **BeamSearch** | <img src="images/beamsearch.gif" width="500" alt="Beam Search"> |
| **GeneticAlgorithm** | <img src="images/geneticalgorithm.gif" width="500" alt="Genetic Algorithm"> |

#### Hình ảnh so sánh hiệu suất
| Thời gian | Số trạng thái đã thăm | Bộ nhớ sử dụng |
|-----------|-----------------------|----------------|
| <img src="images/execution_time_local_search.png" width="300" alt="Time"> | <img src="images/visited_states_local_search.png" width="300" alt="Visited States"> | <img src="images/memory_usage_local_search.png" width="300" alt="Memory"> |

#### Nhận xét về hiệu suất
- **SimpleHillClimbing**: Nhanh nhưng dễ kẹt ở cực trị cục bộ, không phù hợp với các bài toán có không gian trạng thái phức tạp.
- **SteepestHillClimbing**: Cải thiện hơn SimpleHillClimbing bằng cách chọn trạng thái lân cận tốt nhất, nhưng vẫn có nguy cơ kẹt.
- **StochasticHillClimbing**: Tăng tính ngẫu nhiên để tránh cực trị cục bộ, hiệu quả hơn trong một số trường hợp.
- **SimulatedAnnealing**: Có khả năng thoát khỏi cực trị cục bộ nhờ cơ chế chấp nhận giải pháp xấu, nhưng tốn thời gian hơn.
- **BeamSearch**: Giới hạn số trạng thái được xem xét, tiết kiệm bộ nhớ nhưng có thể bỏ qua giải pháp tối ưu.
- **GeneticAlgorithm**: Phù hợp với bài toán lớn, nhưng tốn thời gian do cần tạo và tiến hóa quần thể lớn.

---

### 2.4. Các thuật toán Tìm kiếm trong môi trường phức tạp, không xác định: NoObservationSearch, PartialObservationSearch

#### Thành phần chính của bài toán tìm kiếm
- Mô phỏng môi trường mà thông tin về trạng thái không đầy đủ (chỉ quan sát được một số vị trí).
- **Solution**: Một tập hợp các trạng thái tín ngưỡng (belief states) dẫn đến trạng thái đích.

#### Hình ảnh GIF minh họa
| Thuật toán | GIF |
|------------|-----|
| **NoObservationSearch** | <img src="images/noobservationsearch.gif" width="500" alt="No Observation"> |
| **PartialObservationSearch** | <img src="images/partialobservationsearch.gif" width="500" alt="Partial Observation"> |

#### Hình ảnh so sánh hiệu suất
- Do yêu cầu loại bỏ so sánh nhóm Belief-State Search, hiệu suất của các thuật toán này không được đánh giá trong dự án hiện tại.

#### Nhận xét về hiệu suất
- **NoObservationSearch**: Phù hợp khi không có thông tin quan sát, nhưng tốn tài nguyên do phải xem xét nhiều trạng thái tín ngưỡng.
- **PartialObservationSearch**: Hiệu quả hơn khi có thông tin quan sát, nhưng vẫn phức tạp hơn các thuật toán thông thường.

---

### 2.5. Các thuật toán Tìm kiếm ràng buộc: Backtracking, GenerateAndTest, AC3

#### Thành phần chính của bài toán tìm kiếm
- Bài toán được mô hình hóa như một bài toán thỏa mãn ràng buộc (CSP), với các biến là các ô, giá trị từ 0-8, và ràng buộc về tính duy nhất.
- **Solution**: Một trạng thái hợp lệ thỏa mãn tất cả ràng buộc.

#### Hình ảnh GIF minh họa
| Thuật toán | GIF |
|------------|-----|
| **Backtracking** | <img src="images/backtracking.gif" width="500" alt="Backtracking"> |
| **GenerateAndTest** | <img src="images/generateandtest.gif" width="500" alt="GenerateAndTest"> |
| **AC3** | <img src="images/ac3.gif" width="500" alt="AC3"> |

#### Hình ảnh so sánh hiệu suất
| Thời gian | Số trạng thái đã thăm | Bộ nhớ sử dụng |
|-----------|-----------------------|----------------|
| <img src="images/execution_time_csp.png" width="300" alt="Time"> | <img src="images/visited_states_csp.png" width="300" alt="Visited States"> | <img src="images/memory_usage_csp.png" width="300" alt="Memory"> |

#### Nhận xét về hiệu suất
- **Backtracking**: Hiệu quả trong việc tìm giải pháp CSP, nhưng có thể chậm nếu không gian trạng thái lớn.
- **GenerateAndTest**: Đơn giản nhưng không hiệu quả do phải liệt kê tất cả trạng thái có thể.
- **AC3**: Giảm không gian tìm kiếm bằng cách loại bỏ các giá trị không hợp lệ trước, nhưng phức tạp hơn trong triển khai.

---

### 2.6. Học tăng cường: Q-Learning

#### Thành phần chính của bài toán tìm kiếm
- Học chính sách tối ưu thông qua thử và sai, sử dụng bảng Q để lưu trữ giá trị hành động-trạng thái.
- **Solution**: Một chuỗi các hành động dẫn đến trạng thái đích dựa trên chính sách học được.

#### Hình ảnh GIF minh họa
| Thuật toán | GIF |
|------------|-----|
| **Q-Learning** | <img src="images/qlearning.gif" width="500" alt="Q-Learning"> |

#### Hình ảnh so sánh hiệu suất
| Thời gian | Số trạng thái đã thăm | Bộ nhớ sử dụng |
|-----------|-----------------------|----------------|
| <img src="images/execution_time_optimization_based.png" width="300" alt="Time"> | <img src="images/visited_states_optimization_based.png" width="300" alt="Visited States"> | <img src="images/memory_usage_optimization_based.png" width="300" alt="Memory"> |

#### Nhận xét về hiệu suất
- **Q-Learning**: Tốn thời gian để huấn luyện do cần nhiều tập học, nhưng có thể áp dụng cho các bài toán mà không cần mô hình rõ ràng. Phù hợp hơn cho các bài toán lớn hoặc không xác định.

---

## 3. Kết luận
Dự án đã đạt được các kết quả sau:
- **Triển khai thành công 18 thuật toán** thuộc 6 nhóm khác nhau, từ tìm kiếm không có thông tin đến học tăng cường, áp dụng cho bài toán 8-Puzzle.
- **Xây dựng giao diện người dùng** bằng Tkinter, cho phép nhập trạng thái ban đầu và đích, chọn thuật toán, điều chỉnh tốc độ hiển thị, và xem quá trình giải chi tiết.
- **Đánh giá hiệu suất** của các thuật toán dựa trên thời gian thực thi, số trạng thái đã thăm, và bộ nhớ sử dụng, với kết quả được lưu vào file CSV và hiển thị qua các biểu đồ riêng cho từng nhóm.
- **Nhận xét tổng quan**:
  - Các thuật toán như A* và IDA* thường hiệu quả nhất trong việc tìm đường đi ngắn nhất với chi phí hợp lý.
  - Các thuật toán tìm kiếm cục bộ như SimulatedAnnealing và GeneticAlgorithm phù hợp hơn cho các bài toán phức tạp, nhưng tốn thời gian hơn.
  - Các thuật toán CSP hiệu quả khi bài toán được mô hình hóa dưới dạng ràng buộc, nhưng không linh hoạt với trạng thái ban đầu cố định.
  - Q-Learning cho thấy tiềm năng trong các bài toán không xác định, nhưng cần tối ưu thêm để giảm thời gian huấn luyện.
- **Học được từ dự án**: Hiểu sâu hơn về cách áp dụng các thuật toán AI vào bài toán thực tế, kỹ năng lập trình Python, và cách trực quan hóa dữ liệu hiệu suất.

Dự án có thể được mở rộng bằng cách thêm các heuristic mới, tối ưu hóa thuật toán Q-Learning, hoặc tích hợp các thuật toán tìm kiếm hiện đại hơn.

---

## 👨‍💻 Tác giả
**Trần Hữu Thoại**  
MSSV: `23110334`  
Môn: `Trí Tuệ Nhân Tạo`  
Giáo viên hướng dẫn: `Phan Thị Huyền Trang`