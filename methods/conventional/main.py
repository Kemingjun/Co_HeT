from Util.util import *
from Util.load_data import read_excel
from Util.Solution import Solution
from Util.tensor_util import tensor2path_map
import torch


if __name__ == "__main__":
    # code = [0, 1, 0, 3, 5, 0, 1, 0, 2, 3, 0, 4, 0, 5]
    tensor = torch.tensor([[13,  0,  5],
        [19,  1,  4],
        [17,  0,  3],
        [ 8,  1,  2],
        [18,  0,  5],
        [ 1,  1,  4],
        [ 3,  0,  2],
        [ 0,  1,  3],
        [10,  0,  5],
        [ 2,  0,  4],
        [16,  1,  2],
        [ 4,  1,  5],
        [15,  0,  3],
        [ 7,  1,  2],
        [14,  0,  4],
        [ 9,  1,  5],
        [12,  0,  3],
        [ 5,  1,  4],
        [11,  1,  2],
        [ 6,  0,  5]])
    path_map = tensor2path_map(tensor)

    instance = read_excel("20_20250602215904.xlsx")
    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    solution = Solution(instance, sequence_map, path_init_task_map)
    fitness = solution.get_fitness()
    print(f"fitness:{fitness}")
    pass


