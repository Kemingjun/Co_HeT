import torch
from Util.Config import Config

def tensor2path_map(tensor):
    path_map = {robot: [] for robot in range(1, Config.ROBOT_NUM + 1)}
    for task_info in tensor:
        task = task_info[0]
        for robot in task_info[1:]:
            if robot == -1:
                continue
            path_map[int(robot + 1)].append(int(task + 1))
    return path_map


if __name__ == "__main__":
    tensor = torch.tensor([[18,  1,  8],
         [13,  2,  5],
         [ 6,  0,  6],
         [16,  3, 10],
         [ 2,  0,  7],
         [12,  3, 11],
         [ 4,  2,  9],
         [ 7,  1,  8],
         [ 3,  3,  6],
         [ 1,  3, 11],
         [10,  2,  5],
         [15,  0,  7],
         [ 5,  2,  9],
         [11,  1,  8],
         [17,  3, 11],
         [ 9,  2,  5],
         [ 8,  0,  7],
         [19,  0,  7],
         [ 0,  1,  8],
         [14,  2,  5]])
    path_map = tensor2path_map(tensor)
    pass


