import torch
import torch.nn.functional as F


class paramet_hrsp:

    ROBOT_VELOCITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    DEPOT = torch.tensor([0, 0])

    ROBOT_TYPE_NUM = 2

    ROBOT_NUM = 12

    WEIGHT = 0.5

    ROBOT_NUM_LIST = torch.tensor([4, 8])

    # ROBOT_VELOCITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)
    #
    #
    #
    # ROBOT_NUM = 10
    #
    # WEIGHT = 0.5
    #

    time_norm = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    robot_one_hot = F.one_hot(torch.cat([
        torch.full((n,), i, dtype=torch.long)
        for i, n in enumerate(ROBOT_NUM_LIST)
    ]), num_classes=ROBOT_TYPE_NUM).float().to(device)

    robot_type_indices_list = []
    for i in range(ROBOT_TYPE_NUM):
        robot_type_indices_list.append(torch.full((ROBOT_NUM_LIST[i].item(),), i, dtype=torch.long))

    robot_type_indices = torch.cat(robot_type_indices_list).to(device)

