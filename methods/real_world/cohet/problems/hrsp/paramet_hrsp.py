import torch
import torch.nn.functional as F


class paramet_hrsp:
    ROBOT_TYPE_NUM = 3
    ROBOT_NUM_LIST = torch.tensor([4, 8, 4], dtype=torch.long)
    ROBOT_NUM = int(ROBOT_NUM_LIST.sum().item())

    CARRIER_TYPE = 0
    SHUTTLE_TYPE = 1
    FORKLIFT_TYPE = 2

    WEIGHT = 0.4
    DEPOT_RAW = torch.tensor([0.0, 0.0])
    DEPOT_NORM = torch.tensor([(0.0 + 20.0) / 140.0, 0.0])

    X_OFFSET = 20.0
    X_SCALE = 140.0
    Y_SCALE = 100.0

    CARRIER_VELOCITY = 1.2
    FORKLIFT_VELOCITY = 1.0
    FORKLIFT_PICKUP_TIME = 45.0
    SOURCE_HANDOVER_TIME = 15.0
    CARRIER_SHUTTLE_COUPLING_TIME = 8.0
    CARRIER_SHUTTLE_DECOUPLING_TIME = 8.0
    SHUTTLE_UNLOADING_TIME = 30.0
    DELIVERY_STATION_PROCESSING_TIME = 60.0

    LEFT_SUPPLY_X = -20.0
    RIGHT_SUPPLY_X = 120.0
    LEFT_HANDOVER_X = 0.0
    RIGHT_HANDOVER_X = 100.0
    STATION_Y_SLOTS = torch.arange(5.0, 100.0, 5.0)
    DELIVERY_X_SLOTS = torch.tensor([20.0, 40.0, 60.0, 80.0])
    DEADLINE_BASE = 300.0
    DEADLINE_STEP = 40.0
    DEADLINE_NOISE = 40.0

    time_norm = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    robot_one_hot = F.one_hot(
        torch.cat([torch.full((int(n.item()),), i, dtype=torch.long) for i, n in enumerate(ROBOT_NUM_LIST)]),
        num_classes=ROBOT_TYPE_NUM,
    ).float().to(device)

    robot_type_indices = torch.cat(
        [torch.full((int(n.item()),), i, dtype=torch.long) for i, n in enumerate(ROBOT_NUM_LIST)]
    ).to(device)

