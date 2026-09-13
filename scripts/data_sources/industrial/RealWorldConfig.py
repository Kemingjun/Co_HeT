class RealWorldConfig:
    # Robot type order:
    # 1: carrier, 2: shuttle, 3: autonomous forklift
    ROBOT_TYPE_NUM = 3

    # Fleet setting for the industrial simulation case study.
    ROBOT_NUM_LIST = [4, 8, 4]
    INDEX_LIST = [4, 12, 16]
    TYPE_LIST = [[1, 5], [5, 13], [13, 17]]
    ROBOT_NUM = 16

    # Instance generation does not use this weight.
    # Benchmark objectives are defined in the evaluation configuration.
    WEIGHT = 0.5

    # Default shared depot used by the existing conventional code.
    DEPOT = [0, 0]

    # Position naming
    SUPPLY_BUFFER_NAME = "Supply Buffer"
    HANDOVER_STATION_NAME = "Handover Station"
    DELIVERY_STATION_NAME = "Delivery Station"

    # Industrial simulation layout parameters in meters.
    MAIN_X_MIN = 0
    MAIN_X_MAX = 100
    MAIN_Y_MIN = 0
    MAIN_Y_MAX = 100
    LEFT_SUPPLY_X = -20
    RIGHT_SUPPLY_X = 120
    LEFT_HANDOVER_X = 0
    RIGHT_HANDOVER_X = 100
    STATION_Y_SLOTS = list(range(5, 100, 5))
    DELIVERY_X_SLOTS = [20, 40, 60, 80]

    # Industrial simulation instance generation settings.
    REAL_WORLD_INSTANCE_SIZES = [10, 20, 40, 60]
    REAL_WORLD_INSTANCE_COUNT = 20  # generate_batch default; the test100 entry selects 100.
    DEADLINE_BASE = 300
    DEADLINE_STEP = 40
    DEADLINE_NOISE = 40

    # Robot type mapping
    CARRIER_TYPE = 1
    SHUTTLE_TYPE = 2
    FORKLIFT_TYPE = 3

    # Fixed operation times in seconds.
    # Parameterized from on-site data for the industrial simulation.
    FORKLIFT_PICKUP_TIME = 45
    FORKLIFT_HANDOVER_POSITIONING_TIME = 20
    SOURCE_HANDOVER_TIME = 15
    CARRIER_SHUTTLE_COUPLING_TIME = 8.0
    CARRIER_SHUTTLE_DECOUPLING_TIME = 8.0
    SHUTTLE_LOADING_TIME = 30.0
    SHUTTLE_UNLOADING_TIME = 30.0
    # Fixed shuttle operation time at the delivery station. Shuttle travel is
    # not modeled explicitly.
    DELIVERY_STATION_PROCESSING_TIME = 60.0

    # Type-specific velocities in m/s.
    CARRIER_VELOCITY = 1.2
    # Forklift speed used in the industrial simulation.
    FORKLIFT_VELOCITY = 1.0

    # Expected extended instance columns.
    TASK_INDEX_COL = "task_index"
    SUPPLY_X_COL = "supply_x"
    SUPPLY_Y_COL = "supply_y"
    HANDOVER_X_COL = "handover_x"
    HANDOVER_Y_COL = "handover_y"
    DELIVERY_X_COL = "delivery_x"
    DELIVERY_Y_COL = "delivery_y"
    DEADLINE_COL = "deadline"
    REQUIRED_ROBOT_COL = "required_robot"
