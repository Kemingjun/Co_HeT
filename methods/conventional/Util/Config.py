"""Frozen synthetic HRSP configuration with runtime kappa selection."""


class Config:
    DISTANCE_METRIC = "manhattan"
    WEIGHT = 0.5
    DEPOT = [0, 0]
    VELOCITY = 1

    @classmethod
    def configure(cls, kappa: int) -> None:
        if kappa == 2:
            cls.ROBOT_TYPE_NUM = 2
            cls.ROBOT_NUM_LIST = [4, 8]
            cls.INDEX_LIST = [4, 12]
            cls.TYPE_LIST = [[1, 5], [5, 13]]
            cls.ROBOT_NUM = 12
            return
        if kappa == 3:
            cls.ROBOT_TYPE_NUM = 3
            cls.ROBOT_NUM_LIST = [4, 6, 8]
            cls.INDEX_LIST = [4, 10, 18]
            cls.TYPE_LIST = [[1, 5], [5, 11], [11, 19]]
            cls.ROBOT_NUM = 18
            return
        raise ValueError(f"Unsupported kappa {kappa}; expected 2 or 3")


Config.configure(2)
