"""Numerical constants from the result-producing round-2 Gurobi package."""

from dataclasses import dataclass

BIG_M = 100.0
OBJECTIVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RobotConfig:
    kappa: int
    robot_num_list: tuple[int, ...]
    index_list: tuple[int, ...]
    type_list: tuple[tuple[int, int], ...]
    robot_num: int


def robot_config(kappa: int) -> RobotConfig:
    if kappa == 2:
        return RobotConfig(2, (4, 8), (4, 12), ((1, 5), (5, 13)), 12)
    if kappa == 3:
        return RobotConfig(3, (4, 6, 8), (4, 10, 18), ((1, 5), (5, 11), (11, 19)), 18)
    raise ValueError(f"Unsupported kappa {kappa}; expected 2 or 3")
