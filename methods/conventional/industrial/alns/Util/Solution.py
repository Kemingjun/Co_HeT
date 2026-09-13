from Util.util import (
    Config,
    bisect,
    copy_dict_int_dict,
    copy_dict_int_int,
    copy_dict_int_list,
    evaluate_real_world_schedule,
)


class Solution:
    def __init__(self, instance, sequence_map, path_init_task_map):
        self.instance = instance
        self.sequence_map = sequence_map
        self.path_init_task_map = path_init_task_map

        self.distance = None
        self.travel_time = None
        self.tardiness = None
        self.fitness = None
        self.feasible = None
        self.path_map = None

        self.task_num = len([task for task in self.sequence_map.keys()])
        self.info_map = {task: dict() for task in sequence_map.keys()}
        self.code = self.get_code()
        self.hash_key = hash(tuple(self.code))

    def get_path_map(self):
        if self.path_map is not None:
            return copy_dict_int_list(self.path_map)
        path_map = {}
        for path_index, init_task in self.path_init_task_map.items():
            if init_task == 0:
                path_map[path_index] = []
                continue
            path = [init_task]
            robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
            next_task = self.sequence_map[init_task][f'robot_{robot_type}_next_task']
            while next_task != 0:
                path.append(next_task)
                next_task = self.sequence_map[next_task][f'robot_{robot_type}_next_task']
            path_map[path_index] = path
        self.path_map = path_map
        return copy_dict_int_list(self.path_map)

    def get_code(self):
        code = [0]
        for path_index in range(1, Config.ROBOT_NUM + 1):
            init_task = self.path_init_task_map[path_index]
            if init_task != 0:
                code.append(init_task)
                robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
                next_task = self.sequence_map[init_task][f'robot_{robot_type}_next_task']
                while next_task != 0:
                    code.append(next_task)
                    next_task = self.sequence_map[next_task][f'robot_{robot_type}_next_task']
            if path_index != Config.ROBOT_NUM:
                code.append(0)
        return code

    def get_fitness(self):
        if self.fitness is not None:
            return self.fitness

        fitness, total_distance, total_travel_time, total_tardiness, feasible, info_map = evaluate_real_world_schedule(
            self.instance,
            self.sequence_map,
            self.path_init_task_map,
            self.info_map,
        )
        self.fitness = fitness
        self.distance = total_distance
        self.travel_time = total_travel_time
        self.tardiness = total_tardiness
        self.feasible = feasible
        self.info_map = info_map
        return self.fitness

    def get_path_init_task_map(self):
        return copy_dict_int_int(self.path_init_task_map)

    def get_sequence_map(self):
        return copy_dict_int_dict(self.sequence_map)
