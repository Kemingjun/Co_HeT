from Util.util import *
from Util.load_data import read_excel
# import bisect
# from Util.Config import Config


class Solution:
    def __init__(self, instance, sequence_map, path_init_task_map):
        self.instance = instance
        self.sequence_map = sequence_map
        self.path_init_task_map = path_init_task_map

        self.distance = None
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
        return path_map

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
            if path_index != Config.ROBOT_NUM :
                code.append(0)
        return code



    def get_fitness(self):
        if self.fitness is not None:
            return self.fitness
        total_distance = 0
        total_tardiness = 0

        enabled_task_list = []
        task_require_robot_state = [[0 for _ in range(Config.ROBOT_TYPE_NUM)] for i in range(self.task_num)]

        for path_index, init_task in self.path_init_task_map.items():
            if init_task != 0:
                robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
                task_require_robot_state[init_task - 1][robot_type - 1] = 1
                self.info_map[init_task][f'robot_{robot_type}_pre_complete_time'] = 0
                if self.instance[init_task - 1][5] == task_require_robot_state[init_task - 1]:
                    enabled_task_list.append(init_task)

        completed_task_num = 0
        while completed_task_num < self.task_num:
            if len(enabled_task_list) == 0:
                self.feasible = False
                self.fitness = 1e6
                return self.fitness
            idx = random.randrange(len(enabled_task_list))
            enabled_task = enabled_task_list.pop(idx)

            source_position = [self.instance[enabled_task - 1][1], self.instance[enabled_task - 1][2]]

            arrival_time_map = {}
            robot_type_next_task_map = {}
            task_distance = 0
            for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
                if self.instance[enabled_task - 1][5][robot_type - 1] == 0:
                    continue
                pre_complete_time = self.info_map[enabled_task][f'robot_{robot_type}_pre_complete_time']
                if self.sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] == 0:
                    pre_position = Config.DEPOT
                else:
                    pre_position = [self.instance[self.sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1][1],
                                       self.instance[self.sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1][2]]

                distance = get_distance(pre_position, source_position)
                travel_time = distance / Config.VELOCITY
                arrival_time = pre_complete_time + travel_time
                task_distance += distance

                arrival_time_map[robot_type] = arrival_time

                self.info_map[enabled_task][f"robot_{robot_type}_arrival_time"] = arrival_time

                next_task = self.sequence_map[enabled_task][f'robot_{robot_type}_next_task']
                if next_task != 0:
                    task_require_robot_state[next_task - 1][robot_type - 1] = 1
                    if task_require_robot_state[next_task - 1] == self.instance[next_task - 1][5]:
                        enabled_task_list.append(next_task)
                    robot_type_next_task_map[robot_type] = next_task

            execute_time = max(arrival_time_map.values())

            complete_time_list = []
            for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
                if self.instance[enabled_task - 1][5][robot_type - 1] == 0:
                    continue
                complete_time = execute_time + self.instance[enabled_task - 1][4][robot_type - 1]
                next_task = self.sequence_map[enabled_task][f'robot_{robot_type}_next_task']
                if next_task != 0:
                    self.info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = complete_time

                complete_time_list.append(complete_time)

            final_complete_time = max(complete_time_list)



            # complete_time = execute_time + self.instance[enabled_task - 1][4]

            self.info_map[enabled_task]["execute_time"] = execute_time
            self.info_map[enabled_task]["complete_time"] = final_complete_time

            # for robot_type, next_task in robot_type_next_task_map.items():
            #     self.info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = complete_time

            task_tardiness = max(final_complete_time - self.instance[enabled_task - 1][3], 0)

            total_distance += task_distance
            total_tardiness += task_tardiness

            completed_task_num += 1

            self.info_map[enabled_task]['distance'] = task_distance
            self.info_map[enabled_task]['tardiness'] = task_tardiness
            self.info_map[enabled_task]['cost'] = task_distance * Config.WEIGHT + task_tardiness * (1 - Config.WEIGHT)

        self.fitness = total_distance * Config.WEIGHT + total_tardiness * (1 - Config.WEIGHT)
        self.distance = total_distance
        self.tardiness = total_tardiness
        self.feasible = True

        return self.fitness


    def get_path_init_task_map(self):
        """
        
        :return:
        """
        return copy_dict_int_int(self.path_init_task_map)

    def get_sequence_map(self):
        """
        
        :return:
        """
        return copy_dict_int_dict(self.sequence_map)



#
#
#
#
#
#
# if __name__ == "__main__":
#     code = [0,1,0,3,5,0,1,0,2,3,0,4,0,5]
#     path_map = code2path_map(code)
#     instance = read_excel("5_20250506152634.xlsx")
#     sequence_map, path_init_task_map = path_map2sequence_map(path_map)
#     solution = Solution(instance, sequence_map, path_init_task_map)
#     get_path_map = solution.get_path_map()
#     solution.get_fitness()
#     pass


