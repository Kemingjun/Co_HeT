import pandas as pd
# from Model_Util.util import cal_distance
import json
import os
optimal_solution_dict = {}


# def init_optimal_solution():
#     global optimal_solution_dict
#     with open("C:/Users/13360/Desktop/CHAGV/AHASP_python/Util/optimal_solution_final.json", "r") as file:
#         optimal_solution_dict = json.load(file)
#     pass
class optimalSolution:
    def __init__(self, instance, code, fitness):
        self.instance = instance
        self.code = code
        self.fitness = fitness

    def to_dict(self):
        return {"instance": self.instance, "code": self.code, "fitness": self.fitness}

    @classmethod
    def from_dict(cls, dict_data):
        return cls(dict_data["instance"], dict_data["code"], dict_data["fitness"])

def init_optimal_solution(type=2):
    global optimal_solution_dict
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    json_path = os.path.join(project_root, "result", f"optimal_solution_type_{type}.json")

    with open(json_path, "r", encoding="utf-8") as file:
        optimal_solution_dict = json.load(file)

def update_optimal_solution(type=2):
    global optimal_solution_dict


    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    json_path = os.path.join(project_root, "result", f"optimal_solution_type_{type}.json")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(optimal_solution_dict, file, indent=4, separators=(",", ": "))

def generate_optimal_solution(type=2):
    data_list = []
    for size in [20, 50, 100]:
        for index in range(1, 21):
            instance = f"N{size}_K2_M12_I{index}"
            data_list.append(optimalSolution(instance, [], 10**6))
    data_to_store = {obj.instance: obj.to_dict() for obj in data_list}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    json_path = os.path.join(project_root, "result", f"optimal_solution_type_{type}.json")
    with open(json_path, "w") as file:
        json.dump(data_to_store, file, indent=4)
# generate_optimal_solution()


# init_optimal_solution()


