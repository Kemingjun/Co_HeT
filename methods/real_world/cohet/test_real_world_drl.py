from pathlib import Path
import sys
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONVENTIONAL_ROOT = ROOT.parent / "real-world-ALNS-Gurobi"
if str(CONVENTIONAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONVENTIONAL_ROOT))

from problems.hrsp.problem_hrsp import HRSPDataset  # noqa: E402
from problems.hrsp.state_hrsp import StateHRSP  # noqa: E402
from reinforce_baselines import get_baseline_dataset_graph_size  # noqa: E402
from Util.load_data import read_excel  # noqa: E402
from Util.solution_convert import action_sequence_to_path_map, path_map_to_solution  # noqa: E402


class RealWorldDRLTest(unittest.TestCase):
    def test_random_dataset_has_real_world_fields(self):
        data = HRSPDataset(size=4, num_samples=2)
        sample = data[0]

        self.assertEqual(sample["supply_raw"].shape, (4, 2))
        self.assertEqual(sample["handover_raw"].shape, (4, 2))
        self.assertEqual(sample["delivery_raw"].shape, (4, 2))
        self.assertEqual(sample["supply_norm"].max().item() <= 1.0, True)
        self.assertTrue(torch.equal(sample["required_robot"], torch.ones(4, 3)))

    def test_random_dataset_generation_is_vectorized_enough_for_training(self):
        start = time.perf_counter()
        data = HRSPDataset(size=10, num_samples=65536)
        elapsed = time.perf_counter() - start
        sample = data[0]

        self.assertEqual(len(data), 65536)
        self.assertEqual(sample["supply_raw"].shape, (10, 2))
        self.assertEqual(sample["required_robot"].shape, (10, 3))
        self.assertLess(elapsed, 5.0)

    def test_excel_dataset_reads_conventional_real_world_schema(self):
        data = HRSPDataset(filename="RW_N10_K3_M16_I1.xlsx")
        sample = data[0]

        self.assertEqual(sample["supply_raw"].shape, (10, 2))
        self.assertEqual(sample["handover_raw"].shape, (10, 2))
        self.assertEqual(sample["delivery_raw"].shape, (10, 2))
        self.assertEqual(sample["deadline"].shape, (10,))

    def test_rollout_baseline_graph_size_check_uses_real_world_fields(self):
        data = HRSPDataset(size=40, num_samples=2)

        self.assertEqual(get_baseline_dataset_graph_size(data), 40)

    def test_state_cost_matches_conventional_evaluator_for_fixed_actions(self):
        sample = HRSPDataset(filename="RW_N10_K3_M16_I1.xlsx")[0]
        batched = {key: value[:2].unsqueeze(0) for key, value in sample.items()}
        state = StateHRSP.initialize(batched)
        actions_zero_based = [
            [0, 0, 4, 12],
            [1, 0, 4, 12],
        ]
        for task, carrier, shuttle, forklift in actions_zero_based:
            selected_task = torch.tensor([task], dtype=torch.long)
            selected_robot_one_hot = torch.zeros(1, 16)
            selected_robot_one_hot[0, carrier] = 1
            selected_robot_one_hot[0, shuttle] = 1
            selected_robot_one_hot[0, forklift] = 1
            state = state.update(selected_task, selected_robot_one_hot)

        instance = read_excel("RW_N10_K3_M16_I1.xlsx")[:2]
        actions_one_based = [[task + 1, carrier + 1, shuttle + 1, forklift + 1] for task, carrier, shuttle, forklift in actions_zero_based]
        path_map = action_sequence_to_path_map(actions_one_based)
        solution = path_map_to_solution(instance, path_map)
        conventional_cost = solution.get_fitness()
        drl_cost = (state.length * 0.4 + state.tardiness * 0.6).item()

        self.assertAlmostEqual(drl_cost, conventional_cost, places=4)
        self.assertAlmostEqual(state.length.item(), solution.distance, places=4)
        self.assertAlmostEqual(state.tardiness.item(), solution.tardiness, places=4)


if __name__ == "__main__":
    unittest.main()


