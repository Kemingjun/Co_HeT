# Synthetic conventional baselines

Run from the repository root. `--dry-run` validates the selected workbooks and configuration without calling a solver or writing output:

```bash
conda run -n my310env python scripts/run_conventional.py --solver alns N20_K2_M12_I1 --dry-run
conda run -n my310env python scripts/run_conventional.py --solver gurobi --kappas 2 3 --sizes 10 20 50 --dry-run
```

After reviewing the plan, remove `--dry-run` to solve. Use `--repetitions` explicitly for the desired recorded protocol; the default is one repetition. Matrix selection defaults to instances 1 through 100 in numeric order. `--instance-start` and `--instance-end` select a subset. Each selected cell must contain exactly the canonical 100 workbooks. Supported metaheuristics are `alns`, `iga`, `dabc`, and `diwo`, for kappa 2 or 3 and n=10,20,50,100. The result-producing Gurobi configuration supports n=10,20,50.

The metaheuristic parameters are `iteration_limit=100` and `duration=3600`. The `count <= iteration_limit` condition can complete 101 outer iterations. The time condition is checked between outer iterations (with additional inner checks in IGA/DABC), so it is not a strict wall-clock cutoff; initialization precedes the reported algorithm timer. The independent process timeout is 5400 seconds. Seeds use `202409 + hash((method.upper(), rep)) % 10000000`, with zero-based repetitions and `PYTHONHASHSEED=0` set before each isolated worker starts.

Gurobi uses the c0-c14 model, Big-M=100, Manhattan travel, weights 0.5/0.5, and size-specific solver settings. Its defaults are a 3600-second limit, target MIP gap 0, one thread and seed 0; the external timeout is 4500 seconds. Numerical parameter overrides require `--diagnostic`. Time-limited incumbents are recorded as such. Results include solver status, incumbent, lower bound, achieved MIP gap, runtime and solution count, including no-incumbent outcomes and differences between solver and schedule objectives. The dummy arc for an unused robot is exported as an empty path; no return travel is charged.

Results go to a new timestamped directory under `outputs/conventional`, or a directory specified with `--results-root` and `--run-id`. A run stores its numerical configuration and per-attempt raw JSON and logs. Existing runs require `--resume --run-id NAME` and exactly matching configuration. Successful or terminal Gurobi attempts are skipped; failed and timed-out workers receive a new attempt number. Existing results and logs are retained. The process exits nonzero when an outcome failed, has no incumbent, or fails objective consistency. `methods/conventional/main.py` is a compatibility entrypoint to the same runner.
