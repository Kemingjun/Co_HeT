# Industrial simulation case study

This industrial simulation case study is based on a battery-plate transfer scenario. Each task requires one carrier, one shuttle and one forklift from the fleet of 4 carriers, 8 shuttles and 4 forklifts. Webots illustrates the workflow; the quantitative benchmark uses the industrial scheduling simulator. Results characterize performance under the modeled conditions, without physical-robot deployment validation.

## Settings

| Setting | Value |
|---|---|
| Task scales | 10, 20, 30, 40; 100 independent test instances per scale |
| Fleet | 4 carriers, 8 shuttles, 4 forklifts |
| Travel | Manhattan distance; open routes from depot (0,0) |
| Carrier / forklift speed | 1.2 / 1.0 m/s |
| Forklift pickup / handover | 45 / 15 s |
| Carrier–shuttle coupling / decoupling | 8 / 8 s |
| Shuttle unloading / station processing | 30 / 60 s |
| Objective | 0.5 × total travel time + 0.5 × total tardiness |
| Test generation seed | 20260906; instance seed = master seed + 1000 × n + index |
| DRL training / evaluation seed | 1234 / 20260825 |

Carrier travel includes reaching its shuttle, traveling to handover, and delivery. Forklift travel includes reaching supply and then handover. The carried shuttle does not add another travel-time term. Waiting and operation times affect availability and tardiness, rather than travel time. Tardiness uses shuttle release time.

The Co-HeT industrial scheduling environment provides its [parameter configuration](../../../methods/real_world/cohet/problems/hrsp/paramet_hrsp.py) and [state-transition implementation](../../../methods/real_world/cohet/problems/hrsp/state_hrsp.py). These implement travel, synchronized handover, delivery, robot availability and tardiness in the scheduling workflow.

The industrial layout, robot speeds, and operation times are parameterized using data collected on site at an industrial battery-plate transfer facility. The accompanying photograph illustrates the operational setting. Test instances are generated using these parameters and the documented task-generation rules. The reported performance results are obtained in simulation.

The [test-set description](../../../instances/real_world_test100_seed20260906/README.md) and [generation configuration](../../../instances/real_world_test100_seed20260906/generation_config.json) specify the generated layout and deadlines. Supply and handover positions share a randomly selected side; station positions are sampled from fixed slots. Deadlines follow a task-index rule with bounded noise and are shuffled. The [generator](../../../scripts/generate_industrial_testset.py) produces 100 independent test instances at each of n = 10, 20, 30, 40, using the seeds above.

Each of the six DRL methods is retrained on generated industrial simulation data for each task scale, yielding 24 industrial checkpoints. The comparison uses these retrained models rather than directly transferring the synthetic-benchmark checkpoints. Training uses the industrial travel-time and tardiness objective. The [training configuration](../../../scripts/industrial/training/configs/experiment.json) and [checkpoint guide](../../../checkpoints/README.md) give the settings and model locations.

## Included results

- [DRL results](drl/per_instance_results.csv): 2,400 per-instance records from the six retrained models under Sample-1280.
- [ALNS results](alns/per_rep_results.csv): 2,000 records, comprising five repetitions for each of 400 instances. Repetitions are averaged per instance for reported method means.
- [Gurobi results](gurobi/per_instance_results.csv): 300 time-limited incumbents for n = 10, 20, 30, with original certificates and routes. At n = 40, all 100 runs reached the 3600 s limit: 79 had no incumbent and 21 returned feasible incumbents. A dash in the comparison table indicates that a complete 100-instance objective comparison is unavailable.
- The CSV reports schedule objectives evaluated from saved routes alongside the original solver incumbent objective. Bounds and MIP gaps refer to that original incumbent, not the route-derived schedule objective. All original values are preserved; runtime is the time reported by the solver.

The [baseline guide](../../../methods/conventional/README.md) lists ALNS settings.

DRL computation times are arithmetic means of the recorded per-instance runtimes over the 100 test instances at each scale, using Sample-1280 inference.

## Use

From the repository root, training and evaluation entry points are:

```bash
conda run -n my310env python -B scripts/industrial/training/train_drl.py --method cohet --size 20 --run-id industrial_cohet_n20 --dry-run
conda run -n my310env python -B scripts/industrial/drl/evaluate_drl.py --method cohet --size 20 --checkpoint checkpoints/real_world/cohet/size_20/epoch-99.pt --run-id industrial_eval_n20 --dry-run
conda run -n my310env python -B scripts/industrial/alns/run_alns.py --help
conda run -n my310env python -B scripts/industrial/gurobi/run_gurobi.py --help
```

Removing `--dry-run` starts computation. Use an available CUDA device for the documented DRL timing protocol. Gurobi requires a separately configured license.

Rebuild statistics from the saved results without training, inference or optimization:

```bash
conda run -n my310env python -B scripts/analysis/industrial/rebuild_results.py --output-dir outputs/industrial_summary
```

The command requires a new output directory and preserves the included records.
