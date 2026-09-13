# Independent industrial simulation test set

Dataset ID: `real_world_test100_seed20260906`. Four sizes (10, 20, 30, 40), 100 instances each.
Names repeat across datasets: identify instances by dataset ID, size, and index.

Generation uses the generate_instance function with master seed 20260906.
Per-instance seed = 20260906 + size * 1000 + index. See generation_config.json for parameters.
Left/right sides are equiprobable. Supply and handover share a side; y slots are 5:5:95.
Supply x: -20/120; handover x: 0/100; delivery x: 20/40/60/80.
Deadline = 300 + 40 * task_index + int(Uniform[-1,1] * 40), followed by shuffle.
The integer conversion truncates toward zero.

Coordinates are in meters; deadlines are in seconds. These simulation test instances
are generated using parameters collected from the industrial site.
Evaluation uses Manhattan distances, open paths, and 0.5 travel_time + 0.5 tardiness,
with carrier speed 1.2 and forklift speed 1.0; shuttle travel is not double-counted.

The generator requires a new output directory.
