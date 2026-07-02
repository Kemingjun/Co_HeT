# Conventional Baselines

This release includes the conventional baselines used for comparison with Co-HeT:

- **Gurobi**: exact mixed-integer programming formulation.
- **ALNS**: adaptive large neighborhood search.
- **IGA**: iterated greedy algorithm.
- **DABC**: discrete artificial bee colony.
- **DIWO**: discrete invasive weed optimization.

The packaged code is adapted from the experiment implementation used in the paper. Runtime limits, stopping criteria, and instance choices should be set consistently with the reported experimental protocol when reproducing tables.

Example:

```bash
python scripts/run_conventional.py --solver alns N20_K2_M12_I1
```
