# Conventional Baselines

This release includes the conventional baselines used for comparison with Co-HeT:

- **Gurobi**: MILP formulation solved with Gurobi.
- **ALNS**: adaptive large neighborhood search.
- **IGA**: iterated greedy algorithm.
- **DABC**: discrete artificial bee colony.
- **DIWO**: discrete invasive weed optimization.

The [baseline guide](../methods/conventional/README.md) specifies operator selection, budget parameters, stopping conditions, repetitions and Gurobi settings.

Example:

```bash
conda run -n my310env python scripts/run_conventional.py --solver alns N20_K2_M12_I1 --dry-run
```
