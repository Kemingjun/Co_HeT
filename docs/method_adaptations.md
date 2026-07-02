# DRL Baseline Adaptations

All learning-based methods are evaluated under the same CHRSP formulation, reward function, action space, data generation process, and inference protocol.

## AM

The attention model baseline is adapted to generate CHRSP task-coalition composite actions.

## HDRL

HDRL preserves history-aware dispatching and route-context-aware decoding while using the CHRSP state transition and objective.

## TDRL

TDRL preserves token-style state coding and GRU-based dynamic token updates. The output action is adapted to synchronized task and heterogeneous robot coalition decisions.

## MVMoE

MVMoE adds sparse mixture-of-experts layers to the AM-style model while retaining the same reward and inference protocol.

## ECHO

ECHO is adapted through dual-modality task encoding and historical-resource-aware decoding. Its original vehicle-node action is replaced by the task-coalition composite action required by CHRSP.
