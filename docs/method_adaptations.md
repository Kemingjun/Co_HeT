# Learning-based methods

All six implementations use the CHRSP task–coalition action, synchronization rules and objective. The five baselines are adaptations of the cited methods, with their characteristic network designs retained.

| Method | Preserved design | CHRSP adaptation |
|---|---|---|
| Co-HeT | Separate task and robot attention, bidirectional contextual fusion, hierarchical decoder | Select a task and construct its complete heterogeneous coalition. |
| AM | Attention encoder–decoder | Encode CHRSP features and decode synchronized task–coalition assignments. |
| HDRL | History-aware dispatching and route context | Use collaborative task histories and robot availability in the decoder. |
| TDRL | Token representations and recurrent dynamic updates | Represent task–robot states and select complete coalitions. |
| MVMoE | Sparse mixture-of-experts routing | Apply routing within the adapted encoder/decoder; retain CHRSP feasibility masks and rewards. |
| ECHO | Dual-modality encoding and PFCA context | Replace vehicle–node decisions with task–coalition decisions and collaborative resource context. |

The common main settings include 128-dimensional embeddings and hidden states, one encoder layer, Adam with learning rate 1e-4, and a rollout baseline. MVMoE uses eight experts and top-2 routing in the recorded main configurations; ECHO uses PFCA and an edge dimension of 128. Consult the adjacent checkpoint `args.json` before recreating an individual model.

| Method | Trainable parameters at n=20, k=2 / k=3 (millions) |
|---|---|
| Co-HeT | 1.006 / 1.007 |
| AM | 0.659 / 0.660 |
| MVMoE | 1.814 / 1.814 |
| HDRL | 1.072 / 1.072 |
| TDRL | 2.387 / 3.053 |
| ECHO | 0.759 / 0.759 |

Consult the adjacent `args.json` for each model's precision, architecture and training settings. TDRL uses a fixed robot-count dimension. Architecture variants are implemented in `methods/learning/cohet_ablation`. Industrial models were retrained on generated industrial simulation data at each task scale.

Original project acknowledgements and links are retained in the root README. The repository's MIT license and upstream file notices are preserved.
