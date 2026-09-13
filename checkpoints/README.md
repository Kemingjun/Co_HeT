# Pretrained checkpoints

Models are stored as `epoch-99.pt` beside their `args.json`, which records model-specific settings and configuration sources.

| Directory | Contents | Implementation |
|---|---|---|
| `cohet`, `am`, `mvmoe`, `hdrl`, `tdrl`, `echo` | 48 main models: four task scales, two or three functional types, seed 1234 | `methods/learning/<method>/type_<k>` |
| `multiseed` | 12 additional n=20, k=2 models, seeds 2345 and 3456 | The corresponding main k=2 method |
| `ablation/type_2/size_20` | Eight architectural variants, including Full and Flat | `methods/learning/cohet_ablation` |
| `decoder_scaling` | Full and Flat at n=20,30,40,50 | `methods/learning/cohet_ablation`; Full n=50 uses the main `cohet/type_2` implementation |
| `weight_sensitivity` | Nine weights 0.1 through 0.9; weight 0.5 reuses the main model | `methods/learning/cohet_weight` |
| `real_world` | 24 models retrained on industrial simulation data: six methods at n=10,20,30,40 | `methods/real_world/<method>` |

There are 109 checkpoint paths, each with an adjacent configuration file.

Pass the model directory or `.pt` file to the relevant loader. Main and multi-seed models use [the unified evaluation entry](../scripts/eval_drl.py); the robot-type argument selects the matching method implementation. Specialized models are described in [reproducibility](../docs/reproducibility.md). Architecture variants use the `ablation_variant` field in their configuration.

