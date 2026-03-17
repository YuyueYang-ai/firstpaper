# OmniGS Phase 2 Main Parameters

This note records the main tunable parameters currently used by the Phase 2 pipeline in OmniGS.

Relevant code entry points:

- `include/compact_gaussian.h`
- `examples/train_phase2_residual_field.cpp`
- `cfg/lonlat/office_compact_phase2_30k_formal.yaml`
- `cfg/lonlat/omniscenes_turtlebot_pyebaekRoom_compact_phase2_32000_formal.yaml`

## 1. Mode Switch

`Compression.mode`

- `baseline`
  - Disable compact pipeline and Phase 2.
- `compact`
  - Enable Phase 1 compact pipeline only.
- `compact_phase2`
  - Enable Phase 1 compact pipeline and Phase 2 frozen-package workflow.

Current formal configs use:

- `Compression.mode: "compact_phase2"`

## 2. Phase 2 Frozen-Package Parameters

These parameters control when and how the explicit Gaussian model is frozen and exported for Phase 2 training.

`Phase2.enable`

- Type: bool
- Code default: `false`
- Meaning: enable Phase 2 preparation inside the main training pipeline.

`Phase2.save_frozen_snapshot`

- Type: bool
- Code default: `false`
- Meaning: export a frozen package at the configured freeze iteration / checkpoint.

`Phase2.freeze_topology_iter`

- Type: int
- Code default: `-1`
- Meaning: iteration at which topology is considered frozen for Phase 2.

`Phase2.sort_by_morton`

- Type: bool
- Code default: `true`
- Meaning: sort anchors by Morton order before exporting the frozen package.

`Phase2.normalize_xyz`

- Type: bool
- Code default: `true`
- Meaning: store normalized XYZ for field input.

`Phase2.mask_features_rest_by_sh_level`

- Type: bool
- Code default: `true`
- Meaning: mask `features_rest` according to active SH level before creating the frozen target.

Current formal values:

- `Phase2.enable: 1`
- `Phase2.save_frozen_snapshot: 1`
- `Phase2.freeze_topology_iter: 15000`
- `Phase2.sort_by_morton: 1`
- `Phase2.normalize_xyz: 1`
- `Phase2.mask_features_rest_by_sh_level: 1`

## 3. Phase 2 Locality-Residual Base

These parameters define whether Phase 2 learns residuals on top of a Phase 1 locality base.

Code location:

- `include/compact_gaussian.h`

`use_locality_base`

- Struct field: `Phase2ResidualFieldOptions.use_locality_base`
- Code default: `true`
- Meaning: build a blockwise locality base and learn only residuals over it.

`locality_high_sh_block_size`

- Struct field: `Phase2ResidualFieldOptions.locality_high_sh_block_size`
- Code default: `64`
- Meaning: block size for high-SH anchors when building the locality base.

`locality_low_sh_block_size`

- Struct field: `Phase2ResidualFieldOptions.locality_low_sh_block_size`
- Code default: `128`
- Meaning: block size for low-SH anchors when building the locality base.

These are currently code-defaulted and not explicitly overridden in the formal YAMLs.

## 4. Phase 2 Field Training Parameters

These parameters are parsed by:

- `examples/train_phase2_residual_field.cpp`

and defined in:

- `include/compact_gaussian.h`

### Network Size

`Phase2Field.num_fourier_frequencies`

- Type: int
- Code default: `6`
- Current formal value: `6`
- Meaning: Fourier feature frequency count for normalized XYZ input.

`Phase2Field.hidden_dim`

- Type: int
- Code default: `128`
- Current formal value: `128`
- Meaning: hidden width of the residual MLP.

`Phase2Field.num_hidden_layers`

- Type: int
- Code default: `3`
- Current formal value: `3`
- Meaning: number of hidden layers in the residual MLP.

### Optimization

`Phase2Field.batch_size`

- Type: int
- Code default: `8192`
- Current formal value: `8192`
- Meaning: point batch size for field training.

`Phase2Field.max_steps`

- Type: int
- Code default: `4000`
- Current formal value: `4000`
- Meaning: total optimization steps for Phase 2 field training.

`Phase2Field.log_interval`

- Type: int
- Code default: `200`
- Current formal value: `200`
- Meaning: logging interval during training.

`Phase2Field.eval_interval`

- Type: int
- Code default: `500`
- Current formal value: `500`
- Meaning: evaluation interval during training.

`Phase2Field.learning_rate`

- Type: float
- Code default: `1e-3`
- Current formal value: `1e-3`
- Meaning: optimizer learning rate.

`Phase2Field.weight_decay`

- Type: float
- Code default: `1e-6`
- Current formal value: `1e-6`
- Meaning: optimizer weight decay.

### Input Feature Switches

`Phase2Field.include_features_dc`

- Type: bool
- Code default: `true`
- Current formal value: `1`
- Meaning: include `features_dc` in field input.

`Phase2Field.include_opacity`

- Type: bool
- Code default: `true`
- Current formal value: `1`
- Meaning: include opacity in field input.

`Phase2Field.include_scaling`

- Type: bool
- Code default: `true`
- Current formal value: code default
- Meaning: include scaling in field input.

`Phase2Field.include_rotation`

- Type: bool
- Code default: `true`
- Current formal value: code default
- Meaning: include rotation in field input.

`Phase2Field.block_embedding_dim`

- Type: int
- Code default: `8`
- Current formal value: code default
- Meaning: embedding dimension for block-level locality ids.

## 5. Phase 2 Output / Deployment Parameters

`Phase2Field.save_decoded_compact`

- Type: bool
- Code default: `true`
- Current formal value: `1`
- Meaning: save the dense decoded compact package for debugging / evaluation compatibility.

`Phase2Field.save_phase2_compact`

- Type: bool
- Code default: `true`
- Current formal value: code default
- Meaning: save the true `phase2_compact` package (`anchors + field weights`).

`Phase2Field.decoded_xyz_quant_bits`

- Type: int
- Code default: `16`
- Current formal value: `16`
- Meaning: quantization bits for decoded compact XYZ when `save_decoded_compact` is enabled.

`Phase2Field.decoded_attribute_quant_bits`

- Type: int
- Code default: `16`
- Current formal value: `16`
- Meaning: quantization bits for decoded compact attributes.

`Phase2Field.decoded_rotation_quant_bits`

- Type: int
- Code default: `16`
- Current formal value: `16`
- Meaning: quantization bits for decoded compact rotation.

## 6. Current Formal Preset Summary

Current Phase 2 formal preset is effectively:

- `Compression.mode = compact_phase2`
- Frozen package enabled
- Freeze iteration = `15000`
- Morton sort enabled
- XYZ normalization enabled
- SH-level masking enabled
- Locality residual base enabled
- High-SH block size = `64`
- Low-SH block size = `128`
- Fourier frequencies = `6`
- Hidden dim = `128`
- Hidden layers = `3`
- Batch size = `8192`
- Max steps = `4000`
- LR = `1e-3`
- Weight decay = `1e-6`
- Input uses `features_dc`, `opacity`, `scaling`, `rotation`, `block embedding`
- `save_decoded_compact = true`
- `save_phase2_compact = true`

## 7. Parameters Worth Tuning First

If Phase 2 quality needs improvement, the highest-priority knobs are:

- `Phase2Field.block_embedding_dim`
- `Phase2Field.hidden_dim`
- `Phase2Field.num_hidden_layers`
- `Phase2Field.num_fourier_frequencies`
- `Phase2Field.learning_rate`
- `Phase2Field.max_steps`
- `Phase2ResidualFieldOptions.locality_high_sh_block_size`
- `Phase2ResidualFieldOptions.locality_low_sh_block_size`

Recommended first sweep:

- `block_embedding_dim = 8 / 16 / 32`
- `hidden_dim = 128 / 192 / 256`
- `num_hidden_layers = 3 / 4`
- `num_fourier_frequencies = 6 / 8`
- `learning_rate = 1e-3 / 5e-4`
