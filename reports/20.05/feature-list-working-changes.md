# Working Feature List Changes

This note documents the latest feature-inventory update after the regenerated Phase 1 + Phase 2 extraction audit.

## Applied drop

Dropped these constant features from the active Phase 2 feature metadata/matrix and future Phase 2 extraction output:

- `graph_largest_component_ratio`
- `par_voiced_frame_ratio`
- `ft_cluster_count`
- `ft_switch_count`

Active Phase 2 feature count is now 472.

## Current degeneracy status

- Varying active features: 470
- Constant active features: 0
- All-missing active features: 2

The only all-missing active features are:

- `rd_pause_per_reference_token`: pending audio coverage for `READING` rows.
- `cmd_duration_per_token`: pending audio coverage for `COMMAND` rows.

## Remaining caution flags

- `ft_*`: only 5 covered rows; exclude from current modelling/interpretation until true fluency coverage is available.
- `fc_*`: 27 covered rows; valid but very sparse.
- `ms_*`: 22 covered rows; valid but very sparse.
- `sr_*`, `rep_*`, and `cmd_*`: sparse task-specific families, so they should be interpreted task-wise rather than as universal markers.
- `sx_dep_length_q25`: quasi-constant in the regenerated matrix, with 99.5% of nonmissing rows equal to 1.0. It is not fully degenerate, but it is low-information and should be watched in feature-selection outputs.
