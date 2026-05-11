# Phase 2 Findings

Phase 2 extends the additive Phase 1 benchmark with four richer feature sets aligned to the Lindsay et al. and Balagopalan et al. feature-based AD papers:

1. Prompt-aware `PD_CTP` semantic / content-unit features
2. Task-aware `READING` and `CONVERSATION` semantic modules
3. Richer lexicosyntactic features
4. Richer acoustic / paralinguistic features

## Feature inventory

The live Phase 2 table contains `375` handcrafted features in total:

| Group | Count |
| --- | ---: |
| `ac` | 105 |
| `pause` | 17 |
| `par` | 112 |
| `syn` | 34 |
| `sx` | 31 |
| `lex` | 12 |
| `len` | 11 |
| `disc` | 9 |
| `graph` | 6 |
| `pd` | 20 |
| `rd` | 10 |
| `fc` | 8 |

This exactly matches the overall raw feature count reported in Lindsay et al. (`375`) and materially exceeds the Phase 1 live inventory (`194`).

Compared with the paper reference sets:

- The live paralinguistic / acoustic block is now `234 = ac + pause + par`, which is larger than Lindsay et al. (`208`) and Balagopalan et al. acoustic (`187`).
- The live prompt-aware `PD_CTP` semantic block is `20` features, which is effectively at Lindsay semantic-block scale (`21`).
- The live richer text block is larger than the earlier universal syntax-only pass, but still not as large as the full Balagopalan lexicosyntactic inventory (`297`) because full constituency production-rule extraction and some lexical norm families are still not implemented.

## Prompt-family coverage

Prompt-aware semantic extraction was resolved for the following main prompt families:

- `cookie_theft`: `2252` rows
- `lion_scene`: `155` rows
- `cat_tree`: `92` rows
- `car_trip`: `74` rows
- `umbrella_rain`: `3` rows
- `UNRESOLVED`: `1551` rows

This means the prompt-aware semantic layer is strongest for `cookie_theft`, which remains the cleanest cross-lingual claim-bearing subset.

## Best run summary

Best completed runs are summarized in [phase2_best_run_summary.csv](</C:/Users/dhruv/Documents/06 Coding/MultiConAD/tables/phase2/rich_sweep/phase2_best_run_summary.csv>).

Headline results:

- `benchmark_wide_phase2`: `SVM rbf`, `all_phase2`, `k=100`, balanced accuracy `0.775`
- `pd_ctp_pooled_phase2`: `LR`, `all_phase2`, `k=100`, balanced accuracy `0.774`
- `cookie_theft_pooled_phase2`: `SVM rbf`, `rich_universal`, `k=100`, balanced accuracy `0.760`
- `language_en_pd_ctp_phase2`: `LR`, `all_phase2`, `k=100`, balanced accuracy `0.748`
- `language_zh_pd_ctp_phase2`: `LR`, `all_phase2`, `k=100`, balanced accuracy `0.938`
- `language_el_pd_ctp_phase2`: `RF`, `rich_universal`, `k=100`, balanced accuracy `0.875`
- `language_es_reading_phase2`: `LR`, `task_specific_semantic`, `k=50`, balanced accuracy `0.729`

## What changed relative to Phase 1

- Benchmark-wide accuracy improved from the earlier rich Phase 1 benchmark-wide sweep (`0.749`) to `0.775`.
- Pooled `PD_CTP` improved from the earlier rich Phase 1 pooled `PD_CTP` result (`0.751`) to `0.774`.
- English `PD_CTP` changed only slightly, suggesting the main gain is not a simple English-only artifact.
- Spanish `READING` benefited most directly from the task-specific module: the best subset is `task_specific_semantic`, not the old universal block.
- Chinese `PD_CTP` benefited strongly from adding prompt-aware `PD_CTP` features.
- Greek `PD_CTP` still prefers the richer universal acoustic/paralinguistic block over the combined semantic stack.

## Feature-importance interpretation

The best benchmark-wide Phase 2 run mixes old and new feature groups:

- syntax (`syn_*`)
- prompt-aware `PD_CTP` semantics (`pd_*`)
- richer paralinguistic features (`par_*`)

The best pooled `PD_CTP` Phase 2 run is still dominated by syntax and lexical structure, but the prompt-aware semantic block is now genuinely contributing:

- `lex_function_word_ratio`
- `syn_mean_dependency_length`
- `syn_std_dependency_length`
- `syn_content_function_ratio`
- `pd_unique_unit_efficiency`
- `pd_keyword_to_nonkeyword_ratio`
- `sx_sentence_length_cv`
- `sx_upos_entropy`

Within languages, the Phase 2 picture remains language-specific:

- English `PD_CTP`: dominated by syntax and richer lexicosyntactic structure
- Chinese `PD_CTP`: strongly influenced by prompt-aware `pd_*` semantics plus pause/discourse cues
- Greek `PD_CTP`: still dominated by acoustic and richer paralinguistic descriptors
- Spanish `READING`: dominated by the new reading-reference alignment features

## Main conclusion

Phase 2 is now live and materially richer than Phase 1. It improves the pooled multilingual benchmark and shows that prompt-aware semantics help where prompt references are reliable, especially for Chinese `cookie_theft` and Spanish `READING`.

At the same time, the strongest cross-lingual semantic claim remains narrow:

- `cookie_theft` is the most defensible shared prompt family
- broader `PD_CTP` still mixes multiple picture-description prompts
- feature importance does not collapse to one small universal multilingual biomarker set

The practical read is:

1. richer handcrafted features are worth keeping
2. prompt-aware semantic modules do help
3. the gains are task- and language-dependent rather than globally uniform
