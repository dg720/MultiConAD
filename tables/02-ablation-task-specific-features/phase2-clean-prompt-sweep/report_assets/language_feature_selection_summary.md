# Language Feature Selection Summary

This summary is based on the clean prompt/task slices.
There is no true no-selection full-feature condition in these clean-slice sweeps; the nearest comparisons are:
- the fuller `k=100` version of the same winning feature family
- the best `all_phase2` result for the same slice

| slice_slug | language | task | run_name | best_subset | best_top_k | best_num_selected_features | best_model | best_accuracy | best_balanced_accuracy | best_auroc | same_subset_k100_accuracy | same_subset_k100_balanced_accuracy | same_subset_k100_model | delta_vs_same_subset_k100_accuracy | delta_vs_same_subset_k100_balanced_accuracy | best_all_phase2_top_k | best_all_phase2_model | best_all_phase2_accuracy | best_all_phase2_balanced_accuracy | delta_vs_best_all_phase2_accuracy | delta_vs_best_all_phase2_balanced_accuracy | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en_cookie_theft | English | Cookie Theft | language_en_cookie_theft_clean_phase2 | rich_universal | 100 | 100 | rf:n200 | 0.841709 | 0.708839 | 0.809463 | 0.841709 | 0.708839 | rf:n200 | 0 | 0 | 50 | svm:rbf_c1 | 0.839196 | 0.731164 | 0.00251256 | -0.0223249 | No true no-selection full-feature run exists for clean slices; k=100 and best all_phase2 are the nearest available comparisons. |
| zh_cookie_theft | Chinese | Cookie Theft | language_zh_cookie_theft_clean_phase2 | semantic_plus_pause | 50 | 41 | rf:n200 | 0.92 | 0.875 | 0.911765 | 0.92 | 0.875 | rf:n200 | 0 | 0 | 50 | lr:c1 | 0.84 | 0.816176 | 0.08 | 0.0588235 | No true no-selection full-feature run exists for clean slices; k=100 and best all_phase2 are the nearest available comparisons. |
| el_lion_scene | Greek | Lion Scene | language_el_lion_scene_clean_phase2 | rich_universal | 50 | 50 | lr:c1 | 0.909091 | 0.909091 | 0.884298 | 0.818182 | 0.818182 | lr:c1 | 0.0909091 | 0.0909091 | 100 | lr:c1 | 0.818182 | 0.818182 | 0.0909091 | 0.0909091 | No true no-selection full-feature run exists for clean slices; k=100 and best all_phase2 are the nearest available comparisons. |
| es_reading | Spanish | Reading | language_es_reading_clean_phase2 | task_specific_semantic | 50 | 9 | lr:c1 | 0.758621 | 0.728682 | 0.792248 | 0.758621 | 0.728682 | lr:c1 | 0 | 0 | 100 | rf:n200 | 0.741379 | 0.521705 | 0.0172414 | 0.206977 | No true no-selection full-feature run exists for clean slices; k=100 and best all_phase2 are the nearest available comparisons. |

## Top Permutation Features

### English Cookie Theft

| feature_name | feature_group | importance_mean | importance_std |
| --- | --- | --- | --- |
| syn_aux_ratio | syn | 0.0882334 | 0.015252 |
| syn_pronoun_ratio | syn | 0.0669165 | 0.0246155 |
| syn_mean_dependency_length | syn | 0.0497545 | 0.00575846 |
| syn_num_ratio | syn | 0.0479517 | 0.0124105 |
| syn_clause_density | syn | 0.0344921 | 0.0151097 |
| graph_density | graph | 0.0328896 | 0.0108695 |
| syn_determiner_ratio | syn | 0.0260532 | 0.021065 |
| len_utterance_count | len | 0.0171362 | 0.00922597 |
| syn_utterance_count | syn | 0.0171362 | 0.00922597 |
| lex_function_word_ratio | lex | 0.0150879 | 0.00878559 |

### Chinese Cookie Theft

| feature_name | feature_group | importance_mean | importance_std |
| --- | --- | --- | --- |
| len_speech_duration | len | 0.198529 | 0.064312 |
| lex_mattr_20 | lex | 0.117647 | 0.0415945 |
| disc_repeated_trigram_ratio | disc | 0.111029 | 0.0370139 |
| syn_adp_ratio | syn | 0.1 | 0.05 |
| syn_determiner_noun_ratio | syn | 0.0941176 | 0.0335829 |
| syn_nsubj_ratio | syn | 0.0933824 | 0.0279605 |
| graph_node_count | graph | 0.0882353 | 0.0359468 |
| len_type_count | len | 0.0882353 | 0.0359468 |
| disc_adjacent_utterance_similarity_mean | disc | 0.0816176 | 0.0370139 |
| disc_local_coherence | disc | 0.0816176 | 0.0370139 |

### Greek Lion Scene

| feature_name | feature_group | importance_mean | importance_std |
| --- | --- | --- | --- |
| ac_egemaps_f3frequency_sma3nz_stddevnorm | ac | 0.145455 | 0.0530087 |
| ac_egemaps_jitterlocal_sma3nz_amean | ac | 0.136364 | 0.028748 |
| lex_honore | lex | 0.1 | 0.0181818 |
| par_mfcc_12_skew | par | 0.1 | 0.0340151 |
| par_mfcc_10_std | par | 0.1 | 0.0340151 |
| par_mfcc_5_skew | par | 0.1 | 0.0181818 |
| par_mfcc_9_kurtosis | par | 0.1 | 0.0340151 |
| ac_egemaps_spectralflux_sma3_stddevnorm | ac | 0.0909091 | 0.049793 |
| par_mfcc_8_kurtosis | par | 0.0909091 | 0.028748 |
| ac_egemaps_mfcc4_sma3_stddevnorm | ac | 0.0909091 | 0.049793 |

### Spanish Reading

| feature_name | feature_group | importance_mean | importance_std |
| --- | --- | --- | --- |
| rd_sequence_match_ratio | rd | 0.253023 | 0.0478907 |
| rd_content_word_recall_ratio | rd | 0.0545736 | 0.0513049 |
| rd_repetition_ratio | rd | 0.0246512 | 0.0350963 |
| rd_reference_bigram_coverage | rd | 0.0122481 | 0.0360858 |
| rd_reference_order_score | rd | -0.00868217 | 0.016374 |
| rd_reference_token_coverage | rd | -0.0130233 | 0.0261983 |
| rd_omission_ratio | rd | -0.0130233 | 0.0261983 |
| rd_insertion_ratio | rd | -0.02 | 0.0179095 |
| rd_prompt_similarity | rd | -0.0223256 | 0.0201097 |
