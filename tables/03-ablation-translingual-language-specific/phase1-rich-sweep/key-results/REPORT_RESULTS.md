# Phase 1 Rich Sweep Report

## How the cited papers present results

The report structure here follows the presentation logic used in the three reference papers.

- Lindsay, Tröger, and König (2021) use worked feature tables with short examples and plain-language explanations. That style is useful for prompt-specific or clinically interpretable features, so the feature-example table below mirrors that pattern.
- Balagopalan et al. (2021) separate three things cleanly: feature inventory tables, model-performance tables, and a feature-differentiation table with class means and model weights. The inventory and differentiation tables below follow that design.
- Laguarta and Subirana (2021) present subject-level saliency with a radar chart to support explainability and longitudinal monitoring. The patient-level saliency figure below adapts that visual idea for our best interpretable English PD_CTP model.

## Main results

| analysis | best_model | feature_subset | top_k | balanced_accuracy | macro_f1 | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benchmark-wide | SVM (rbf) | all_universal | 50 | 0.749 | 0.752 | 0.818 | 0.736 |
| Pooled PD_CTP | SVM (linear) | all_universal | 100 | 0.751 | 0.752 | 0.813 | 0.727 |
| English PD_CTP | SVM (linear) | text_only | 50 | 0.756 | 0.770 | 0.825 | 0.742 |
| Chinese PD_CTP | Decision Tree | pause_only | 20 | 0.986 | 0.964 | 0.986 | 0.889 |
| Greek PD_CTP | SVM (linear) | acoustic_only | 50 | 0.925 | 0.925 | 0.944 | 0.973 |

The main benchmark-wide winner is a nonlinear SVM on the full universal handcrafted set, but the more interpretable primary claim remains the pooled `PD_CTP` slice, where a linear SVM on the universal feature stack reaches balanced accuracy `0.751` and AUROC `0.813`.

## Table 1. Worked examples for key report features

Example text used for the calculations: "The boy is taking a cookie from the jar. The mother is washing the dishes. The girl is watching them."

| feature_name | feature_family | explanation | example_value |
| --- | --- | --- | --- |
| len_token_count | len | Total number of word tokens in the response. | 20 |
| len_type_count | len | Number of distinct word forms used at least once. | 14 |
| len_tokens_per_utterance | len | Average number of tokens per utterance after splitting the response. | 6.667 |
| lex_lemma_type_token_ratio | lex | Distinct lemmas divided by total tokens; reduces inflectional duplication. | 0.700 |
| lex_function_word_ratio | lex | Share of tokens that are function words such as determiners, pronouns, auxiliaries, and conjunctions. | 0.550 |
| syn_determiner_ratio | syn | Share of tokens tagged as determiners, e.g. 'the', 'a'. | 0.300 |
| syn_aux_ratio | syn | Share of tokens tagged as auxiliary verbs, e.g. 'is', 'are'. | 0.150 |
| syn_noun_ratio | syn | Share of tokens tagged as common nouns. | 0.300 |
| syn_content_function_ratio | syn | Content-word count divided by function-word count. | 0.818 |
| syn_mean_dependency_length | syn | Average absolute token distance between a word and its syntactic head. | 1.588 |

## Table 2. Summary of extracted lexico-syntactic and discourse features

| feature_type | num_features | brief_description |
| --- | --- | --- |
| Length and response volume | 11 | Counts, utterance length, and speech-rate derived length measures. |
| Lexical diversity and richness | 12 | Type-token style diversity, lexical richness, repetition, and word-form measures. |
| Discourse repetition and coherence | 9 | Repetition, local coherence, and utterance-to-utterance semantic similarity. |
| Syntactic composition | 34 | POS ratios, dependency relations, subordination, and parse-complexity measures. |
| Speech graph topology | 6 | Graph density, loops, components, and degree over token-transition graphs. |

## Table 3. Summary of extracted acoustic and pause features

| feature_type | num_features | brief_description |
| --- | --- | --- |
| Pause and silence timing | 17 | Counts, durations, and timing ratios from silence and filled-pause behavior. |
| openSMILE eGeMAPS | 88 | Functionals from the eGeMAPSv02 descriptor set. |
| MFCC summary statistics | 10 | Mean and standard deviation of low-order MFCCs from librosa fallback extraction. |
| Energy and zero-crossing | 4 | Signal-energy and zero-crossing descriptors. |
| Pitch and duration fallback | 3 | Audio duration and coarse pitch descriptors used as fallback features. |

## Table 4. Status of task-specific semantic modules

| feature_type | num_features | brief_description |
| --- | --- | --- |
| Universal handcrafted semantics in Phase 1 | 0 | No prompt-specific content-unit features are extracted in the current Phase 1 benchmark. |
| Planned PD_CTP semantic module | 9 | Phase 2 target block: content-unit coverage, density, object/action balance, and semantic similarity. |
| Planned FLUENCY semantic module | 12 | Phase 2 target block: valid items, intrusions, clustering, switching, and item rate. |
| Planned STORY_NARRATIVE semantic module | 7 | Phase 2 target block: proposition coverage, event ordering, and story-reference similarity. |
| Planned CONVERSATION semantic module | 7 | Phase 2 target block: topic coherence, topic switching, and named-entity / repetition dynamics. |

The current rich sweep is still a Phase 1 universal-feature analysis. The Phase 2 task-specific semantic modules are planned but not yet extracted into the live benchmark table, which is why the prompt-specific content-unit block remains a next-step item rather than part of the current sweep.

## Figure 1. Patient-level saliency map

Selected held-out English `PD_CTP` pair from dataset `Pitt`:

- AD subject `511` with AD probability `1.000`
- HC subject `113` with AD probability `0.003`

AD excerpt: "(.) well , I needta read it . [+ exc] 14181_14982 +< tell you . [+ exc] [+ gram] well , that's the boy . 22172_23197 (.) he wants the cookies . 27979_29241 and this must be the mother . 31510_32910 is that all ? [+ exc] 35298_35821..."

HC excerpt: "little boy getting in the cookie jar . [+ gram] 8288_9898 and the little girl holding out her hand for some cookies . 11124_15390 and a stool the boy is on is about to tip over . 15612_19878 and a lady is drying a dish . 20661_24027..."

![English PD_CTP saliency map](report_assets/english_pd_ctp_saliency_map.png)

The saliency figure uses the top non-missing linear-SVM features from the English `PD_CTP` winner and plots absolute per-feature contribution magnitudes after imputation and scaling. It is a contribution-style diagnostic view, not a causal claim.

## Figure 2. Language-specific t-SNE comparison

Each row is a language-specific slice. The left column uses the strongest pooled multilingual feature set available for that slice, while the right column uses the best monolingual feature set for that language.

![Language t-SNE comparison](report_assets/language_tsne_comparison_grid.png)

## Figure 3. Language-specific UMAP comparison

The same comparison is shown with UMAP. In practice, the t-SNE panels are currently easier to interpret in this benchmark than the UMAP panels.

![Language UMAP comparison](report_assets/language_umap_comparison_grid.png)

Why English showed distinct clusters in the earlier pooled plot: English contains the largest number of samples and the widest mixture of corpora, transcript conventions, and prompt packaging. The projection was therefore reflecting residual dataset/task structure in addition to diagnosis. The language-specific panels are a cleaner way to inspect separation.

## Table 5. Feature differentiation analysis for pooled PD_CTP top features

| feature_name | feature_type | mu_ad | mu_hc | cohens_d | label_correlation | svm_weight | bonferroni_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| syn_num_ratio | syn | 0.032 | 0.006 | 0.773 | 0.387 | -0.032 | 0.000 |
| len_syllables_per_second | len | 0.754 | 2.014 | -1.506 | -0.517 | -0.525 | 0.000 |
| len_utterance_count | len | 9.311 | 14.230 | -0.586 | -0.249 | -0.070 | 0.000 |
| syn_utterance_count | syn | 9.311 | 14.230 | -0.586 | -0.249 | -0.070 | 0.000 |
| pause_speech_rate_words_per_second | pause | 0.753 | 1.787 | -1.490 | -0.532 | -0.226 | 0.000 |
| len_tokens_per_second | len | 0.753 | 1.787 | -1.490 | -0.532 | -0.226 | 0.000 |
| syn_interjection_ratio | syn | 0.015 | 0.003 | 0.657 | 0.340 | 0.171 | 0.000 |
| syn_mean_dependency_length | syn | 4.043 | 2.824 | 0.892 | 0.430 | 0.631 | 0.000 |
| syn_pronoun_ratio | syn | 0.067 | 0.031 | 0.586 | 0.271 | 0.326 | 0.000 |
| syn_cconj_ratio | syn | 0.031 | 0.012 | 0.586 | 0.275 | 0.126 | 0.000 |
| pause_articulation_rate_words_per_speaking_second | pause | 1.094 | 2.435 | -1.373 | -0.507 | 0.824 | 0.000 |
| syn_aux_ratio | syn | 0.044 | 0.023 | 0.506 | 0.232 | -0.197 | 0.000 |
| syn_propn_ratio | syn | 0.010 | 0.002 | 0.539 | 0.271 | 0.084 | 0.000 |
| lex_function_word_ratio | lex | 0.263 | 0.156 | 0.466 | 0.205 | -0.761 | 0.000 |
| syn_std_dependency_length | syn | 6.149 | 3.468 | 0.719 | 0.353 | -0.827 | 0.000 |

Interpretation of Table 5:

- This table follows the Balagopalan et al. pattern: class means, association with the binary label, and model weight are shown together.
- The analysis is computed on the grouped training split for the pooled `PD_CTP` linear-SVM winner, using the selected features from that run.
- Because no MMSE target is available consistently across the benchmark, the correlation column reports point-biserial correlation with the AD label rather than severity correlation.

## Table 6. Top ANOVA-ranked features

### Benchmark-wide

| feature_name | f_score | p_value |
| --- | --- | --- |
| syn_num_ratio | 106.886 | 0.000 |
| syn_interjection_ratio | 80.552 | 0.000 |
| syn_mean_dependency_length | 74.656 | 0.000 |
| syn_propn_ratio | 70.007 | 0.000 |
| syn_utterance_count | 59.454 | 0.000 |
| len_utterance_count | 59.454 | 0.000 |
| syn_std_dependency_length | 59.027 | 0.000 |
| syn_clause_density | 55.556 | 0.000 |
| syn_subordination_ratio | 52.347 | 0.000 |
| syn_pronoun_noun_ratio | 49.699 | 0.000 |
| syn_max_dependency_length | 46.638 | 0.000 |
| syn_pronoun_ratio | 45.138 | 0.000 |
| syn_cconj_ratio | 39.770 | 0.000 |
| lex_std_token_length | 34.731 | 0.000 |
| syn_aux_verb_ratio | 33.776 | 0.000 |

### Pooled PD_CTP

| feature_name | f_score | p_value |
| --- | --- | --- |
| syn_num_ratio | 233.363 | 0.000 |
| syn_propn_ratio | 170.573 | 0.000 |
| syn_mean_dependency_length | 157.030 | 0.000 |
| syn_interjection_ratio | 154.649 | 0.000 |
| syn_std_dependency_length | 129.398 | 0.000 |
| syn_max_dependency_length | 119.195 | 0.000 |
| syn_clause_density | 116.917 | 0.000 |
| syn_subordination_ratio | 115.546 | 0.000 |
| syn_utterance_count | 87.330 | 0.000 |
| len_utterance_count | 87.330 | 0.000 |
| lex_std_token_length | 82.083 | 0.000 |
| syn_max_tree_depth | 82.023 | 0.000 |
| syn_mean_tree_depth | 74.627 | 0.000 |
| syn_part_ratio | 66.099 | 0.000 |
| syn_obl_ratio | 56.260 | 0.000 |

These are train-only univariate screening scores. They are useful for understanding which single features separate classes most strongly before the final model is fit, but they are not the same thing as final model importance.

## Table 7. Best held-out result per method, benchmark-wide

| model | variant | feature_subset | top_k | num_features | balanced_accuracy | precision | recall | specificity | macro_f1 | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dt | depth20 | text_plus_pause_plus_acoustic | 100 | 100 | 0.691 | 0.623 | 0.538 | 0.844 | 0.697 | 0.697 | 0.510 |
| lr | c1 | all_universal | all | 193 | 0.747 | 0.624 | 0.695 | 0.799 | 0.740 | 0.792 | 0.698 |
| rf | n200 | text_plus_pause_plus_acoustic | 50 | 50 | 0.679 | 0.908 | 0.377 | 0.982 | 0.697 | 0.820 | 0.756 |
| svm | linear_c1 | all_universal | 50 | 50 | 0.741 | 0.601 | 0.708 | 0.775 | 0.730 | 0.797 | 0.678 |
| svm | rbf_c1 | all_universal | 50 | 50 | 0.749 | 0.674 | 0.648 | 0.850 | 0.752 | 0.818 | 0.736 |

## Table 8. Best held-out result per method, pooled PD_CTP

| model | variant | feature_subset | top_k | num_features | balanced_accuracy | precision | recall | specificity | macro_f1 | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dt | depth20 | text_plus_pause | 50 | 50 | 0.721 | 0.553 | 0.654 | 0.788 | 0.709 | 0.725 | 0.494 |
| lr | c1 | all_universal | all | 193 | 0.736 | 0.576 | 0.669 | 0.802 | 0.724 | 0.818 | 0.656 |
| rf | n200 | all_universal | all | 193 | 0.731 | 0.892 | 0.485 | 0.976 | 0.762 | 0.860 | 0.789 |
| svm | linear_c1 | all_universal | 100 | 100 | 0.751 | 0.649 | 0.640 | 0.861 | 0.752 | 0.813 | 0.727 |
| svm | rbf_c1 | text_only | 50 | 50 | 0.745 | 0.669 | 0.610 | 0.879 | 0.751 | 0.804 | 0.688 |

These method tables follow the Balagopalan-style model comparison format more directly: each model family is shown with its best-performing feature subset and `top-k` setting on the same held-out split.

## Cross-lingual discussion

### All-task monolingual overlap

| feature_name | languages_present | num_languages | mean_rank | mean_importance |
| --- | --- | --- | --- | --- |
| lex_mattr_20 | en,es,zh | 3 | 9.667 | 0.046 |
| disc_repeated_bigram_ratio | en,es,zh | 3 | 14.000 | 0.021 |
| lex_lemma_type_token_ratio | en,es,zh | 3 | 14.333 | 0.026 |
| lex_repetition_rate | en,es,zh | 3 | 19.333 | 0.027 |
| lex_type_token_ratio | en,es,zh | 3 | 22.000 | 0.019 |
| lex_content_word_ratio | en,es,zh | 3 | 22.000 | 0.013 |
| disc_repeated_unigram_ratio | en,es,zh | 3 | 22.333 | 0.017 |
| lex_mattr_10 | en,es | 2 | 4.000 | 0.037 |
| len_type_count | en,zh | 2 | 6.500 | 0.020 |
| lex_brunet | en,es | 2 | 10.000 | 0.010 |

Across `en`, `es`, and `zh` all-task models, the most repeated overlap is lexical diversity and repetition structure rather than acoustics. `lex_mattr_20`, `lex_lemma_type_token_ratio`, and repetition-based discourse measures recur across languages. That is the clearest broad multilingual stability signal in the current benchmark.

### PD_CTP cross-lingual overlap

| feature_group | languages_present | num_languages | mean_rank | mean_importance |
| --- | --- | --- | --- | --- |
| len | en,zh | 2 | 1.000 | 0.084 |
| ac | el | 1 | 1.000 | 0.061 |
| pause | zh | 1 | 2.000 | 0.050 |
| lex | en | 1 | 2.000 | 0.029 |
| syn | en | 1 | 3.000 | 0.022 |
| disc | en | 1 | 4.000 | 0.006 |

Within `PD_CTP`, the feature picture is much less stable across languages. English is led by lexical and syntactic structure, Chinese by pause/rate measures, and Greek by acoustic descriptors. That means the current benchmark does not yet support a claim that one small universal top-feature set explains `PD_CTP` impairment consistently across languages.

## Report takeaways

1. The additive rich benchmark is now complete: proper `openSMILE` acoustics, archive-aligned model families, nonlinear SVM coverage, and held-out permutation importance for every best model.
2. The benchmark-wide winner benefits from nonlinear interactions, but the strongest interpretable result is still the pooled `PD_CTP` slice.
3. Cross-lingual stability is stronger for broad lexical/discourse patterns than for a single shared `PD_CTP` biomarker set.
4. The next scientifically meaningful step is Phase 2 task-specific semantic extraction, especially content-unit coverage for `PD_CTP` and separate semantic modules for `READING`, `FLUENCY`, and `CONVERSATION`.

## What else from Balagopalan is still worth trying

- `Gaussian NB` and the shallow `NN` baseline. Those were part of the original ADReSS comparison and would complete the classical-model family beyond the current `LR / DT / RF / SVM` set.
- `LOSO` or grouped leave-one-subject-out validation on a cleaner matched subset. Balagopalan reported both held-out and cross-validation settings; for small task-specific slices this could expose instability better than a single split.
- `BERT + handcrafted feature fusion` as a secondary comparison, not as the interpretability core.
