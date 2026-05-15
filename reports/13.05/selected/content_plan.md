# Content Plan For `reports/05-13/main.tex`

This plan assumes:

- keep Section `4.1` (`Dataset Breakdown`) as-is
- keep Tables `9-11` as-is
- keep Section `4.4` (`Experimental Plan`) as-is
- replace only the narrative/content from Section `4.2` onward using the selected bundle below

## Proposed outline from 4.2 onward

### `4.2 Full Results & Ensembling`

- Keep the current binary result tables (`tab:binary_mono`, `tab:binary_multi`, `tab:binary_trans`) unchanged.
- Replace the current prose around those tables with a short results-focused summary:
  - strongest monolingual language/model pattern
  - where multilingual / translated training helps
  - where it hurts
  - one brief sentence on late fusion / best-case comparison
- End with one concise transition from benchmark results into feature analysis.

Suggested paragraph order:

1. One opening paragraph summarising the pooled-language result pattern.
2. One short paragraph comparing language-specific best cases against pooled or translated settings.
3. One transition paragraph explaining why feature analysis is needed next.

### `4.3 Feature Benchmarking`

#### `4.3.1 Feature Inventory`

- Replace the current inventory table/text with a shorter Phase 1 / Phase 2 feature inventory summary.
- Focus on:
  - universal core feature count
  - richer Phase 2 additions
  - task-specific feature count
  - one sentence on why task-aware analysis is necessary

#### `4.3.2 Feature Selection / Statistical Methods`

- Introduce the three methods briefly:
  - ANOVA + permutation for predictive ranking
  - Kruskal-Wallis + correction for conservative univariate significance
  - Welch + Bonferroni + Cohen's `d` for mean-difference-first saliency
- Keep this as short explanatory prose, not a long mathematical appendix.

#### `4.3.3 Phase 1 Language Results`

- Add a compact table or paragraph summary of the best `all_universal` setup per language.
- Highlight pruning sensitivity only briefly:
  - pooled model prefers `k=50`
  - English / Greek / Spanish best cases differ
  - Chinese is comparatively stable across pruning

#### `4.3.4 What The Best Models Actually Use`

- Use the top-feature summaries to describe what dominates each language:
  - Chinese: acoustic-heavy with some lexical/pause features
  - English: syntax/discourse-heavy
  - Greek: strongly acoustic
  - Spanish: lexical/discourse/graph-heavy

#### `4.3.5 Significance vs Model Dependence`

- Replace the current generic feature-ranking discussion with a cleaner comparison:
  - significance and permutation answer different questions
  - overlap is high only in English
  - mismatch is strongest in Chinese, still substantial in Greek/Spanish
- Use one concise conclusion line: predictive saliency is not automatically a biomarker table.

#### `4.3.6 Saliency Figures`

- Show the pooled-language overlay and the local-language overlay as the main figure pair.
- Add a short interpretation paragraph:
  - shared multilingual signal exists
  - language-specific feature selection sharpens separation
  - Greek depends most on local adaptation
- Optionally add the local best-case pair figure as a secondary figure or an appendix-style in-text reference if space is tight.

## File-to-section mapping

| File | Role | Target location in report | How to use it |
| --- | --- | --- | --- |
| `01_phase1_combined_pool_language_summary.txt` | results summary | `4.2` closing paragraph and `4.3.3` | Use the best per-language / pooled accuracies and pruning sensitivity to summarise where the universal feature pool works best. |
| `02_phase1_combined_pool_top_features.txt` | commentary / feature summary | `4.3.4` | Use for one short paragraph describing dominant feature families by language. |
| `11_overall_feature_inventory_summary.txt` | inventory source | `4.3.1` | Use to replace the current inventory text/table with a shorter count-and-category summary. |
| `13_phase1_language_significance_vs_permutation.txt` | commentary / comparison | `4.3.5` | Main source for the significance-vs-permutation narrative and per-language overlap statements. |
| `16_bonferroni_saliency_vs_permutation.txt` | commentary / comparison | `4.3.5` and `4.3.6` | Use to explain when Bonferroni saliency is cleaner and when permutation saliency remains more faithful to the model. |
| `17_statistical_methods.md` | methods source | `4.3.2` | Condense into a short prose explanation of the three statistical/selection views. |
| `25_saliency_overlay_pooled_welch_median_iqr_hc_oriented_by_language.png` | figure | `4.3.6` | First figure in the saliency comparison pair; use as the shared multilingual shortlist view. |
| `29_saliency_overlay_local_welch_median_iqr_hc_oriented_by_language.png` | figure | `4.3.6` | Second figure in the saliency comparison pair; use as the language-specific shortlist view. |
| `31_selected_comparison_note.txt` | interpretation note | `4.3.6` | Primary source for the caption/interpretation of the pooled-vs-local saliency figure pair. |
| `32_saliency_pair_local_welch_hc_oriented_bestcase_by_language.png` | figure | `4.3.6` secondary figure or optional appendix-style placement | Use only if space allows; supports the case-level “near-empty AD vs near-full HC” interpretation. |
| `33_local_welch_bestcase_pairs.csv` | data support | supports `32`; mention only if needed | Do not render directly in the main report; use only as backing metadata for the best-case pair figure or caption. |
| `34_local_welch_bestcase_pairs_note.txt` | commentary / caption support | supports `32` in `4.3.6` | Source for explaining how the best-case AD/HC examples were chosen and what they show. |

## Replacement guidance for the existing report

### Keep unchanged

- Section `4.1`
- Table `9`
- Tables `10-11` as requested
- Section `4.4`

### Replace

- the current narrative immediately before and after the binary result tables in `4.2`
- the current generic `4.3` feature inventory prose
- the current broad feature-model / feature-importance / t-SNE narrative if it is not directly supported by the selected bundle

### Drop from the current `4.3` draft

- material that depends on non-selected tables or figures
- repeated warnings about benchmark heterogeneity once the point has been made once
- extra interpretability claims not supported by the selected significance / saliency notes

## Writing target

- Keep `4.2` short and benchmark-facing.
- Let `4.3` do the interpretive work.
- Prefer a few concrete comparative claims over a long methods recap.
- Treat the pooled-vs-local saliency figure pair as the main scientific bridge from benchmark accuracy to biomarker interpretation.
