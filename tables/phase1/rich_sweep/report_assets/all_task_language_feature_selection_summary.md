# Phase 1 All-Task Feature Selection Summary

Text baseline columns are accuracy percentages from `paper_vs_ours_3tables.txt`. Feature benchmark columns are single-seed raw accuracy values from the Phase 1 rich sweep; balanced accuracy is retained in the CSV for reference.

| Language | Best Text Mono Acc | Best Text Combined Acc | Best Feature Local Acc | Local Feature Set | Delta vs Global Feature | Delta vs Full Universal | Top Local Feature Groups |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Chinese | 84.1 | 83.9 | 87.5 | text_plus_pause k=100 svm:rbf_c1 | +8.5 | +6.2 | len, pause, lex, disc |
| English | 88.9 | 87.8 | 86.4 | text_plus_pause_plus_graph k=100 rf:n200 | +7.4 | +0.6 | lex, syn, len, disc |
| Greek | 79.1 | 76.7 | 85.7 | acoustic_only k=100 svm:rbf_c1 | +6.7 | +2.4 | ac |
| Spanish | 74.9 | 75.6 | 85.2 | all_universal k=all rf:n200 | +6.2 | +0.0 | len, disc, lex, syn |

- **Chinese**: Chinese benefits from a monolingual text+pause fit; pause and rate features become more discriminative once English and Greek are removed.
- **English**: English improves materially under raw accuracy once the feature fit is allowed to specialize within language; the strongest slice uses a broader text-plus-graph feature mix.
- **Greek**: Greek benefits the most from monolingual specialization; acoustic-only features dominate and outperform the pooled benchmark clearly.
- **Spanish**: Spanish improves over the pooled benchmark, but the best raw-accuracy model no longer prefers a tiny monolingual text subset; the full universal feature pool is competitive once accuracy becomes primary.
