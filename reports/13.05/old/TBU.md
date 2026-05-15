# TBU

## Pending Phase 2 correction

The current Phase 2 clean-slice report tables are **interim**.

Phase 2 differs from Phase 1 because it introduces **task-specific features**. However, the final reported Phase 2 results should **not** be restricted to one hand-picked subset such as `rich_universal`, `semantic_plus_pause`, or `task_specific_semantic`.

The intended final Phase 2 rule is:

- use the broader `all_phase2` pool
- let task-specific and richer universal features compete jointly
- select top `k` features from that combined Phase 2 pool
- then fit the classifier on that selected joint feature set

In other words: task-specific features should be **added to the overall Phase 2 pool**, not used to define isolated final reporting subsets.

## How important features are currently selected

There are **two different stages** here:

1. **Feature selection for model fitting**

- features are ranked with **ANOVA `f_classif`** on the **training split only**
- the run keeps the top `k` ranked features from the chosen pool or subset
- `k` is therefore a pruning parameter, not a significance threshold

2. **Feature importance for interpretation**

- the primary importance ranking is **held-out permutation importance**
- this is computed **after** the model is fit, on the held-out test split
- native model importances are only secondary:
  - `LR` / linear `SVM`: coefficient magnitude
  - `DT` / `RF`: impurity-based feature importance

So the current “top features” tables are **not** driven only by coefficient size, and they are **not** filtered by a hard ANOVA significance cutoff such as `p < 0.05`.

The current logic is:

- ANOVA ranking chooses which features enter the model
- permutation importance is the main post hoc ranking used to interpret the fitted model

## Are the feature-based models smaller / less tuned than the full benchmark?

Yes, in practice the feature sweeps currently use a **narrower search space** than the full benchmark.

What the feature sweeps currently do:

- one split per run
- a fixed classical model family:
  - `LR`
  - `DT`
  - `RF`
  - linear `SVM`
  - RBF `SVM`
- one very small parameter setting per family:
  - `LR C=1`
  - `DT max_depth=20`
  - `RF n_estimators=200`
  - `SVM linear C=1`
  - `SVM rbf C=1`

What the text benchmark does that is broader:

- evaluates multiple representation settings
- compares monolingual / multilingual / translated settings
- includes **fusion ensembles**
- aggregates over multiple seeds

So the feature-based experiments are not necessarily using “smaller” models in an absolute sense, but they **are** using:

- less hyperparameter tuning
- fewer model variants
- no ensemble fusion
- a narrower evaluation protocol

## Follow-up experiment to run

### Goal

Take the best feature pools and feature counts, then tune the **full model family** for best performance rather than relying on one fixed parameter setting per family.

### Recommended design

1. **Phase 1 follow-up**

- use the **combined feature pool only**
  - `all_universal`
- evaluate each language and the pooled benchmark
- carry forward promising `k` values from the current pruning tables:
  - `all`, `100`, `50`, `25`, `10`

2. **Phase 2 follow-up**

- use the **full combined Phase 2 pool**
  - `all_phase2`
- do **not** hard-restrict the final comparison to one subset
- allow task-specific features to compete inside the combined pool
- again carry forward promising `k` values

3. **Tune the full classical model family**

- `LR`: tune `C`
- `DT`: tune `max_depth`, `min_samples_split`, `min_samples_leaf`
- `RF`: tune `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`
- linear `SVM`: tune `C`
- RBF `SVM`: tune `C` and `gamma`

4. **Add ensemble follow-up**

- once the best tuned single models are known, test:
  - soft-vote ensembles
  - simple score averaging over top tuned models

5. **Use raw accuracy as the primary report metric**

- keep balanced accuracy, AUROC, and AUPRC as supporting metrics
- report raw accuracy first to stay aligned with the benchmark text tables

6. **Track cross-language feature stability explicitly**

- identify which features recur as important across languages
- these are the strongest candidates for genuinely translingual / generalisable biomarkers
- useful summaries include:
  - number of languages in which a feature appears in the top `N`
  - average rank across languages
  - average permutation importance across languages
  - overlap at the feature-group level as well as the individual-feature level

Suggested outputs:

- a compact table of features shared across `en / zh / el / es`
- a “feature stability” figure, for example:
  - average-rank bar chart
  - heatmap of rank by feature x language
  - overlap summary at the feature-group level

This would make the translingual story much clearer:

- language-specific features explain what is most diagnostic locally
- cross-language recurring features explain what is most generalisable

7. **Test hard ANOVA thresholding as a principled inclusion rule**

- current runs use ANOVA mainly as a ranking device for top-`k` pruning
- a future follow-up should also test hard inclusion rules such as:
  - Bonferroni-corrected threshold
  - FDR-controlled threshold
  - fixed nominal threshold for exploratory comparison

This would let us compare:

- top-`k` selection for predictive performance
- significance-threshold inclusion for a more principled statistical feature set

8. **Report ANOVA significance levels alongside selected features**

- we do already compute ANOVA rankings and associated `p` values in the raw experiment artifacts
- however, we are not yet systematically reporting ANOVA significance levels next to the final selected-feature tables in the report folder
- a follow-up report table should add, for each selected feature:
  - ANOVA `f` score
  - raw `p` value
  - corrected significance status where relevant
    - Bonferroni
    - FDR

This would make it easier to distinguish:

- features selected mainly because they help prediction in combination
- features that are also individually statistically strong under univariate testing

### Practical report framing

The current feature-based runs are useful for:

- identifying which feature pools matter
- identifying which feature groups recur
- identifying promising pruning levels

The next experiment should answer a different question:

- **if we take the best handcrafted feature pools and then tune the full model family properly, how close can feature-based methods get to the strongest benchmark models?**
