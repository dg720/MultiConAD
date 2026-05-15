Create two visualisations for analysing whether AD speech features are translingual, language-sensitive, task-sensitive, or language-task confounded.

Context:
We have extracted a feature matrix from multilingual Alzheimer's speech data. Each sample has:
- diagnosis label: AD or HC/control for the binary feature experiments;
- language label: EN, EL, ZH, or ES;
- source-level task coverage from the paper/table convention:
  - PD = Picture Description
  - FT = Fluency Task
  - SR = Story Retelling
  - FC = Free Conversation
  - NA = Narrative
- live repo sample-level task labels from `data/processed/phase1/phase1_manifest.jsonl`, including:
  - `PD_CTP`, `READING`, `COMMAND`, `CONVERSATION`, `MIXED_PROTOCOL`, `PICTURE_RECALL`, `REPETITION`, `MOTOR_SPEECH`, `PROCEDURAL`, and `OTHER`;
- feature values from core, richer non-task, and task-specific groups.

Important dataset convention:
The paper/table task columns are source-level coverage indicators, not always one-to-one sample-level labels in the current manifest. For example, a source may include FT or SR tasks, while the live benchmark may expose those rows as `PD_CTP`, `OTHER`, `PICTURE_RECALL`, or another more specific manifest label after preprocessing. Therefore, every visualisation must report both:
- the paper-style/source task label where it is inferable: PD, FT, SR, FC, NA;
- the live manifest task label actually used in the feature matrix.

Use the paper-style labels in figure text for consistency:
- PD = Picture Description
- FT = Fluency Task
- SR = Story Retelling
- FC = Free Conversation
- NA = Narrative

Current live AD/HC cell coverage:
Only a subset of language-task cells has enough AD and HC groups for stable effect estimates. With a minimum of 10 AD groups and 10 HC groups, the robust live cells are:
- EN COMMAND
- ZH OTHER
- EL PD_CTP
- EN PD_CTP
- ZH PD_CTP
- ES READING

This means the figures can evaluate recurrence and confounding, but they should avoid claiming clean language-specificity for cells that are only observed in one language-task combination.

The key methodological principle is:
Features are considered language-sensitive only if their AD/control separation remains stable after accounting for task domain; otherwise, they are treated as task-sensitive, language-task confounded, or insufficiently supported.

Overall method:
1. Keep the predictive top-k experiments aligned with the existing feature-set runs:
   - ranking modes: ANOVA and Welch t-test + Bonferroni;
   - k values: 5, 10, 20, 50, 100, and all;
   - same model families, tuning protocol, grouping logic, and seed handling as the comparable classical ML runs.
2. Use simplified signed Cohen's d language-task analysis as the main interpretability layer:
   - compute AD/HC separation inside valid language-task cells;
   - summarise diagnostic strength, direction consistency, language specificity, and task specificity;
   - classify features as translingual candidates, language-sensitive candidates, task-sensitive, language-task confounded, or unstable/insufficient.
3. Use regression interaction models only as a robustness check for top selected features, not as the default visualisation method.

Create two figures:

1. A two-axis language-vs-task specificity scatter plot.
2. A language-task feature recurrence heatmap.

The figures should focus on AD/control effect patterns, not raw feature values.

==================================================
FIGURE 1: LANGUAGE VS TASK SPECIFICITY SCATTER
==================================================

Goal:
Create a scatter plot where each point is one feature. The x-axis measures task specificity and the y-axis measures language specificity.

The plot should help classify features into four regions:
- low task specificity, low language specificity: translingual/general candidate features;
- low task specificity, high language specificity: language-sensitive candidates;
- high task specificity, low language specificity: task-sensitive features;
- high task specificity, high language specificity: language-task confounded features.

Primary method for this repo:
Use cell-wise AD/HC effect sizes rather than a full interaction model as the default, because the live data are not fully crossed by language and task.

For each feature:
1. Use only samples where the feature is applicable and non-missing.
2. Collapse repeated rows by `group_id` before cell-level effect estimation, or otherwise ensure participants/groups are not overweighted.
3. For each valid language-task cell, compute Cohen's d for AD vs HC.
4. Mark a cell valid only if it has enough AD and HC groups; default minimum: at least 10 AD groups and 10 HC groups.
5. Compute:
   - diagnostic strength = mean absolute Cohen's d across valid cells;
   - direction consistency = proportion of valid cells sharing the modal AD/HC direction;
   - task specificity = variance of cell effect sizes across task labels after averaging across languages where possible;
   - language specificity = variance of cell effect sizes across languages after averaging across task labels where possible.

Secondary method:
For subsets with enough crossing, especially PD across EN/EL/ZH, optionally fit:

feature_value_f ~ diagnosis + language + diagnosis:language

For broader task analysis, the full model below can be attempted only when cell coverage supports it:

feature_value_f ~ diagnosis + language + task + diagnosis:language + diagnosis:task

If the model is rank-deficient or unstable, use the cell-wise effect-size method above.

Important:
The scores must reflect variation in the AD/control gap, not baseline language or task differences in feature values.

Visual design:
- x-axis: task specificity score.
- y-axis: language specificity score.
- each point: one feature.
- point colour: actual feature family:
  - length/rate: `len`
  - lexical: `lex`
  - pauses/fluency: `pause`
  - discourse/graph: `disc`, `graph`
  - syntax/phrase production: `syn`, `sx`, `pr`
  - acoustic/prosodic/paralinguistic: `ac`, `par`
  - task-specific: `pd`, `rd`, `fc`, `ft`, `sr`, `cmd`, `rep`, `ms`, `na`
- point size: diagnostic strength.
- point transparency: evidence strength, based on the number of valid language-task cells.
- marker edge or shape: direction consistency.

Quadrants:
Add faint vertical and horizontal threshold lines. Use median or 75th percentile thresholds after inspecting score distributions. Label the four regions:
- Translingual candidates
- Language-sensitive candidates
- Task-sensitive
- Language-task confounded

Feature labels:
Only label the most important/interpretable features:
- top 5 strongest translingual candidates;
- top 5 highest language-specificity candidates;
- top 5 highest task-specificity candidates;
- known clinically relevant features such as pause rate, speech rate, F0 variability, PD content-unit count, type-token ratio, and MATTR.

Recommended split:
Produce two scatter variants if possible:
1. Main: valid live language-task cells, with strong confounding caveats.
2. PD-only: EN/EL/ZH PD_CTP cells, focused on cleaner task-controlled language sensitivity.

Output:
- `language_vs_task_specificity_scatter.png`
- `language_vs_task_specificity_scatter.pdf`
- `language_vs_task_specificity_scores.csv`

The CSV should include:
- feature name
- feature family
- diagnostic strength
- language specificity score
- task specificity score
- direction consistency score
- number of valid language-task cells
- paper-style task labels observed, where inferable
- live manifest task labels observed
- assigned interpretation category

Suggested interpretation categories:
- translingual candidate
- language-sensitive candidate
- task-sensitive
- language-task confounded
- unstable / insufficient evidence

==================================================
FIGURE 2: LANGUAGE-TASK FEATURE RECURRENCE HEATMAP
==================================================

Goal:
Create a heatmap showing where selected features are significant or highly ranked across language-task cells.

Rows:
Features.

Columns:
Use readable paper-style task labels where possible, while retaining manifest labels in metadata. Example column labels:
- EN PD
- EL PD
- ZH PD
- ES Reading / RD
- EN Command / CMD
- ZH Other / unresolved source task

Cells:
Show whether the feature separates AD vs HC in that language-task cell, including direction and strength.

Preferred cell encoding:
- red shades: higher in AD
- blue shades: lower in AD
- white: not significant / not selected
- grey: not applicable to that task
- low-opacity grey or hatch: insufficient data

Use Cohen's d as colour intensity.

Overlay symbols:
- `*` = significant after Bonferroni correction
- `.` = selected in top-k but not significant after correction
- `NA` = task-specific feature not applicable

Important:
The heatmap must show directionality. A feature that recurs but flips AD/HC direction should not be interpreted as a clean translingual marker.

Discovery phase:
For each valid language-task cell:
1. Rank applicable features by AD/HC separation using Welch t-test + Bonferroni and absolute effect size.
2. Optionally include ANOVA ranking for comparison, because our model sweeps report both ANOVA and Welch+Bonferroni.
3. Store top 100, 50, 20, 10, and 5 features, plus the all-feature reference, to match the current top-k experiment grid.
4. Record effect direction: higher in AD, lower in AD, or inconsistent.
5. Record corrected significance using 5% Bonferroni correction.

Heatmap inclusion rule:
Include a feature if it satisfies at least one of:
1. Robust recurrence:
   - appears in the top 50 in at least two valid language-task cells.
2. Strong local signal:
   - appears in the top 10 in at least one valid language-task cell and has |Cohen's d| > 0.5.
3. Clinically important family:
   - belongs to pause/fluency, speech rate, F0/prosody, lexical diversity, syntactic complexity, information density, or task-specific content groups, and appears in the top 100 somewhere.

Then cap the main heatmap to 25-40 features using a stability score.

Suggested stability score:
- top 5 = 4 points
- top 10 = 3 points
- top 20 = 2 points
- top 50 = 1 point
- not selected = 0 points

Sort features by:
1. interpretation category:
   - translingual
   - language-sensitive
   - task-sensitive
   - language-task confounded
   - unstable / insufficient evidence
2. feature family
3. stability score, descending

Interpretation categories:
- Translingual:
  Feature is selected/significant in multiple languages and at least one comparable task family, with consistent AD/HC direction. Strongest evidence currently comes from recurrence across EN/EL/ZH PD.
- Language-sensitive:
  Feature is selected/significant mainly in one language after controlling for task where possible, especially within PD.
- Task-sensitive:
  Feature is selected/significant across multiple languages within the same task family, but not across task families.
- Language-task confounded:
  Feature appears mainly in one language-task cell, so language and task effects cannot be separated.
- Unstable / insufficient evidence:
  Feature appears inconsistently, flips AD/HC direction, or appears only in underpowered cells.

Visual design:
- Main heatmap should be compact and readable: 25-35 features.
- Use grouped row annotations for feature family.
- Use grouped column annotations for language and task.
- Clearly mark NA cells for task-specific features that cannot apply to a task.
- Cluster rows only after filtering to avoid clustering noise.
- Prefer column order grouped by task for the main figure, because task confounding is the central methodological risk.
- Optionally provide a second version grouped by language.

Recommended main heatmap:
- Rows: 25-35 selected features.
- Columns: valid language-task cells plus clearly marked insufficient cells if needed for transparency.
- Cell colour: Cohen's d, centred at zero.
- Grey: not applicable.
- White: not selected/non-significant.
- Symbol overlay: significance/top-k status.
- Side annotation: feature family.
- Right-side label: interpretation category.

Optional appendix heatmap:
Create a larger appendix heatmap using the full union of top 100 features, clustered, for transparency. The main report should use the filtered heatmap.

Output:
- `language_task_feature_heatmap.png`
- `language_task_feature_heatmap.pdf`
- `language_task_feature_heatmap_data.csv`

The CSV should include:
- feature name
- feature family
- selected cells
- top-k ranks per cell
- Cohen's d per cell
- p-value and corrected p-value per cell
- direction per cell
- stability score
- paper-style task labels
- live manifest task labels
- interpretation category

==================================================
GENERAL DESIGN CONSIDERATIONS
==================================================

The two figures should work together:
- The heatmap shows the direct evidence: where each feature appears and in which direction.
- The scatter plot summarises whether the pattern looks language-sensitive, task-sensitive, translingual, or confounded.

Avoid overclaiming:
- Do not label features as language-specific simply because they appear in one language.
- If a feature appears only in one language-task cell, label it as language-task confounded.
- Prefer terms like "language-sensitive candidate" and "translingual candidate feature" rather than definitive biomarker claims.

Handle sparse data carefully:
- Minimum cell size must be enforced before computing cell-level effects.
- If a cell has too few AD or HC groups, mark it as insufficient rather than treating the feature as absent.
- Use transparency or confidence annotations to show evidence strength.
- The paper-style task coverage table shows broader source task availability than the live manifest exposes, so missing live cells should not be described as missing from the original source.

Model/validation consistency:
- Use the same train/test grouping logic as the modelling experiments.
- Rank features on training folds/splits where the visualisation is linked to model performance.
- For purely descriptive full-dataset figures, label them explicitly as descriptive and do not mix them with held-out performance claims.
- Use grouped `group_id` aggregation or grouped resampling to avoid overweighting participants with multiple samples.

ASR and transcript caveat:
- ASR-derived datasets keep one combined transcript if both participant and interviewer are audible.
- Participant/interviewer separation is only reliable for speaker-coded sources such as CHA/TSV.
- Pause and acoustic features are not diarisation-aware, so interviewer prompts and recording gaps can affect these features in ASR-derived samples.

Output location:
Save visualisation outputs under:

`tables/03-ablation-translingual-language-specific/language-task-visualisations/`

Use:
- `result-tables/` for readable text summaries;
- `result-tables/csv/` for score/rank/effect CSVs;
- `report_assets/` for PNG/PDF figures;
- `summaries/` for JSON run metadata.

Final expected deliverables:
1. `report_assets/language_task_feature_heatmap.pdf`
2. `report_assets/language_task_feature_heatmap.png`
3. `result-tables/csv/language_task_feature_heatmap_data.csv`
4. `report_assets/language_vs_task_specificity_scatter.pdf`
5. `report_assets/language_vs_task_specificity_scatter.png`
6. `result-tables/csv/language_vs_task_specificity_scores.csv`
7. `result-tables/language_task_visualisation_readme.txt`
