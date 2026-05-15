# Statistical Methods

This note formalizes the three feature-selection / feature-interpretation patterns discussed for Phase 1.

Notation for one feature \(f\):

- AD samples: \(x_1, \dots, x_{n_1}\)
- HC samples: \(y_1, \dots, y_{n_2}\)
- AD mean: \(\bar{x}\)
- HC mean: \(\bar{y}\)
- AD sample variance: \(s_x^2\)
- HC sample variance: \(s_y^2\)
- Number of tested features in one language or run: \(m\)

## 1. ANOVA + permutation

This is the current predictive pipeline used for the main Phase 1 feature sweeps.

### Step 1: univariate ANOVA ranking

For each feature \(f_j\), compute an ANOVA \(F\)-statistic on the training split only:

\[
F_j = \frac{\text{between-class variance for } f_j}{\text{within-class variance for } f_j}
\]

In the two-class case, this is equivalent to asking whether the class means differ relative to within-class noise.

Features are ranked by decreasing \(F_j\), and the run keeps:

- the top \(k\) ranked features, or
- all usable features if \(k = \text{all}\)

This is a ranking rule, not a corrected significance rule.

### Step 2: fit classifier on selected features

Let the selected feature set be \(S\). A classifier \(g(\cdot)\) is fit on the training data restricted to \(S\).

### Step 3: held-out permutation importance

For each selected feature \(f_j \in S\):

1. Evaluate baseline test accuracy:

\[
A_{\text{base}} = \text{Acc}(g, X_{\text{test}})
\]

2. Randomly permute column \(f_j\) in the test set, breaking its relation to the label.

3. Recompute accuracy:

\[
A_{j,r}^{\text{perm}} = \text{Acc}(g, X_{\text{test}}^{(j,r)})
\]

where \(r\) indexes repeated shuffles.

4. Define permutation importance as the average accuracy drop:

\[
I_j = \frac{1}{R}\sum_{r=1}^{R} \left(A_{\text{base}} - A_{j,r}^{\text{perm}}\right)
\]

Interpretation:

- large \(I_j\): the fitted model relies strongly on feature \(f_j\)
- small \(I_j\): the model can largely recover without that feature

This is model-faithful importance, not a direct statistical test of AD vs HC mean difference.

## 2. Kruskal-Wallis + Bonferroni

This is the nonparametric language-specific significance approach used in the current Phase 1 significance tables.

### Step 1: per-feature Kruskal-Wallis test

For each feature \(f_j\), pool all AD and HC values and replace them with ranks. Let:

- \(R_x\) be the sum of ranks for AD
- \(R_y\) be the sum of ranks for HC
- \(N = n_1 + n_2\)

The Kruskal-Wallis statistic for two groups is:

\[
H = \frac{12}{N(N+1)}\left(\frac{R_x^2}{n_1} + \frac{R_y^2}{n_2}\right) - 3(N+1)
\]

This tests the null hypothesis:

\[
H_0: \text{AD and HC are drawn from the same feature distribution}
\]

It is a rank-based group-difference test. It does not directly test difference in means.

### Step 2: Bonferroni correction

Because we test many features, raw \(p\)-values are adjusted:

\[
p_j^{\text{bonf}} = \min(m \cdot p_j, 1)
\]

Equivalently, a feature is Bonferroni-significant at family-wise level \(\alpha = 0.05\) if:

\[
p_j < \frac{0.05}{m}
\]

Interpretation:

- very strict control of false positives
- suitable when we want a conservative list of individually differentiating features

## 3. Welch t-test + Bonferroni + Cohen's d

This is the proposed mean-difference-first alternative, closer to Balagopalan-style feature differentiation.

### Step 1: per-feature Welch t-test

For each feature \(f_j\), test:

\[
H_0: \mu_{AD} = \mu_{HC}
\]

using the Welch statistic:

\[
t_j = \frac{\bar{x} - \bar{y}}{\sqrt{\frac{s_x^2}{n_1} + \frac{s_y^2}{n_2}}}
\]

with Welch-Satterthwaite degrees of freedom:

\[
\nu_j \approx
\frac{\left(\frac{s_x^2}{n_1} + \frac{s_y^2}{n_2}\right)^2}
{\frac{\left(\frac{s_x^2}{n_1}\right)^2}{n_1-1} + \frac{\left(\frac{s_y^2}{n_2}\right)^2}{n_2-1}}
\]

This explicitly tests mean difference while allowing unequal variances and unequal sample sizes.

### Step 2: Bonferroni correction

Again adjust each feature's \(p\)-value across the \(m\) tested features:

\[
p_j^{\text{bonf}} = \min(m \cdot p_j, 1)
\]

Keep only the surviving features:

\[
\mathcal{S}_{\text{sig}} = \{ f_j : p_j^{\text{bonf}} < 0.05 \}
\]

### Step 3: rank surviving features by absolute Cohen's \(d\)

Among the corrected-significant features, compute a standardized mean difference:

\[
d_j = \frac{\bar{x} - \bar{y}}{s_{\text{pooled}}}
\]

with pooled standard deviation:

\[
s_{\text{pooled}} = \sqrt{\frac{s_x^2 + s_y^2}{2}}
\]

Then rank by:

\[
|d_j|
\]

Interpretation:

- sign of \(d_j\): which group has the higher mean
- magnitude of \(|d_j|\): how strongly separated the means are in standard deviation units

Why rank by \(|d|\) after significance filtering:

- Bonferroni says the feature survives multiple-testing control
- \(|d|\) then prioritizes the features with the clearest standardized AD/HC separation
- this is usually better than ranking only by \(p\), because \(p\) is affected by sample size as well as effect magnitude

## Practical comparison

### ANOVA + permutation

Best when the goal is:

- predictive performance
- model-faithful interpretation
- identifying what the fitted classifier actually used

### Kruskal-Wallis + Bonferroni

Best when the goal is:

- conservative nonparametric feature screening
- feature-wise AD/HC distribution differences
- robustness to non-normal feature distributions

### Welch t-test + Bonferroni + Cohen's \(d\)

Best when the goal is:

- conservative mean-difference screening
- cleaner AD-vs-HC feature separation for saliency-style plots
- ranking statistically credible features by separation strength rather than by model dependence
