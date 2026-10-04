# Campaign Experiment Analysis: Technical Documentation

This document explains how the project works, module by module, and why
each design decision was made. For setup and usage see `README.md`; for a
reproduction walkthrough see `PROJECT_INSTRUCTIONS.md`.

Every number quoted here comes from `reports/results.json`, produced by
`campaign-experiment run` on the real data. Where a number would change if
the analysis were rerun with different settings, the setting is named.

---

## 1. Executive summary

A randomised experiment sent 64,000 retail customers either a Mens email, a
Womens email, or nothing. The project answers whether either email raised
revenue, whether that is profitable, and which to send, following a plan
committed before any outcome was analysed.

| Question | Answer |
|---|---|
| Did randomisation work? | Yes. SRM p 0.904; largest covariate imbalance \|SMD\| 0.014 |
| Did the Mens email raise revenue? | Yes, +$0.77 per customer (95% CI +$0.50 to +$1.06) |
| Did the Womens email? | Yes, +$0.42 per customer (95% CI +$0.17 to +$0.68) |
| Which is better? | Mens, by $0.35 per customer (95% CI +$0.03 to +$0.66, p 0.03) |
| Is it profitable? | At the assumed 40% margin and $0.05 per email, Mens nets +$0.26 per email; it breaks even at a 6.5% margin |
| Recommendation | Send the Mens email |

---

## 2. Design principles

1. **Plan first.** `docs/TEST_PLAN.md` fixes the primary metric, tests,
   thresholds, segments and decision rule. Its commit precedes the first
   commit that computes an effect.
2. **Validity before effects.** Sample ratio and covariate balance are
   computed before the primary test, and the decision rule requires them.
3. **Define metrics once.** Per-arm and per-segment metrics live in SQL
   files. The Python statistics consume their output.
4. **Two independent routes to each interval.** Welch from summary
   statistics and a bootstrap from rows; they must agree.
5. **Every family of tests is corrected.** Holm within the primary pair,
   the secondary four, and each email's segment family.
6. **No number typed by hand.** The report and memo are rendered from the
   results dictionary, and a test compares the committed reports with a
   fresh run byte for byte.
7. **Report what happened.** Findings that weaken the story (a negligible
   CUPED gain, a non-binding robustness check) are reported, with the
   reason.

---

## 3. Architecture

```
               params.yaml
                    │  load_config(): typed, validated, frozen
                    v
   ┌──────────── Config ─────────────┐
   │                                  │
   v                                  v
data.py                         analysis.py ──────────────────────────────┐
 ensure_dataset()                 1 checks.srm_check, covariate_balance   │
  ├ cache hit (SHA-256 ok)        2 stats.welch_test + bootstrap_diff_ci  │
  ├ download primary (HTTP)       3 winsorised checks (planned, post-hoc) │
  └ fallback mirror (HTTPS)       4 stats.two_proportion_ztest            │
 validate() -> add_features()     5 head-to-head Welch                    │
        │                         6 variance_reduction (CUPED, Lin)       │
        v                         7 power at observed baseline            │
   DataFrame ──> metrics_sql.py   8 segments.segment_effects              │
                 DuckDB + SQL     9 economics + decision                  │
                                 10 peeking simulation, uplift T-learner  │
                                                                          v
                                  report.py: results.json, REPORT.md, DECISION_MEMO.md
                                  figures.py: four PNG figures
```

The pipeline is a single process with no server. DuckDB runs in memory on
the validated DataFrame, which keeps SQL as the metric definition language
without operating a database.

---

## 4. Module reference

### 4.1 `config.py`

Loads `params.yaml` into frozen dataclasses (`DataConfig`,
`ExperimentConfig`, and so on). `_section` rejects unknown and missing keys,
so a typo such as `alhpa` fails at load time rather than silently using a
default. `_validate` checks cross-field rules: allocation sums to 1, one
share per arm, alpha in (0, 1), the uplift arm is a treatment arm.

### 4.2 `data.py`

| Function | Responsibility |
|---|---|
| `ensure_dataset` | Return a verified local file; download only if no verified copy is cached |
| `_fetch` | Download to `*.part`, rename on success; retry with backoff 1 s, 2 s |
| `sha256_of` | Streamed digest (1 MB blocks) |
| `validate` | Collect every schema and integrity problem, then raise once |
| `add_features` | Derive `arm` (ordered categorical) and `purchase_line` |
| `load_dataset` | The single entry point used by the pipeline |

**Integrity rules enforced** (all hold on the real file):

- Exact column set; no nulls; numeric types where expected.
- Binary columns in {0, 1}; `recency` in 1..12; `history`, `spend` >= 0.
- Every arm present and no unknown arm.
- `conversion = 1` implies `visit = 1`.
- `spend > 0` exactly when `conversion = 1`.
- Every customer bought mens or womens merchandise (or both).
- `history_segment` agrees with the dollar value of `history`.

**Duplicates are kept.** 6,562 rows are identical to another row. There is
no customer id, so two customers with the same attributes and no purchase
produce identical rows. Deleting them would change the experiment's sample.

### 4.3 SQL layer: `sql/*.sql` and `metrics_sql.py`

`arm_metrics.sql` returns per-arm count, visits, conversions, revenue, mean
and sample variance of spend, rates, and average order value (descriptive
only, via `FILTER (WHERE conversion = 1)`). Window functions
(`SUM(n) OVER ()`) add each arm's share of customers and revenue.

`segment_metrics.sql` uses `UNPIVOT` to turn the segment columns into
`(segment, level)` rows, so one `GROUP BY` covers all segments, and
`COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY segment, level)` gives the
arm share within each level. Segment column names are inserted into the
query from a whitelist (`SEGMENT_COLUMNS`); anything else is rejected
before it reaches SQL.

The SQL output is exactly the sufficient statistics (mean, variance, n)
that the Welch and z-tests need, so the inference runs on aggregates.
Tests check the SQL against pandas `groupby`.

### 4.4 `checks.py`

- `srm_check`: chi-square goodness of fit of arm counts against the
  planned allocation. Threshold p < 0.001, because a mismatch signals broken
  assignment or logging, which invalidates every estimate.
- `covariate_balance`: standardised mean difference
  `(mean_t - mean_c) / sqrt((var_t + var_c) / 2)` for five numeric
  covariates and one-hot `zip_code` and `channel` levels. SMD is used rather
  than a p-value because it does not shrink toward "significant" as the
  sample grows.

### 4.5 `power.py`

Normal-approximation MDE for means and proportions, the two-proportion
sample size formula (pooled variance under the null, separate variances
under the alternative), and achieved power. `mde_mean` is tested against
`statsmodels.stats.power.NormalIndPower`.

### 4.6 `stats.py`

| Function | Notes |
|---|---|
| `welch_test` | Welch-Satterthwaite degrees of freedom; t-based CI; handles zero variance |
| `two_proportion_ztest` | Pooled SE for the test statistic, unpooled SE for the CI |
| `bootstrap_diff_ci` | Resamples each arm separately (randomisation fixed arm sizes); drawn in chunks of 100 resamples so memory is bounded |
| `holm` | Step-down adjusted p-values, monotone and capped at 1 |
| `winsorize` | Caps at a value computed once on the pooled sample |

All are tested against `scipy.stats.ttest_ind(equal_var=False)`,
`statsmodels` `proportions_ztest`, `confint_proportions_2indep` and
`multipletests(method="holm")`.

### 4.7 `variance_reduction.py`

- **CUPED:** `Y_adj = Y - theta (X - mean X)` with
  `theta = cov(Y, X) / var(X)` on the pooled two-arm sample. Unbiased
  because X is measured before treatment. It removes a share of variance
  equal to `corr(Y, X)^2`; a test asserts exactly this.
- **Lin (2013) regression adjustment:** OLS of spend on treatment, centred
  covariates and treatment-by-covariate interactions, with HC2 robust
  standard errors (statsmodels). Centring makes the treatment coefficient
  the average effect; interactions keep it consistent even if the linear
  model is misspecified.

### 4.8 `segments.py`

Welch test per (segment, level) from the SQL segment table. Holm is applied
across every level of every segment for one email, because a reader
scanning the table for "where it works" is running all of those tests.
`harm` marks Holm-significant negative effects, which the decision rule
forbids.

### 4.9 `economics.py`

`net profit per email = gross_margin x revenue lift - cost_per_email`. The
profit interval maps the bootstrap CI of the lift through the same formula.
`break_even_margin = cost / lift`; a value above 1 means no margin covers
the cost.

### 4.10 `decision.py`

`ArmDecision` holds the five pre-registered criteria; `qualifies` requires
all of them. `decide` returns the qualifying arm with the highest expected
net profit, or `None`.

### 4.11 `peeking.py`

A/A simulation: both arms share the observed control conversion rate.
Customers arriving between looks are drawn as one binomial count per look,
which gives exactly the same distribution of cumulative conversions as
simulating every customer, using 2,000 x 20 numbers instead of
2,000 x 21,000. False-positive rates are reported for 1, 2, 3, 5, 10 and 20
evenly spaced looks, with and without Bonferroni across looks.

### 4.12 `uplift.py`

T-learner: one logistic regression (standardised features) fitted on
treated customers, one on control customers, using pre-period features
only. Predicted uplift is the difference of the two probabilities. Because
individual uplift is never observed, evaluation is on a held-out 30% split
(stratified by arm and outcome):

- **Decile table:** observed treated-minus-control rate within each
  predicted-uplift decile.
- **Qini curve:** cumulative incremental outcomes when targeting customers
  in score order, `Y_t - Y_c x N_t / N_c` within the top slice.
- **Normalised Qini coefficient:** area between the curve and random
  targeting, divided by the total incremental outcome (0 = random).

### 4.13 `analysis.py`, `report.py`, `figures.py`, `cli.py`

`run_analysis` executes the plan in order and returns one dictionary.
`write_outputs` converts it to JSON-safe types (NaN and infinity become
null), then renders `REPORT.md`, `DECISION_MEMO.md` and four figures.
Every sentence that states a check result reads that result: if the sample
ratio check failed, the memo says so (a test feeds it a failed check). Files
are written with LF line endings and no timestamp, so an unchanged analysis
produces identical bytes on any OS.

The CLI (`campaign-experiment`) has three commands: `download`, `plan`
(MDE grid from assumed baselines, no outcome data) and `run`. Expected
failures (bad config, failed download, failed validation) print one clear
message and exit with code 1.

---

## 5. Data flow

| Step | Input | Transformation | Output |
|---|---|---|---|
| 1 | URL from params | Download, verify SHA-256 | `data/raw/hillstrom.csv` |
| 2 | CSV | `read_csv`, `validate` | Validated DataFrame (64,000 x 12) |
| 3 | DataFrame | `add_features` | + `arm`, `purchase_line` |
| 4 | DataFrame | DuckDB `arm_metrics.sql` | 3 rows of sufficient statistics |
| 5 | Arm counts | Chi-square vs allocation | SRM result |
| 6 | Covariates | SMD per covariate per treatment | 22-row balance table |
| 7 | Arm statistics | Welch per treatment, Holm | Primary estimates |
| 8 | Spend rows | Bootstrap, 2,000 resamples per arm pair | Bootstrap CIs |
| 9 | Spend rows | Cap, Welch, Holm | Robustness checks |
| 10 | Arm counts | z-tests, Holm across four | Secondary estimates |
| 11 | Rows | CUPED, Lin OLS | Adjusted estimates and variance removed |
| 12 | DataFrame | DuckDB `segment_metrics.sql` | 33 (segment, level, arm) rows |
| 13 | Segment table | Welch per level, Holm per email | Segment effects, harm flags |
| 14 | Lift and CI | Margin x lift - cost | Economics |
| 15 | All criteria | Decision rule | Recommended arm |
| 16 | Control rate | A/A simulation | Peeking table |
| 17 | Rows | T-learner, held-out evaluation | Deciles, Qini |
| 18 | Results | Render | `reports/` |

---

## 6. Results in detail

### 6.1 Arm summary

| Arm | Customers | Visit rate | Conversion | Revenue / customer |
|---|---|---|---|---|
| No E-Mail | 21,306 | 10.62% | 0.57% | $0.65 |
| Mens E-Mail | 21,307 | 18.28% | 1.25% | $1.42 |
| Womens E-Mail | 21,387 | 15.14% | 0.88% | $1.08 |

### 6.2 Primary metric

| Arm | Lift | Welch 95% CI | Bootstrap 95% CI | Holm p |
|---|---|---|---|---|
| Mens | +$0.77 | +$0.49 to +$1.05 | +$0.50 to +$1.06 | < 0.0001 |
| Womens | +$0.42 | +$0.17 to +$0.68 | +$0.17 to +$0.68 | 0.0011 |

The two interval methods agree to within about a cent at each end, so the normal
approximation is adequate at this sample size despite the heavy tail.

### 6.3 Robustness and the top-coding finding

The plan capped spend at the 99th percentile of positive spend. That value
is $499.00, which is also the maximum: 12 of 578 purchases are exactly
$499.00. The source file is top-coded, so the planned cap changes nothing
and the check passes trivially. The report says so rather than counting it
as evidence. A post-hoc cap at the 95th percentile ($362.20), labelled as
not pre-registered and excluded from the decision, gives +$0.74 (Mens) and
+$0.41 (Womens), both still significant.

### 6.4 Variance reduction: a negative result

| Arm | Unadjusted SE | CUPED SE | Variance removed | corr(spend, history) |
|---|---|---|---|---|
| Mens | $0.1452 | $0.1452 | 0.044% | 0.021 |
| Womens | $0.1303 | $0.1303 | 0.038% | 0.020 |

Last year's spend is nearly uncorrelated with spend in a two-week window,
mostly because over 99% of customers spend nothing in the window. The full
Lin regression with every covariate does no better. The lesson for future
tests: the CUPED covariate must be the same metric over a comparable
window, such as spend in the two weeks before the send.

### 6.5 Power

At alpha 0.025 per comparison and 80% power, the realised design could
detect a revenue lift of $0.35 per customer, a conversion lift of 0.23 pp,
or a visit lift of 0.92 pp. The planning grid in the test plan estimated
$0.30 to $0.60 for a revenue SD of $10 to $20; the observed control SD was
$11.59.

### 6.6 Segments

The Mens email lift is Holm-significant in all 11 segment levels. The
Womens email lift is significant in 3: new customers (+$0.73), womens-only
buyers (+$0.63) and urban customers (+$0.67). No segment shows a
significant negative effect for either email.

Segments overlap (every customer is in one channel and one newbie level),
so a concentrated effect in one segment appears, diluted, in every segment
that overlaps it. A unit test demonstrates this: removing purchases among
Web customers in one arm also flags harm in both newbie levels.

### 6.7 Peeking

| Looks | False positives, stop at first p < 0.05 | With Bonferroni |
|---|---|---|
| 1 | 4.5% | 4.5% |
| 5 | 13.6% | 3.1% |
| 20 | 23.1% | 1.2% |

### 6.8 Uplift model (Womens email, visit outcome)

Trained on 29,885 customers, evaluated on 12,808 held out. Observed uplift
falls from +9.3 pp in the top decile to -2.6 pp in the bottom decile.
Normalised Qini 0.23; the top 30% by predicted uplift accounts for 55% of
incremental visits. Exploratory: the next step is a randomised test of
targeted sending against sending to everyone.

---

## 7. Error handling

| Situation | Behaviour |
|---|---|
| Unknown or missing key in `params.yaml` | `ConfigError` naming the section and key |
| Invalid value (alpha 1.5, allocation not summing to 1) | `ConfigError` |
| Network failure | Up to 3 attempts with backoff, then the mirror |
| Checksum mismatch | File deleted, next source tried |
| All sources fail | `DataDownloadError` listing each source's failure |
| Corrupted cached file | Detected by digest, deleted, re-downloaded |
| Schema or integrity violation | `DataValidationError` listing every problem at once |
| Segment column not in the whitelist | `ValueError` before any SQL runs |
| Invalid statistical input (n < 2, rate outside [0, 1]) | `ValueError` |
| Any of the above from the CLI | One-line error, exit code 1, no traceback |

---

## 8. Testing

99 tests: 97 run offline in CI, 2 need the network (`pytest -m network`).

| File | Covers |
|---|---|
| `test_config.py` | Loading, unknown keys, missing sections, cross-field rules |
| `test_data.py` | Every validation rule; cache reuse; corrupted cache; checksum mismatch fallback; unreachable source; gzipped mirror; real data (network) |
| `test_power.py` | MDE against statsmodels; sample size; power at MDE and at zero |
| `test_metrics_sql.py` | SQL equals pandas; window shares; UNPIVOT coverage; whitelist |
| `test_checks.py` | SRM pass, fail, unequal plans; balance pass and injected imbalance |
| `test_stats.py` | Welch, z-test, Holm against scipy/statsmodels; A/A false-positive rate; bootstrap agreement and determinism |
| `test_effects.py` | CUPED variance removed equals rho squared; CUPED unbiased under the null; Lin agrees with CUPED; segment harm localisation; economics; decision rule |
| `test_peeking_uplift.py` | Single look holds alpha; peeking inflates; Bonferroni controls; Qini of perfect vs random rankings; T-learner finds an injected responsive segment |
| `test_pipeline.py` | End-to-end on a synthetic experiment with a known winner; no recommendation when nothing works; reproducible bytes; LF endings; CLI; memo reports failed checks; committed reports equal a fresh run (network) |

Synthetic data comes from `tests/conftest.py::make_experiment`, which
produces frames that pass every validation rule and lets a test set the
true effect per arm, the strength of the pre-period covariate, and the
allocation.

---

## 9. Security and data handling

- No secrets, credentials or environment variables are used or needed.
- The canonical download is plain HTTP; integrity is enforced by pinned
  SHA-256 digests, and a mismatch is never loaded.
- YAML is parsed with `yaml.safe_load`.
- SQL identifiers come only from a whitelist; no user input reaches a query.
- The dataset is anonymous: no names, emails or identifiers.
- `data/` is git-ignored; the dataset is never committed.

---

## 10. Limitations

See README, Limitations. In short: revenue is not profit, the window is two
weeks with no unsubscribe data, spend is top-coded at $499, segment and
uplift results are exploratory, and the public dataset's results were known
before this analysis.
