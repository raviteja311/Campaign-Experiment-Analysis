# Project Instructions: From Zero to a Reproduced Analysis

This guide lets a developer who has never seen the project understand it,
set it up, reproduce every number, and extend it, without asking basic
questions. Commands are shown for Windows PowerShell and macOS/Linux where
they differ.

---

## 1. Project Overview

A retailer ran a randomised email experiment on 64,000 customers: one third
got a Mens email, one third a Womens email, one third nothing. This project
analyses it end to end, following a test plan written before any outcome
was analysed, and ends in a one-page decision memo.

Final outputs, all in `reports/`:

- `DECISION_MEMO.md`: the recommendation and its evidence, one page.
- `REPORT.md`: every table behind the memo.
- `results.json`: every number, machine-readable.
- `figures/`: four charts.

## 2. Problem Statement

Before sending an email to all customers, the business must know whether it
raises revenue, whether that revenue covers the cost, and which email to
send. Answering this from raw averages risks four classic mistakes: trusting
a broken randomisation, ignoring noise, running many comparisons without
correction, and ignoring cost. The project is built to avoid each one.

## 3. Goals

1. Fix the method in advance (`docs/TEST_PLAN.md`).
2. Verify the experiment is valid before reading effects.
3. Estimate effects with reliable confidence intervals.
4. Convert effects into a profit decision with stated assumptions.
5. Make every number reproducible by one command and protected by tests.

## 4. Requirements

**Functional**

- Download and verify the dataset; never commit it.
- Validate the schema and integrity rules.
- Compute metrics in SQL.
- Run SRM and balance checks, the primary test, robustness checks,
  secondary tests, variance reduction, segment analysis, economics and the
  decision rule.
- Simulate the cost of peeking; fit and evaluate an uplift model.
- Generate the report, memo and figures.

**Non-functional**

- Reproducible: same input, same bytes out, on any OS.
- Tested offline: CI needs no network.
- Configurable: every tunable value in `params.yaml`.
- Fast: a full run in seconds on a laptop.
- No secrets required.

## 5. Technology Stack

| Tool | Version | Used for |
|---|---|---|
| Python | 3.12 (3.11+) | Everything |
| DuckDB | 1.5.6 | SQL metric layer, in memory |
| pandas / NumPy | 3.0.6 / 2.5.3 | Data handling |
| SciPy | 1.18.1 | Distributions, chi-square, t-test reference |
| statsmodels | 0.15.0 | OLS with HC2 errors; reference implementations in tests |
| scikit-learn | 1.9.1 | Logistic regression for the uplift model |
| matplotlib | 3.11.2 | Figures |
| PyYAML | 6.0.3 | Configuration |
| pytest / ruff | 9.1.1 / 0.16.10 | Tests, formatting and lint |

## 6. Architecture

```
params.yaml -> config.py
                  |
data.py (download, verify, validate) -> DataFrame -> metrics_sql.py (DuckDB)
                  |
analysis.py runs the plan in order:
  validity -> primary -> robustness -> secondary -> head to head
  -> variance reduction -> power -> segments -> economics -> decision
  -> peeking -> uplift
                  |
report.py + figures.py -> reports/
```

One process, no server, no database to operate. See
`PROJECT_DOCUMENTATION.md` section 3 for the full diagram.

## 7. Data Sources

**Kevin Hillstrom's MineThatData E-Mail Analytics Challenge (2008).**

| | Primary | Mirror |
|---|---|---|
| URL | `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv` | `https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz` |
| Format | CSV | gzipped CSV |
| Size | 3,964,977 bytes | 443,070 bytes |
| SHA-256 | `0e5893...aece` | `bab657...234f` |

The two files contain identical data; the mirror writes `0.0` where the
original writes `0`, so their digests differ. Both digests are pinned in
`params.yaml`.

## 8. Remote Data / Large File Strategy

- **Size estimate:** 4 MB. Small enough to load fully into memory; streaming
  or sampling would add complexity with no benefit.
- **Remote access:** downloaded on first use by `ensure_dataset()` into
  `data/raw/`, which is git-ignored. Later runs reuse the cached file after
  re-verifying its SHA-256.
- **Integrity:** the canonical host serves plain HTTP, so a pinned digest,
  not the transport, guarantees the bytes. A mismatch deletes the file and
  tries the HTTPS mirror.
- **Failure safety:** downloads go to a `.part` file renamed only on
  success, so a dropped connection never leaves a half file that looks
  valid.
- **Nothing large is committed.** The only committed outputs are the
  reports (about 350 KB in total), which are the project's deliverable and
  are proven by a test to equal a fresh regeneration.

## 9. Environment Setup

```bash
git clone https://github.com/raviteja311/Campaign-Experiment-Analysis.git
cd Campaign-Experiment-Analysis
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
```

```bash
source .venv/bin/activate             # macOS / Linux
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Install:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

No environment variables, API keys or `.env` file are needed.

## 10. Project Structure

```
params.yaml                 all settings
docs/TEST_PLAN.md           pre-registered plan
src/campaign_experiment/
  config.py                 settings loader and validation
  data.py                   acquisition and validation
  sql/                      metric definitions (SQL)
  metrics_sql.py            DuckDB runner
  checks.py                 SRM, balance
  power.py                  MDE and sample size
  stats.py                  tests, bootstrap, Holm
  variance_reduction.py     CUPED, regression adjustment
  segments.py               heterogeneous effects
  economics.py              profit, break-even
  decision.py               decision rule
  peeking.py                A/A peeking simulation
  uplift.py                 T-learner, Qini
  analysis.py               pipeline orchestration
  report.py, figures.py     outputs
  cli.py, __main__.py       command line
tests/                      99 tests; conftest.py builds synthetic experiments
reports/                    generated outputs (committed)
.github/workflows/ci.yml    CI
```

## 11. Implementation Steps

The repository was built in this order; each step is one commit
(`git log --oneline`):

1. **Scaffold:** packaging, typed config, CI.
2. **Data acquisition:** checksum-verified download, validation, synthetic
   test fixtures.
3. **Pre-register the plan:** `docs/TEST_PLAN.md` and the power module,
   before any outcome was computed.
4. **SQL metric layer and validity checks.**
5. **Effect estimation:** tests, bootstrap, Holm, CUPED, regression
   adjustment, segments, economics, decision rule.
6. **Peeking simulation and uplift model.**
7. **Plan clarification:** the winsorisation cap was corrected to apply to
   positive spend, before analysis.
8. **Pipeline and reports.**
9. **Hardening:** every report statement derived from results, LF output,
   a regeneration test.
10. **Documentation.**

To rebuild it from scratch yourself, follow the same order: the plan must
exist before the code that reads outcomes.

## 12. Important Code Components

| Component | Why it matters |
|---|---|
| `data.validate` | Collects every problem before raising, so one run shows all data issues |
| `sql/arm_metrics.sql` | The metric definitions; produces the sufficient statistics the tests use |
| `stats.welch_test` | Works from (mean, variance, n), so the same code serves arms and segments |
| `stats.bootstrap_diff_ci` | Resamples arms separately, in bounded-memory chunks |
| `stats.holm` | Family-wise error control valid under any dependence |
| `variance_reduction.cuped_effect` | Shows how pre-period data reduces noise, and when it cannot |
| `segments.segment_effects` | Holm across all levels, plus harm detection |
| `decision.ArmDecision` | The pre-registered rule as code, with the failed criteria listed |
| `report._validity_line` | States check results as they came out, never as assumed |

## 13. Model / Algorithm

**Inference.** Welch's t-test for revenue per customer, with
Welch-Satterthwaite degrees of freedom; two-proportion z-tests for rates
(pooled SE for the statistic, unpooled for the interval); percentile
bootstrap with 2,000 resamples; Holm step-down correction.

**Validity.** Chi-square goodness of fit of arm counts (fail below
p 0.001); standardised mean differences (flag above 0.1).

**Variance reduction.** CUPED: `Y - theta (X - mean X)`,
`theta = cov(Y, X) / var(X)`. Lin regression adjustment: OLS with centred
covariates and interactions, HC2 errors.

**Uplift.** T-learner with two standardised logistic regressions;
evaluation by decile table and normalised Qini coefficient on a held-out
stratified 30% split.

**Peeking.** Monte Carlo A/A with binomial increments per look.

## 14. API / Application Flow

There is no HTTP API; the application is a CLI.

```
campaign-experiment run
  -> load_config(params.yaml)
  -> load_dataset(): cache or download -> verify -> validate -> features
  -> run_analysis(): the plan, in order
  -> write_outputs(): results.json, REPORT.md, DECISION_MEMO.md, figures
  -> prints the recommendation and output paths
```

| Command | Network | Writes |
|---|---|---|
| `campaign-experiment download` | First run only | `data/raw/` |
| `campaign-experiment plan [--n-per-arm N]` | No | Nothing (prints) |
| `campaign-experiment run [--output DIR]` | First run only | `reports/` or `DIR` |
| `campaign-experiment --params other.yaml run` | | Uses another config |

Exit code 0 on success, 1 on a configuration, download or validation error.

## 15. Testing

```bash
pytest                       # 97 offline tests (what CI runs)
pytest -m network            # 2 real-data tests
pytest tests/test_stats.py   # one file
ruff format --check .
ruff check .
```

How the tests establish correctness:

- **Against references:** Welch vs `scipy`, z-test and Holm vs
  `statsmodels`, MDE vs `NormalIndPower`, SQL vs pandas.
- **Against known truth:** synthetic experiments from
  `tests/conftest.py::make_experiment` with a chosen effect per arm.
- **Statistical properties:** A/A false-positive rate near 5%; CUPED
  unbiased under the null; CUPED variance removed equals the squared
  correlation; one look holds alpha while many looks inflate it.
- **Reproducibility:** outputs are byte-identical across runs, LF-only, and
  the committed reports equal a fresh run on the real data (network test).

## 16. Common Errors & Solutions

| Symptom | Cause | Fix |
|---|---|---|
| `campaign-experiment: command not found` | Package not installed, or venv inactive | Activate the venv; `pip install -e .` |
| `ModuleNotFoundError: campaign_experiment` | Same | Same |
| `no source produced a verified file` | Offline, or a source changed its bytes | Check the connection; if a source genuinely changed, verify the new file and update its digest in `params.yaml` |
| `cached hillstrom.csv fails its checksum` (warning) | Cached file corrupted or edited | Nothing: it is deleted and re-downloaded automatically |
| `params.yaml [experiment]: unknown keys` | Typo in a key | Fix the key name |
| `data validation failed` | Input file does not match the expected schema | Read the listed problems; do not bypass validation |
| `pytest -m network` fails offline | Needs internet for the first download | Run once online; later runs use the cache |
| Activation blocked in PowerShell | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Report files show as modified after `run` | Code or params changed | Expected; review the diff, commit if intended |

## 17. Git Workflow

- One commit per completed milestone, with a conventional prefix
  (`feat:`, `docs:`, `test:`, `refactor:`, `chore:`) and a body that
  explains why.
- Before committing: `git status`, review the diff, `ruff format --check .`,
  `ruff check .`, `pytest`, and confirm nothing from `data/` or `.venv/` is
  staged.
- If code or `params.yaml` changes the results, run
  `campaign-experiment run` and commit the regenerated `reports/` in the
  same commit, so the network test keeps passing.
- Never rewrite `docs/TEST_PLAN.md` silently. A change goes in its own
  commit that says what changed and whether outcomes had been seen.

## 18. Running the Project

```bash
campaign-experiment plan     # optional: the planning MDE grid
campaign-experiment run      # downloads on first use, then analyses
```

Then open `reports/DECISION_MEMO.md`. A full run takes about seven seconds
after the first download.

## 19. Expected Results

`campaign-experiment run` prints `recommendation: Mens E-Mail`, and:

| Check | Expected value |
|---|---|
| Rows | 64,000 |
| Arm sizes | No E-Mail 21,306; Mens 21,307; Womens 21,387 |
| SRM p | 0.9037 |
| Mens lift | +$0.77 per customer, bootstrap CI +$0.50 to +$1.06 |
| Womens lift | +$0.42 per customer, bootstrap CI +$0.17 to +$0.68 |
| Mens minus Womens | +$0.35, p 0.0305 |
| CUPED variance removed | about 0.04% |
| Peeking, 20 looks | 23.1% false positives |
| Uplift, top 30% | 55% of incremental visits |

`pytest -m network` passing confirms your outputs match the committed ones
byte for byte.

## 20. Limitations

- Margin and cost per email are assumptions; the memo gives break-even
  margins to show how much they matter.
- The outcome window is two weeks; there is no unsubscribe or long-term
  data.
- Spend is top-coded at $499 in the source, which makes the pre-registered
  cap non-binding; a labelled post-hoc check covers the gap.
- Segment and uplift findings are exploratory.
- The dataset is public and its headline results predate this analysis.

## 21. Future Improvements

- A pre-period covariate over a matching two-week window for CUPED.
- Group-sequential stopping boundaries (O'Brien-Fleming alpha spending).
- A randomised test of uplift-targeted sending versus send-to-all.
- Calibration diagnostics for the uplift model's classifiers.
- A non-inferiority guardrail on unsubscribes when such data exists.
