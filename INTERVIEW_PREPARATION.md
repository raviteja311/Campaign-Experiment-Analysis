# Interview Preparation

Answers are grounded in this repository: every number is from
`reports/results.json`, and every method named is implemented and tested
here. Practise saying them out loud; do not memorise word for word.

---

## Part 1. The pitch

**30 seconds.**
"I analysed a real randomised email experiment with 64,000 customers the
way a product team would analyse a live A/B test. I wrote and committed the
test plan before touching outcomes, checked randomisation with a
sample-ratio test and covariate balance, estimated revenue lift with Welch
tests and bootstrap intervals, corrected for multiple comparisons with
Holm, and turned the result into a profit decision. The Mens email raised
revenue per customer by 77 cents, roughly 26 cents of profit per email at
the assumed margin, and beat the Womens email by 35 cents. Everything
regenerates from one command, and a test proves the committed report
matches a fresh run byte for byte."

**2 minutes:** add the pipeline (SQL metric layer on DuckDB, validity
checks, primary test, robustness, segments with Holm, economics with a
break-even margin, encoded decision rule) and two honest findings: CUPED
removed only 0.04% of variance, and the source top-codes spend at $499,
which made a planned robustness check non-binding.

---

## Part 2. Project questions

**Why did you build this project?**
My other projects are about building and evaluating models, and
experimentation is central to customer analytics roles. I wanted a project
where the hard part is statistical judgement and a business decision, not
model accuracy.

**What problem does it solve?**
It answers whether to roll out an email campaign: does it raise revenue,
does that cover the cost, and which version to send, with the uncertainty
and assumptions stated so a stakeholder can act on it.

**Why this architecture?**
It is a batch analysis, so a CLI pipeline is the simplest thing that works.
There is no API because nothing needs serving. Metrics are SQL files run on
in-memory DuckDB, so the definitions read like warehouse queries without
operating a database. Every stage is a small module with one job, which
made each one testable against a reference or a known truth.

**Why these technologies?**

- DuckDB: real SQL (window functions, UNPIVOT) on a DataFrame, in-process.
- SciPy and statsmodels: standard, auditable implementations; I also use
  them as references to test my own functions.
- scikit-learn: logistic regression for the uplift model; interpretable and
  fast.
- pytest and ruff with GitHub Actions: tests and style on every push.

**How does it work end to end?**
Load config. Download the CSV once, verify its SHA-256, validate it. Load
it into DuckDB and compute per-arm statistics. Run SRM and balance checks.
Run Welch tests for revenue per customer with a bootstrap interval, Holm
across the two emails. Run robustness, secondary metrics, head-to-head,
CUPED and regression adjustment, power at the observed baseline, segment
effects with Holm, then economics and the decision rule. Finally simulate
peeking, fit the uplift model, and render the report, memo and figures.

**What were the biggest challenges, and how did you solve them?**

1. *A plan that did not survive contact with the data, twice.* The plan
   capped spend at its 99th percentile, but purchases are about 1% of
   customers, so that percentile is zero. I caught it before analysis and
   committed a clarification (99th percentile of *positive* spend). After
   analysis, I found that even this cap equals the maximum, $499, because
   the source top-codes spend: 12 of 578 purchases are exactly $499. I did
   not switch thresholds quietly. The report says the planned check is
   non-binding, and a 95th-percentile check is added as clearly labelled
   post-hoc.
2. *A memo that asserted things it had not checked.* My first memo template
   said "assignment worked" unconditionally. I rewrote it to read the check
   results, and added a test that feeds it a failed check.
3. *Reproducibility across operating systems.* Windows wrote CRLF line
   endings, so outputs would differ on Linux. I write LF explicitly and
   test it, and a network test regenerates the reports and compares bytes.

**How did you evaluate the system?**
Three ways. Statistical functions are compared with scipy and statsmodels.
Synthetic experiments with known effects check that methods recover the
truth, including an A/A test that the false-positive rate is about 5% and a
check that CUPED is unbiased under the null. The pipeline is tested end to
end on a synthetic experiment with a known winner, and on the real data by
regenerating the committed reports.

**What would you improve?**
Use a pre-period covariate over a matching two-week window so CUPED works.
Replace the Bonferroni illustration with group-sequential boundaries.
Validate the uplift model with a randomised targeted-versus-everyone test.

**What happens if the input or data fails?**
Network failures retry with backoff and fall back to an HTTPS mirror. A
checksum mismatch deletes the file and is never loaded. Validation lists
every problem at once and stops. Config typos fail at load. The CLI prints
one clear error and exits 1. A failed SRM check makes the decision rule
reject every arm and the memo says so.

**How would you scale it?**
The bottleneck would be row-level work: the bootstrap and the regression.
The parametric tests already run on SQL aggregates, so with millions of
rows I would compute sufficient statistics in the warehouse, bootstrap by
resampling aggregated buckets (Poisson bootstrap), and keep the same
Python. The SQL would move from in-memory DuckDB to the warehouse largely
unchanged.

**How would you deploy it?**
As a scheduled job that runs after each experiment's window closes, reads
from the warehouse, and publishes the memo. CI already runs tests on every
change; a deployment would add a scheduler and a place to publish reports.
I would not add a web service unless people needed to query results
interactively.

---

## Part 3. Statistics questions

**Why revenue per customer and not average order value?**
Revenue per customer averages over everyone randomised, zeros included, so
treatment and control are comparable groups. Average order value only
includes purchasers, and the email changes who purchases, so the two groups
of purchasers are no longer randomised. It is reported, never tested.

**Why Welch and not Student's t-test?**
Student assumes equal variances. Treated arms have more purchasers, so
higher variance: control variance was 134 against 315 for Mens. Welch does
not assume equality and loses almost nothing when variances are equal.

**Revenue is extremely skewed. Is a t-test valid?**
With about 21,000 customers per arm the sampling distribution of the mean
is close to normal. I did not rely on that: the percentile bootstrap gives
intervals within about a cent of Welch's.

**What is a sample ratio mismatch and why check it first?**
The arms were meant to be equal thirds. If the counts deviate more than
chance allows, assignment or logging is broken, and the arms may differ in
ways that bias every estimate. Here the chi-square p-value is 0.904. I use
a strict threshold (0.001) because a false alarm would discard a valid test.

**Why standardised mean differences for balance, not p-values?**
P-values shrink with sample size, so with 64,000 rows trivial differences
look significant. SMD measures the size of the imbalance; the largest here
is 0.014 against a 0.1 threshold.

**Why Holm and not Bonferroni?**
Both control the family-wise error rate under any dependence. Holm is
uniformly more powerful: it tests the smallest p-value at alpha/m, the next
at alpha/(m-1), and so on.

**Which families did you correct?**
The two primary comparisons; the four secondary comparisons; and, for each
email, all 11 segment levels. Each family is a set of tests someone would
read together.

**A segment's 95% CI excludes zero but you call it not significant. Why?**
The interval is unadjusted; significance uses the Holm-adjusted p-value.
Womens/Multichannel is an example: interval +$0.26 to +$2.06, Holm p 0.093.

**What is CUPED and why did it not help?**
It subtracts the part of the outcome predicted by a pre-period covariate,
`Y - theta (X - mean X)` with `theta = cov(Y, X) / var(X)`. It stays
unbiased because the covariate was measured before treatment, and it removes
a share of variance equal to the squared correlation. Last year's spend
correlates 0.02 with two-week spend, because over 99% of customers spend
nothing in the window, so it removed 0.04%. The fix is a covariate measuring
the same thing over a comparable window.

**What is the Lin regression adjustment?**
OLS of the outcome on treatment, centred covariates and their interactions
with treatment, with HC2 robust errors. Centring makes the treatment
coefficient the average effect; interactions keep it consistent even when
the linear model is wrong. Here it agreed with CUPED.

**What is an MDE and what was yours?**
The smallest true effect the test would detect with the chosen power. At
alpha 0.025 per comparison and 80% power: $0.35 revenue per customer,
0.23 pp conversion, 0.92 pp visits. It tells you what a null result can rule
out.

**Why is observed power not evidence?**
Power computed at the observed effect is a function of the p-value, so it
adds no information. I report it as descriptive only.

**What is peeking, and what did your simulation show?**
Checking a running test repeatedly and stopping at the first p < 0.05.
Every look is another chance to cross the line by luck. In 2,000 A/A
simulations, one look gave 4.5% false positives; 20 looks gave 23.1%.
Bonferroni across looks restores control but is conservative (1.2% at 20
looks); group-sequential boundaries spend alpha more efficiently.

**How did you turn the lift into a decision?**
Net profit per email = margin x lift - cost. With the assumed 40% margin and
$0.05 per email, Mens nets $0.26. Because margin is an assumption, I also
report the break-even margin, 6.5% for Mens, which shows the decision is
robust to the assumption being quite wrong.

**What did the pre-registration actually protect against?**
Choosing the metric, test, segments or threshold after seeing which gives
the best answer. It cannot stop someone knowing a public dataset's results,
and the plan says so.

---

## Part 4. Uplift modelling questions

**What is a T-learner?**
Two models: P(outcome | treated, x) fitted on treated customers and
P(outcome | control, x) on control customers. Predicted uplift is the
difference. I used logistic regression on pre-period features only.

**How do you evaluate uplift when individual uplift is never observed?**
On held-out data: rank customers by predicted uplift, then compare treated
and control outcomes within each decile. Observed visit uplift falls from
+9.3 pp in the top decile to -2.6 pp in the bottom. The Qini curve
accumulates incremental visits in score order; its normalised area over
random targeting was 0.23, and the top 30% captured 55% of incremental
visits.

**Would you deploy it?**
Not yet. It is exploratory. The next step is a randomised test of
model-targeted sending against sending to everyone.

---

## Part 5. SQL and engineering questions

**Show me a window function you used.**
`n / SUM(n) OVER ()` for each arm's share of customers, and
`COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY segment, level)` for the arm
share within each segment level, which should be about a third under
randomisation.

**Why UNPIVOT?**
To turn four segment columns into `(segment, level)` rows so one GROUP BY
covers them all instead of four near-identical queries. Columns are cast to
text first because UNPIVOT needs one value type.

**Is building SQL from strings safe here?**
Only segment column names are inserted, and they come from a whitelist;
anything else raises before reaching SQL. No user value is ever inserted.

**Why compute tests from SQL aggregates?**
Mean, variance and count are sufficient statistics for Welch and z-tests,
so the same function serves arm-level and segment-level comparisons, and it
would scale to a warehouse unchanged.

**How do you know your statistics code is right?**
Each function is tested against scipy or statsmodels, and against synthetic
data where I set the true effect.

**Why keep duplicate rows?**
There is no customer id. 6,562 rows are identical because different
customers share attributes and did not buy. Deleting them changes the
sample.

**How do you guarantee the data is the right data?**
A pinned SHA-256 per source. The original host is plain HTTP, so the digest
is the integrity guarantee. Downloads write to a temporary file renamed only
on success.

**How is the report kept honest?**
It is rendered from the results, with no timestamp and LF endings, so an
unchanged analysis gives identical bytes. A network test regenerates it and
compares bytes with the committed copy.

---

## Part 6. Connecting it to the role

**How would this apply to a retail bank?**
The same pipeline measures a retention or cross-sell campaign: randomise
eligible customers, pre-register revenue or retention per customer, check
SRM and balance, estimate the lift with corrected tests, and convert it to
profit with explicit margin assumptions. The uplift model is the bridge to
propensity work: it targets customers whose behaviour the campaign changes,
not those who would have acted anyway.

**What would you add for a bank?**
Guardrails this dataset lacks: complaints, opt-outs and product closures as
non-inferiority metrics. A longer window to catch delayed effects.
Pre-period covariates that match the outcome so CUPED earns its keep.

---

## Part 7. Bugs and mistakes worth telling

1. **The vacuous robustness check.** The planned cap equalled the maximum
   spend because of top-coding. I reported it as non-binding rather than
   counting it as a pass.
2. **Wrong test expectations, right code.** My first CUPED test expected
   over 20% variance reduction, but the fixture's correlation was 0.28,
   which predicts 7.8%; the code measured 7.76%. I fixed the fixture and
   added an assertion that variance removed equals the squared correlation.
3. **Segment harm spreads.** I expected an injected harm in Web customers to
   flag only the Web level; it also flagged both newbie levels, correctly,
   because they contain Web customers. The test now asserts the right
   thing, and the report explains overlap.
4. **An unnormalised metric.** My first Qini coefficient was 67, in units of
   visits, which means nothing to a reader. I normalised it by the total
   incremental outcome, giving 0.23 on a 0-is-random scale.
