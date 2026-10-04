# Pre-Registered Test Plan

**Status:** fixed before any outcome analysis. The git commit that adds this
file precedes every commit that computes an effect, so the order can be
checked in the history (`git log --follow docs/TEST_PLAN.md`).

**Honesty note.** The Hillstrom dataset is public and has been analysed
since 2008, so its headline results exist in the literature. This plan
cannot make the author unaware of them. What it does guarantee is that the
metrics, tests, thresholds and decision rule below were written down before
this repository computed them, and that the analysis follows the plan even
where a different choice would have produced a more convenient answer. Any
deviation is listed in the final report under *Deviations from plan*.

Before writing this plan, the author looked only at the arm sizes (the
design) and at internal-consistency rules (for example, that a purchase
always has a visit). No outcome was compared across arms.

---

## 1. Business question

A retailer emailed customers who had bought in the last 12 months. The
marketing team wants to know:

1. Does either email increase revenue per customer over the following two weeks?
2. If so, is the increase worth the cost of sending it?
3. Which email should be sent, and are there customer groups it harms?

## 2. Design

| Item | Value |
|---|---|
| Unit of randomisation | Customer |
| Population | 64,000 customers who purchased in the previous 12 months |
| Arms | `No E-Mail` (control), `Mens E-Mail`, `Womens E-Mail` |
| Planned allocation | 1/3 each |
| Exposure | One email |
| Outcome window | Two weeks after the send |
| Analysis | Fixed horizon, a single look at the end of the window |

## 3. Metrics

| Role | Metric | Definition |
|---|---|---|
| **Primary** | Revenue per customer | Mean of `spend` over **all** customers in the arm, zeros included |
| Secondary | Conversion rate | Share of customers with `conversion = 1` |
| Secondary | Visit rate | Share of customers with `visit = 1` |
| Guardrail (validity) | Sample ratio | Arm counts versus the planned 1/3 split |
| Guardrail (validity) | Covariate balance | Standardised mean difference of every pre-period covariate |
| Guardrail (harm) | Segment no-harm | No pre-specified segment shows a significant **negative** revenue effect |
| Guardrail (robustness) | Winsorised revenue | The primary result holds when spend is capped at the 99th percentile of **positive** spend, pooled across arms (see note below) |

Revenue per customer is the primary metric because it is the one the
business decision is made on. Revenue *per converter* (average order value)
is **not** used: it conditions on a post-treatment outcome, so the
customers in it differ between arms and the comparison is no longer
randomised.

**Clarification added before analysis (second commit to this file).** The
first version said "capped at its 99th percentile". Purchases are rare (the
planning range above is 0.5% to 2% conversion), so the 99th percentile of
spend over *all* customers is likely $0, and that cap would erase every
purchase rather than trim the tail. The cap is therefore the 99th percentile
of spend among customers who spent anything, computed once on all arms
pooled so treatment and control are capped at the same dollar value. This
was changed before any outcome was computed in this repository.

The dataset contains no unsubscribe or complaint data. In a real campaign
those would be the first harm guardrails; here the segment no-harm check
is the closest available substitute, and the limitation is reported.

## 4. Hypotheses and tests

For each treatment arm *t* versus control:

- H0: mean revenue per customer is equal. H1: it differs (two-sided).
- **Primary test:** Welch's t-test on per-customer spend. With about 21,000
  customers per arm the sampling distribution of the mean is close to normal
  despite the heavy right tail; a percentile bootstrap confidence interval
  (2,000 resamples) is reported alongside as a check that does not rely on
  that approximation.
- **Multiplicity:** two primary comparisons, controlled with Holm's method
  at a family-wise alpha of 0.05.
- **Secondary tests:** two-proportion z-tests for conversion and visit rate,
  Holm-corrected across the four secondary comparisons. Reported, not used
  for the decision.
- **Head-to-head:** Mens versus Womens on revenue per customer (Welch),
  reported to say whether the choice between the emails is itself supported.

## 5. Validity checks (run before any effect is read)

| Check | Method | Fails when | Action on failure |
|---|---|---|---|
| Sample ratio mismatch | Chi-square goodness of fit against 1/3 each | p < 0.001 | Stop: effect estimates are not trustworthy |
| Covariate balance | Standardised mean difference, each treatment vs control | any \|SMD\| > 0.1 | Report, and rely on CUPED-adjusted estimates |
| Data integrity | Schema and consistency rules in `data.validate` | any rule fails | Stop |

Covariates checked: `recency`, `history`, `mens`, `womens`, `newbie`, and
one-hot `zip_code` and `channel`.

## 6. Variance reduction (pre-specified, not used for the decision)

CUPED with the pre-period covariate `history` (dollars spent in the year
before the test). theta is estimated on the pooled sample. The adjusted
estimate and the variance reduction achieved are reported to inform the
sizing of future tests. The decision uses the unadjusted primary test, so
the result cannot depend on the choice of covariate.

## 7. Power and minimum detectable effect

Planning uses the realised arm sizes (about 21,300 per arm, fixed by the
design), alpha = 0.025 per comparison (Bonferroni across the two primary
comparisons, which is conservative relative to Holm), and 80% power. The
baseline values are **assumptions**, because they are outcomes and were not
looked at:

| Metric | Assumed control baseline | MDE (absolute) |
|---|---|---|
| Visit rate | 10% / 15% / 20% | 0.90 / 1.07 / 1.20 pp |
| Conversion rate | 0.5% / 1% / 2% | 0.21 / 0.30 / 0.42 pp |
| Revenue per customer | SD $10 / $15 / $20 | $0.30 / $0.45 / $0.60 |

After the analysis, the report restates the MDE at the observed control
baseline, so a non-significant result can be read as "no effect larger than
X" rather than as "no effect".

## 8. Segments (exploratory, Holm-corrected)

Pre-specified segments: `newbie`, `channel`, `purchase_line` (mens only,
womens only, both, derived from `mens` and `womens`), and `zip_code`.

For each treatment arm, the revenue effect is estimated within every level
of every segment (Welch's t-test), and Holm's method is applied across all
levels of all segments for that arm. Segment results generate targeting
hypotheses for the next test. They do **not** change the ship decision,
except through the no-harm guardrail.

## 9. Decision rule

An email arm is recommended for rollout only if **all** of these hold:

1. The validity checks in section 5 pass.
2. Its Holm-adjusted primary p-value is below 0.05 and the effect is positive.
3. The winsorised-revenue sensitivity check agrees in sign and significance.
4. No segment shows a Holm-significant negative revenue effect.
5. Its expected net profit per emailed customer is positive:
   `gross_margin x revenue lift - cost_per_email`, with the assumed
   `gross_margin = 0.40` and `cost_per_email = $0.05`.

Margin and cost are assumptions, not data. The memo therefore also reports
the **break-even margin** (`cost_per_email / revenue lift`), so a reader can
see how far the assumptions could be wrong before the decision flips, and a
profit interval derived from the bootstrap interval of the lift.

If both arms qualify, the arm with the higher expected net profit is
recommended, and the head-to-head test states whether that preference is
statistically supported or only a point estimate.

## 10. Out of scope for the decision

- **Peeking simulation:** an A/A simulation showing how repeatedly checking
  a running test inflates false positives. It is illustrative and uses
  simulated data; the real analysis is a single fixed-horizon look.
- **Uplift model:** a T-learner estimating who responds to the Womens email
  (visit outcome), evaluated on a held-out split. Exploratory input to the
  next test's targeting, not part of this decision.
