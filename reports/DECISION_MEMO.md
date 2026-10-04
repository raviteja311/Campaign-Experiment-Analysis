# Decision Memo: Email Campaign Rollout

**Recommendation: send the Mens E-Mail.** It passed every pre-registered criterion and has the higher expected net profit of the two qualifying emails (Mens E-Mail, Womens E-Mail).

## What the test found

- **Mens E-Mail:** revenue per customer $0.65 -> $1.42, a lift of **+$0.77** (95% CI +$0.50 to +$1.06, Holm p < 0.0001). Conversion 0.57% -> 1.25%. Expected net profit **+$0.26 per email** (95% CI +$0.15 to +$0.37); it breaks even at a gross margin of 6.5%.
- **Womens E-Mail:** revenue per customer $0.65 -> $1.08, a lift of **+$0.42** (95% CI +$0.17 to +$0.68, Holm p 0.0011). Conversion 0.57% -> 0.88%. Expected net profit **+$0.12 per email** (95% CI +$0.02 to +$0.22); it breaks even at a gross margin of 11.8%.
- **Head to head:** the Mens E-Mail out-earns the Womens E-Mail by $0.35 per customer (95% CI +$0.03 to +$0.66, p 0.0305). The preference is statistically supported, though the interval is wide.

## Why the result can be trusted

- Assignment check: arm sizes match the planned allocation (SRM p 0.9037); every pre-period covariate is balanced (largest |SMD| 0.014, threshold 0.1).
- Metrics, tests, thresholds and this decision rule were committed in `docs/TEST_PLAN.md` before any outcome was analysed.
- The lifts hold under a bootstrap interval and when spend is capped at $362.20 (a post-hoc check; see the report).

## What could change the decision

- **Assumed economics.** Profit uses an assumed 40% gross margin and $0.05 per email; the dataset records revenue only. The break-even margins above show how far these can move before the call flips.
- **Two-week window.** Effects on later purchases, fatigue or unsubscribes are not measured; this dataset has no unsubscribe or complaint data, so those guardrails could not be checked.
- **Top-coded spend.** The source caps spend at $499.00 (12 of 578 purchases sit exactly there), so very large orders are understated in every arm.

## Suggested next test

- **Mens E-Mail:** significant revenue lift in 11 of 11 segment levels after Holm correction.
- **Womens E-Mail:** significant revenue lift in 3 of 11 segment levels after Holm correction; not significant in channel=Multichannel, channel=Phone, channel=Web, newbie=0, purchase_line=both, purchase_line=mens_only, zip_code=Rural, zip_code=Surburban.
- A targeting model for the Womens E-Mail (held-out data) ranks customers so that the top 30% account for 55% of incremental visits. Segment and model findings are exploratory: test targeted sending against send-to-all in a new randomised experiment before acting on them.
