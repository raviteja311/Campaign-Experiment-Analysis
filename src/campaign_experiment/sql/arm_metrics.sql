-- One row per arm: the sufficient statistics every test in stats.py needs.
-- Means, variances and counts are enough for z- and Welch tests, so the
-- inference can run on these aggregates without touching row-level data.
WITH per_arm AS (
    SELECT
        segment                      AS arm,
        COUNT(*)                     AS n,
        SUM(visit)                   AS visits,
        SUM(conversion)              AS conversions,
        SUM(spend)                   AS revenue,
        AVG(spend)                   AS revenue_per_customer,
        VAR_SAMP(spend)              AS revenue_var,
        AVG(visit)                   AS visit_rate,
        AVG(conversion)              AS conversion_rate,
        -- Average order value among converters. Descriptive only: it
        -- conditions on a post-treatment outcome, so it is never tested.
        AVG(spend) FILTER (WHERE conversion = 1) AS order_value
    FROM customers
    GROUP BY segment
)
SELECT
    *,
    n / SUM(n) OVER ()               AS share_of_customers,
    revenue / SUM(revenue) OVER ()   AS share_of_revenue
FROM per_arm
ORDER BY arm
