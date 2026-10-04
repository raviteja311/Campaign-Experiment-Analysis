-- Arm-level statistics within every level of every segment column.
-- UNPIVOT turns the segment columns into (segment, level) rows, so one
-- GROUP BY covers all segments. {segment_columns} is filled from a
-- whitelist in metrics_sql.py, never from user input.
WITH long AS (
    UNPIVOT (
        SELECT segment AS arm, spend, conversion, visit, {cast_columns}
        FROM customers
    )
    ON {segment_columns}
    INTO NAME segment VALUE level
)
SELECT
    segment,
    level,
    arm,
    COUNT(*)            AS n,
    AVG(spend)          AS revenue_per_customer,
    VAR_SAMP(spend)     AS revenue_var,
    AVG(conversion)     AS conversion_rate,
    AVG(visit)          AS visit_rate,
    -- Share of the segment level's customers that landed in this arm. Under
    -- randomisation this is close to 1/3 in every level.
    COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY segment, level) AS arm_share
FROM long
GROUP BY segment, level, arm
ORDER BY segment, level, arm
