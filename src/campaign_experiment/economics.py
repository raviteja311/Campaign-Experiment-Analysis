"""Translate a revenue lift into the profit question the business asks.

Margin and cost per email are assumptions from params.yaml, not data, so
every output that depends on them is paired with the break-even margin: the
margin at which sending the email stops paying for itself.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Economics:
    revenue_lift_per_customer: float
    revenue_lift_ci: tuple[float, float]
    gross_margin: float
    cost_per_email: float
    net_profit_per_customer: float
    net_profit_ci: tuple[float, float]
    break_even_margin: float | None
    return_per_dollar: float | None
    rollout_customers: int
    net_profit_at_rollout: float

    def to_dict(self) -> dict:
        return asdict(self)


def campaign_economics(
    lift: float,
    lift_ci: tuple[float, float],
    gross_margin: float,
    cost_per_email: float,
    rollout_customers: int,
) -> Economics:
    """Net profit per emailed customer = margin x revenue lift - cost per email."""
    if not 0 < gross_margin <= 1 or cost_per_email < 0 or rollout_customers < 0:
        raise ValueError("need 0 < margin <= 1, cost >= 0, rollout >= 0")
    low, high = sorted(lift_ci)
    net = gross_margin * lift - cost_per_email
    net_ci = (gross_margin * low - cost_per_email, gross_margin * high - cost_per_email)
    # Break-even only exists when the email raises revenue at all.
    break_even = cost_per_email / lift if lift > 0 else None
    if break_even is not None and break_even > 1:
        break_even = math.inf
    per_dollar = (gross_margin * lift) / cost_per_email if cost_per_email > 0 else None
    return Economics(
        revenue_lift_per_customer=lift,
        revenue_lift_ci=(low, high),
        gross_margin=gross_margin,
        cost_per_email=cost_per_email,
        net_profit_per_customer=net,
        net_profit_ci=net_ci,
        break_even_margin=break_even,
        return_per_dollar=per_dollar,
        rollout_customers=rollout_customers,
        net_profit_at_rollout=net * rollout_customers,
    )
