"""Apply the pre-registered decision rule (docs/TEST_PLAN.md, section 9)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArmDecision:
    arm: str
    validity_passed: bool
    primary_significant_positive: bool
    winsorized_agrees: bool
    no_segment_harm: bool
    profitable: bool

    @property
    def qualifies(self) -> bool:
        return all(
            [
                self.validity_passed,
                self.primary_significant_positive,
                self.winsorized_agrees,
                self.no_segment_harm,
                self.profitable,
            ]
        )

    @property
    def failed_criteria(self) -> list[str]:
        return [name for name, value in asdict(self).items() if value is False]

    def to_dict(self) -> dict:
        return {**asdict(self), "qualifies": self.qualifies, "failed": self.failed_criteria}


def decide(decisions: list[ArmDecision], net_profit: dict[str, float]) -> str | None:
    """The qualifying arm with the highest expected net profit, or None."""
    qualifying = [d.arm for d in decisions if d.qualifies]
    if not qualifying:
        return None
    return max(qualifying, key=lambda arm: net_profit[arm])
