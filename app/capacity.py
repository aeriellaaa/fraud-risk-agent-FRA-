"""
Capacity-Aware Threshold Calculator.

Cross-checking the cost-optimal threshold (decision.py) against realistic review-team
throughput surfaced a real gap: the cost-optimal threshold was chosen purely to minimize
FP_COST*fp + FN_COST*fn, with zero constraint on how many escalated cases a real analyst
team can actually clear per day.

Verified against the real trained model on the actual held-out test set: at the
cost-optimal threshold (0.01), 30.0% of traffic escalates, giving 95.6% recall -- not
the "100%" figure sometimes quoted. At 5,000 transactions/day and this project's own
sourced review-time assumption (~14 min/review), a 5-analyst team can sustain only a
3.4% escalate rate, achieving 39.7% recall.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CapacityAssumptions:
    team_size: int
    minutes_per_review: float = 14.0
    shift_hours: float = 8.0
    daily_transaction_volume: int = 5000

    @property
    def reviews_per_analyst_per_day(self) -> float:
        return (self.shift_hours * 60) / self.minutes_per_review

    @property
    def max_daily_escalations(self) -> float:
        return self.team_size * self.reviews_per_analyst_per_day

    @property
    def max_sustainable_escalate_rate(self) -> float:
        return self.max_daily_escalations / self.daily_transaction_volume


@dataclass
class CapacityConstrainedThreshold:
    assumptions: CapacityAssumptions
    auto_approve_threshold: float
    auto_reject_threshold: float
    escalate_rate: float
    recall_at_this_operating_point: float
    fn_at_auto_approve: int
    note: str


def find_capacity_constrained_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    assumptions: CapacityAssumptions,
    auto_reject_threshold: float = 0.15,
    search_step: float = 0.001,
    search_max: float = 0.15,
) -> CapacityConstrainedThreshold:
    max_rate = assumptions.max_sustainable_escalate_rate
    n = len(probs)

    for t in np.arange(search_step, search_max, search_step):
        escalate_mask = (probs > t) & (probs < auto_reject_threshold)
        escalate_rate = escalate_mask.sum() / n
        if escalate_rate <= max_rate:
            auto_approve_mask = probs <= t
            reject_mask = probs >= auto_reject_threshold
            fn = int(((auto_approve_mask) & (y_true == 1)).sum())
            tp_by_reject = int(((reject_mask) & (y_true == 1)).sum())
            tp_by_escalation = int(((escalate_mask) & (y_true == 1)).sum())
            total_fraud = int((y_true == 1).sum())
            recall = (tp_by_reject + tp_by_escalation) / total_fraud if total_fraud else 0.0
            return CapacityConstrainedThreshold(
                assumptions=assumptions,
                auto_approve_threshold=round(float(t), 4),
                auto_reject_threshold=auto_reject_threshold,
                escalate_rate=round(escalate_rate, 4),
                recall_at_this_operating_point=round(recall, 4),
                fn_at_auto_approve=fn,
                note=(
                    f"At {assumptions.team_size} analyst(s) ({assumptions.max_daily_escalations:.0f} "
                    f"reviews/day capacity, {assumptions.daily_transaction_volume:,} txns/day assumed), "
                    f"the cost-optimal threshold (0.01) is NOT reachable without a growing backlog. "
                    f"This threshold ({t:.4f}) is the closest capacity-respecting alternative."
                ),
            )

    raise ValueError(
        f"No threshold up to {search_max} satisfies the capacity constraint "
        f"(max sustainable escalate rate: {max_rate:.2%})."
    )


def compare_cost_optimal_vs_capacity_constrained(
    probs: np.ndarray,
    y_true: np.ndarray,
    cost_optimal_threshold: float,
    team_sizes: list[int],
    daily_transaction_volume: int = 5000,
) -> dict:
    n = len(probs)
    total_fraud = int((y_true == 1).sum())
    escalate_mask_opt = (probs > cost_optimal_threshold) & (probs < 0.15)
    reject_mask_opt = probs >= 0.15
    tp_opt = int(((reject_mask_opt) & (y_true == 1)).sum()) + int(((escalate_mask_opt) & (y_true == 1)).sum())
    recall_opt = tp_opt / total_fraud if total_fraud else 0.0
    escalate_rate_opt = escalate_mask_opt.sum() / n

    rows = [{
        "scenario": "Cost-optimal (unconstrained capacity)", "team_size": None,
        "auto_approve_threshold": cost_optimal_threshold,
        "escalate_rate": round(escalate_rate_opt, 4), "recall": round(recall_opt, 4),
        "sustainable": False,
        "note": "Assumes unlimited review capacity -- not achievable with any real team at this volume.",
    }]
    for size in team_sizes:
        assumptions = CapacityAssumptions(team_size=size, daily_transaction_volume=daily_transaction_volume)
        try:
            result = find_capacity_constrained_threshold(probs, y_true, assumptions)
            rows.append({
                "scenario": f"{size}-analyst team", "team_size": size,
                "auto_approve_threshold": result.auto_approve_threshold,
                "escalate_rate": result.escalate_rate, "recall": result.recall_at_this_operating_point,
                "sustainable": True, "note": result.note,
            })
        except ValueError as e:
            rows.append({
                "scenario": f"{size}-analyst team", "team_size": size,
                "auto_approve_threshold": None, "escalate_rate": None, "recall": None,
                "sustainable": False, "note": str(e),
            })
    return {"daily_transaction_volume": daily_transaction_volume, "rows": rows}
