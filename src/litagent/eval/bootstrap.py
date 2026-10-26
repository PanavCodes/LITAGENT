"""Bootstrap sampling protocols for rigorous metric confidence intervals.

Per EVALUATION.md §4 (FR-EVL-5):
- Paper protocol: 12 resamples of 20 questions (10 vector, 10 graph);
  t-distribution 95% margin of error (t_0.975,11 = 2.201).
- Extended protocol: 1,000 resamples with replacement of full test set;
  mean, SD, 95% percentile interval.
- Paired bootstrap for pairwise configuration comparisons.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

logger = logging.getLogger(__name__)

# Student-t critical value for df = 11, alpha = 0.05 (two-tailed 95%)
T_CRITICAL_DF11 = 2.201


def run_paper_bootstrap(
    sample_records: list[dict[str, Any]],
    metric_keys: list[str],
    n_resamples: int = 12,
    sample_size: int = 20,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Execute the reference paper's 12 x 20 bootstrap protocol."""
    rng = random.Random(seed)

    # Stratify by retriever type
    vec_records = [r for r in sample_records if r.get("expected_route") == "vector"]
    graph_records = [r for r in sample_records if r.get("expected_route") == "graph"]

    half = sample_size // 2
    resample_means: dict[str, list[float]] = {m: [] for m in metric_keys}

    for _ in range(n_resamples):
        batch: list[dict[str, Any]] = []
        if vec_records and len(vec_records) >= half:
            batch.extend(rng.sample(vec_records, half))
        elif vec_records:
            batch.extend(rng.choices(vec_records, k=half))

        if graph_records and len(graph_records) >= half:
            batch.extend(rng.sample(graph_records, half))
        elif graph_records:
            batch.extend(rng.choices(graph_records, k=half))

        if not batch:
            batch = rng.sample(sample_records, min(len(sample_records), sample_size))

        for m in metric_keys:
            vals = [float(r.get(m, 0.0)) for r in batch if m in r]
            m_val = sum(vals) / len(vals) if vals else 0.0
            resample_means[m].append(m_val)

    summary: dict[str, dict[str, float]] = {}
    for m in metric_keys:
        vals = resample_means[m]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        sd = math.sqrt(variance)
        moe_95 = (T_CRITICAL_DF11 * sd) / math.sqrt(len(vals))
        summary[m] = {
            "mean": round(mean, 4),
            "sd": round(sd, 4),
            "moe_95": round(moe_95, 4),
            "ci_lower": round(mean - moe_95, 4),
            "ci_upper": round(mean + moe_95, 4),
            "n_resamples": len(vals),
        }
    return summary


def run_extended_bootstrap(
    sample_records: list[dict[str, Any]],
    metric_keys: list[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Execute 1,000 resamples with replacement for full test set."""
    if not sample_records:
        return {}
    rng = random.Random(seed)
    n = len(sample_records)

    resample_means: dict[str, list[float]] = {m: [] for m in metric_keys}

    for _ in range(n_resamples):
        batch = [sample_records[rng.randint(0, n - 1)] for _ in range(n)]
        for m in metric_keys:
            vals = [float(r.get(m, 0.0)) for r in batch if m in r]
            m_val = sum(vals) / len(vals) if vals else 0.0
            resample_means[m].append(m_val)

    summary: dict[str, dict[str, float]] = {}
    for m in metric_keys:
        vals = sorted(resample_means[m])
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        sd = math.sqrt(variance)
        p025 = vals[int(0.025 * len(vals))]
        p975 = vals[int(0.975 * len(vals))]
        summary[m] = {
            "mean": round(mean, 4),
            "sd": round(sd, 4),
            "p025": round(p025, 4),
            "p975": round(p975, 4),
            "n_resamples": len(vals),
        }
    return summary


def run_paired_bootstrap(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    metric: str,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap for comparing config A against config B."""
    n = min(len(records_a), len(records_b))
    if n == 0:
        return {"mean_diff": 0.0, "win_rate_a": 0.0}
    rng = random.Random(seed)

    diffs: list[float] = []
    wins_a = 0

    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        val_a = sum(float(records_a[i].get(metric, 0.0)) for i in indices) / n
        val_b = sum(float(records_b[i].get(metric, 0.0)) for i in indices) / n
        d = val_a - val_b
        diffs.append(d)
        if d > 0:
            wins_a += 1

    mean_diff = sum(diffs) / len(diffs)
    return {
        "mean_diff": round(mean_diff, 4),
        "win_rate_a": round(wins_a / n_resamples, 4),
    }
