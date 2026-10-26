"""Uncertainty calibration and evaluation module.

Per EVALUATION.md §6 (FR-UNC-4):
- Binary correctness label definition:
  - Graph: set-F1 >= 0.8
  - Vector: faithfulness >= 0.7 and answer_relevancy >= 0.7
  - Out-of-scope: good = abstained
- Logistic regression fit on dev split only
- AUROC, ECE (10 bins), reliability plot, selective-answering curve on test split
- Signal ablation (drop S1, drop S2, drop S5)
- Saves results/calibration.json and results/figures/calibration_curve.png
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_correctness_label(record: dict[str, Any]) -> int:
    """Determine binary correctness label per EVALUATION.md §3.3."""
    expected_route = record.get("expected_route", "")
    is_abstain = bool(record.get("is_abstain", False))

    if expected_route == "out_of_scope":
        return 1 if is_abstain else 0

    if is_abstain:
        return 0

    if expected_route == "graph":
        return 1 if float(record.get("graph_set_f1", 0.0)) >= 0.8 else 0

    # Vector or both
    faith = float(record.get("faithfulness", 0.0))
    relev = float(record.get("answer_relevancy", 0.0))
    return 1 if faith >= 0.7 and relev >= 0.7 else 0


def compute_ece(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    if not labels or not probs:
        return 0.0
    bin_size = 1.0 / n_bins
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        bin_low = i * bin_size
        bin_high = (i + 1) * bin_size
        indices = [
            idx for idx, p in enumerate(probs)
            if bin_low <= p < bin_high or (i == n_bins - 1 and p == 1.0)
        ]
        if not indices:
            continue
        bin_acc = sum(labels[idx] for idx in indices) / len(indices)
        bin_conf = sum(probs[idx] for idx in indices) / len(indices)
        ece += (len(indices) / total) * abs(bin_acc - bin_conf)

    return round(ece, 4)


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute Area Under the Receiver Operating Characteristic curve via Mann-Whitney U."""
    if not labels or not scores:
        return 0.5
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return 0.5

    # Count pairs where pos > neg
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return round(wins / (len(pos) * len(neg)), 4)


def fit_logistic_dev(dev_records: list[dict[str, Any]]) -> dict[str, float]:
    """Fit logistic weights on dev split only: label ~ S1, S2, S5."""
    # Simple, robust numerical logistic fit
    # Features: s1 (margin), s2 (grade), s5 (graph_ok)
    X = []
    y = []
    for r in dev_records:
        signals = r.get("signals", {})
        s1 = float(signals.get("S1", 0.5))
        s2 = float(signals.get("S2", 0.5))
        s5 = float(signals.get("S5", 0.5))
        lbl = compute_correctness_label(r)
        X.append([s1, s2, s5])
        y.append(lbl)

    if not y:
        return {"w_S1": 0.35, "w_S2": 0.35, "w_S5": 0.30, "bias": 0.0}

    # Use standard initialized weights and calibrate gradient descent
    w = [0.35, 0.35, 0.30]
    bias = 0.0
    lr = 0.1
    for _ in range(50):
        for xi, yi in zip(X, y):
            z = bias + sum(wi * x for wi, x in zip(w, xi))
            pred = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, z))))
            err = yi - pred
            bias += lr * err
            for k in range(3):
                w[k] += lr * err * xi[k]

    return {
        "w_S1": round(w[0], 4),
        "w_S2": round(w[1], 4),
        "w_S5": round(w[2], 4),
        "bias": round(bias, 4),
    }


def run_calibration_analysis(
    dev_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Run full calibration pipeline on dev and test splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit on dev only
    weights = fit_logistic_dev(dev_records)

    # 2. Test predictions
    test_labels: list[int] = []
    test_uncertainties: list[float] = []
    test_fitted_probs: list[float] = []

    s1_vals: list[float] = []
    s2_vals: list[float] = []
    s5_vals: list[float] = []

    for r in test_records:
        lbl = compute_correctness_label(r)
        test_labels.append(lbl)
        u = float(r.get("uncertainty_score", 0.5))
        test_uncertainties.append(u)

        signals = r.get("signals", {})
        s1 = float(signals.get("S1", 0.5))
        s2 = float(signals.get("S2", 0.5))
        s5 = float(signals.get("S5", 0.5))
        s1_vals.append(s1)
        s2_vals.append(s2)
        s5_vals.append(s5)

        z = weights["bias"] + weights["w_S1"] * s1 + weights["w_S2"] * s2 + weights["w_S5"] * s5
        p = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, z))))
        test_fitted_probs.append(round(p, 4))

    # Test confidence = 1 - uncertainty
    conf_scores = [1.0 - u for u in test_uncertainties]

    # Metrics
    auroc_conf = compute_auroc(test_labels, conf_scores)
    ece_conf = compute_ece(test_labels, conf_scores)
    auroc_fitted = compute_auroc(test_labels, test_fitted_probs)
    ece_fitted = compute_ece(test_labels, test_fitted_probs)

    # Signal ablations
    auroc_drop_s1 = compute_auroc(test_labels, [0.5 * s2 + 0.5 * s5 for s2, s5 in zip(s2_vals, s5_vals)])
    auroc_drop_s2 = compute_auroc(test_labels, [0.5 * s1 + 0.5 * s5 for s1, s5 in zip(s1_vals, s5_vals)])
    auroc_drop_s5 = compute_auroc(test_labels, [0.5 * s1 + 0.5 * s2 for s1, s2 in zip(s1_vals, s2_vals)])

    results = {
        "dev_samples": len(dev_records),
        "test_samples": len(test_records),
        "fitted_weights_on_dev": weights,
        "test_auroc_confidence": auroc_conf,
        "test_ece_confidence": ece_conf,
        "test_auroc_fitted": auroc_fitted,
        "test_ece_fitted": ece_fitted,
        "signal_ablation": {
            "all_signals_auroc": auroc_conf,
            "drop_S1_auroc": auroc_drop_s1,
            "drop_S2_auroc": auroc_drop_s2,
            "drop_S5_auroc": auroc_drop_s5,
        },
    }

    # Save JSON
    calib_json = output_dir / "calibration.json"
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot calibration curve with matplotlib
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        # 10 bins
        bin_confs = []
        bin_accs = []
        for i in range(10):
            low, high = i * 0.1, (i + 1) * 0.1
            idxs = [k for k, p in enumerate(conf_scores) if low <= p < high or (i == 9 and p == 1.0)]
            if idxs:
                bin_confs.append(sum(conf_scores[k] for k in idxs) / len(idxs))
                bin_accs.append(sum(test_labels[k] for k in idxs) / len(idxs))

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        if bin_confs:
            ax.plot(bin_confs, bin_accs, "s-", color="#2563eb", label=f"LitAgent Confidence (ECE={ece_conf})")
        ax.set_xlabel("Predicted Confidence (1 - Uncertainty)")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_title("LitAgent Uncertainty Calibration Curve (Test Split)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plot_path = figures_dir / "calibration_curve.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        results["plot_path"] = str(plot_path)
    except Exception as e:
        logger.warning("Could not render calibration plot: %s", e)

    return results
