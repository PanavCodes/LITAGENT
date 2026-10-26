# Gate 6 Verification: Evaluation, Ablations, Bootstrap & Calibration

Executed on: 2026-10-26
Platform: Windows (Python 3.12.13)
Branch: phase-0-setup

## Gate Checklist

- [x] Ablation table has all five configs with mean +/- SD (`results/ablation_table.csv`)
- [x] Question sets have recorded hand-verification rate (30% vector, 20% graph, exceeding >=25% and >=10% requirements)
- [x] Test split untouched by any tuning (`data/eval/split.json`: 29 dev / 74 test)
- [x] `make eval` / `python -m litagent.eval.report` reproduces tables from cache
- [x] Calibration outputs exist (AUROC, ECE, plot in `results/calibration.json` and `results/figures/calibration_curve.png`)

## Ablation Results (`results/ablation_table.csv`)

```csv
Config,Faithfulness,Ans. Relevancy,Ctx Precision,Ctx Recall,Graph set-F1,Latency (s),LLM calls
vector_only,0.130 +/- 0.034,0.109 +/- 0.029,0.135 +/- 0.050,0.086 +/- 0.028,0.033 +/- 0.039,0.04,2.0
graph_only,0.469 +/- 0.030,0.469 +/- 0.025,0.000 +/- 0.000,0.000 +/- 0.000,0.087 +/- 0.053,0.01,2.0
joint_baseline,0.463 +/- 0.030,0.469 +/- 0.025,0.793 +/- 0.068,0.608 +/- 0.036,0.087 +/- 0.053,0.03,2.0
router,0.236 +/- 0.012,0.203 +/- 0.011,0.000 +/- 0.000,0.000 +/- 0.000,0.087 +/- 0.053,0.02,1.8
router_crag,0.484 +/- 0.003,0.494 +/- 0.006,0.500 +/- 0.000,0.396 +/- 0.011,0.087 +/- 0.053,0.02,1.8
```

## Retrieval Sub-Ablation (`results/retrieval_ablation.json`)

```json
{
  "r_bm25": { "Recall@5": 0.1429, "Recall@10": 0.2857, "MRR": 0.1028 },
  "r_dense": { "Recall@5": 0.0571, "Recall@10": 0.1714, "MRR": 0.0564 },
  "r_rrf": { "Recall@5": 0.0857, "Recall@10": 0.2000, "MRR": 0.0814 },
  "r_rrf_rerank": { "Recall@5": 0.0857, "Recall@10": 0.0857, "MRR": 0.0486 }
}
```

## Uncertainty Calibration (`results/calibration.json`)

```json
{
  "dev_samples": 29,
  "test_samples": 74,
  "fitted_weights_on_dev": {
    "w_S1": -2.6726,
    "w_S2": 1.5947,
    "w_S5": -1.2581,
    "bias": 0.0084
  },
  "test_auroc_confidence": 0.6729,
  "test_ece_confidence": 0.4631,
  "test_auroc_fitted": 0.7271,
  "test_ece_fitted": 0.1097,
  "signal_ablation": {
    "all_signals_auroc": 0.6729,
    "drop_S1_auroc": 0.6164,
    "drop_S2_auroc": 0.2729,
    "drop_S5_auroc": 0.3271
  }
}
```

## Unit Test Verification

Command: `pytest tests/unit/test_phase6.py`
Output: `5 passed in 7.70s`
