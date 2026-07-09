# Ablation Study — Week 2

## Objective
Prove that adding external context features (ambient temperature, factory load density) improves machine failure prediction.

## Method
Trained identical Random Forest models on:
1. Internal sensor features only (baseline)
2. Internal + external contextual features (enriched)

## Results

| Model | Macro F1 Score |
|---|---|
| Baseline (internal only) | 0.7933 |
| Enriched (internal + external) | 0.7876 |

**Improvement: -0.0057 (-0.72%)**

## Conclusion
External features did not show improvement in this run.
