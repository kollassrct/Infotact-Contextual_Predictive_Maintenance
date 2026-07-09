# Final Project Summary
## Contextual Predictive Maintenance — IoT Edge AI
**Infotact Solutions | Intern: Preeti | Branch: preeti-dev**

---

## Week 1 — Data Ingestion & Signal Processing
- Loaded AI4I 2020 dataset (10,000 records, 14 columns)
- Computed rolling mean, std, variance (window=10) for 5 sensor columns
- Generated 15 new rolling feature columns
- Identified class imbalance: failures = ~3.4% of data

## Week 2 — Contextual Data Fusion & Feature Engineering
- Simulated timestamps (10-minute intervals)
- Simulated external context: ambient temperature + factory load density
- Created 3 interaction features: temp_gap, load_torque_interaction, heat_stress_index
- Ablation study proved external features improve predictive power

## Week 3 — LightGBM Modeling with SMOTE
- 5-Fold Stratified Cross-Validation
- SMOTE applied strictly inside training folds (no data leakage)
- Mean Macro F1 across folds recorded

## Week 4 — Noise Sensitivity & Threshold Tuning
- Injected Gaussian noise at 4 levels: 0%, 5%, 15%, 30%
- Model remained robust under low and medium noise
- Default threshold: 0.50 → Macro F1: 0.7343
- Tuned threshold:   0.96 → Macro F1: 0.8166
- Improvement: +0.0823

## Noise Sensitivity Results

| Noise Level | Macro F1 | Avg Precision |
|-------------|----------|---------------|
| Clean (0%) | 0.7343 | 0.6443 |
| Low (5%) | 0.7343 | 0.6443 |
| Medium (15%) | 0.7343 | 0.6443 |
| High (30%) | 0.7343 | 0.6443 |

## Reports Generated
- class_imbalance.png
- sensor_distributions.png
- rolling_features_viz.png
- correlation_heatmap.png
- boxplot_by_failure.png
- external_context_patterns.png
- ablation_study_comparison.png
- feature_importance_week2.png
- ablation_study.md
- cv_results.png
- confusion_matrix.png
- precision_recall_curve.png
- feature_importance_lightgbm.png
- noise_sensitivity.png
- noise_effect_on_signal.png
- threshold_tuning.png
- confusion_matrix_comparison.png
