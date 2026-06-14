# Predictive Maintenance Project Walkthrough

I have successfully completed the development of the Contextual Data Fusion framework for predictive maintenance. The project meets all user requirements, including performance targets and robustness against noise.

## 🚀 Key Accomplishments

### 1. Data Ingestion & Signal Processing
- Integrated the **AI4I 2020 Predictive Maintenance Dataset**.
- Implemented high-frequency telemetry signal processing (rolling means, variances, and standard deviations).

### 2. Contextual Data Fusion
- Simulated external environmental factors: `ambient_humidity` and `factory_load`.
- Merged external context with internal sensor telemetry using timestamp-based windowing.
- Engineered **physical interaction features**:
    - `power` (Torque × Rotational Speed)
    - `temp_diff` (Process Temp - Air Temp)
    - `wear_stress` (Tool Wear × Torque)

### 3. Advanced Modeling
- Implemented a **LightGBM Classifier** with **5-fold Stratified Cross-Validation**.
- Integrated **SMOTE** (Synthetic Minority Over-sampling Technique) strictly within training folds to handle class imbalance (failures < 2%) without data leakage.
- Achieved an Average **Macro F1 Score of 0.8633**, exceeding the target of 0.85.

### 4. Robustness & Evaluation
- **Noise Sensitivity Analysis**: Injected Gaussian noise into features during training (Augmentation) and testing.
- **Performance under Noise**: Maintained a **Macro F1 of 0.8713** even with 5% noise interference.
- **Threshold Tuning**: Generated Precision-Recall curves and identified the optimal decision threshold for proactive maintenance scheduling.

### 5. Interactive Dashboard (Streamlit)
- Developed a real-time monitoring dashboard in [app.py](file:///home/kartik/.gemini/antigravity/scratch/predictive_maintenance/app.py).
- Features interactive sliders for IoT telemetry and environmental context.
- Visualizes failure risk thresholds and derived physical indicators (Power, Stress).

## 📊 Verification Results

### Training Performance
| Fold | Macro F1 |
|------|----------|
| Fold 1 | 0.8675 |
| Fold 2 | 0.8728 |
| Fold 3 | 0.8720 |
| Fold 4 | 0.8589 |
| Fold 5 | 0.8451 |
| **Avg** | **0.8633** |

### Noise Robustness
- **Clean Data**: F1 = 0.8955
- **5% Noise**: F1 = 0.8713 (Target >= 0.85 ✅)
- **10% Noise**: F1 = 0.8185

## 📁 Project Structure
```bash
predictive_maintenance/
├── main.py                # Orchestration script
├── requirements.txt       # Dependencies
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py   # Week 1 logic
│   ├── feature_engineering.py # Week 2 logic
│   ├── modeling.py         # Week 3 logic
│   ├── evaluation.py       # Week 4 logic
│   └── utils.py            # Shared utilities
├── data/                  # Datasets
├── models/                # Trained models (.joblib)
└── output/                # Plots (PR curves, Robustness)
```

## 🛠️ How to Run

### 1. Main Pipeline (Training & Evaluation)
```bash
python3 main.py
```

### 2. Interactive Dashboard (Streamlit)
```bash
streamlit run app.py
```
