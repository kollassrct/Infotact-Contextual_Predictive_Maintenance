# Contextual Predictive Maintenance (IoT Edge AI)

## Overview

This project is part of the Infotact Technical Internship Program for Advanced Data Science & Machine Learning.

The objective is to develop an intelligent predictive maintenance system that combines industrial IoT telemetry data with external contextual information to predict machine failures before they occur.

Unlike traditional predictive maintenance systems that rely only on machine sensor data, this project explores contextual data fusion by integrating environmental and operational factors to improve prediction accuracy.

---

## Business Problem

Unexpected machine failures lead to:

- Production downtime
- Increased maintenance costs
- Reduced operational efficiency
- Equipment damage

This project aims to shift maintenance strategies from reactive maintenance to proactive maintenance using Machine Learning.

---

## Project Objectives

- Analyze industrial IoT sensor telemetry
- Engineer signal-processing features
- Integrate contextual external data
- Handle highly imbalanced failure datasets
- Build robust failure prediction models
- Evaluate model performance under noisy conditions
- Document the complete ML lifecycle using GitHub

---

## Dataset

### AI4I Predictive Maintenance Dataset

The dataset contains machine telemetry information including:

| Feature | Description |
|----------|------------|
| Air Temperature | Ambient air temperature |
| Process Temperature | Machine process temperature |
| Rotational Speed | Machine RPM |
| Torque | Applied torque |
| Tool Wear | Tool wear duration |
| Machine Failure | Target variable |

---

# Project Roadmap

## Week 1: IoT Telemetry Ingestion & Signal Processing

### Objectives

- Load and validate dataset
- Perform data quality checks
- Explore sensor telemetry
- Create rolling statistical features

### Features to Generate

- Rolling Mean
- Rolling Standard Deviation
- Rolling Variance

### Deliverables

- Data ingestion notebook
- Data validation report
- Feature engineering notebook
- Clean dataset

---

## Week 2: Contextual Data Fusion & Feature Engineering

### Objectives

- Simulate or integrate contextual information
- Merge external and internal data
- Generate enriched features
- Conduct ablation study

### Contextual Features

- Ambient Temperature
- Load Density
- Environmental Conditions
- Operational Context

### Deliverables

- Context-enriched dataset
- Feature engineering report
- Ablation study results

---

## Week 3: Imbalanced Classification & LightGBM Modeling

### Objectives

- Implement 5-Fold Stratified Cross Validation
- Apply SMOTE within training folds
- Train LightGBM classifier
- Evaluate predictive performance

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Macro F1 Score

### Deliverables

- Trained LightGBM model
- Cross-validation results
- Model evaluation report

---

## Week 4: Noise Sensitivity Analysis & Threshold Tuning

### Objectives

- Inject synthetic sensor noise
- Evaluate model robustness
- Generate Precision-Recall curves
- Tune classification thresholds

### Deliverables

- Robustness analysis
- Threshold optimization report
- Final predictive maintenance pipeline

---

# Technology Stack

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- LightGBM
- Imbalanced-Learn (SMOTE)

## Tools

- Jupyter Notebook
- Git
- GitHub

---

# Repository Structure

```text
contextual-predictive-maintenance/
│
├── data/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_signal_processing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── reports/
│
├── src/
│
├── images/
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

---

# Learning Outcomes

Through this project, I aim to gain practical experience in:

- Data Ingestion Pipelines
- Time-Series Signal Processing
- Feature Engineering
- Data Fusion Techniques
- Imbalanced Classification
- LightGBM Modeling
- Model Evaluation
- ML Engineering Best Practices
- GitHub Project Management

---

# Project Status

### Week 1

- [x] Project Initialization
- [x] Dataset Selection
- [ ] Data Ingestion
- [ ] Data Validation
- [ ] Signal Processing Features

### Week 2

- [ ] Contextual Data Integration
- [ ] Feature Engineering
- [ ] Ablation Study

### Week 3

- [ ] SMOTE Pipeline
- [ ] LightGBM Training
- [ ] Cross Validation

### Week 4

- [ ] Noise Analysis
- [ ] Threshold Optimization
- [ ] Final Evaluation

---

# Author

**Satya Sri Ram Charan Teja Kolla**

Final Year B.Tech Student  
Aspiring Data Scientist | Machine Learning Enthusiast | GATE 2027 Aspirant

---

# License

This project is developed for educational and internship purposes.
