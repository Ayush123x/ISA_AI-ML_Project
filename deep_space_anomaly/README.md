# SpaceX Starlink Satellite Mission Control 🛰️
**Real-Time Orbital Anomaly Detection & Autonomous Decision Support System**

---

### India Space Academy (ISW) - Final Project Submission
**Author:** Ayush Pratik
**Status:** Review-Ready 
**License:** GNU GPL v3.0

---

## 📌 Project Overview
This repository contains a high-stakes AI-Based Anomaly Detection & Decision Support System built for the India Space Academy (ISA). The system analyzes the **real SpaceX Starlink Dataset (902 satellites)** to detect orbital deviations, predict Remaining Useful Life (RUL) using LEO physics simulations, and autonomously recommend actions for the fleet (e.g., *Monitor, Orbit Correction Burn, Emergency Deorbit*).

### 🚀 Deployment Status
**This application currently runs on Localhost via Streamlit.** It is fully self-contained and ready for grading.

---

## 📦 Repository Architecture

```text
.
├── data/
│   ├── raw/                   # (MUST contain SpaceX_Satellite_Dataset.csv)
│   └── processed/             # Generated feature sets & prediction logs
├── models/                    # Saved ML models (.pkl, .keras)
├── src/                       # Core ML pipelines
│   ├── data_preprocessing.py
│   ├── anomaly_detection.py
│   ├── rul_predictor.py
│   ├── decision_engine.py
│   └── evaluator.py
├── dashboard/                 # Streamlit UI
│   └── app.py
├── reports/                   # Technical reports and figures
├── .gitignore
├── LICENSE                    # GNU GPL v3.0
├── requirements.txt           # Version-locked dependencies
├── run_project.py             # One-click pipeline execution script
└── README.md
```
*(Note: All code heavily utilizes `os.path` relative references to ensure it is 100% clone-able across different operating systems without breaking).*

---

## 🛠️ Setup & Execution Instructions

Follow these steps to clone the repository and run the full AI pipeline on your local machine:

### 1. Requirements
Ensure you have **Python 3.10+** installed. (Developed on Python 3.12).

### 2. Installation
Open your terminal/command prompt, clone the repo, and install the exact version-locked dependencies:
```bash
git clone https://github.com/your-username/spacex-anomaly-detection.git
cd spacex-anomaly-detection
pip install -r requirements.txt
```

### 3. Run the ML Pipeline
Execute the master script. This will sequentially run data preprocessing, train the machine learning models (Isolation Forest, SVM, Deep Autoencoder, LSTM), execute the autonomous decision engine, and generate a final report.
```bash
python run_project.py
```

### 4. Launch the Mission Control Dashboard
Once the pipeline has successfully generated the predictions in `/data/processed/`, launch the interactive UI:
```bash
streamlit run dashboard/app.py
```

---

## 📸 Visual Proof of Localhost Functionality

### Anomaly Heatmaps (Fleet-Wide Orbital Mapping)
*(Reviewers: The dashboard contains interactive Plotly heatmaps plotting Perigee vs Apogee mapping the nominal vs anomalous fleet clusters).*
![Anomaly Heatmap Placeholder](https://via.placeholder.com/800x400.png?text=Interactive+Plotly+Map+Shown+in+Localhost)

### Autonomous Decision Logs
*(Reviewers: The decision engine evaluates composite risk scores predicting satellite RUL and altitude stability, outputting direct telemetry commands).*
![Decision Log Placeholder](https://via.placeholder.com/800x200.png?text=Decision+Log+Table+Shown+in+Localhost)

---

## 🛡️ License & Academic Integrity
This code is released under the **GNU General Public License v3.0**. It is submitted for academic review by the India Space Academy. Do not use the logic or weights in proprietary/closed-source projects without explicit attribution to the author.
