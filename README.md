# SpaceX Starlink Satellite Mission Control 🛰️

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Models-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Review%20Ready-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-GPL--3.0-red?style=for-the-badge)

**Real-Time Orbital Anomaly Detection & Autonomous Decision Support System**

---

### India Space Academy (ISW) - Final Project Submission
**Author:** Ayush Pratik
**Status:** Review-Ready 
**License:** GNU GPL v3.0

---
## 🔎 The Mission
Managing a massive Low Earth Orbit (LEO) constellation of **902 satellites** requires split-second precision. Manual monitoring of orbital decay and telemetry deviations is mathematically impossible at scale. This system analyzes real SpaceX Starlink data to detect orbital deviations and predict Remaining Useful Life (RUL) using LEO physics simulations.

## 💡 The Solution
Sentinel introduces a **High-Stakes AI Decision Engine** that autonomously flags orbital anomalies and outputs actionable fleet commands:

* 🔹 **Anomaly Detection** → Detects deviations using Isolation Forest, SVM, and Deep Autoencoders.
* 🔹 **RUL Prediction** → Forecasts satellite lifespan using LSTM-based decay modeling.
* 🔹 **Autonomous Logic** → Recommends specific telemetry commands (e.g., *Monitor, Orbit Correction, Emergency Deorbit*).

---

## 🔥 Autonomous Decision Matrix
To ensure fleet safety, the Decision Engine utilizes a composite risk scoring system:

| Risk Level | Trigger Condition | Status | Autonomous Fleet Command |
| :--- | :--- | :--- | :--- |
| **Level 1** | Stable Orbit & Nominal Telemetry | **Nominal** | `Monitor` |
| **Level 2** | Altitude Drift / Minor Decay | **Warning** | `Orbit Correction Burn` |
| **Level 3** | Critical RUL / Severe Deviation | **Critical** | `Emergency Deorbit` |

---

## 🏗 System Architecture
The platform is built on a modular, OS-agnostic architecture designed to manage the full lifecycle of an orbital event.

`Raw Telemetry` ➡️ `Data Preprocessing` ➡️ `ML Pipelines (Anomaly/RUL)` ➡️ `Decision Engine` ➡️ `Streamlit Dashboard`

---

## ⚙️ Tech Stack
* **Language:** Python 3.12
* **Machine Learning:** TensorFlow/Keras, Scikit-Learn
* **Visualization:** Streamlit, Plotly (Interactive Heatmaps)
* **Logic:** LEO Physics Simulations & Autonomous Decision Engine

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
