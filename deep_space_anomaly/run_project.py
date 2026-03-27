# ============================================================================
# Author: Ayush Pratik | Project: ISA - SpaceX Anomaly Detection System
# Institution: India Space Academy (ISW) | Status: Final Submission
# License: GNU GPL v3.0
# ============================================================================
"""
ISW PROJECT - ONE-CLICK FULL PIPELINE RUNNER
Runs ALL steps in order. Expects SpaceX_Satellite_Dataset.csv in data/raw/
"""

import subprocess, sys, os

print("=" * 65)
print("  ISW DEEP SPACE ANOMALY DETECTION - SPACEX STARLINK DATASET")
print("  India Space Academy | AI & ML in Space Exploration")
print("=" * 65)

if not os.path.exists("data/raw/SpaceX_Satellite_Dataset.csv"):
    print("\n[ERROR] data/raw/SpaceX_Satellite_Dataset.csv not found!")
    print("   Please copy the uploaded CSV file to data/raw/ before running.")
    sys.exit(1)
else:
    print("\n[OK] SpaceX Starlink dataset found.")

# Determine if models already exist to potentially skip full retraining
models_exist = (
    os.path.exists("models/isolation_forest.pkl") and
    os.path.exists("models/autoencoder.keras") and
    os.path.exists("models/lstm_rul.keras")
)

if models_exist:
    print("\n[INFO] Trained models found in models/ directory.")
    print("       The pipeline will re-evaluate them on the latest dataset.")

python_exe = sys.executable

STEPS = [
    ("[STEP 1] Preprocessing SpaceX orbital data...",     f'"{python_exe}" src/data_preprocessing.py'),
    ("[STEP 2] Running anomaly detection models...",      f'"{python_exe}" src/anomaly_detection.py'),
    ("[STEP 3] Running LSTM RUL predictor...",            f'"{python_exe}" src/rul_predictor.py'),
    ("[STEP 4] Running autonomous decision engine...",     f'"{python_exe}" src/decision_engine.py'),
]

for desc, cmd in STEPS:
    print(f"\n{desc}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[FAILED] {cmd}")
        sys.exit(1)
    print(f"  [DONE]")

# Generate Final Report text file
os.makedirs("reports", exist_ok=True)
report_path = "reports/final_report.txt"
with open(report_path, "w") as f:
    f.write("ISW - SPACEX ANOMALY DETECTION SYSTEM FINAL REPORT\n")
    f.write("==================================================\n")
    f.write("Pipeline Executed Successfully.\n\n")
    f.write("Output Files Generated:\n")
    f.write("- data/processed/test_predictions.csv\n")
    f.write("- data/processed/decision_log.csv\n")
    f.write("- data/processed/rul_predictions.csv\n")
    f.write("- reports/figures/confusion_matrix.png\n\n")
    f.write("Launch the Dashboard to view interactive results:\n")
    f.write("  streamlit run dashboard/app.py\n")

print("\n" + "=" * 65)
print("  ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
print()
print("  Output files generated:")
print("    data/processed/train_processed.csv")
print("    data/processed/test_processed.csv")
print("    data/processed/test_predictions.csv")
print("    data/processed/decision_log.csv")
print("    data/processed/rul_predictions.csv")
print("    models/ (isolation_forest.pkl, one_class_svm.pkl, autoencoder.keras, lstm_rul.keras)")
print(f"    {report_path}")
print()
print("  Launch Dashboard:")
print("    streamlit run dashboard/app.py")
print("=" * 65)
