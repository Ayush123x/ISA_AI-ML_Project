# GitHub Repository Finalization — Implementation Plan

## Goal
Finalize the SpaceX Starlink Anomaly Detection project into a professional, clone-ready, review-ready GitHub repository for the India Space Academy (ISA) submission.

---

## Proposed Changes

### Configuration Files

#### [NEW] [LICENSE](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/LICENSE)
- Full text of the **GNU GPL v3.0** license.

#### [NEW] [.gitignore](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/.gitignore)
- Ignore `__pycache__/`, `.DS_Store`, `.venv/`, `*.pyc`, IDE files, etc.

#### [MODIFY] [requirements.txt](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/requirements.txt)
- Update to match the **actually installed** package versions on your system:
  `pandas==2.3.3`, `scikit-learn==1.8.0`, `tensorflow==2.21.0`, `streamlit==1.55.0`, `plotly==6.6.0`, etc.

---

### Branding Headers (all [.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/run_project.py) files)

#### [MODIFY] All 7 Python files
Add this standard header block to the top of every [.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/run_project.py) file:
```python
# ============================================================================
# Author: Ayush Pratik | Project: ISA - SpaceX Anomaly Detection System
# Institution: India Space Academy (ISW) | Status: Final Submission
# License: GNU GPL v3.0
# ============================================================================
```

Files: [data_preprocessing.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/src/data_preprocessing.py), [anomaly_detection.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/src/anomaly_detection.py), [rul_predictor.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/src/rul_predictor.py), [decision_engine.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/src/decision_engine.py), [evaluator.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/src/evaluator.py), [dashboard/app.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/dashboard/app.py), [run_project.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/run_project.py)

> [!IMPORTANT]
> The author name is set to "Ayush Pratik" based on your Windows username. Please confirm if this is correct or provide the name you'd prefer.

---

### Pipeline Runner

#### [MODIFY] [run_project.py](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/run_project.py)
- Add **model existence check**: if `models/` contains trained models, skip retraining and go straight to scoring.
- Generate `reports/final_report.txt` with a summary of all pipeline results.
- Fix the printed model file extensions ([.h5](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/models/autoencoder.h5) → [.keras](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/models/lstm_rul.keras)).
- Use `sys.executable` instead of `"python"` for cross-platform compatibility.

---

### Documentation

#### [MODIFY] [README.md](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/README.md)
Complete rewrite with:
- Professional header with ISA branding
- Deployment status (Localhost via Streamlit)
- Full setup instructions (`pip install -r requirements.txt`, `python run_project.py`)
- File structure tree
- Placeholder sections for "Anomaly Heatmaps" and "Decision Logs"
- Confirmation that all paths are relative

---

## Verification Plan

### Automated Test
Run the full pipeline from a clean state to confirm clone-ability:
```bash
python run_project.py
```
This should:
1. Load [data/raw/SpaceX_Satellite_Dataset.csv](file:///c:/Users/prati/.gemini/antigravity/scratch/deep_space_anomaly/data/raw/SpaceX_Satellite_Dataset.csv)
2. Preprocess, train models, score, generate decisions
3. Create `reports/final_report.txt`
4. Exit with code 0

### Manual Verification
1. Confirm `reports/final_report.txt` exists and contains results
2. Run `python -m streamlit run dashboard/app.py` and confirm it loads
3. Verify no absolute paths (`C:\Users\...`) exist in any source file via `grep`
