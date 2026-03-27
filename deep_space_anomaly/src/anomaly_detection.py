# ============================================================================
# Author: Ayush Pratik | Project: ISA - SpaceX Anomaly Detection System
# Institution: India Space Academy (ISW) | Status: Final Submission
# License: GNU GPL v3.0
# ============================================================================
"""
Three complementary anomaly detection models on SpaceX Starlink orbital data.
Each model approaches the problem differently — ensemble catches what individuals miss.
"""

import numpy as np
import pandas as pd
import joblib, os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, average_precision_score)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

os.makedirs("models", exist_ok=True)

FEATURE_COLS = [
    'Perigee (km)', 'Apogee (km)', 'Eccentricity',
    'Inclination (degrees)', 'Period (minutes)', 'Launch Mass (kg.)',
    'mean_altitude_km', 'altitude_spread_km', 'ecc_deviation',
    'inc_deviation_deg', 'period_deviation_min', 'days_since_launch_norm'
]

def load_features(csv_path, label_col='anomaly'):
    df = pd.read_csv(csv_path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[label_col].values if label_col in df.columns else None
    return X, y, df

# Model 1: Isolation Forest
def train_isolation_forest(X_normal):
    model = IsolationForest(n_estimators=300, contamination=0.08,
                            max_samples='auto', random_state=42, n_jobs=-1)
    model.fit(X_normal)
    joblib.dump(model, "models/isolation_forest.pkl")
    print("[OK] Isolation Forest trained.")
    return model

def score_isolation_forest(X, model=None):
    if model is None: model = joblib.load("models/isolation_forest.pkl")
    raw_scores  = model.decision_function(X)
    norm_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    preds       = np.where(model.predict(X) == -1, 1, 0)
    return preds, norm_scores

# Model 2: One-Class SVM
def train_one_class_svm(X_normal):
    model = OneClassSVM(kernel='rbf', nu=0.08, gamma='scale')
    model.fit(X_normal)
    joblib.dump(model, "models/one_class_svm.pkl")
    print("[OK] One-Class SVM trained.")
    return model

def score_one_class_svm(X, model=None):
    if model is None: model = joblib.load("models/one_class_svm.pkl")
    raw_scores  = model.decision_function(X)
    norm_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    preds       = np.where(model.predict(X) == -1, 1, 0)
    return preds, norm_scores

# Model 3: Deep Learning Autoencoder
def build_autoencoder(input_dim):
    """Architecture: input -> 32 -> 16 -> 8 (latent) -> 16 -> 32 -> input"""
    inputs  = Input(shape=(input_dim,), name='orbital_input')
    x = Dense(32, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    encoded = Dense(8, activation='relu', name='latent_space')(x)
    x = Dense(16, activation='relu')(encoded)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    decoded = Dense(input_dim, activation='sigmoid', name='reconstruction')(x)
    autoencoder = Model(inputs, decoded, name='OrbitalAutoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(X_normal, epochs=100, batch_size=32):
    model   = build_autoencoder(X_normal.shape[1])
    es      = EarlyStopping(monitor='val_loss', patience=10,
                            restore_best_weights=True, verbose=0)
    history = model.fit(X_normal, X_normal, epochs=epochs, batch_size=batch_size,
                        validation_split=0.15, callbacks=[es], verbose=0)
    model.save("models/autoencoder.keras")
    print(f"[OK] Autoencoder trained. Best val_loss: {min(history.history['val_loss']):.6f}")
    return model

def score_autoencoder(X, model=None, threshold=None):
    if model is None: model = load_model("models/autoencoder.keras")
    recon       = model.predict(X, verbose=0)
    mse         = np.mean(np.power(X - recon, 2), axis=1)
    if threshold is None:
        threshold = np.percentile(mse, 92)
    preds       = (mse > threshold).astype(int)
    norm_scores = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
    return preds, norm_scores, threshold

# Ensemble Majority Vote
def ensemble_predict(X, vote_threshold=2):
    pred_if,  score_if           = score_isolation_forest(X)
    pred_svm, score_svm          = score_one_class_svm(X)
    pred_ae,  score_ae, ae_thresh = score_autoencoder(X)
    votes          = pred_if + pred_svm + pred_ae
    ensemble_pred  = (votes >= vote_threshold).astype(int)
    ensemble_score = (score_if + score_svm + score_ae) / 3.0
    return ensemble_pred, ensemble_score, {
        'isolation_forest': (pred_if,  score_if),
        'one_class_svm':    (pred_svm, score_svm),
        'autoencoder':      (pred_ae,  score_ae),
    }

def print_evaluation(y_true, y_pred, y_score, model_name):
    print(f"\n{'='*55}\n  {model_name}\n{'='*55}")
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Handle case where only 1 class is present in y_true
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        if unique_classes[0] == 0:
            target_names = ['Nominal']
        else:
            target_names = ['Anomaly']
        print("  [Note: Test set contains only one class]")
    else:
        target_names = ['Nominal', 'Anomaly']
        
    try:
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    except Exception as e:
        print(f"  Classification report error: {e}")
        
    try:
        cm = confusion_matrix(y_true, y_pred)
        if len(cm) == 1:
            # Manually construct 2x2 confusion matrix
            if unique_classes[0] == 0:
                tn = cm[0,0] if np.sum(y_pred==0) > 0 else 0
                fp = len(y_true) - tn
                fn, tp = 0, 0
            else:
                tp = cm[0,0] if np.sum(y_pred==1) > 0 else 0
                fn = len(y_true) - tp
                tn, fp = 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
    except Exception as e:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    far = fp / (fp + tn + 1e-8)
    
    try:
        if len(unique_classes) > 1:
            auc = roc_auc_score(y_true, y_score)
            ap  = average_precision_score(y_true, y_score)
        else:
            auc = np.nan
            ap = np.nan
    except Exception:
        auc = np.nan
        ap = np.nan
    print(f"  Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  False Alarm Rate: {far:.4f} ({far*100:.2f}%)")
    print(f"  ROC-AUC:          {auc:.4f}")
    print(f"  Avg Precision:    {ap:.4f}")
    return {'model': model_name, 'auc': auc, 'ap': ap, 'far': far,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

if __name__ == "__main__":
    print("\n[STEP 2] TRAINING ANOMALY DETECTION MODELS ON SPACEX STARLINK DATA...\n")
    X_train, y_train, train_df = load_features("data/processed/train_processed.csv")
    X_test,  y_test,  test_df  = load_features("data/processed/test_processed.csv")
    X_train_normal = X_train[y_train == 0]
    print(f"Training on {len(X_train_normal)} nominal satellites "
          f"({len(X_train)-len(X_train_normal)} anomalies held out)")
    train_isolation_forest(X_train_normal)
    train_one_class_svm(X_train_normal)
    train_autoencoder(X_train_normal)
    print("\n[RESULTS] EVALUATION ON TEST SET:")
    pred_if,  score_if           = score_isolation_forest(X_test)
    pred_svm, score_svm          = score_one_class_svm(X_test)
    pred_ae,  score_ae, _        = score_autoencoder(X_test)
    ens_pred, ens_score, _       = ensemble_predict(X_test)
    results = []
    for name, pred, score in [
        ("Isolation Forest",     pred_if,  score_if),
        ("One-Class SVM",        pred_svm, score_svm),
        ("Autoencoder (DL)",     pred_ae,  score_ae),
        ("[ENSEMBLE] (Voted)",   ens_pred, ens_score),
    ]:
        r = print_evaluation(y_test, pred, score, name)
        results.append(r)
    test_df['pred_ensemble'] = ens_pred
    test_df['anomaly_score'] = ens_score
    test_df['pred_if']       = pred_if
    test_df['pred_svm']      = pred_svm
    test_df['pred_ae']       = pred_ae
    test_df.to_csv("data/processed/test_predictions.csv", index=False)
    print("\n[OK] Predictions saved to data/processed/test_predictions.csv")
