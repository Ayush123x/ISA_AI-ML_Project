# ============================================================================
# Author: Ayush Pratik | Project: ISA - SpaceX Anomaly Detection System
# Institution: India Space Academy (ISW) | Status: Final Submission
# License: GNU GPL v3.0
# ============================================================================
"""
LSTM-based Remaining Useful Life (RUL) Predictor for Starlink satellites.
Since the dataset is a snapshot (not time-series per satellite), we simulate
orbital decay sequences per satellite using LEO physics:
  - Perigee decreases ~2 km/month due to atmospheric drag at 550 km
  - Eccentricity oscillates from solar radiation pressure
  - Inclination drifts slowly from J2 perturbation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

SEQUENCE_LENGTH = 10

def simulate_orbital_decay_sequences(df, feature_cols, seq_len=SEQUENCE_LENGTH, seed=42):
    """
    Simulate seq_len time-steps of orbital evolution per satellite.
    Physics: LEO at 550 km loses ~2 km/month altitude.
    """
    np.random.seed(seed)
    X_sequences, y_ruls = [], []
    for _, row in df.iterrows():
        RUL_base = row['RUL']
        seq = []
        for step in range(seq_len):
            t = step / seq_len
            perigee      = row['Perigee (km)']          - t * 2.5 + np.random.normal(0, 0.5)
            apogee       = row['Apogee (km)']           - t * 2.0 + np.random.normal(0, 0.5)
            eccentricity = row['Eccentricity']          + np.random.normal(0, 1e-5)
            inclination  = row['Inclination (degrees)'] + np.random.normal(0, 0.005)
            period       = row['Period (minutes)']      - t * 0.05 + np.random.normal(0, 0.01)
            mass         = row['Launch Mass (kg.)']     - t * 0.5
            mean_alt     = (perigee + apogee) / 2
            alt_spread   = apogee - perigee
            ecc_dev      = abs(eccentricity - 0.000145)
            inc_dev      = abs(inclination - 53.0)
            per_dev      = abs(period - 95.6)
            days_norm    = row.get('days_since_launch_norm', 0.5) + t * 0.02
            step_features = [perigee, apogee, eccentricity, inclination, period, mass,
                             mean_alt, alt_spread, ecc_dev, inc_dev, per_dev, days_norm]
            step_features = np.clip(step_features, 0, 1)
            seq.append(step_features)
        rul_remaining = max(0, RUL_base - (seq_len - 1) * (365 / seq_len))
        X_sequences.append(np.array(seq))
        y_ruls.append(rul_remaining)
    return np.array(X_sequences), np.array(y_ruls)

def build_lstm_rul(seq_len, n_features):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='relu', name='rul_output')
    ], name='StarLinkRUL_LSTM')
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    return model

def train_rul_model(train_df, feature_cols):
    print("  Generating orbital decay sequences for LSTM training...")
    X_train, y_train = simulate_orbital_decay_sequences(train_df, feature_cols)
    print(f"  LSTM input shape: {X_train.shape}  RUL range: [{y_train.min():.0f}, {y_train.max():.0f}] days")
    model = build_lstm_rul(SEQUENCE_LENGTH, X_train.shape[2])
    model.summary()
    es  = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
    model.fit(X_train, y_train, epochs=200, batch_size=32,
              validation_split=0.15, callbacks=[es, rlr], verbose=1)
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_rul.keras")
    print("[OK] LSTM RUL model saved.")
    return model

def evaluate_rul(test_df, feature_cols, model=None):
    if model is None: model = load_model("models/lstm_rul.keras")
    X_test, y_test = simulate_orbital_decay_sequences(test_df, feature_cols)
    y_pred = model.predict(X_test, verbose=0).flatten()
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n[RESULTS] RUL PREDICTION (SpaceX Starlink):")
    print(f"  MAE:  {mae:.1f} days ({mae/365*12:.1f} months)")
    print(f"  RMSE: {rmse:.1f} days")
    for i in range(min(5, len(y_test))):
        print(f"  Satellite {i+1}: True RUL={y_test[i]:.0f}d | Predicted={y_pred[i]:.0f}d")
    test_df = test_df.copy()
    test_df['RUL_predicted'] = np.nan
    test_df.iloc[:len(y_pred), test_df.columns.get_loc('RUL_predicted')] = y_pred
    test_df[['Current Official Name of Satellite', 'RUL', 'RUL_predicted']].to_csv(
        "data/processed/rul_predictions.csv", index=False)
    print("  RUL predictions saved.")
    return y_pred, y_test, mae, rmse

if __name__ == "__main__":
    print("\n[STEP 3] TRAINING LSTM RUL PREDICTOR ON STARLINK ORBITAL DATA...\n")
    train_df  = pd.read_csv("data/processed/train_processed.csv")
    test_df   = pd.read_csv("data/processed/test_processed.csv")
    SKIP_COLS = {'Satellite ID(Fake)', 'Current Official Name of Satellite',
                 'Country/Org of UN Registry', 'Country of Operator/Owner',
                 'Users', 'Class of Orbit', 'Type of Orbit', 'Date of Launch',
                 'Contractor', 'Country of Contractor', 'Launch Site',
                 'Launch Vehicle', 'COSPAR Number', 'NORAD Number',
                 'anomaly', 'RUL', 'days_since_launch', 'lifetime_days',
                 'launch_batch', 'Longitude of GEO (degrees)'}
    feat_cols = [c for c in train_df.columns if c not in SKIP_COLS]
    model     = train_rul_model(train_df, feat_cols)
    evaluate_rul(test_df, feat_cols, model)
