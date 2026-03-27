# ============================================================================
# Author: Ayush Pratik | Project: ISA - SpaceX Anomaly Detection System
# Institution: India Space Academy (ISW) | Status: Final Submission
# License: GNU GPL v3.0
# ============================================================================
"""
SpaceX Starlink Satellite Dataset Preprocessor
Converts orbital parameter registry -> ML-ready anomaly detection dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import joblib, os

# Column name constants (exact match to CSV headers)
SAT_ID_COL    = 'Satellite ID(Fake)'
SAT_NAME_COL  = 'Current Official Name of Satellite'
PERIGEE_COL   = 'Perigee (km)'
APOGEE_COL    = 'Apogee (km)'
ECC_COL       = 'Eccentricity'
INC_COL       = 'Inclination (degrees)'
PERIOD_COL    = 'Period (minutes)'
MASS_COL      = 'Launch Mass (kg.)'
LAUNCH_DATE   = 'Date of Launch'
LIFETIME_COL  = 'Expected Lifetime (yrs.)'
NORAD_COL     = 'NORAD Number'

REFERENCE_DATE = pd.Timestamp('2024-01-01')  # Simulate "today" for RUL

ORBITAL_FEATURES = [PERIGEE_COL, APOGEE_COL, ECC_COL, INC_COL, PERIOD_COL, MASS_COL]

def load_raw(path="data/raw/SpaceX_Satellite_Dataset.csv"):
    df = pd.read_csv(path)
    df[LAUNCH_DATE] = pd.to_datetime(df[LAUNCH_DATE])
    print(f"Loaded {len(df)} satellite records")
    print(f"  Date range: {df[LAUNCH_DATE].min().date()} to {df[LAUNCH_DATE].max().date()}")
    return df

def engineer_rul(df):
    """
    RUL (days) = Expected Lifetime - Days Since Launch.
    Expected Lifetime defaults to 4.0 years for satellites without stated value.
    """
    df[LIFETIME_COL] = df[LIFETIME_COL].fillna(4.0)
    df['days_since_launch'] = (REFERENCE_DATE - df[LAUNCH_DATE]).dt.days
    df['lifetime_days']     = df[LIFETIME_COL] * 365.25
    df['RUL']               = (df['lifetime_days'] - df['days_since_launch']).clip(lower=0)
    df['RUL']               = df['RUL'].round(1)
    return df

def engineer_orbital_features(df):
    """Derive additional physically meaningful orbital features."""
    df['mean_altitude_km']     = (df[PERIGEE_COL] + df[APOGEE_COL]) / 2.0
    df['altitude_spread_km']   = df[APOGEE_COL] - df[PERIGEE_COL]
    fleet_ecc_mean             = df[ECC_COL].mean()
    df['ecc_deviation']        = np.abs(df[ECC_COL] - fleet_ecc_mean)
    df['inc_deviation_deg']    = np.abs(df[INC_COL] - 53.0)
    fleet_period_mean          = df[PERIOD_COL].mean()
    df['period_deviation_min'] = np.abs(df[PERIOD_COL] - fleet_period_mean)
    df['launch_batch']         = df[LAUNCH_DATE].rank(method='dense').astype(int)
    df['days_since_launch_norm'] = df['days_since_launch'] / df['days_since_launch'].max()
    return df

def label_anomalies(df):
    """
    Z-score > 3.0 on orbital features = anomaly label.
    Also labels RUL-critical satellites (< 30 days remaining) as anomalous.
    This is physically meaningful: a Starlink satellite deviating 3 sigma
    from the fleet norm in Perigee/Eccentricity/Inclination is genuinely anomalous.
    """
    anomaly_features = [PERIGEE_COL, APOGEE_COL, ECC_COL, INC_COL,
                        PERIOD_COL, 'altitude_spread_km', 'ecc_deviation']
    z_scores        = np.abs(stats.zscore(df[anomaly_features]))
    orbital_anomaly = (z_scores > 3.0).any(axis=1).astype(int)
    rul_critical    = (df['RUL'] < 30).astype(int)
    df['anomaly']   = np.clip(orbital_anomaly + rul_critical, 0, 1)

    n = df['anomaly'].sum()
    print(f"  Anomaly labeling: {n} anomalous ({n/len(df)*100:.1f}%)")
    print(f"  - Orbital outliers: {orbital_anomaly.sum()}")
    print(f"  - RUL-critical (<30 days): {rul_critical.sum()}")
    return df

def get_all_features(df):
    base = ORBITAL_FEATURES
    eng  = ['mean_altitude_km', 'altitude_spread_km', 'ecc_deviation',
            'inc_deviation_deg', 'period_deviation_min', 'days_since_launch_norm']
    return base + eng

def normalize(df, feature_cols, scaler=None):
    os.makedirs("models", exist_ok=True)
    if scaler is None:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, "models/scaler.pkl")
        print("  Scaler fitted and saved.")
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler

def preprocess(path="data/raw/SpaceX_Satellite_Dataset.csv", is_train=True, scaler=None):
    df           = load_raw(path)
    df           = engineer_rul(df)
    df           = engineer_orbital_features(df)
    df           = label_anomalies(df)
    feature_cols = get_all_features(df)
    df, scaler   = normalize(df, feature_cols, scaler=scaler if not is_train else None)
    os.makedirs("data/processed", exist_ok=True)
    tag = "train" if is_train else "test"
    df.to_csv(f"data/processed/{tag}_processed.csv", index=False)
    print(f"[OK] {tag.upper()} data saved: {df.shape[0]} rows x {len(feature_cols)} features")
    return df, feature_cols, scaler

def prepare_and_split(path="data/raw/SpaceX_Satellite_Dataset.csv", test_size=0.25, seed=42):
    """
    Load full fleet, engineer features, label anomalies fleet-wide,
    then do a STRATIFIED random split to ensure both train and test
    have anomalous and nominal satellites.
    """
    from sklearn.model_selection import train_test_split as sk_split
    df           = load_raw(path)
    df           = engineer_rul(df)
    df           = engineer_orbital_features(df)
    df           = label_anomalies(df)
    train_df, test_df = sk_split(df, test_size=test_size, random_state=seed,
                                  stratify=df['anomaly'])
    os.makedirs("data/raw", exist_ok=True)
    train_df.to_csv("data/raw/train_raw.csv", index=False)
    test_df.to_csv("data/raw/test_raw.csv",   index=False)
    print(f"  Train: {len(train_df)} satellites ({int(train_df['anomaly'].sum())} anomalies) | "
          f"Test: {len(test_df)} satellites ({int(test_df['anomaly'].sum())} anomalies)")
    return train_df, test_df

def preprocess_split(path="data/raw/SpaceX_Satellite_Dataset.csv", is_train=True, scaler=None):
    """Preprocess a pre-labeled raw CSV (anomaly column already exists)."""
    df           = load_raw(path)
    df           = engineer_rul(df)
    df           = engineer_orbital_features(df)
    if 'anomaly' not in df.columns:
        df = label_anomalies(df)
    feature_cols = get_all_features(df)
    df, scaler   = normalize(df, feature_cols, scaler=scaler if not is_train else None)
    os.makedirs("data/processed", exist_ok=True)
    tag = "train" if is_train else "test"
    df.to_csv(f"data/processed/{tag}_processed.csv", index=False)
    n_anom = int(df['anomaly'].sum())
    print(f"[OK] {tag.upper()} data saved: {df.shape[0]} rows x {len(feature_cols)} features "
          f"({n_anom} anomalies, {n_anom/len(df)*100:.1f}%)")
    return df, feature_cols, scaler

if __name__ == "__main__":
    print("\n[STEP 1] PREPROCESSING SpaceX Starlink Dataset...")
    prepare_and_split()
    train_df, features, scaler = preprocess_split("data/raw/train_raw.csv", is_train=True)
    test_df,  _,       _      = preprocess_split("data/raw/test_raw.csv",   is_train=False, scaler=scaler)
    print(f"\nFeatures used ({len(features)}): {features}")

