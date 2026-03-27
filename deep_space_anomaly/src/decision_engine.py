# ============================================================================
# Author: Ayush Pratik | Project: ISA - SpaceX Anomaly Detection System
# Institution: India Space Academy (ISW) | Status: Final Submission
# License: GNU GPL v3.0
# ============================================================================
"""
Autonomous Decision Support System for Starlink Deep Space Operations.
No Earth contact assumed for emergency decisions (simulates comm delay).
"""

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

# Risk Levels
NOMINAL   = "NOMINAL"
CAUTION   = "CAUTION"
WARNING   = "WARNING"
CRITICAL  = "CRITICAL"
EMERGENCY = "EMERGENCY"

# Corrective Actions (Starlink-specific)
ACT_NONE              = "NO_ACTION_REQUIRED"
ACT_MONITOR           = "INCREASE_TELEMETRY_POLLING"
ACT_ORBIT_ADJUST      = "INITIATE_ORBIT_CORRECTION_BURN"
ACT_DEORBIT_PREP      = "PREPARE_CONTROLLED_DEORBIT_SEQUENCE"
ACT_EMERGENCY_DEORBIT = "EXECUTE_EMERGENCY_DEORBIT_BURN"
ACT_ALERT_GROUND      = "TRANSMIT_PRIORITY_ALERT_TO_SPACEX_GROUND_STATION"

@dataclass
class SatelliteState:
    sat_id:             int
    sat_name:           str
    anomaly_detected:   bool
    anomaly_score:      float
    rul_days:           float
    perigee_km:         float
    eccentricity:       float
    altitude_spread_km: float
    risk_score:         float = 0.0

@dataclass
class Decision:
    risk_level:   str
    action:       str
    confidence:   float
    reason:       str
    alert_ground: bool

def compute_risk_score(state: SatelliteState) -> float:
    """
    Composite risk score weights:
      0.35 - anomaly detected (binary)
      0.25 - anomaly severity score
      0.25 - RUL criticality (0 at >=365d, 1.0 at 0d)
      0.10 - orbital instability (altitude spread)
      0.05 - low perigee alarm (<400 km = rapid decay)
    """
    score  = 0.0
    if state.anomaly_detected:
        score += 0.35
    score += 0.25 * min(state.anomaly_score, 1.0)
    rul_norm    = max(0.0, min(1.0, 1.0 - (state.rul_days / 365.0)))
    score      += 0.25 * rul_norm
    spread_norm = min(state.altitude_spread_km / 50.0, 1.0)
    score      += 0.10 * spread_norm
    if state.perigee_km < 400:
        score  += 0.05
    return round(min(score, 1.0), 4)

def determine_risk_level(risk_score: float) -> str:
    if risk_score < 0.15: return NOMINAL
    if risk_score < 0.35: return CAUTION
    if risk_score < 0.55: return WARNING
    if risk_score < 0.75: return CRITICAL
    return EMERGENCY

def decide_action(state: SatelliteState) -> Decision:
    state.risk_score = compute_risk_score(state)
    level            = determine_risk_level(state.risk_score)

    if state.risk_score >= 0.75 or (state.anomaly_detected and state.rul_days < 7):
        return Decision(EMERGENCY, ACT_EMERGENCY_DEORBIT, 0.97,
            f"EMERGENCY: {state.sat_name} RUL={state.rul_days:.0f}d score={state.risk_score:.3f}. "
            f"Uncontrolled reentry risk.", True)

    if state.risk_score >= 0.55 or state.rul_days < 30:
        return Decision(CRITICAL, ACT_DEORBIT_PREP, 0.92,
            f"CRITICAL: {state.sat_name} preparing controlled deorbit. "
            f"RUL={state.rul_days:.0f}d perigee={state.perigee_km:.0f}km.", True)

    if state.risk_score >= 0.35 or (state.anomaly_detected and state.altitude_spread_km > 10):
        return Decision(WARNING, ACT_ORBIT_ADJUST, 0.86,
            f"WARNING: {state.sat_name} orbit correction burn needed. "
            f"Alt_spread={state.altitude_spread_km:.1f}km ecc={state.eccentricity:.6f}.", False)

    if state.risk_score >= 0.15 or state.anomaly_detected:
        return Decision(CAUTION, ACT_MONITOR, 0.79,
            f"CAUTION: {state.sat_name} minor orbital deviation. "
            f"score={state.risk_score:.3f}", False)

    return Decision(NOMINAL, ACT_NONE, 0.99,
        f"{state.sat_name} all orbital parameters nominal.", False)

def process_satellite_fleet(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        state = SatelliteState(
            sat_id            = int(row.get('Satellite ID(Fake)', 0)),
            sat_name          = str(row.get('Current Official Name of Satellite', 'Unknown')),
            anomaly_detected  = bool(row.get('pred_ensemble', row.get('anomaly', 0))),
            anomaly_score     = float(row.get('anomaly_score', 0.0)),
            rul_days          = float(row.get('RUL', 365)),
            perigee_km        = float(row.get('Perigee (km)', 548)),
            eccentricity      = float(row.get('Eccentricity', 0.000145)),
            altitude_spread_km= float(row.get('altitude_spread_km', 2))
        )
        state.risk_score = compute_risk_score(state)
        decision         = decide_action(state)
        results.append({
            'Satellite':    state.sat_name,
            'RUL (days)':  state.rul_days,
            'Risk Score':  state.risk_score,
            'Risk Level':  decision.risk_level,
            'Action':      decision.action.replace('_', ' '),
            'Alert Ground':'YES' if decision.alert_ground else 'No',
            'Confidence':  decision.confidence,
            'Reason':      decision.reason
        })
    result_df = pd.DataFrame(results)
    result_df.to_csv("data/processed/decision_log.csv", index=False)
    return result_df

if __name__ == "__main__":
    print("\n[STEP 4] TESTING AUTONOMOUS DECISION ENGINE - SPACEX STARLINK FLEET\n")
    test_fleet = [
        SatelliteState(1,   "Starlink-1007", False, 0.05, 280.0, 549.0, 0.000145, 2.0),
        SatelliteState(100, "Starlink-1100", True,  0.45,  95.0, 541.0, 0.000380, 8.5),
        SatelliteState(300, "Starlink-1300", True,  0.72,  22.0, 532.0, 0.000620, 18.0),
        SatelliteState(500, "Starlink-1500", True,  0.92,   3.0, 301.0, 0.001200, 38.0),
    ]
    print(f"{'Satellite':<18} {'RUL':>6} {'Risk':>6} {'Level':<12} {'Action'}")
    print("-" * 75)
    for state in test_fleet:
        state.risk_score = compute_risk_score(state)
        d = decide_action(state)
        print(f"{state.sat_name:<18} {state.rul_days:>5.0f}d {state.risk_score:>6.3f} "
              f"{d.risk_level:<12} {d.action.replace('_',' ')[:35]}")
    try:
        pred_df   = pd.read_csv("data/processed/test_predictions.csv")
        decisions = process_satellite_fleet(pred_df)
        print(f"\n[OK] Processed {len(decisions)} satellites")
        print(decisions['Risk Level'].value_counts())
    except FileNotFoundError:
        print("\n(Run anomaly_detection.py first to generate test_predictions.csv)")
