"""
health_alert_engine.py
======================
Backend-ready real-time alert engine for the Healthcare Risk Prediction Pipeline.

Design
------
Stage 1 (offline) : XGBoost multiclass classifier trained on single tabular
                    vital-sign readings. Static, memoryless.
Stage 2 (this file): Runtime logic layered on top of the static model:
                      * per-patient rolling buffer
                      * sudden-change detection between consecutive readings
                      * repeated-prediction detection over the buffer
                      * combined alert escalation

Alert-level invariant (enforced by decide_final_alert)
------------------------------------------------------
  - Current Critical model prediction -> final alert is ALWAYS 'critical'.
  - Current Abnormal model prediction -> final alert is AT LEAST 'warning'.
  - Time logic may escalate alerts, never downgrade the prediction baseline.

Typical backend usage
---------------------
    from health_alert_engine import load_artifacts, predict_risk_and_alert

    load_artifacts(artifacts_dir="artifacts")   # once at startup

    resp = predict_risk_and_alert({
        "patient_id":  "P001",
        "timestamp":   "2026-04-24T10:30:00",
        "heart_rate":  115,
        "spo2":         92,
        "temperature":  38.1,
    })
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

ALERT_CONFIG = {
    "buffer_size":                10,
    "min_readings_for_time_logic": 3,
    "abnormal_count_warning":      3,
    "critical_count_alert":        2,
    "spo2_sudden_drop":            3.0,
    "hr_sudden_change":           25.0,
    "temp_sudden_increase":        0.8,
}

REQUIRED_INPUT_FIELDS = ["patient_id", "timestamp", "heart_rate", "spo2", "temperature"]


# --------------------------------------------------------------------------- #
# Module-level state                                                          #
# --------------------------------------------------------------------------- #

_ARTIFACTS: dict = {}
_PATIENT_BUFFERS: dict = defaultdict(lambda: deque(maxlen=ALERT_CONFIG["buffer_size"]))


# --------------------------------------------------------------------------- #
# Artifact loading                                                            #
# --------------------------------------------------------------------------- #

def load_artifacts(artifacts_dir: str = "artifacts") -> dict:
    base          = Path(artifacts_dir)
    model_path    = base / "model"         / "xgb_risk_model.joblib"
    features_path = base / "preprocessing" / "features.json"
    labels_path   = base / "preprocessing" / "label_mapping.json"

    for p in (model_path, features_path, labels_path):
        if not p.exists():
            raise FileNotFoundError(f"Artifact missing: {p}")

    _ARTIFACTS["model"]         = joblib.load(model_path)
    _ARTIFACTS["features_meta"] = json.loads(features_path.read_text())
    _ARTIFACTS["feature_names"] = _ARTIFACTS["features_meta"]["model_features"]
    _ARTIFACTS["label_map"]     = {
        int(k): v for k, v in json.loads(labels_path.read_text()).items()
    }
    return _ARTIFACTS


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #

def validate_reading(reading_dict: dict) -> Tuple[bool, Optional[str]]:
    if not isinstance(reading_dict, dict):
        return False, "reading must be a dictionary"
    missing = [f for f in REQUIRED_INPUT_FIELDS if f not in reading_dict]
    if missing:
        return False, f"missing required fields: {missing}"
    for field in ("heart_rate", "spo2", "temperature"):
        val = reading_dict[field]
        if val is None:
            return False, f"field '{field}' is null"
        try:
            float(val)
        except (TypeError, ValueError):
            return False, f"field '{field}' must be numeric, got {val!r}"
    if not str(reading_dict["patient_id"]).strip():
        return False, "patient_id is empty"
    return True, None


# --------------------------------------------------------------------------- #
# Preprocessing                                                               #
# --------------------------------------------------------------------------- #

def preprocess_single_reading(reading_dict: dict,
                              feature_names: Optional[list] = None) -> pd.DataFrame:
    if feature_names is None:
        feature_names = _ARTIFACTS["feature_names"]

    ts = reading_dict.get("timestamp")
    try:
        hour = pd.to_datetime(ts).hour if ts is not None else 12
    except Exception:
        hour = 12
    is_night = int(hour >= 22 or hour <= 5)

    row = {
        "heart_rate":  float(reading_dict["heart_rate"]),
        "spo2":        float(reading_dict["spo2"]),
        "temperature": float(reading_dict["temperature"]),
        "hour_of_day": int(hour),
        "is_night":    is_night,
    }
    return pd.DataFrame([row], columns=feature_names)


def predict_single_reading(reading_dict: dict) -> dict:
    X          = preprocess_single_reading(reading_dict)
    probs      = _ARTIFACTS["model"].predict_proba(X)[0]
    pred_class = int(np.argmax(probs))
    label_map  = _ARTIFACTS["label_map"]
    return {
        "predicted_class":     pred_class,
        "predicted_label":     label_map.get(pred_class, str(pred_class)),
        "class_probabilities": {
            label_map.get(i, str(i)): float(round(p, 4)) for i, p in enumerate(probs)
        },
    }


# --------------------------------------------------------------------------- #
# Rolling buffer                                                              #
# --------------------------------------------------------------------------- #

def update_patient_buffer(patient_id: str, reading_record: dict) -> None:
    _PATIENT_BUFFERS[patient_id].append(reading_record)


def get_patient_history(patient_id: str) -> list:
    return list(_PATIENT_BUFFERS.get(patient_id, []))


def reset_patient_buffer(patient_id: str) -> None:
    _PATIENT_BUFFERS.pop(patient_id, None)


# --------------------------------------------------------------------------- #
# Time-logic detectors                                                        #
# --------------------------------------------------------------------------- #

def detect_sudden_change(patient_history: list,
                         config: Optional[dict] = None) -> dict:
    cfg = config or ALERT_CONFIG
    result = {"detected": False, "spo2_drop": 0.0, "temp_rise": 0.0,
              "hr_delta": 0.0, "reasons": []}
    if len(patient_history) < 2:
        return result

    current  = patient_history[-1]
    previous = patient_history[-2]

    spo2_drop = float(previous["spo2"])       - float(current["spo2"])
    temp_rise = float(current["temperature"]) - float(previous["temperature"])
    hr_delta  = abs(float(current["heart_rate"]) - float(previous["heart_rate"]))

    result["spo2_drop"] = round(spo2_drop, 2)
    result["temp_rise"] = round(temp_rise, 2)
    result["hr_delta"]  = round(hr_delta,  2)

    if spo2_drop >= cfg["spo2_sudden_drop"]:
        result["detected"] = True
        result["reasons"].append(f"SpO2 dropped {spo2_drop:.1f}% vs previous reading")
    if temp_rise >= cfg["temp_sudden_increase"]:
        result["detected"] = True
        result["reasons"].append(f"Temperature rose {temp_rise:.1f}C vs previous reading")
    if hr_delta >= cfg["hr_sudden_change"]:
        result["detected"] = True
        result["reasons"].append(f"Heart rate changed by {hr_delta:.0f} bpm vs previous reading")
    return result


def evaluate_repeated_predictions(patient_history: list,
                                  config: Optional[dict] = None) -> dict:
    cfg = config or ALERT_CONFIG
    result = {"detected": False, "abnormal_count": 0, "critical_count": 0,
              "window_size": len(patient_history), "reasons": []}
    if len(patient_history) < cfg["min_readings_for_time_logic"]:
        return result

    preds = [r.get("predicted_class") for r in patient_history
             if r.get("predicted_class") is not None]
    n_abn  = sum(1 for p in preds if p == 1)
    n_crit = sum(1 for p in preds if p == 2)
    result["abnormal_count"] = n_abn
    result["critical_count"] = n_crit

    if n_abn >= cfg["abnormal_count_warning"]:
        result["detected"] = True
        result["reasons"].append(f"{n_abn} Abnormal predictions in last {len(preds)} readings")
    if n_crit >= cfg["critical_count_alert"]:
        result["detected"] = True
        result["reasons"].append(f"{n_crit} Critical predictions in last {len(preds)} readings")
    return result


# --------------------------------------------------------------------------- #
# Alert decision                                                              #
# --------------------------------------------------------------------------- #

_LEVEL_ORDER = {"normal": 0, "warning": 1, "critical": 2}

def _escalate(current: str, proposed: str) -> str:
    return proposed if _LEVEL_ORDER[proposed] > _LEVEL_ORDER[current] else current


_ALERT_MESSAGES = {
    "normal":   "No urgent action. Patient vitals appear stable.",
    "warning":  "Abnormal pattern detected. Monitor patient closely.",
    "critical": "Urgent abnormal condition detected. Send immediate alert.",
}

_RECOMMENDED_ACTIONS = {
    "normal":   "Continue routine monitoring.",
    "warning":  "Notify attending nurse; increase monitoring frequency.",
    "critical": "Dispatch medical team immediately; escalate to physician.",
}

# Baseline alert level produced directly by the current model prediction.
# The final alert level is ALWAYS >= this baseline (enforced below).
_BASELINE_FROM_PREDICTION = {0: "normal", 1: "warning", 2: "critical"}


def decide_final_alert(single_prediction_result: dict,
                       sudden_change_result: dict,
                       repeated_prediction_result: dict) -> dict:
    """
    Combine Stage 1 + time logic into a single alert level.

    SPEC-ENFORCED INVARIANT:
      - Critical prediction (class=2) -> alert_level is ALWAYS 'critical'.
      - Abnormal prediction (class=1) -> alert_level is AT LEAST 'warning'.
      - Time logic may raise the alert, never lower the prediction baseline.
    """
    pred_class = single_prediction_result["predicted_class"]
    reasons    = []

    # --- baseline alert level from the static model ---------------------------
    baseline_level = _BASELINE_FROM_PREDICTION.get(pred_class, "normal")
    level = baseline_level

    if pred_class == 2:
        reasons.append("model predicted Critical on current reading")
    elif pred_class == 1:
        reasons.append("model predicted Abnormal on current reading")

    is_abn_or_crit = pred_class in (1, 2)

    # --- repeated-prediction escalation ---------------------------------------
    if repeated_prediction_result["detected"]:
        if repeated_prediction_result["critical_count"] >= ALERT_CONFIG["critical_count_alert"]:
            level = _escalate(level, "critical")
        elif repeated_prediction_result["abnormal_count"] >= ALERT_CONFIG["abnormal_count_warning"]:
            level = _escalate(level, "warning")
        reasons.extend(repeated_prediction_result["reasons"])

    # --- sudden-change escalation ---------------------------------------------
    if sudden_change_result["detected"]:
        if sudden_change_result["spo2_drop"] >= ALERT_CONFIG["spo2_sudden_drop"]:
            level = _escalate(level, "critical")       # desaturation always critical
        if sudden_change_result["hr_delta"] >= ALERT_CONFIG["hr_sudden_change"]:
            level = _escalate(level, "critical" if is_abn_or_crit else "warning")
        if sudden_change_result["temp_rise"] >= ALERT_CONFIG["temp_sudden_increase"]:
            level = _escalate(level, "critical" if is_abn_or_crit else "warning")
        reasons.extend(sudden_change_result["reasons"])

    # --- co-occurrence escalation ---------------------------------------------
    if repeated_prediction_result["detected"] and sudden_change_result["detected"]:
        level = _escalate(level, "critical")
        reasons.append("repeated abnormality combined with sudden vital change")

    # --- SAFETY CLAMP (defensive): time logic must never lower the baseline ---
    level = _escalate(baseline_level, level)

    return {
        "alert_level":        level,
        "alert_message":      _ALERT_MESSAGES[level],
        "recommended_action": _RECOMMENDED_ACTIONS[level],
        "reasons":            reasons,
    }


# --------------------------------------------------------------------------- #
# Main backend API                                                            #
# --------------------------------------------------------------------------- #

def predict_risk_and_alert(reading_dict: dict) -> dict:
    is_valid, err = validate_reading(reading_dict)
    if not is_valid:
        return {
            "error":          err,
            "input_reading":  reading_dict,
            "final_alert": {
                "alert_level":        "normal",
                "alert_message":      "Invalid input; alerting skipped.",
                "recommended_action": "Fix input payload and retry.",
            },
        }

    if "model" not in _ARTIFACTS:
        raise RuntimeError(
            "Artifacts not loaded. Call load_artifacts(artifacts_dir=...) at startup."
        )

    patient_id = str(reading_dict["patient_id"])
    timestamp  = reading_dict.get("timestamp")

    try:
        prediction = predict_single_reading(reading_dict)
    except Exception as e:
        return {"error": f"model prediction failed: {e}", "input_reading": reading_dict}

    reading_record = {
        "timestamp":           timestamp,
        "heart_rate":          float(reading_dict["heart_rate"]),
        "spo2":                float(reading_dict["spo2"]),
        "temperature":         float(reading_dict["temperature"]),
        "predicted_class":     prediction["predicted_class"],
        "predicted_label":     prediction["predicted_label"],
        "class_probabilities": prediction["class_probabilities"],
    }
    update_patient_buffer(patient_id, reading_record)
    history = get_patient_history(patient_id)

    sudden_change = detect_sudden_change(history, ALERT_CONFIG)
    repeated_pred = evaluate_repeated_predictions(history, ALERT_CONFIG)
    final_alert   = decide_final_alert(prediction, sudden_change, repeated_pred)

    return {
        "patient_id": patient_id,
        "timestamp":  timestamp,
        "input_reading": {
            "heart_rate":  reading_record["heart_rate"],
            "spo2":        reading_record["spo2"],
            "temperature": reading_record["temperature"],
        },
        "model_prediction": {
            "predicted_class":     prediction["predicted_class"],
            "predicted_label":     prediction["predicted_label"],
            "class_probabilities": prediction["class_probabilities"],
        },
        "time_logic": {
            "repeated_prediction_detected": repeated_pred["detected"],
            "sudden_change_detected":       sudden_change["detected"],
            "buffer_size":                  len(history),
            "abnormal_count":               repeated_pred["abnormal_count"],
            "critical_count":               repeated_pred["critical_count"],
            "spo2_drop":                    sudden_change["spo2_drop"],
            "temp_rise":                    sudden_change["temp_rise"],
            "hr_delta":                     sudden_change["hr_delta"],
            "reasons":                      (repeated_pred["reasons"]
                                             + sudden_change["reasons"]),
        },
        "final_alert": final_alert,
    }
