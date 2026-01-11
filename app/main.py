# main.py
from __future__ import annotations

import sys, os
sys.path.insert(0, os.getcwd()+"/../")

import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from src.utils import parse_engine_cc, parse_value_and_rpm, _OWNER_MAP, _TRANSMISSION_MAP

import gradio as gr
from gradio.routes import mount_gradio_app
import requests

API_URL = "https://lexuanhoang1904-carprice.hf.space"
# =========================================================
# Config: artifact paths (produced by your Jupyter blocks)
# =========================================================
TE_ARTIFACTS_PATH = "app/te_artifacts.joblib"
SCALER_ARTIFACTS_PATH = "app/scaler_artifacts.joblib"
PROCESSED_COLS_PATH = "app/processed_columns.joblib"
MODEL_PATH = "src/models/car_price_model_randomf.pkl"
MODEL_VERSION = "rf-v1-log1p"

# If you don't have processed_columns.joblib yet, set this to your hard-coded 29 columns
# and the code will still work.
# IMPORTANT: order matters.
# FALLBACK_PROCESSED_COLUMNS: List[str] = [
#     "Year",
#     "Kilometer",
#     "Length",
#     "Width",
#     "Height",
#     "Seating Capacity",
#     "Fuel Tank Capacity",
#     "Owner_ord",
#     "Transmission_bin",
#     "Fuel Type_CNG + CNG",
#     "Fuel Type_Diesel",
#     "Fuel Type_Electric",
#     "Fuel Type_Hybrid",
#     "Fuel Type_LPG",
#     "Fuel Type_Petrol",
#     "Fuel Type_Petrol + CNG",
#     "Fuel Type_Petrol + LPG",
#     "Seller Type_Corporate",
#     "Seller Type_Individual",
#     "Drivetrain_FWD",
#     "Drivetrain_RWD",
#     "Model_te",
#     "Make_te",
#     "MaxPower_value",
#     "MaxPower_rpm",
#     "MaxTorque_value",
#     "MaxTorque_rpm",
#     "Engine_cc",
# ]

# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(title="Car Price Prediction API", version="0.2.0")


# =========================================================
# Request/Response schemas
# =========================================================
FuelType = Literal[
    "CNG + CNG",
    "Diesel",
    "Electric",
    "Hybrid",
    "LPG",
    "Petrol",
    "Petrol + CNG",
    "Petrol + LPG",
]

Transmission = Literal["Manual", "Automatic"]
SellerType = Literal["Corporate", "Individual"]
Drivetrain = Literal["FWD", "RWD"]

Owner = Literal[
    "UnRegistered Car",
    "First",
    "Second",
    "Third",
    "Fourth",
    "4 or More",
]


class RawCarFeatures(BaseModel):
    # Categorical (restricted so inference can't receive unseen categories)
    Make: str = Field(..., min_length=1, description="Car make (e.g., Honda)")
    Model: str = Field(..., min_length=1, description="Car model (e.g., Amaze 1.2 VX i-VTEC)")
    Fuel_Type: FuelType = Field(..., alias="Fuel Type")
    Transmission: Transmission
    Seller_Type: SellerType = Field(..., alias="Seller Type")
    Drivetrain: Drivetrain
    Owner: Owner

    # Numeric
    Year: int = Field(..., ge=1980, le=2035)
    Kilometer: float = Field(..., ge=0)
    Length: float = Field(..., ge=0)
    Width: float = Field(..., ge=0)
    Height: float = Field(..., ge=0)
    Seating_Capacity: float = Field(..., ge=1, le=20, alias="Seating Capacity")
    Fuel_Tank_Capacity: float = Field(..., ge=0, le=200, alias="Fuel Tank Capacity")

    # Messy text fields from dataset
    Engine: str = Field(..., description='e.g. "1198 cc"')
    Max_Power: str = Field(..., alias="Max Power", description='e.g. "87 bhp @ 6000 rpm"')
    Max_Torque: str = Field(..., alias="Max Torque", description='e.g. "109 Nm @ 4500 rpm"')

    class Config:
        populate_by_name = True


class PredictResponse(BaseModel):
    predicted_price: float
    model_version: str
    internal_features: Dict[str, Any]


# =========================================================
# Artifact loading
# =========================================================
@lru_cache(maxsize=1)
def load_te_artifacts() -> Dict[str, Any]:
    if not os.path.exists(TE_ARTIFACTS_PATH):
        raise RuntimeError(
            f"Missing TE artifacts at '{TE_ARTIFACTS_PATH}'. "
            "Create it in your notebook with Block A and copy it next to main.py."
        )
    return joblib.load(TE_ARTIFACTS_PATH)


@lru_cache(maxsize=1)
def load_scaler_artifacts() -> Dict[str, Any]:
    if not os.path.exists(SCALER_ARTIFACTS_PATH):
        raise RuntimeError(
            f"Missing scaler artifacts at '{SCALER_ARTIFACTS_PATH}'. "
            "Create it in your notebook with Block B and copy it next to main.py."
        )
    return joblib.load(SCALER_ARTIFACTS_PATH)


@lru_cache(maxsize=1)
def load_processed_columns() -> List[str]:
    if os.path.exists(PROCESSED_COLS_PATH):
        obj = joblib.load(PROCESSED_COLS_PATH)
        cols = obj.get("processed_columns")
        if not cols or not isinstance(cols, list):
            raise RuntimeError(f"processed_columns.joblib exists but is invalid: {obj}")
        return cols
    return FALLBACK_PROCESSED_COLUMNS

@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Missing model at '{MODEL_PATH}'. Train and save your RandomForestRegressor "
            f"to this path (joblib)."
        )
    return joblib.load(MODEL_PATH)

# =========================================================
# Parsing helpers (match your notebook logic)
# =========================================================
_num_re = r"(\d+(?:\.\d+)?)"


def _parse_value_and_rpm(s: str) -> Tuple[float, float]:
    """
    Examples:
      '87 bhp @ 6000 rpm' -> (87.0, 6000.0)
      '109 Nm @ 4500 rpm' -> (109.0, 4500.0)
      '74 bhp'            -> (74.0, nan)
    """
    return parse_value_and_rpm(s)



def _parse_engine_cc(s: str) -> float:
    """
    Examples:
      '1198 cc' -> 1198.0
      '1,198 cc' -> 1198.0
      '1498 CC' -> 1498.0
    """
    return parse_engine_cc(s)


# =========================================================
# Deterministic preprocessing
# =========================================================
# _OWNER_MAP = {
#     "UnRegistered Car": 0,
#     "First": 1,
#     "Second": 2,
#     "Third": 3,
#     "Fourth": 4,
#     "4 or More": 5,
# }

# _TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}


def preprocess_to_internal(x: RawCarFeatures) -> Dict[str, Any]:
    """
    Convert RawCarFeatures -> dict of processed features in *processed_columns* space.

    - No pandas get_dummies (we do explicit one-hots)
    - Make/Model use saved TE maps + global mean fallback
    - Scaling is not applied here (we apply later in to_scaled_vector)
    """
    processed_cols = load_processed_columns()
    te = load_te_artifacts()
    global_mean = float(te["global_mean_price"])
    make_map: Dict[str, float] = te["make_te_map"]
    model_map: Dict[str, float] = te["model_te_map"]

    feats: Dict[str, Any] = {}

    # ---- numeric passthrough ----
    feats["Year"] = float(x.Year)
    feats["Kilometer"] = float(x.Kilometer)
    feats["Length"] = float(x.Length)
    feats["Width"] = float(x.Width)
    feats["Height"] = float(x.Height)
    feats["Seating Capacity"] = float(x.Seating_Capacity)
    feats["Fuel Tank Capacity"] = float(x.Fuel_Tank_Capacity)

    # ---- ordinal/binary ----
    feats["Owner_ord"] = float(_OWNER_MAP[x.Owner])
    feats["Transmission_bin"] = float(_TRANSMISSION_MAP[x.Transmission])

    # ---- one-hots (must match your training dummy columns) ----
    # Fuel Type
    for ft in [
        "Fuel Type_CNG + CNG",
        "Fuel Type_Diesel",
        "Fuel Type_Electric",
        "Fuel Type_Hybrid",
        "Fuel Type_LPG",
        "Fuel Type_Petrol",
        "Fuel Type_Petrol + CNG",
        "Fuel Type_Petrol + LPG",
    ]:
        feats[ft] = 0.0
    feats[f"Fuel Type_{x.Fuel_Type}"] = 1.0

    # Seller Type
    for st in ["Seller Type_Corporate", "Seller Type_Individual"]:
        feats[st] = 0.0
    feats[f"Seller Type_{x.Seller_Type}"] = 1.0

    # Drivetrain
    for dt in ["Drivetrain_FWD", "Drivetrain_RWD"]:
        feats[dt] = 0.0
    feats[f"Drivetrain_{x.Drivetrain}"] = 1.0

    # ---- target encodings (deterministic lookup) ----
    feats["Make_te"] = float(make_map.get(x.Make, global_mean))
    feats["Model_te"] = float(model_map.get(x.Model, global_mean))

    # ---- parse max power/torque ----
    mp_val, mp_rpm = _parse_value_and_rpm(x.Max_Power)
    mt_val, mt_rpm = _parse_value_and_rpm(x.Max_Torque)

    feats["MaxPower_value"] = float(mp_val) if not math.isnan(mp_val) else math.nan
    feats["MaxPower_rpm"] = float(mp_rpm) if not math.isnan(mp_rpm) else math.nan
    feats["MaxTorque_value"] = float(mt_val) if not math.isnan(mt_val) else math.nan
    feats["MaxTorque_rpm"] = float(mt_rpm) if not math.isnan(mt_rpm) else math.nan

    # ---- parse engine ----
    feats["Engine_cc"] = float(_parse_engine_cc(x.Engine))

    # ---- ensure all processed columns exist; fill missing with NaN ----
    for c in processed_cols:
        if c not in feats:
            feats[c] = math.nan

    # ---- simple NaN handling (server-side) ----
    # You said UI will require all fields; NaNs can still appear if parsing fails.
    # Here we keep NaNs; later we can error out if any remain in numeric_to_scale.
    return feats


def to_scaled_vector(internal: Dict[str, Any]) -> np.ndarray:
    """
    Apply your saved StandardScaler to the numeric columns in the saved order.
    """
    processed_cols = load_processed_columns()
    art = load_scaler_artifacts()

    scaler = art["scaler"]
    num_cols: List[str] = list(art["num_cols"])  # order matters

    # Build full vector in processed_cols order
    x_full = np.array([float(internal[c]) for c in processed_cols], dtype=np.float32)

    # Extract numeric subvector in num_cols order
    # Ensure these exist and are finite (or your scaler will blow up)
    num_vals = []
    for c in num_cols:
        if c not in internal:
            raise HTTPException(status_code=400, detail=f"Missing required numeric column after preprocessing: '{c}'")
        v = float(internal[c])
        if not np.isfinite(v):
            raise HTTPException(
                status_code=400,
                detail=f"Preprocessing produced non-finite value for '{c}' (got {internal[c]}). "
                       f"Likely parsing failed (Engine/Max Power/Max Torque).",
            )
        num_vals.append(v)

    x_num = np.array(num_vals, dtype=np.float32).reshape(1, -1)
    x_num_scaled = scaler.transform(x_num).astype(np.float32).reshape(-1)

    # Put scaled values back into full vector
    idx = [processed_cols.index(c) for c in num_cols]
    x_full[idx] = x_num_scaled

    return x_full


# =========================================================
# Dummy model (Step 2: still fake prediction)
# =========================================================
def dummy_model_from_scaled(x_vec: np.ndarray) -> float:
    """
    Deterministic pseudo prediction using scaled features.
    Replace with your real model later.
    """
    # A stable deterministic function (no randomness)
    # The exact values don't matter for step 2.
    s = float(np.sum(x_vec))
    pred = 500000.0 + 25000.0 * math.tanh(s / 10.0)
    return float(pred)


# =========================================================
# Routes
# =========================================================
@app.get("/health")
def health():
    # Also validates artifacts exist/loadable
    _ = load_processed_columns()
    _ = load_te_artifacts()
    _ = load_scaler_artifacts()
    return {"status": "ok", "model_version": "dummy-v0", "artifacts": "loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(features: RawCarFeatures, debug: bool = Query(False, description="If true, return internal features")):
    internal = preprocess_to_internal(features)
    x_vec = to_scaled_vector(internal)
    model = load_model()

    # sklearn expects 2D: (batch, features)
    x_in = x_vec.reshape(1, -1)

    # Your training target is log1p(price), so prediction is in log space
    pred_log = float(model.predict(x_in)[0])

    # Convert back to price space: price = exp(log1p(price)) - 1
    pred_price = float(np.expm1(pred_log))

    return PredictResponse(
        predicted_price=pred_price,
        model_version=MODEL_VERSION,
        internal_features=(
            {**internal, "_pred_log1p": pred_log} if debug else {}
        ),
    )

def send_request(
    Make: str,
    Model: str,
    Fuel_Type: str,
    Transmission: str,
    Seller_Type: str,
    Drivetrain: str,
    Owner: str,
    Year: int,
    Kilometer: float,
    Length: float,
    Width: float,
    Height: float,
    Seating_Capacity: float,
    Fuel_Tank_Capacity: float,
    Engine: str,
    Max_Power: str,
    Max_Torque: str,
):
    payload = {
        "Make": Make,
        "Model": Model,
        "Fuel Type": Fuel_Type,
        "Transmission": Transmission,
        "Seller Type": Seller_Type,
        "Drivetrain": Drivetrain,
        "Owner": Owner,
        "Year": int(Year),
        "Kilometer": float(Kilometer),
        "Length": float(Length),
        "Width": float(Width),
        "Height": float(Height),
        "Seating Capacity": float(Seating_Capacity),
        "Fuel Tank Capacity": float(Fuel_Tank_Capacity),
        "Engine": Engine,
        "Max Power": Max_Power,
        "Max Torque": Max_Torque,
    }

    r = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()
    return float(data["predicted_price"])


def build_demo():
    te = load_te_artifacts()
    make_map = te["make_te_map"]
    model_map = te["model_te_map"]
    
    with gr.Blocks() as demo:
        gr.Markdown("# Car Price Predictor")

        with gr.Row():
            Make = gr.Dropdown(
                choices=sorted(make_map.keys()),
                label="Make",
                value=list(make_map.keys())[0],
            )

            Model = gr.Dropdown(
                choices=sorted(model_map.keys()),
                label="Model",
                value=list(model_map.keys())[0],
            )


        with gr.Row():
            Fuel_Type = gr.Dropdown(
                choices=["CNG + CNG","Diesel","Electric","Hybrid","LPG","Petrol","Petrol + CNG","Petrol + LPG"],
                label="Fuel Type",
                value="Petrol",
            )
            Transmission = gr.Dropdown(choices=["Manual","Automatic"], label="Transmission", value="Manual")
            Seller_Type = gr.Dropdown(choices=["Corporate","Individual"], label="Seller Type", value="Individual")
            Drivetrain = gr.Dropdown(choices=["FWD","RWD"], label="Drivetrain", value="FWD")
            Owner = gr.Dropdown(
                choices=["UnRegistered Car","First","Second","Third","Fourth","4 or More"],
                label="Owner",
                value="First",
            )

        with gr.Row():
            Year = gr.Number(label="Year", value=2017, precision=0)
            Kilometer = gr.Number(label="Kilometer", value=87150)
            Seating_Capacity = gr.Number(label="Seating Capacity", value=5)
            Fuel_Tank_Capacity = gr.Number(label="Fuel Tank Capacity", value=35)

        with gr.Row():
            Length = gr.Number(label="Length", value=3990)
            Width = gr.Number(label="Width", value=1680)
            Height = gr.Number(label="Height", value=1505)

        with gr.Row():
            Engine = gr.Textbox(label='Engine (e.g., "1198 cc")', value="1198 cc")
            Max_Power = gr.Textbox(label='Max Power (e.g., "87 bhp @ 6000 rpm")', value="87 bhp @ 6000 rpm")
            Max_Torque = gr.Textbox(label='Max Torque (e.g., "109 Nm @ 4500 rpm")', value="109 Nm @ 4500 rpm")

        btn = gr.Button("Predict")
        out = gr.Number(label="Predicted Price")

        btn.click(
            fn=send_request,
            inputs=[
                Make, Model, Fuel_Type, Transmission, Seller_Type, Drivetrain, Owner,
                Year, Kilometer, Length, Width, Height, Seating_Capacity, Fuel_Tank_Capacity,
                Engine, Max_Power, Max_Torque
            ],
            outputs=[out],
        )

    return demo

demo = build_demo()
app = mount_gradio_app(app, demo, path="/")
