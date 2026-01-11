from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
import hashlib

app = FastAPI(title="Car Price API", version="0.1.0")

# -------------------------
# Option A: Public contract
# -------------------------
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
Owner = Literal["First", "Second", "Third", "UnRegistered Car"]
SellerType = Literal["Corporate", "Individual"]
Drivetrain = Literal["FWD", "RWD"]  # extend later if your data has AWD


class RawCarFeatures(BaseModel):
    make: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=128)

    year: int = Field(..., ge=1980, le=2030)
    kilometer: int = Field(..., ge=0, le=1_000_000)

    fuel_type: FuelType
    transmission: Transmission
    owner: Owner
    seller_type: SellerType
    drivetrain: Drivetrain

    # Optional numeric features (UI can provide them; if missing, you’ll impute later in preprocessing step 2)
    engine_cc: Optional[int] = Field(None, ge=500, le=10000)

    max_power_value: Optional[float] = Field(None, ge=0, le=2000)  # bhp
    max_power_rpm: Optional[int] = Field(None, ge=0, le=20000)

    max_torque_value: Optional[float] = Field(None, ge=0, le=5000)  # Nm
    max_torque_rpm: Optional[int] = Field(None, ge=0, le=20000)

    length: Optional[float] = Field(None, ge=2000, le=6000)   # mm
    width: Optional[float] = Field(None, ge=1200, le=3000)    # mm
    height: Optional[float] = Field(None, ge=1000, le=3000)   # mm

    seating_capacity: Optional[int] = Field(None, ge=2, le=10)
    fuel_tank_capacity: Optional[float] = Field(None, ge=10, le=200)


class PredictResponse(BaseModel):
    predicted_price: int
    model_version: str
    features: RawCarFeatures
    internal_features: Dict[str, Any]  # helpful for debugging step 1/2; remove later


# -------------------------
# Internal feature mapping
# (matches your processed columns)
# -------------------------
OWNER_ORD = {"First": 0, "Second": 1, "Third": 2, "UnRegistered Car": 3}
TRANSMISSION_BIN = {"Manual": 0, "Automatic": 1}


PROCESSED_COLUMNS = [
    # numeric
    "Year", "Kilometer", "Length", "Width", "Height",
    "Seating Capacity", "Fuel Tank Capacity",
    "Owner_ord", "Transmission_bin",

    # one-hot fuel type
    "Fuel Type_CNG + CNG",
    "Fuel Type_Diesel",
    "Fuel Type_Electric",
    "Fuel Type_Hybrid",
    "Fuel Type_LPG",
    "Fuel Type_Petrol",
    "Fuel Type_Petrol + CNG",
    "Fuel Type_Petrol + LPG",

    # one-hot seller type
    "Seller Type_Corporate",
    "Seller Type_Individual",

    # one-hot drivetrain
    "Drivetrain_FWD",
    "Drivetrain_RWD",

    # target-encoded
    "Model_te",
    "Make_te",

    # parsed power/torque
    "MaxPower_value", "MaxPower_rpm",
    "MaxTorque_value", "MaxTorque_rpm",

    # engine cc
    "Engine_cc",
]


def stable_te_placeholder(text: str, lo: float = -1.0, hi: float = 1.0) -> float:
    """
    Step 1 placeholder for target encoding (Make_te, Model_te).
    Deterministic mapping: string -> float in [lo, hi].
    In Step 2 you will replace with real lookup tables learned from training data.
    """
    h = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF  # [0,1]
    return lo + (hi - lo) * x


def preprocess_to_internal(x: RawCarFeatures) -> Dict[str, Any]:
    # start with all-zero vector
    feats: Dict[str, Any] = {c: 0.0 for c in PROCESSED_COLUMNS}

    # numeric passthrough (None -> 0 for step 1; in step 2 you’ll impute properly)
    feats["Year"] = float(x.year)
    feats["Kilometer"] = float(x.kilometer)

    feats["Length"] = float(x.length) if x.length is not None else 0.0
    feats["Width"] = float(x.width) if x.width is not None else 0.0
    feats["Height"] = float(x.height) if x.height is not None else 0.0

    feats["Seating Capacity"] = float(x.seating_capacity) if x.seating_capacity is not None else 0.0
    feats["Fuel Tank Capacity"] = float(x.fuel_tank_capacity) if x.fuel_tank_capacity is not None else 0.0

    feats["Owner_ord"] = float(OWNER_ORD[x.owner])
    feats["Transmission_bin"] = float(TRANSMISSION_BIN[x.transmission])

    # fuel one-hot
    feats[f"Fuel Type_{x.fuel_type}"] = 1.0

    # seller one-hot
    feats[f"Seller Type_{x.seller_type}"] = 1.0

    # drivetrain one-hot
    feats[f"Drivetrain_{x.drivetrain}"] = 1.0

    # target-encoding placeholders (deterministic)
    feats["Make_te"] = stable_te_placeholder(x.make)
    feats["Model_te"] = stable_te_placeholder(x.model)

    # power/torque
    feats["MaxPower_value"] = float(x.max_power_value) if x.max_power_value is not None else 0.0
    feats["MaxPower_rpm"] = float(x.max_power_rpm) if x.max_power_rpm is not None else 0.0
    feats["MaxTorque_value"] = float(x.max_torque_value) if x.max_torque_value is not None else 0.0
    feats["MaxTorque_rpm"] = float(x.max_torque_rpm) if x.max_torque_rpm is not None else 0.0

    # engine cc
    feats["Engine_cc"] = float(x.engine_cc) if x.engine_cc is not None else 0.0

    return feats


# -------------------------
# Dummy model (Step 1 only)
# -------------------------
def dummy_model(internal_feats: Dict[str, Any]) -> int:
    # deterministic pseudo-price from internal features
    base = 300_000
    base += (internal_feats["Year"] - 2010) * 12_000
    base -= int(internal_feats["Kilometer"] * 0.8)

    # give some bump based on TE placeholders so make/model affect output deterministically
    base += int(internal_feats["Make_te"] * 20_000)
    base += int(internal_feats["Model_te"] * 30_000)

    # clamp
    return max(50_000, min(int(base), 5_000_000))


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(features: RawCarFeatures):
    internal = preprocess_to_internal(features)
    pred = dummy_model(internal)
    return {
        "predicted_price": pred,
        "model_version": "dummy-v0",
        "features": features,
        "internal_features": internal,  # remove later if you don’t want to expose this
    }
