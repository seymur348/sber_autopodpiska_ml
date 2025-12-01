from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import dill

app = FastAPI()


# ===========================
# LOAD MODEL & FREQUENCY MAPS
# ===========================
with open("../models/model.pkl", "rb") as f:
    model = dill.load(f)

with open("../models/freq_maps.pkl", "rb") as f:
    freq_maps = dill.load(f)


# ===========================
# COLUMN LISTS (must match training!)
# ===========================
onehot_cols = [
    "device_category",
    "utm_medium",
    "geo_country"
]

freq_cols = [
    "utm_source",
    "utm_campaign",
    "utm_adcontent",
    "device_browser",
    "geo_city"
]

numeric_cols = [
    "visit_hour",
    "visit_weekday",
    "visit_month",
    "is_weekend",
    "hits",
    "pageviews",
    "events",
    "screen_w",
    "screen_h"
]

ALL_COLS = onehot_cols + freq_cols + numeric_cols


# ===========================
# FastAPI INPUT SCHEMA
# ===========================
class Input(BaseModel):
    device_category: str
    utm_medium: str
    geo_country: str

    utm_source: str
    utm_campaign: str
    utm_adcontent: str
    device_browser: str
    geo_city: str

    visit_hour: int
    visit_weekday: int
    visit_month: int
    is_weekend: int
    hits: float
    pageviews: float
    events: float
    screen_w: int
    screen_h: int


# ===========================
# Frequency Encoding
# ===========================
def apply_freq_encoding(df):
    for col in freq_cols:
        mapping = freq_maps[col]
        df[col] = df[col].map(mapping).fillna(0.0)
    return df


# ===========================
# Prepare Input Function
# ===========================
def prepare_input(data: Input):
    df = pd.DataFrame([data.dict()])[ALL_COLS]

    # Convert one-hot columns to string
    for col in onehot_cols:
        df[col] = df[col].astype(str)

    # Apply frequency encoding
    df = apply_freq_encoding(df)

    return df


# ===========================
# PREDICT ENDPOINT
# ===========================
@app.post("/predict")
def predict(data: Input):
    df = prepare_input(data)

    proba = float(model.predict_proba(df)[0][1])
    pred = int(proba >= 0.5)

    return {
        "prediction": pred,
        "probability": proba
    }


# ===========================
# Root Endpoint
# ===================
