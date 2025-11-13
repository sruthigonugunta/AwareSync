# train.py — scientifically grounded glycemic-impact model
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

# ============================
# 1. Load foods dataset
# ============================
BASE = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(BASE, "data", "foods.csv"))

df.columns = [c.strip().lower() for c in df.columns]
df["category"] = df["category"].str.lower()
df["item"] = df["item"].astype(str).str.strip()

# ============================
# 2. Estimate GI based on category
# ============================
GI_MAP = {
    "carbs": 70,
    "snack": 65,
    "veggie": 20,
    "protein": 10,
}

def estimate_gi(row):
    return GI_MAP.get(row["category"], 50)

df["gi"] = df.apply(estimate_gi, axis=1)

# ============================
# 3. Compute Glycemic Load
# ============================
df["net_carbs"] = (df["carbs_g"] - df["fiber_g"]).clip(lower=0)
df["glycemic_load"] = df["net_carbs"] * df["gi"] / 100

# ============================
# 4. Compute final glycemic impact
# ============================
df["glycemic_impact"] = (
    df["glycemic_load"]
    + 0.5 * df["added_sugar_g"]
    + 0.001 * df["sodium_mg"]
    + 0.3 * df["sat_fat_g"]
    - 0.4 * df["fiber_g"]
).clip(lower=0)

# ============================
# 5. Features + Labels
# ============================
y = df["glycemic_impact"].values
X = df[
    [
        "category",
        "calories",
        "carbs_g",
        "protein_g",
        "unsat_fat_g",
        "sat_fat_g",
        "fiber_g",
        "sugar_g",
        "added_sugar_g",
        "sodium_mg",
        "price_per_unit",
        "price_per_serving",
        "serving_size_g"
    ]
].copy()

num_cols = [
    "calories", "carbs_g", "protein_g", "unsat_fat_g", "sat_fat_g",
    "fiber_g", "sugar_g", "added_sugar_g", "sodium_mg",
    "price_per_unit", "price_per_serving", "serving_size_g"
]

cat_cols = ["category"]

pre = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

pipe = Pipeline(
    [
        ("pre", pre),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=12
        )),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

pipe.fit(X_train, y_train)
r2 = pipe.score(X_test, y_test)

# ============================
# 7. Save model + metrics
# ============================
model_dir = os.path.join(BASE, "model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(pipe, os.path.join(model_dir, "model.pkl"))

with open(os.path.join(model_dir, "metrics.json"), "w") as f:
    json.dump({"r2": float(r2), "train": len(X_train), "test": len(X_test)}, f)

print("Training complete.")
print("R2 =", r2)