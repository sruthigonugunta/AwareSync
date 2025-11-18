# api/main.py — AwareSync API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint, confloat
import pandas as pd
import joblib
import numpy as np
import os
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "templates"))

app = FastAPI(title="AwareSync API")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local prototype
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load data + model
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root

foods = pd.read_csv(os.path.join(ROOT_DIR, "data", "foods.csv"))
foods.columns = [c.strip().lower() for c in foods.columns]

# make sure numeric
for col in [
    "calories", "carbs_g", "protein_g", "unsat_fat_g", "sat_fat_g",
    "fiber_g", "sugar_g", "added_sugar_g", "sodium_mg",
    "price_per_unit", "price_per_serving", "serving_size_g",
]:
    if col in foods.columns:
        foods[col] = pd.to_numeric(foods[col], errors="coerce").fillna(0)

foods["price_per_serving"] = foods["price_per_serving"].fillna(0)

model = joblib.load(os.path.join(ROOT_DIR, "model", "model.pkl"))

# -----------------------------
# Diet lists
# -----------------------------
WHITE_MEAT_LIST = ["chicken", "shrimp", "salmon","fish", "turkey"]
RED_MEAT_LIST   = ["beef", "pork", "meat", "sausage", "hamburger"]
CHEESE_LIST     = ["paneer", "ricotta", "gorgonzola", "cheese", "pizza", "milk", "yogurt", "ice cream", "cake", "chocolate", "mozzerella", "chedder", "colby jack", "pepper jack", "gouda", "colby pepper jack", "parmesan", "feta", "Prosciutto", "sour cream", "whipped cream"]

# -----------------------------
# Pydantic Models
# -----------------------------
class UserInput(BaseModel):
    budget: confloat(gt=0) = 60.0
    people: conint(ge=1) = 1
    height_cm: float = 168
    weight_kg: float = 75
    activity: int = 0
    diet: str = "any"
    restrict_low_sodium: bool = True
    max_added_sugar_g: float = 8
    target_carbs_per_meal_g: float = 45
    location: str = "Atlanta, GA"
    store_preference: str = "any"  # NEW


class Request(BaseModel):
    user: UserInput

# -----------------------------
# Diet Filtering
# -----------------------------
def diet_filter(df: pd.DataFrame, diet: str) -> pd.DataFrame:
    items = df["item"].str.lower()

    is_meat = items.str.contains("|".join(WHITE_MEAT_LIST + RED_MEAT_LIST), na=False)
    is_red = items.str.contains("|".join(RED_MEAT_LIST), na=False)
    is_cheese = items.str.contains("|".join(CHEESE_LIST), na=False)

    if diet == "any":
        return df
    if diet == "vegetarian":
        return df[~is_meat]
    if diet == "vegan":
        return df[~(is_meat | is_cheese)]
    if diet == "white_meat_only":
        return df[~is_red]

    return df

# -----------------------------
# Additional rules (store, sodium, sugar)
# -----------------------------
def apply_rules(df: pd.DataFrame, user: UserInput) -> pd.DataFrame:
    df = diet_filter(df, user.diet)

    # store preference (optional)
    pref = (getattr(user, "store_preference", "") or "").strip().lower()

    if pref and pref not in ("any", "no preference", "none"):
        if "store" in df.columns:
            filtered = df[df["store"].str.lower().str.contains(pref, na=False)]
            # if nothing matched, fall back to no store filter
            if not filtered.empty:
                df = filtered

    if user.restrict_low_sodium:
        if "sodium_mg" in df.columns:
            df = df[df["sodium_mg"] <= 300]

    if "added_sugar_g" in df.columns:
        df = df[df["added_sugar_g"] <= user.max_added_sugar_g]

    return df.reset_index(drop=True)

# -----------------------------
# Ranking with Model
# -----------------------------
FEATURE_COLS = [
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
    "serving_size_g",
]

def predict_ranking(df: pd.DataFrame, user: UserInput) -> pd.DataFrame:
    if df.empty:
        return df

    X = df[FEATURE_COLS]

    preds = model.predict(X)

    out = df.copy()
    out["predicted_glycemic_impact"] = preds.astype(float)
    out["price_per_serving"] = df["price_per_serving"].astype(float)

    out["health_score"] = (
        -out["predicted_glycemic_impact"]
        - 0.04 * out["sugar_g"]
        + 0.03 * out["fiber_g"]
    ).astype(float)

    return out.sort_values(
        ["health_score", "price_per_serving"],
        ascending=[False, True],
    ).reset_index(drop=True)

# -----------------------------
# Buckets
# -----------------------------
def bucketize(df: pd.DataFrame):
    veg = df[df["category"] == "veggie"].reset_index(drop=True)
    carbs = df[df["category"].isin(["carbs", "snack"])].reset_index(drop=True)
    prots = df[df["category"] == "protein"].reset_index(drop=True)
    return veg, carbs, prots

# -----------------------------
# Helpers
# -----------------------------
def pick_carbs(carb_df: pd.DataFrame, target: float):
    if carb_df.empty:
        return None
    idx = (carb_df["carbs_g"] - target).abs().idxmin()
    return carb_df.loc[idx]

def pick_protein(df: pd.DataFrame, diet: str):
    if df.empty:
        return None
    items = df["item"].str.lower()
    if diet == "white_meat_only":
        allowed = df[items.str.contains("chicken|shrimp|salmon", na=False)]
        if not allowed.empty:
            return allowed.iloc[0]
        return None
    return df.iloc[0]

def build_plate(veg: pd.DataFrame, carbs: pd.DataFrame, prots: pd.DataFrame, user: UserInput):
    v = veg.iloc[0] if not veg.empty else None
    c = pick_carbs(carbs, user.target_carbs_per_meal_g)
    p = pick_protein(prots, user.diet)
    if v is None or c is None or p is None:
        return None
    return [v, p, c]

# -----------------------------
# Weekly Meal Planner
# -----------------------------
def plan_week(ranked: pd.DataFrame, user: UserInput):
    veg, carbs, prots = bucketize(ranked)

    total_cost = 0.0
    items_used = []
    meals = []

    for day in range(1, 8):
        meal = build_plate(veg, carbs, prots, user)
        if meal is None:
            break

        cost = sum(float(x["price_per_serving"]) for x in meal) * int(user.people)
        cost = float(cost)

        if total_cost + cost > user.budget:
            break

        total_cost += cost

        meals.append({
            "day": int(day),
            "breakfast": [str(x["item"]) for x in meal],
            "lunch":     [str(x["item"]) for x in meal],
            "dinner":    [str(x["item"]) for x in meal],
        })

        for x in meal:
            items_used.append({
                "item": str(x["item"]),
                "category": str(x["category"]),
                "price_per_serving": float(x["price_per_serving"]),
                "carbs_g": float(x["carbs_g"]),
                "protein_g": float(x["protein_g"]),
                "fat_g": float(x["sat_fat_g"] + x["unsat_fat_g"]),
                "fiber_g": float(x["fiber_g"]),
                "added_sugar_g": float(x["added_sugar_g"]),
                "predicted_glycemic_impact": float(x["predicted_glycemic_impact"]),
                "store": str(x["store"]) if "store" in x.index else "",
            })

        # rotate buckets so we don't choose same items in same order every day
        if len(veg) > 1:
            veg = pd.concat([veg.iloc[1:], veg.iloc[:1]], ignore_index=True)
        if len(carbs) > 1:
            carbs = pd.concat([carbs.iloc[1:], carbs.iloc[:1]], ignore_index=True)
        if len(prots) > 1:
            prots = pd.concat([prots.iloc[1:], prots.iloc[:1]], ignore_index=True)

    return float(total_cost), items_used, meals

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/recommend")
def recommend(req: Request):
    user = req.user

    filtered = apply_rules(foods, user)
    ranked = predict_ranking(filtered, user)

    total_cost, items_used, meals = plan_week(ranked, user)

    return {
        "budget_used": float(total_cost),
        "diet": str(user.diet),
        "target_carbs_per_meal_g": float(user.target_carbs_per_meal_g),
        "items": items_used,
        "meals": meals,
    }
@app.get("/resources", response_class=HTMLResponse)
def resources_page(request: Request):
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "resources.csv"))
    data = df.to_dict(orient="records")
    return templates.TemplateResponse(
        "resources.html",
        {"request": request, "resources": data}
    )


@app.get("/events", response_class=HTMLResponse)
def events_page(request: Request):
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "events.csv"))
    data = df.to_dict(orient="records")
    return templates.TemplateResponse(
        "events.html",
        {"request": request, "events": data}
    )

