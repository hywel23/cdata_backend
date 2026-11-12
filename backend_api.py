from pathlib import Path
from functools import lru_cache
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "processed_fyp_dataset.csv"

if not DATA_FILE.exists():
    DATA_FILE = BASE_DIR / "processed_fyp_dataset(1).csv"

MODEL_FILE = BASE_DIR / "linear_regression_model.pkl"

PRICING_FILE = (
    BASE_DIR / "pricing_optimization_results.csv"
)

GOVERNANCE_FILE = (
    BASE_DIR / "governance_logs"
    / "latest_pricing_decision.json"
)

XAI_DIR = BASE_DIR / "xai_results"

# 2. INITILIZE FASTAPI

app = FastAPI(
    title="CDATA Decision Support API",
    description=(
        "Retail pricing decision support under "
        "crude oil market volatility."
    ),
    version="1.0.0"
)

# Local development origins.
# Add the Flutter web origin if necessary.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://cdata-fyp.web.app",
        "https://cdata-fyp.firebaseapp.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 3. LOAD MACHINE LEARNING MODEL

@lru_cache(maxsize=1)
def get_model():

    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Trained model not found."
        )

    return joblib.load(MODEL_FILE)

# 4. LOAD LATEST DATASET SNAPSHOT

def get_latest_data():

    if not DATA_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Processed dataset not found."
        )

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"]
    )

    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="Dataset is empty."
        )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    return df, df.iloc[-1]

# 5. SALES PREDICTION

def calculate_sales_prediction(df):

    model = get_model()

    FEATURES = list(
        model.feature_names_in_
    )

    missing = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing features: {missing}"
        )

    input_data = df.iloc[[-1]][FEATURES].copy()

    input_data = input_data.astype(float)

    prediction = float(
        model.predict(input_data)[0]
    )

    return max(0.0, prediction)

# 6. LOAD PRICING RECOMMENDATION

def get_pricing_recommendation():

    if not PRICING_FILE.exists():
        return None

    pricing_df = pd.read_csv(PRICING_FILE)

    if pricing_df.empty:
        return None

    # Prefer the optimizer's flagged candidate.

    if "Recommended" in pricing_df.columns:

        recommended_rows = pricing_df[
            pricing_df["Recommended"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        ]

    else:
        recommended_rows= pd.DataFrame()

    if not recommended_rows.empty:

        best = recommended_rows.iloc[0]

    else:

        best = pricing_df.loc[
            pricing_df[
                "Expected_Gross_Contribution"
            ].idxmax()
        ]

    return {
        "candidate_price_rm": float(
            best["Candidate_Price"]
        ),
        "predicted_sales": float(
            best["Predicted_Sales"]
        ),
        "expected_revenue_rm": float(
            best["Expected_Revenue"]
        ),
        "expected_gross_contribution_rm": float(
            best["Expected_Gross_Contribution"]
        ),
        "margin_pct": float(
            best["Margin_%"]
        ),
        "price_change_pct": float(
            best["Price_Change_%"]
        )
    }

# 7. LOAD GOVERNANCE DECISION

def get_governance_decision():

    if not GOVERNANCE_FILE.exists():
        return None

    with open(
        GOVERNANCE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)

# 8. LOAD XAI EXPLANATIONS

def get_xai_results():

    global_file = (
        XAI_DIR / "global_feature_importance.csv"
    )

    local_file = (
        XAI_DIR / "local_explanation.csv"
    )

    results = {
        "global_importance": [],
        "local_explanation": []
    }

    if global_file.exists():

        global_df = pd.read_csv(global_file)

        for _, row in global_df.head(5).iterrows():

            results["global_importance"].append({
                "feature": str(row["Feature"]),
                "mae_increase": float(
                    row["MAE_Increase"]
                )
            })

    if local_file.exists():

        local_df = pd.read_csv(local_file)

        for _, row in local_df.head(5).iterrows():

            results["local_explanation"].append({
                "feature": str(row["Feature"]),
                "contribution": float(
                    row["Contribution"]
                )
            })

    return results

# 9. API HEALTH CHECK

@app.get("/")
def home():

    return {
        "system": "CDATA",
        "message": "CDATA Backend API is running.",
        "version": "1.0.0"
    }

@app.get("/api/health")
def health():

    return {
        "status": "running",
        "dataset_available": DATA_FILE.exists(),
        "model_available": MODEL_FILE.exists(),
        "pricing_available": PRICING_FILE.exists(),
        "governance_available": GOVERNANCE_FILE.exists()
    }

# 10. MARKET DATA ENDPOINT

@app.get("/api/market")
def market():

    df, latest = get_latest_data()

    return {
        "date": latest["Date"].strftime("%Y-%m-%d"),
        "brent_price": float(
            latest["Brent_Price"]
        ),
        "wti_price": float(
            latest["WTI_Price"]
        ),
        "ovx": float(
            latest["OVX"]
        ),
        "cpi": float(
            latest["CPI"]
        )
    }

# 11. SALES FORECAST ENDPOINT

@app.get("/api/prediction")
def prediction():

    df, latest = get_latest_data()

    predicted_sales = (
        calculate_sales_prediction(df)
    )

    return {
        "data_date": latest["Date"].strftime(
            "%Y-%m-%d"
        ),
        "predicted_daily_sales": round(
            predicted_sales, 2
        ),
        "model": "Linear Regression",
        "prediction_type": (
            "Latest historical observation"
        )
    }

# 12. PRICING ENDPOINT

@app.get("/api/pricing")
def pricing():

    _, latest = get_latest_data()

    candidate = get_pricing_recommendation()

    governance = get_governance_decision()

    current_price = float(
        latest["Current_Retail_Price"]
    )

    date = latest["Date"].strftime("%Y-%m-%d")

    # Validate the the audit belongs to
    # the current dataset and candidate

    audit_valid = False

    if candidate is not None and governance is not None:

        audit_valid = (
            governance.get("data_date") == date
            and abs(
                float(
                    governance.get(
                        "candidate_price_rm", -1
                    )
                ) - candidate["candidate_price_rm"]
            ) < 0.005
            and abs(
                float(
                    governance.get(
                        "current_price_rm", -1
                    )
                ) - current_price
            ) < 0.005
        )

    if audit_valid:
        status = governance["status"]
    else:
        status = "NOT_VALIDATED"

    return {
        "data_date": date,
        "current_price_rm": current_price,
        "candidate": candidate,
        "governance_status": status,
        "active_price_rm": current_price,
        "executed": False,
        "note": (
            "Prices are not automatically changed."
        )
    }

# 13. GOVERNANCE ENDPOINT

@app.get("/api/governance")
def governance():

    decision = get_governance_decision()

    if decision is None:
        raise HTTPException(
            status_code=404,
            detail="Governance record not found."
        )

    return decision

# 14. EXPLAINABLE AI ENDPOINT

@app.get("/api/xai")
def xai():

    return get_xai_results()

# 15. UNIFIED DASHBOARD ENDPOINT

@app.get("/api/dashboard")
def dashboard():

    return {
        "system": "CDATA",
        "market": market(),
        "prediction": prediction(),
        "pricing": pricing(),
        "xai": xai()
    }

from fastapi import Query

@app.get('/api/market/history')
def market_history(days: int = Query(default=90, ge=7, le=365)):
    df, _ = get_latest_data()
    required = ['Date', 'Brent_Price', 'WTI_Price']
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f'Missing columns: {missing}')

    subset = df[required].tail(days).copy()
    rows = []
    for _, row in subset.iterrows():
        rows.append({
            'date': str(row['Date'])[:10],
            'brent_price': float(row['Brent_Price']),
            'wti_price': float(row['WTI_Price']),
        })
    return {'rows': rows, 'count': len(rows), 'source': 'processed_fyp_dataset.csv'}

from pydantic import BaseModel, Field
import numpy as np


class CDATAScenarioInput(BaseModel):
    proposed_price_rm: float = Field(gt=0, le=100000)
    brent_change_pct: float = Field(default=0, ge=-20, le=20)
    wti_change_pct: float = Field(default=0, ge=-20, le=20)
    unit_cost_change_pct: float = Field(default=0, ge=-20, le=20)


@app.post('/api/simulate')
def simulate_pricing_scenario(request: CDATAScenarioInput):
    """Hypothetical *same-day* model response; never executes retail prices."""
    df, latest = get_latest_data()
    model = get_model()

    if not hasattr(model, 'feature_names_in_'):
        raise HTTPException(status_code=503, detail='Saved model does not expose feature names.')
    features = list(model.feature_names_in_)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise HTTPException(status_code=503, detail=f'Dataset lacks model features: {missing}')

    if len(df) < 31:
        raise HTTPException(status_code=503, detail='At least 31 historical rows are required.')

    try:
        current_price = float(latest['Current_Retail_Price'])
        current_cost = float(latest['Estimated_Dynamic_COGS'])
        inventory = max(0, float(latest['Inventory_Level']))
        old_brent = float(latest['Brent_Price'])
        old_wti = float(latest['WTI_Price'])
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=f'Invalid retail/market data: {exc}')

    if current_price <= 0 or old_brent <= 0 or old_wti <= 0:
        raise HTTPException(status_code=503, detail='Historical price inputs must be positive.')

    price = request.proposed_price_rm
    brent = old_brent * (1 + request.brent_change_pct / 100)
    wti = old_wti * (1 + request.wti_change_pct / 100)
    cost = current_cost * (1 + request.unit_cost_change_pct / 100)

    baseline_features = df.iloc[[-1]][features].copy()
    scenario_features = baseline_features.copy()
    idx = scenario_features.index[-1]

    # Immediate spot-price shock: historical lags remain unchanged.
    # Contemporaneous derived features are updated to avoid inconsistent inputs.
    updates = {
        'Current_Retail_Price': price,
        'Estimated_Dynamic_COGS': cost,
        'Brent_Price': brent,
        'WTI_Price': wti,
    }
    if 'Brent_MA_7' in features:
        updates['Brent_MA_7'] = float(latest['Brent_MA_7']) + (brent - old_brent) / 7
    if 'Brent_MA_30' in features:
        updates['Brent_MA_30'] = float(latest['Brent_MA_30']) + (brent - old_brent) / 30
    if 'WTI_MA_7' in features:
        updates['WTI_MA_7'] = float(latest['WTI_MA_7']) + (wti - old_wti) / 7
    if 'Brent_Return_1D_%' in features:
        prev = float(df['Brent_Price'].iloc[-2])
        if prev <= 0:
            raise HTTPException(status_code=503, detail='Invalid prior-day Brent price.')
        updates['Brent_Return_1D_%'] = (brent / prev - 1) * 100
    if 'Brent_Return_7D_%' in features:
        prev7 = float(df['Brent_Price'].iloc[-8])
        if prev7 <= 0:
            raise HTTPException(status_code=503, detail='Invalid seven-observation Brent lag.')
        updates['Brent_Return_7D_%'] = (brent / prev7 - 1) * 100
    if 'Price_Shift_From_Lag_%' in features:
        lag3 = float(latest['Brent_Lag_3'])
        if lag3 <= 0:
            raise HTTPException(status_code=503, detail='Invalid Brent three-day lag.')
        updates['Price_Shift_From_Lag_%'] = (brent / lag3 - 1) * 100
    if 'Brent_Volatility_14d_%' in features:
        recent_returns = df['Brent_Price'].astype(float).pct_change().mul(100).tail(14).copy()
        previous_brent = float(df['Brent_Price'].iloc[-2])
        recent_returns.iloc[-1] = (brent / previous_brent - 1) * 100
        updates['Brent_Volatility_14d_%'] = float(recent_returns.std(ddof=1))

    for feature, value in updates.items():
        if feature in features:
            scenario_features.at[idx, feature] = value

    try:
        baseline_pred = max(0, float(model.predict(baseline_features)[0]))
        scenario_pred = max(0, float(model.predict(scenario_features)[0]))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f'Model prediction failed: {exc}')

    baseline_sellable = min(baseline_pred, inventory)
    scenario_sellable = min(scenario_pred, inventory)
    baseline_revenue = current_price * baseline_sellable
    scenario_revenue = price * scenario_sellable
    baseline_contribution = (current_price - current_cost) * baseline_sellable
    scenario_contribution = (price - cost) * scenario_sellable
    contribution_change = scenario_contribution - baseline_contribution
    contribution_change_pct = (contribution_change / abs(baseline_contribution) * 100
                               if abs(baseline_contribution) > 0.001 else None)

    price_change_pct = (price / current_price - 1) * 100
    margin_pct = (price - cost) / price * 100

    # Price support and governance are based on the *training* set only.
    training_rows = df.iloc[:int(len(df) * 0.8)]
    min_trained_price = float(training_rows['Current_Retail_Price'].min())
    max_trained_price = float(training_rows['Current_Retail_Price'].max())
    blocks = []
    if abs(price_change_pct) > 5 + 1e-8:
        blocks.append('Price adjustment exceeds the illustrative ±5% limit.')
    if margin_pct < 25:
        blocks.append('Projected unit margin falls below the illustrative 25% floor.')
    if not (min_trained_price <= price <= max_trained_price):
        blocks.append('Proposed price is outside the historical training-price range.')
    approval_required = abs(price_change_pct) > 2
    improvement_pct = contribution_change_pct if contribution_change_pct is not None else 0

    if blocks:
        status = 'BLOCKED_BY_GOVERNANCE'
    elif improvement_pct < 0.5:
        status = 'HOLD_CURRENT_PRICE'
    elif approval_required:
        status = 'PENDING_APPROVAL'
    else:
        status = 'ELIGIBLE_FOR_REVIEW'

    return {
        'data_date': str(latest['Date'])[:10],
        'simulation_type': 'historical_same_day_hypothetical',
        'model': type(model).__name__,
        'inputs': {
            'proposed_price_rm': round(price, 2),
            'brent_change_pct': request.brent_change_pct,
            'wti_change_pct': request.wti_change_pct,
            'unit_cost_change_pct': request.unit_cost_change_pct,
        },
        'baseline': {
            'price_rm': round(current_price, 2),
            'brent_price': round(old_brent, 2),
            'wti_price': round(old_wti, 2),
            'unit_cost_rm': round(current_cost, 2),
            'predicted_demand_units': round(baseline_pred, 2),
            'sellable_units': round(baseline_sellable, 2),
            'revenue_rm': round(baseline_revenue, 2),
            'gross_contribution_rm': round(baseline_contribution, 2),
        },
        'scenario': {
            'price_rm': round(price, 2),
            'brent_price': round(brent, 2),
            'wti_price': round(wti, 2),
            'unit_cost_rm': round(cost, 2),
            'predicted_demand_units': round(scenario_pred, 2),
            'sellable_units': round(scenario_sellable, 2),
            'revenue_rm': round(scenario_revenue, 2),
            'gross_contribution_rm': round(scenario_contribution, 2),
            'gross_margin_pct': round(margin_pct, 2),
        },
        'comparison': {
            'demand_change_units': round(scenario_pred - baseline_pred, 2),
            'contribution_change_rm': round(contribution_change, 2),
            'contribution_change_pct': (round(contribution_change_pct, 2)
                                        if contribution_change_pct is not None else None),
            'price_change_pct': round(price_change_pct, 2),
        },
        'governance': {
            'status': status,
            'approval_required': approval_required,
            'hard_rule_violations': blocks,
            'trained_price_range_rm': [round(min_trained_price, 2), round(max_trained_price, 2)],
            'margin_floor_pct': 25,
            'max_price_change_pct': 5,
            'minimum_improvement_pct': 0.5,
        },
        'limitations': [
            'Model responses reflect learned associations, not causal effects.',
            'Market shocks are hypothetical changes to one historical observation, not forecasts.',
            'Cost changes are manually specified and not inferred from oil-price movements.',
            'Results are simulated only: no prices, approvals or audit records are changed.',
        ],
    }

# ========================================
# CDATA FIREBASE APPROVAL WORKFLOW
# ========================================

from proposal_workflow import register_proposal_routes

register_proposal_routes(app)
