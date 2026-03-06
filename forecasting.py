# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# %%
# load forecast dataset
df = pd.read_csv("data/processed/part1_forecast_data.csv", parse_dates=["date"])
print(f"loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"date range: {df['date'].min()} to {df['date'].max()}")
print(df.head().to_string())


# %%
# pick the target column (occupancy rate)

target = None
for col in df.columns:
    if "occ" in col.lower() and df[col].dtype in ["float64", "int64"]:
        target = col
        break

if target is None:
    # fallback: try revpar
    for col in df.columns:
        if "revpar" in col.lower() and df[col].dtype in ["float64", "int64"]:
            target = col
            break

print(f"target: '{target}'")
print(f"mean: {df[target].mean():.2f}, std: {df[target].std():.2f}")


# %%
drop_cols = ["date", "period", target]
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in drop_cols]

# drop any feature that's more than 30% NaN
threshold = 0.3
keep = []
for col in feature_cols:
    pct_missing = df[col].isnull().mean()
    if pct_missing <= threshold:
        keep.append(col)
    else:
        print(f"  dropping '{col}' ({pct_missing:.0%} missing)")

feature_cols = keep
print(f"\nfeatures: {len(feature_cols)} columns")
print(feature_cols)


# %%
# drop rows where target is NaN, fill remaining NaN in features

df_model = df[["date", target] + feature_cols].dropna(subset=[target]).copy()
df_model[feature_cols] = df_model[feature_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)

print(f"modelling dataset: {df_model.shape[0]} rows, {len(feature_cols)} features")
print(f"remaining NaN: {df_model[feature_cols].isnull().sum().sum()}")


# %%
# train/test split — use last 20% as test (time series, no random shuffle)

X = df_model[feature_cols]
y = df_model[target]

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df_model["date"].iloc[split_idx:]

print(f"train: {len(X_train)} rows ({df_model['date'].iloc[0].strftime('%Y-%m')} to {df_model['date'].iloc[split_idx-1].strftime('%Y-%m')})")
print(f"test:  {len(X_test)} rows ({df_model['date'].iloc[split_idx].strftime('%Y-%m')} to {df_model['date'].iloc[-1].strftime('%Y-%m')})")


# %%
# train models

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                             random_state=42, verbosity=0),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = mean_absolute_percentage_error(y_test, pred) * 100

    results[name] = {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE": round(mape, 2),
        "predictions": pred,
        "model": model,
    }
    print(f"  {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")


# %%
# model comparison

comp = pd.DataFrame({
    name: {"MAE": r["MAE"], "RMSE": r["RMSE"], "MAPE": r["MAPE"]}
    for name, r in results.items()
}).T.sort_values("MAE")

print(comp.to_string())

best_name = comp["MAE"].idxmin()
print(f"\nbest model: {best_name}")

fig, ax = plt.subplots(figsize=(10, 5))
comp[["MAE", "RMSE"]].plot(kind="bar", ax=ax, color=["#2c3e50", "#e74c3c"])
ax.set_title("Forecast Model Comparison", fontweight="bold")
ax.set_ylabel("Error")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_comparison_forecast.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
# actual vs predicted plot for best model

best_pred = results[best_name]["predictions"]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_test, y_test.values, "o-", label="Actual", color="#2c3e50", linewidth=1.5)
ax.plot(dates_test, best_pred, "o-", label=f"Predicted ({best_name})", color="#e74c3c", linewidth=1.5)
ax.set_title(f"Forecast vs Actual — {best_name}", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel(target)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/forecast_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
# feature importance from xgboost

xgb_model = results["XGBoost"]["model"]
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("top 15 features:")
print(importances.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top15 = importances.head(15)
ax.barh(top15["feature"], top15["importance"], color="#2c3e50")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance (XGBoost)", fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
# cross-validation with time series split

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(xgb_model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
print(f"cross-validation MAE: {-cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")


# %%
# save everything

best_model = results[best_name]["model"]
joblib.dump(best_model, "models/forecast_xgb_model.pkl")
importances.to_csv("data/processed/feature_importance.csv", index=False)
comp.to_csv("data/processed/model_comparison_forecast.csv")

print("saved:")
print("  models/forecast_xgb_model.pkl")
print("  data/processed/feature_importance.csv")
print("  data/processed/model_comparison_forecast.csv")
# %%
