"""
train_models.py
---------------
FALLBACK SCRIPT -- not the source of the deployed models.

The models the app loads are exported directly by Section 7 of
MLDP_Program_Codes.ipynb, which also verifies that reloading them reproduces the
notebook's test-set predictions bit-for-bit.

This script performs the same export standalone, using the identical pipeline,
hyperparameters and train/test split, so the artefacts can be regenerated without
opening the notebook -- for example after a scikit-learn upgrade makes the committed
.pkl files unreadable. Expect RMSE 0.3513 (heating) and 1.0291 (cooling); anything
else means something has drifted.

Run:  python train_models.py
Out:  models/heating_model.pkl, models/cooling_model.pkl, models/metadata.json
"""
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
HERE = pathlib.Path(__file__).parent
OUT = HERE / "models"
OUT.mkdir(exist_ok=True)

COLUMN_MAP = {"X1": "RelCompactness", "X2": "SurfaceArea", "X3": "WallArea",
              "X4": "RoofArea", "X5": "Height", "X6": "Orientation",
              "X7": "GlazingArea", "X8": "GlazingDistrib",
              "Y1": "HeatingLoad", "Y2": "CoolingLoad"}
TARGETS = ["HeatingLoad", "CoolingLoad"]
NOMINAL = ["Orientation", "GlazingDistrib"]
REDUNDANT = ["SurfaceArea"]

# ----------------------------------------------------------------- data & features
df = pd.read_excel(HERE / "ENB2012_data.xlsx").rename(columns=COLUMN_MAP)
clean = df.copy()
for col in NOMINAL:
    clean[col] = clean[col].astype("category")


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Identical to Section 3.2 of the notebook. Row-wise and deterministic."""
    out = data.copy()
    out["GlazedArea_abs"] = out["GlazingArea"] * out["RoofArea"]
    out["WallToRoof"] = out["WallArea"] / out["RoofArea"]
    out["SurfaceToVolume"] = out["SurfaceArea"] / (out["RoofArea"] * out["Height"])
    out["IsTwoStorey"] = (out["Height"] > 5).astype(int)
    out["NoGlazing"] = (out["GlazingArea"] == 0).astype(int)
    return out


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    X = data.drop(columns=TARGETS, errors="ignore").copy()
    X = add_engineered_features(X)
    return X.drop(columns=REDUNDANT, errors="ignore")


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    nominal = [c for c in NOMINAL if c in X.columns]
    numeric = [c for c in X.columns if c not in nominal]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("nom", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal),
    ])


X_full, Y_full = build_feature_matrix(clean), clean[TARGETS]
X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full, test_size=0.2, random_state=RANDOM_STATE)

# --------------------------------------------- final tuned models (from Section 5.3)
ESTIMATORS = {
    "HeatingLoad": GradientBoostingRegressor(
        random_state=RANDOM_STATE, learning_rate=0.05, max_depth=5,
        max_features=0.7, min_samples_leaf=3, n_estimators=800, subsample=0.8),
    "CoolingLoad": HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, l2_regularization=0.0, learning_rate=0.1,
        max_depth=5, max_iter=800, max_leaf_nodes=63, min_samples_leaf=10),
}

metadata = {"targets": TARGETS, "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "features": list(X_full.columns), "metrics": {}, "importance": {},
            "extrapolation": {}, "business_bar": {}}

shape_id = clean.groupby("RelCompactness", observed=True).ngroup()

for tgt, est in ESTIMATORS.items():
    pipe = Pipeline([("prep", make_preprocessor(X_train)), ("model", clone(est))])
    pipe.fit(X_train, Y_train[tgt])
    pred = pipe.predict(X_test)

    metadata["metrics"][tgt] = {
        "RMSE": float(np.sqrt(mean_squared_error(Y_test[tgt], pred))),
        "MAE": float(mean_absolute_error(Y_test[tgt], pred)),
        "R2": float(r2_score(Y_test[tgt], pred)),
        "MAPE": float(mean_absolute_percentage_error(Y_test[tgt], pred) * 100),
        "model_name": type(est).__name__,
        "mean_load": float(clean[tgt].mean()),
    }
    metadata["business_bar"][tgt] = float(0.05 * clean[tgt].mean())

    perm = permutation_importance(pipe, X_test, Y_test[tgt],
                                  scoring="neg_root_mean_squared_error",
                                  n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    metadata["importance"][tgt] = {f: float(v) for f, v
                                   in zip(X_test.columns, perm.importances_mean)}

    grouped = cross_val_score(clone(pipe), X_full, Y_full[tgt], groups=shape_id,
                              cv=GroupKFold(n_splits=6),
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
    metadata["extrapolation"][tgt] = {"grouped_rmse": float(-grouped.mean()),
                                      "worst_fold": float(-grouped.min())}

    fname = f"{tgt.replace('Load', '').lower()}_model.pkl"
    with open(OUT / fname, "wb") as fh:
        pickle.dump(pipe, fh, protocol=5)
    print(f"{tgt:<12} RMSE={metadata['metrics'][tgt]['RMSE']:.4f} "
          f"R2={metadata['metrics'][tgt]['R2']:.4f}  -> {fname}")

# --------------------------------------- validated design envelope (the 12 shapes)
shapes = (clean[["RelCompactness", "SurfaceArea", "WallArea", "RoofArea", "Height"]]
          .drop_duplicates().sort_values("RelCompactness").reset_index(drop=True))
metadata["shapes"] = shapes.to_dict(orient="records")
metadata["glazing_areas"] = sorted(clean["GlazingArea"].unique().tolist())
metadata["glazing_distribs"] = sorted(int(v) for v in clean["GlazingDistrib"].unique())
metadata["orientations"] = sorted(int(v) for v in clean["Orientation"].unique())
metadata["load_ranges"] = {t: {"min": float(clean[t].min()), "max": float(clean[t].max())}
                           for t in TARGETS}

with open(OUT / "metadata.json", "w") as fh:
    json.dump(metadata, fh, indent=2)

print(f"\n{len(shapes)} validated building shapes recorded.")
print(f"Artefacts written to {OUT}/")
