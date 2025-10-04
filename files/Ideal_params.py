# ExoTerra.py
# Automated training pipeline for two targets: disposition and goldilocks
# Uses stacking (XGBoost + RF + LR) with SelectKBest feature selection
# Hardcoded best features and hyperparameters from Ideal_params.py output
# Run with: python ExoTerra.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import time

RND = 0

# ------------------ Config ------------------
INPUT_CSV = "exo_merged.csv"   # your merged CSV path
SAVE_PREFIX = "best_exo_fixed_model"
FINAL_CV = 5                   # final CV for reporting
N_JOBS = -1

# ------------------ Utilities & Loading ------------------
def safe_read_csv(path):
    return pd.read_csv(path, comment="#", low_memory=False)

print("Loading:", INPUT_CSV)
dt = safe_read_csv(INPUT_CSV)
print("Initial rows:", len(dt))

# ------------------ Basic normalization & robust casting ------------------
col_map = {
    "pl_orbper": "orbital_period", "koi_period": "orbital_period",
    "pl_trandep": "transit_depth", "koi_depth": "transit_depth",
    "pl_rade": "planet_radius", "koi_prad": "planet_radius",
    "st_rad": "stellar_rad", "koi_srad": "stellar_rad",
    "st_teff": "stellar_teff", "koi_steff": "stellar_teff",
    "pl_insol": "insolation_flux", "koi_insol": "insolation_flux",
    "pl_trandur": "transit_duration", "koi_duration": "transit_duration",
    "koi_pdisposition": "disposition", "tfopwg_disp": "disposition", "disposition": "disposition"
}
dt = dt.rename(columns={k: v for k,v in col_map.items() if k in dt.columns})

if "mission" not in dt.columns:
    # Simple inference
    if any("koi_" in c for c in dt.columns):
        dt["mission"] = "kepler"
    else:
        dt["mission"] = "merged"

# ------------------ Feature engineering ------------------
numeric_candidates = ["orbital_period", "transit_depth", "planet_radius", "stellar_rad",
                      "insolation_flux", "transit_duration", "stellar_teff"]
for c in numeric_candidates:
    if c in dt.columns:
        dt[c] = pd.to_numeric(dt[c], errors="coerce")

if {"planet_radius", "stellar_rad"}.issubset(dt.columns):
    dt["planet_to_star_radius_ratio"] = dt["planet_radius"] / (dt["stellar_rad"] + 1e-12)
if {"transit_depth", "transit_duration"}.issubset(dt.columns):
    dt["depth_to_duration"] = dt["transit_depth"] / (dt["transit_duration"] + 1e-12)
if {"planet_radius", "orbital_period"}.issubset(dt.columns):
    dt["radius_per_period"] = dt["planet_radius"] / (dt["orbital_period"] + 1e-12)

for c in ["orbital_period", "insolation_flux", "transit_depth", "planet_radius"]:
    if c in dt.columns:
        dt[f"log_{c}"] = np.log1p(dt[c].astype(float))

if {"stellar_rad", "stellar_teff"}.issubset(dt.columns):
    dt["stellar_lum_proxy"] = (dt["stellar_rad"].astype(float).fillna(0)**2) * (dt["stellar_teff"].astype(float).fillna(0)**4)
    dt["stellar_lum_proxy"] = np.log1p(dt["stellar_lum_proxy"])

# ------------------ Target mapping ------------------
def map_disposition(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if any(k in s for k in ("CONFIRM", "CONFIRMED", "C")):
        return 2
    if any(k in s for k in ("CAND", "CANDIDATE", "PC", "PLANET CANDIDATE")):
        return 1
    if any(k in s for k in ("FALSE", "FP", "FALSE POSITIVE", "FALSE_POSITIVE")):
        return 0
    try:
        return int(float(s))
    except Exception:
        return np.nan

if "disposition" not in dt.columns:
    raise KeyError("No 'disposition' column found in merged data.")

dt["disposition_num"] = dt["disposition"].apply(map_disposition)
dt = dt.dropna(subset=["disposition_num"]).reset_index(drop=True)
dt["disposition_num"] = dt["disposition_num"].astype(int)
print("After dropping missing disposition rows:", len(dt))

# Goldilocks label (equilibrium temp window)
if "equilibrium_temp" in dt.columns:
    dt["goldilocks"] = ((pd.to_numeric(dt["equilibrium_temp"], errors="coerce") >= 180) &
                        (pd.to_numeric(dt["equilibrium_temp"], errors="coerce") <= 310)).astype(int)
else:
    dt["goldilocks"] = 0

# ------------------ Hardcoded best features and hyperparameters ------------------

# Hardcoded best features determined from previous randomized search and feature selection
best_features = [
    "insolation_flux", "transit_depth", "planet_radius", "stellar_rad", "orbital_period",
    "stellar_teff", "transit_duration", "planet_to_star_radius_ratio",
    "depth_to_duration", "radius_per_period", "log_orbital_period", "log_insolation_flux"
]

categorical_cols = ["mission"] if "mission" in dt.columns else []

# Filter features to those present in data
best_features = [f for f in best_features if f in dt.columns]

print("Using hardcoded numeric features:", best_features)
print("Categorical features:", categorical_cols)

for c in best_features:
    dt[c] = pd.to_numeric(dt[c], errors="coerce")

# ------------------ Prepare datasets ------------------
def prepare_X_y(target_col):
    y = dt[target_col]
    mask = y.notna()
    X = dt.loc[mask, best_features + categorical_cols].copy()
    y = y[mask]
    return X, y

# ------------------ Preprocessor & pipeline ------------------
try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OHE)
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, best_features),
    ("cat", categorical_transformer, categorical_cols)
], remainder="drop")

# Hardcoded best hyperparameters from previous randomized search
# These values should be replaced with actual best params from Ideal_params.py output
best_params = {
    "model__xgb__n_estimators": 200,
    "model__xgb__max_depth": 4,
    "model__xgb__learning_rate": 0.1,
    "model__xgb__subsample": 0.8,
    "model__xgb__colsample_bytree": 0.8,
    "model__rf__n_estimators": 200,
    "model__rf__max_depth": 10,
    "selector__k": min(12, len(best_features) + len(categorical_cols)),
}

def create_pipeline():
    selector = SelectKBest(score_func=f_classif, k=best_params["selector__k"])
    base_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
        random_state=RND,
        n_estimators=best_params["model__xgb__n_estimators"],
        max_depth=best_params["model__xgb__max_depth"],
        learning_rate=best_params["model__xgb__learning_rate"],
        subsample=best_params["model__xgb__subsample"],
        colsample_bytree=best_params["model__xgb__colsample_bytree"]
    )
    base_rf = RandomForestClassifier(
        n_jobs=1,
        random_state=RND,
        n_estimators=best_params["model__rf__n_estimators"],
        max_depth=best_params["model__rf__max_depth"]
    )
    final_lr = LogisticRegression(max_iter=2000, random_state=RND)
    stack = StackingClassifier(
        estimators=[("xgb", base_xgb), ("rf", base_rf)],
        final_estimator=final_lr,
        n_jobs=1,
        passthrough=False
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("model", stack)
    ])
    return pipeline

cv_final = StratifiedKFold(n_splits=FINAL_CV, shuffle=True, random_state=RND)

def run_fixed_eval(target_name):
    print(f"\n--- Processing target: {target_name} ---")
    X, y = prepare_X_y(target_name)
    if X.shape[0] == 0:
        raise RuntimeError(f"No data available for target {target_name}")
    pipeline = create_pipeline()

    start = time.time()
    print(f"Starting training and cross-validation for {target_name}...")
    pipeline.fit(X, y)
    elapsed = time.time() - start
    print(f"Training done in {elapsed/60:.2f} minutes")

    cv_scores = cross_val_score(pipeline, X, y, cv=cv_final, scoring="accuracy", n_jobs=N_JOBS)
    print(f"Cross-validated accuracy ({FINAL_CV}-fold) for {target_name}: {cv_scores.mean():.4f}")

    # Save model
    save_path = f"{SAVE_PREFIX}_{target_name}.joblib"
    joblib.dump(pipeline, save_path)
    print(f"Saved pipeline for {target_name} to: {save_path}")

    # Print selected features
    try:
        preproc = pipeline.named_steps["preprocessor"]
        selector_step = pipeline.named_steps["selector"]
        num_cols = best_features
        cat_cols = []
        if categorical_cols:
            ohe_step = preproc.named_transformers_["cat"].named_steps["onehot"]
            try:
                ohe_names = ohe_step.get_feature_names_out(categorical_cols)
            except Exception:
                cats = ohe_step.categories_
                ohe_names = []
                for col, cats_list in zip(categorical_cols, cats):
                    for val in cats_list:
                        ohe_names.append(f"{col}_{val}")
            cat_cols = list(ohe_names)
        feature_names = list(num_cols) + list(cat_cols)
        selected_mask = selector_step.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_mask]
        print(f"Selected features used by model for {target_name}:", selected_features)
    except Exception:
        pass

# ------------------ Run for both targets ------------------
run_fixed_eval("disposition_num")
run_fixed_eval("goldilocks")

print("Done.")