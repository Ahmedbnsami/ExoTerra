import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import joblib

# Load dataset
dt = pd.read_csv("exo_merged.csv")

# --- Feature engineering ---
if "planet_radius" in dt.columns and "stellar_rad" in dt.columns:
    dt["planet_to_star_radius_ratio"] = dt["planet_radius"] / dt["stellar_rad"]

if "transit_depth" in dt.columns and "transit_duration" in dt.columns:
    dt["depth_to_duration"] = dt["transit_depth"] / dt["transit_duration"]

if "orbital_period" in dt.columns and "planet_radius" in dt.columns:
    dt["radius_per_period"] = dt["planet_radius"] / (dt["orbital_period"] + 1e-6)

for col in ["orbital_period", "insolation_flux", "transit_depth"]:
    if col in dt.columns:
        dt[f"log_{col}"] = np.log1p(pd.to_numeric(dt[col], errors="coerce"))

numeric_features = [
    "insolation_flux", "transit_depth", "stellar_rad", "planet_radius",
    "orbital_period", "stellar_logg", "equilibrium_temp", "stellar_teff",
    "transit_duration",
    "planet_to_star_radius_ratio", "depth_to_duration", "radius_per_period",
    "log_orbital_period", "log_insolation_flux", "log_transit_depth"
]

# Filter features that exist in dataframe
features = [f for f in numeric_features if f in dt.columns]

# Define function to train and evaluate model for a given target
def train_and_evaluate(target):
    # Drop rows with missing target
    data = dt.dropna(subset=[target])
    X = data[features]
    y = data[target]

    # Define base learners
    estimators = [
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, solver='liblinear'))
    ]

    # Define stacking classifier
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, solver='liblinear'),
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    # Pipeline: impute missing, scale, select features, then stacking
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k='all')),
        ('stacking', stack)
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-validated accuracy for target '{target}': {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Fit on full data
    pipeline.fit(X, y)

    # Get selected features (all features are selected since k='all', but we can check scores)
    selector = pipeline.named_steps['selector']
    scores_ = selector.scores_
    pvalues_ = selector.pvalues_
    selected_features = [f for f, p in zip(features, pvalues_) if p is not None and p < 0.05]
    if not selected_features:
        selected_features = features
    print(f"Selected features for target '{target}': {selected_features}")

    # Save model
    joblib.dump(pipeline, f"{target}_stacked_model.joblib")

    return pipeline

# Train and evaluate for 'disposition'
model_disposition = train_and_evaluate('disposition')

# Train and evaluate for 'goldilocks'
model_goldilocks = train_and_evaluate('goldilocks')