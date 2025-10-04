# ExoTerra.py - deployment-ready version with dynamic input, prediction, logging, error handling

import pandas as pd
import numpy as np
import warnings
import time
import joblib
import json
import logging
import os
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RND = 42

# Configure logging
logger = logging.getLogger("ExoTerra")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

# ------------------ Config ------------------
DEFAULT_FINAL_CV = 5
DEFAULT_N_JOBS = -1
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ Hardcoded features and categorical columns ------------------
best_features = [
    'insolation_flux','transit_depth','planet_radius','stellar_rad','orbital_period',
    'stellar_teff','transit_duration','planet_to_star_radius_ratio',
    'depth_to_duration','radius_per_period','log_orbital_period','log_insolation_flux'
]
categorical_cols = ['mission']

# ------------------ Default hyperparameters ------------------
default_best_params = {
    'selector__k': len(best_features)+len(categorical_cols),
    'model__xgb__n_estimators': 200,
    'model__xgb__max_depth': 6,
    'model__xgb__learning_rate': 0.05,
    'model__xgb__subsample': 0.7,
    'model__xgb__colsample_bytree': 0.8,
    'model__rf__n_estimators': 100,
    'model__rf__max_depth': None
}

# ------------------ Utility functions ------------------

def _map_disposition(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().upper()
    if any(k in s for k in ('CONFIRM','CONFIRMED','C')): return 2
    if any(k in s for k in ('CAND','CANDIDATE','PC','PLANET CANDIDATE')): return 1
    if any(k in s for k in ('FALSE','FP','FALSE POSITIVE','FALSE_POSITIVE')): return 0
    try: return int(float(s))
    except: return np.nan

def _feature_engineering(df):
    numeric_candidates = ['orbital_period','transit_depth','planet_radius','stellar_rad','insolation_flux','transit_duration','stellar_teff']
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Only create derived features if source columns exist
    if 'planet_radius' in df.columns and 'stellar_rad' in df.columns:
        df['planet_to_star_radius_ratio'] = df['planet_radius'] / (df['stellar_rad'] + 1e-12)
    if 'transit_depth' in df.columns and 'transit_duration' in df.columns:
        df['depth_to_duration'] = df['transit_depth'] / (df['transit_duration'] + 1e-12)
    if 'planet_radius' in df.columns and 'orbital_period' in df.columns:
        df['radius_per_period'] = df['planet_radius'] / (df['orbital_period'] + 1e-12)
    for c in ['orbital_period','insolation_flux','transit_depth','planet_radius']:
        if c in df.columns:
            df[f'log_{c}'] = np.log1p(df[c].astype(float))

    # Only create stellar_lum_proxy if both columns exist
    if 'stellar_rad' in df.columns and 'stellar_teff' in df.columns:
        df['stellar_lum_proxy'] = np.log1p((df['stellar_rad'].fillna(0)**2) * (df['stellar_teff'].fillna(0)**4))

    return df


def _prepare_dataframe(input_data):
    """
    Accepts either a CSV path or a pandas DataFrame.
    Returns processed DataFrame ready for modeling.
    """
    try:
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            logger.info("Input is a DataFrame with %d rows", len(df))
        elif isinstance(input_data, str):
            logger.info(f"Loading CSV from path: {input_data}")
            df = pd.read_csv(input_data, comment='#', low_memory=False)
            logger.info("Loaded CSV with %d rows", len(df))
        else:
            raise ValueError("Input data must be a pandas DataFrame or a CSV file path string.")

        # Column renaming / normalization
        col_map = {
            'pl_orbper': 'orbital_period', 'koi_period': 'orbital_period',
            'pl_trandep': 'transit_depth', 'koi_depth': 'transit_depth',
            'pl_rade': 'planet_radius', 'koi_prad': 'planet_radius',
            'st_rad': 'stellar_rad', 'koi_srad': 'stellar_rad',
            'st_teff': 'stellar_teff', 'koi_steff': 'stellar_teff',
            'pl_insol': 'insolation_flux', 'koi_insol': 'insolation_flux',
            'pl_trandur': 'transit_duration', 'koi_duration': 'transit_duration',
            'koi_pdisposition': 'disposition', 'tfopwg_disp': 'disposition', 'disposition': 'disposition'
        }
        rename_cols = {k: v for k,v in col_map.items() if k in df.columns}
        if rename_cols:
            df.rename(columns=rename_cols, inplace=True)
            logger.info(f"Renamed columns: {rename_cols}")

        # Add mission column if missing
        if 'mission' not in df.columns:
            df['mission'] = 'kepler' if any('koi_' in c for c in df.columns) else 'merged'
            logger.info("Added missing 'mission' column with default values")

        # Map disposition to numeric for target if present
        if 'disposition' in df.columns:
            df['disposition_num'] = df['disposition'].apply(_map_disposition).astype('Int64')
            logger.info("Mapped 'disposition' to numeric 'disposition_num'")

        # Add goldilocks target if equilibrium_temp present
        if 'equilibrium_temp' in df.columns:
            eq_temp = pd.to_numeric(df['equilibrium_temp'], errors='coerce')
            df['goldilocks'] = ((eq_temp >= 180) & (eq_temp <= 310)).astype(int)
            logger.info("Added 'goldilocks' target based on 'equilibrium_temp'")
        else:
            df['goldilocks'] = 0

        # Feature engineering
        df = _feature_engineering(df)

        # Filter best_features to columns present
        filtered_best_features = [f for f in best_features if f in df.columns]
        # Ensure categorical columns present
        filtered_categorical_cols = [c for c in categorical_cols if c in df.columns]

        if not filtered_best_features:
            raise ValueError("None of the required best features are present in the data.")

        logger.info(f"Using numeric features: {filtered_best_features}")
        logger.info(f"Using categorical features: {filtered_categorical_cols}")

        return df, filtered_best_features, filtered_categorical_cols

    except Exception as e:
        logger.error(f"Error preparing DataFrame: {e}", exc_info=True)
        raise

def _create_preprocessor(numeric_features, categorical_features):
    try:
        OHE = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                       ('scaler', StandardScaler())])

        categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                           ('onehot', OHE)])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')

        return preprocessor
    except Exception as e:
        logger.error(f"Error creating preprocessor: {e}", exc_info=True)
        raise

def _create_pipeline(numeric_features, categorical_features, best_params):
    try:
        preprocessor = _create_preprocessor(numeric_features, categorical_features)

        selector = SelectKBest(score_func=f_classif, k=best_params.get('selector__k', len(numeric_features)+len(categorical_features)))

        base_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=RND,
                                 n_estimators=best_params.get('model__xgb__n_estimators', 200),
                                 max_depth=best_params.get('model__xgb__max_depth', 6),
                                 learning_rate=best_params.get('model__xgb__learning_rate', 0.05),
                                 subsample=best_params.get('model__xgb__subsample', 0.7),
                                 colsample_bytree=best_params.get('model__xgb__colsample_bytree', 0.8))

        base_rf = RandomForestClassifier(n_estimators=best_params.get('model__rf__n_estimators', 100),
                                         max_depth=best_params.get('model__rf__max_depth', None),
                                         random_state=RND, n_jobs=1)

        final_lr = LogisticRegression(max_iter=2000, random_state=RND)

        stack = StackingClassifier(estimators=[('xgb', base_xgb), ('rf', base_rf)],
                                   final_estimator=final_lr, n_jobs=1, passthrough=False)

        pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', stack)])

        return pipeline
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}", exc_info=True)
        raise

def _save_pipeline(pipeline, target_name):
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"best_exo_model_{target_name}_{timestamp}.joblib"
        save_path = os.path.join(SAVE_DIR, filename)
        joblib.dump(pipeline, save_path)
        logger.info(f"Saved model to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Error saving pipeline: {e}", exc_info=True)
        raise

# ------------------ Core functions ------------------

class ExoTerraModel:
    def __init__(self, input_data, target_name, best_params=None):
        """
        input_data: CSV filepath or pandas DataFrame
        target_name: target column name ('disposition_num' or 'goldilocks')
        best_params: optional dictionary to override hyperparameters
        """
        self.target_name = target_name
        self.best_params = default_best_params.copy()
        if best_params:
            self.best_params.update(best_params)
        self.pipeline = None
        self.data = None
        self.numeric_features = None
        self.categorical_features = None

        try:
            df, num_feats, cat_feats = _prepare_dataframe(input_data)
            self.data = df
            self.numeric_features = num_feats
            self.categorical_features = cat_feats

            if target_name not in df.columns:
                raise ValueError(f"Target column '{target_name}' not found in data.")

            # Filter out rows with NA target
            self.data = self.data.dropna(subset=[target_name])
            if self.data.empty:
                raise ValueError(f"No rows with non-null target '{target_name}' after filtering.")

            logger.info(f"ExoTerraModel initialized for target '{target_name}' with {len(self.data)} rows")

        except Exception as e:
            logger.error(f"Failed to initialize ExoTerraModel: {e}", exc_info=True)
            raise

    def train(self):
        try:
            self.pipeline = _create_pipeline(self.numeric_features, self.categorical_features, self.best_params)
            X = self.data[self.numeric_features + self.categorical_features]
            y = self.data[self.target_name]

            start_time = time.time()
            logger.info(f"Training model for target '{self.target_name}'...")
            self.pipeline.fit(X, y)
            elapsed = time.time() - start_time
            logger.info(f"Training completed in {elapsed/60:.2f} minutes")

            # Cross-validation scores
            cv_final = StratifiedKFold(n_splits=DEFAULT_FINAL_CV, shuffle=True, random_state=RND)
            logger.info(f"Performing {DEFAULT_FINAL_CV}-fold cross-validation...")
            cv_scores = cross_val_score(self.pipeline, X, y, cv=cv_final, scoring='accuracy', n_jobs=DEFAULT_N_JOBS)
            logger.info(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

            # Save model
            save_path = _save_pipeline(self.pipeline, self.target_name)

            return {
                'cv_mean_accuracy': float(cv_scores.mean()),
                'cv_std_accuracy': float(cv_scores.std()),
                'training_time_sec': elapsed,
                'model_path': save_path
            }
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise

    def predict(self, df_new):
        """
        Predict on new data (DataFrame).
        Returns JSON string with predictions mapped to row indices, CV stats, and timing.
        Does NOT retrain model.
        """
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not trained. Call train() first.")

        try:
            df_new_processed, num_feats_new, cat_feats_new = _prepare_dataframe(df_new)

            # Check if features match training features
            missing_num = [f for f in self.numeric_features if f not in df_new_processed.columns]
            missing_cat = [c for c in self.categorical_features if c not in df_new_processed.columns]
            if missing_num or missing_cat:
                raise ValueError(f"New data missing required features. Missing numeric: {missing_num}, categorical: {missing_cat}")

            print("Before X_new:", self.numeric_features + self.categorical_features)
            X_new = df_new_processed[self.numeric_features + self.categorical_features]

            start_time = time.time()
            print("X_new columns:", X_new.columns)
            preds = self.pipeline.predict(X_new)
            elapsed = time.time() - start_time

            # Cross-validation on training data for stats
            X_train = self.data[self.numeric_features + self.categorical_features]
            y_train = self.data[self.target_name]
            cv_final = StratifiedKFold(n_splits=DEFAULT_FINAL_CV, shuffle=True, random_state=RND)
            cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv_final, scoring='accuracy', n_jobs=DEFAULT_N_JOBS)

            # Map predictions to row indices
            predictions_dict = {int(idx): int(pred) if np.issubdtype(type(pred), np.integer) else pred for idx, pred in zip(df_new_processed.index, preds)}

            output = {
                'target': self.target_name,
                'cv_mean_accuracy': float(cv_scores.mean()),
                'cv_std_accuracy': float(cv_scores.std()),
                'prediction_time_sec': elapsed,
                'predictions': predictions_dict
            }
            return json.dumps(output)
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise

# ------------------ Deployment functions ------------------

def train_model(input_data, target_name, hyperparams_override=None):
    """
    Train model on input data (CSV path or DataFrame) for given target.
    hyperparams_override: dict to override default hyperparameters
    Returns dict with training stats and model path.
    """
    try:
        model = ExoTerraModel(input_data, target_name, best_params=hyperparams_override)
        results = model.train()
        return results
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

def predict_new_data(model, df_new):
    """
    Predict on new DataFrame using trained ExoTerraModel instance.
    Returns JSON string with predictions and stats.
    """
    try:
        return model.predict(df_new)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise

# ------------------ Offline training and evaluation example ------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate ExoTerra models.")
    parser.add_argument('--csv', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--target', type=str, choices=['disposition_num', 'goldilocks'], required=True, help="Target column to train on")
    parser.add_argument('--override_params', type=str, default=None, help="JSON string to override hyperparameters")
    args = parser.parse_args()

    try:
        override_params = None
        if args.override_params:
            override_params = json.loads(args.override_params)
            logger.info(f"Overriding hyperparameters with: {override_params}")

        logger.info(f"Starting training on target '{args.target}' with CSV '{args.csv}'")
        results = train_model(args.csv, args.target, hyperparams_override=override_params)
        logger.info(f"Training results: {results}")

        # Load trained model for prediction demo
        model_files = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith(f"best_exo_model_{args.target}_")])
        if not model_files:
            logger.error("No saved model found after training.")
            exit(1)
        latest_model_path = os.path.join(SAVE_DIR, model_files[-1])
        logger.info(f"Loading model from {latest_model_path} for prediction demo")
        trained_model = joblib.load(latest_model_path)

        # Wrap loaded pipeline in ExoTerraModel for prediction interface
        # We need to reconstruct ExoTerraModel with data info for predict method to work properly
        # So we do a minimal hack here:
        exo_model = ExoTerraModel(args.csv, args.target)
        exo_model.pipeline = trained_model

        # Predict on training data as demo
        logger.info("Running prediction demo on training data")
        pred_json = exo_model.predict(exo_model.data)
        print(pred_json)

        logger.info("Done.")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        exit(1)
 # ------------------ Web/Backend JSON Output Example ------------------

# This is what `ExoTerraModel.predict(df_new)` returns as JSON string
example_output = {
    'target': 'disposition_num',
    'cv_mean_accuracy': 0.8509,
    'cv_std_accuracy': 0.0021,
    'prediction_time_sec': 0.15,
    'predictions': {
        0: 1, # candidate
        1: 2, # confirmed
        2: 0, # false positive
        3: 2,
        4: 1
    }
}

# Backend can directly load this JSON and send to frontend
# e.g., json.dumps(example_output)