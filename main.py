!pip install pygeohash catboost -q

import pygeohash as pgh
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
print("Initializing the demand prediction pipeline...")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_idx = test['Index']


# ---------------------------------------------------------
# 2. Feature Engineering and Preprocessing
# ---------------------------------------------------------
def preprocess_data(df_train, df_test):
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    df_test['demand'] = -1

    full = pd.concat([df_train, df_test], ignore_index=True)

    print("Generating spatial and geometric features...")

    # Base geographic coordinates
    full['Latitude'] = full['geohash'].apply(
        lambda x: pgh.decode(str(x))[0] if pd.notnull(x) else np.nan
    )
    full['Longitude'] = full['geohash'].apply(
        lambda x: pgh.decode(str(x))[1] if pd.notnull(x) else np.nan
    )

    # Macro- and micro-level geographic regions
    full['geohash_macro'] = full['geohash'].astype(str).str[:4]
    full['geohash_micro'] = full['geohash'].astype(str).str[:5]

    # Geohash frequency encoding
    freq_map = full['geohash'].value_counts().to_dict()
    full['geohash_density'] = full['geohash'].map(freq_map)

    # Distance and angle relative to the geographic center
    mean_lat, mean_lon = full['Latitude'].mean(), full['Longitude'].mean()

    full['Dist_to_Center'] = np.sqrt(
        (full['Latitude'] - mean_lat) ** 2
        + (full['Longitude'] - mean_lon) ** 2
    )

    full['Angle_to_Center'] = np.arctan2(
        full['Latitude'] - mean_lat,
        full['Longitude'] - mean_lon
    )

    print("Generating temporal features...")

    if 'timestamp' in full.columns:
        ts = full['timestamp'].astype(str).str.split(':', expand=True)

        full['Hour'] = (
            pd.to_numeric(ts[0], errors='coerce')
            .fillna(0)
            .astype(int)
        )
        full['Minute'] = (
            pd.to_numeric(ts[1], errors='coerce')
            .fillna(0)
            .astype(int)
        )

        time_minutes = full['Hour'] * 60 + full['Minute']
        full['time_sin'] = np.sin(2 * np.pi * time_minutes / 1440)
        full['time_cos'] = np.cos(2 * np.pi * time_minutes / 1440)

        # Rush-hour and overnight indicators
        full['is_morning_peak'] = (
            (full['Hour'] >= 7) & (full['Hour'] <= 9)
        ).astype(int)
        full['is_evening_peak'] = (
            (full['Hour'] >= 17) & (full['Hour'] <= 19)
        ).astype(int)
        full['is_dead_of_night'] = (
            (full['Hour'] >= 1) & (full['Hour'] <= 4)
        ).astype(int)

        full.drop(columns=['timestamp'], inplace=True)

    if 'day' in full.columns:
        full['day_of_week'] = full['day'] % 7
        full['is_weekend'] = (full['day_of_week'] >= 5).astype(int)

    print("Processing categorical features...")

    # Categorical features for LightGBM, XGBoost, and CatBoost
    cat_cols = [
        'geohash',
        'geohash_macro',
        'geohash_micro',
        'RoadType',
        'Weather',
        'LargeVehicles',
        'Landmarks'
    ]

    # Preserve string representations for CatBoost
    for col in cat_cols:
        if col in full.columns:
            full[f"{col}_str"] = full[col].astype(str)
            full[col] = full[col].astype('category')

    # Separate training and test datasets
    X = full[full['is_train'] == 1].drop(
        columns=['is_train', 'Index', 'demand'],
        errors='ignore'
    )
    y = full[full['is_train'] == 1]['demand']
    X_test = full[full['is_train'] == 0].drop(
        columns=['is_train', 'Index', 'demand'],
        errors='ignore'
    )

    return X, y, X_test, cat_cols


X, y, X_test, base_cat_cols = preprocess_data(train, test)

# Prepare feature sets for LightGBM/XGBoost and CatBoost
lgb_xgb_features = [c for c in X.columns if not c.endswith('_str')]
cb_features = [
    c for c in X.columns
    if c.endswith('_str') or X[c].dtype != 'category'
]
cb_cat_features = [c for c in cb_features if c.endswith('_str')]


# ---------------------------------------------------------
# 3. Model Training and Cross-Validation
# ---------------------------------------------------------
print("Training ensemble models (LightGBM, XGBoost, and CatBoost)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cb_oof = np.zeros(len(X))

lgb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))
cb_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")

    # 3.1 LightGBM
    print("Training LightGBM...")

    model_lgb = lgb.LGBMRegressor(
        n_estimators=3500,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model_lgb.fit(
        X.iloc[train_idx][lgb_xgb_features],
        y.iloc[train_idx],
        eval_set=[
            (
                X.iloc[val_idx][lgb_xgb_features],
                y.iloc[val_idx]
            )
        ],
        categorical_feature=base_cat_cols,
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )

    lgb_oof[val_idx] = model_lgb.predict(
        X.iloc[val_idx][lgb_xgb_features]
    )
    lgb_test += (
        model_lgb.predict(X_test[lgb_xgb_features]) / kf.n_splits
    )

    # 3.2 XGBoost
    print("Training XGBoost (histogram tree method)...")

    model_xgb = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=150
    )

    model_xgb.fit(
        X.iloc[train_idx][lgb_xgb_features],
        y.iloc[train_idx],
        eval_set=[
            (
                X.iloc[val_idx][lgb_xgb_features],
                y.iloc[val_idx]
            )
        ],
        verbose=False
    )

    xgb_oof[val_idx] = model_xgb.predict(
        X.iloc[val_idx][lgb_xgb_features]
    )
    xgb_test += (
        model_xgb.predict(X_test[lgb_xgb_features]) / kf.n_splits
    )

    # 3.3 CatBoost
    print("Training CatBoost...")

    train_pool = Pool(
        X.iloc[train_idx][cb_features],
        y.iloc[train_idx],
        cat_features=cb_cat_features
    )
    val_pool = Pool(
        X.iloc[val_idx][cb_features],
        y.iloc[val_idx],
        cat_features=cb_cat_features
    )
    test_pool = Pool(
        X_test[cb_features],
        cat_features=cb_cat_features
    )

    model_cb = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        eval_metric='RMSE',
        random_seed=42,
        task_type="CPU",
        early_stopping_rounds=150,
        verbose=False
    )

    model_cb.fit(train_pool, eval_set=val_pool)

    cb_oof[val_idx] = model_cb.predict(val_pool)
    cb_test += (
        model_cb.predict(test_pool) / kf.n_splits
    )


# ---------------------------------------------------------
# 4. Model Evaluation and Ensemble Predictions
# ---------------------------------------------------------
print(f"\n[Validation] LightGBM R2: {r2_score(y, lgb_oof):.5f}")
print(f"[Validation] XGBoost R2:  {r2_score(y, xgb_oof):.5f}")
print(f"[Validation] CatBoost R2: {r2_score(y, cb_oof):.5f}")

# Standard weighted ensemble
standard_oof = (
    (0.4 * lgb_oof)
    + (0.3 * xgb_oof)
    + (0.3 * cb_oof)
)
print(f"[Standard Ensemble] R2: {r2_score(y, standard_oof):.5f}")

# Power averaging
power = 2
power_oof = (
    (lgb_oof ** power) * 0.4
    + (xgb_oof ** power) * 0.3
    + (cb_oof ** power) * 0.3
) ** (1 / power)

print(f"[Power Average] R2: {r2_score(y, power_oof):.5f}")


# ---------------------------------------------------------
# 5. Submission File Generation
# ---------------------------------------------------------
# Generate final test predictions using power averaging
final_test_preds = (
    (lgb_test ** power) * 0.4
    + (xgb_test ** power) * 0.3
    + (cb_test ** power) * 0.3
) ** (1 / power)

submission = pd.DataFrame({
    'Index': test_idx,
    'demand': final_test_preds
})

submission.to_csv('submission.csv', index=False)

print("\nSubmission file saved as 'submission.csv'.")
