"""
ml_model.py
===========
Machine Learning & Analytics Layer for Agricultural Commodity Price Analytics.

Responsibilities:
    1. Train a distributed RandomForestRegressor (Spark MLlib).
    2. Evaluate model performance (RMSE, MAE, R² Score).
    3. Extract and rank feature importances (Business Intelligence).
    4. Perform recursive 30-day price forecasting.

Author: Agricultural Analytics Team
Version: 1.0.0
"""

import logging
import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MLModel")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Modal_Price_lag_1",
    "Modal_Price_lag_7",
    "Modal_Price_lag_14",
    "Modal_Price_lag_30",
    "Price_Rolling_Avg_7",
    "Price_Rolling_Avg_30",
    "Price_Change_7D",
    "Price_Change_30D",
    "Price_Volatility_7D",
    "Price_Volatility_30D",
]
TARGET_COL   = "Modal_Price"
FEATURES_VEC = "features"


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_model(
    train_df: DataFrame,
    num_trees: int = 100,
    max_depth: int = 10,
    seed: int = 42,
    features_col: str = FEATURES_VEC,
    label_col: str = TARGET_COL,
) -> RandomForestRegressionModel:
    """
    Trains a distributed Random Forest Regressor using Spark MLlib.

    The Random Forest algorithm is chosen because it:
    - Handles non-linear relationships between lagged price features.
    - Is robust to outliers remaining after cleansing.
    - Natively provides feature importance scores for Business Intelligence.
    - Scales across distributed Spark executors.

    Parameters
    ----------
    train_df : DataFrame
        Training set with ``features`` vector and ``Modal_Price`` label.
    num_trees : int
        Number of decision trees in the ensemble (default: 100).
    max_depth : int
        Maximum depth of each decision tree (default: 10).
    seed : int
        Random seed for reproducibility.
    features_col : str
        Name of the assembled feature vector column.
    label_col : str
        Name of the target/label column.

    Returns
    -------
    RandomForestRegressionModel
        Trained model ready for evaluation and inference.
    """
    logger.info(
        "Training RandomForestRegressor | trees=%d | max_depth=%d | seed=%d",
        num_trees, max_depth, seed,
    )

    rf = RandomForestRegressor(
        featuresCol=features_col,
        labelCol=label_col,
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=seed,
        subsamplingRate=0.8,        # 80% row sampling per tree (bagging)
        featureSubsetStrategy="auto",  # sqrt(n_features) for regression
    )

    model = rf.fit(train_df)
    logger.info("✔ Model training complete. Trees trained: %d", len(model.trees))
    return model


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: RandomForestRegressionModel,
    test_df: DataFrame,
    features_col: str = FEATURES_VEC,
    label_col: str = TARGET_COL,
) -> Dict[str, float]:
    """
    Evaluates the trained model on the test set and computes key metrics.

    Metrics Computed
    ----------------
    - **RMSE** (Root Mean Squared Error): Penalises large errors heavily.
      Lower is better. Unit: same as Modal_Price (₹).
    - **MAE**  (Mean Absolute Error): Average absolute deviation.
      Lower is better. More interpretable than RMSE.
    - **R²**   (Coefficient of Determination): Proportion of variance explained.
      Range: (−∞, 1]. Closer to 1.0 is better.

    Parameters
    ----------
    model : RandomForestRegressionModel
        Trained Spark ML model.
    test_df : DataFrame
        Held-out test set (data from 2026 onward).
    features_col : str
        Feature vector column name.
    label_col : str
        Ground-truth label column name.

    Returns
    -------
    dict
        Keys: ``rmse``, ``mae``, ``r2``. Values are Python floats.
    """
    logger.info("Evaluating model on test set …")

    test_count = test_df.limit(1).count()
    if test_count == 0:
        raise ValueError(
            "Cannot evaluate the model because the test set is empty. "
            "Choose a broader scope or a test year with available data."
        )

    predictions = model.transform(test_df)

    evaluators = {
        "rmse": RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse"),
        "mae":  RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae"),
        "r2":   RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2"),
    }

    metrics = {name: float(evaluator.evaluate(predictions)) for name, evaluator in evaluators.items()}

    logger.info(
        "✔ Evaluation — RMSE: %.2f | MAE: %.2f | R²: %.4f",
        metrics["rmse"], metrics["mae"], metrics["r2"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Feature Importance (Business Intelligence)
# ---------------------------------------------------------------------------

def get_feature_importances(
    model: RandomForestRegressionModel,
    feature_cols: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """
    Extracts and ranks feature importances from the trained Random Forest.

    Feature importance in a Random Forest is the mean decrease in impurity
    (variance for regression) across all trees, normalised to sum to 1.0.

    Business Interpretation
    -----------------------
    The relative importance of ``Modal_Price_lag_7`` vs ``Modal_Price_lag_14``
    vs ``Modal_Price_lag_30`` answers the core Business Intelligence question:

        "How strongly do past price signals (7, 14, 30 days ago) influence
         today's modal price?"

    Parameters
    ----------
    model : RandomForestRegressionModel
        Trained model with ``featureImportances`` attribute.
    feature_cols : list of str, optional
        Ordered list of feature names matching the VectorAssembler order.

    Returns
    -------
    list of dict
        Sorted (descending) list of:
        ``{"feature": str, "importance": float, "importance_pct": float}``
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    raw_importances = model.featureImportances.toArray()

    importances = [
        {
            "feature":        name,
            "importance":     float(score),
            "importance_pct": round(float(score) * 100, 2),
        }
        for name, score in zip(feature_cols, raw_importances)
    ]

    # Sort descending by importance
    importances.sort(key=lambda x: x["importance"], reverse=True)

    logger.info("Feature importances (top 3):")
    for entry in importances[:3]:
        logger.info("  %-30s %.2f%%", entry["feature"], entry["importance_pct"])

    return importances


# ---------------------------------------------------------------------------
# Recursive 30-Day Forecasting
# ---------------------------------------------------------------------------

def recursive_forecast(
    model: RandomForestRegressionModel,
    spark: SparkSession,
    seed_data: pd.DataFrame,
    forecast_days: int = 30,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generates a recursive multi-step price forecast for the next N days.

    Methodology
    -----------
    This function uses a **recursive (auto-regressive) forecasting** strategy:
    for each future day ``t``, the model predicts ``Modal_Price_t``, and this
    predicted value is then fed back as a lag feature for day ``t+1``.

    Steps
    -----
    1. Start with the most recent N days of actual data (``seed_data``).
    2. For each future day:
       a. Compute lag-7, lag-14, lag-30 from the rolling price history.
       b. Compute the 7-day rolling average from price history.
       c. Estimate Min_Price and Max_Price using recent price spread ratios.
       d. Run model inference via Spark.
       e. Append predicted price to the history buffer.
    3. Return all 30 forecast records.

    Parameters
    ----------
    model : RandomForestRegressionModel
        Trained Spark ML model.
    spark : SparkSession
        Active Spark session (used for distributed inference).
    seed_data : pd.DataFrame
        Pandas DataFrame of the most recent actual records.
        Must contain: Modal_Price and Arrival_Date. Min_Price and Max_Price
        are optional and are used only to estimate the forecast band.
    forecast_days : int
        Number of future days to project (default: 30).
    feature_cols : list of str, optional
        Feature column names in assembler order.

    Returns
    -------
    pd.DataFrame
        Forecast results with columns:
        ``Day``, ``Forecast_Date``, ``Predicted_Modal_Price``,
        ``Predicted_Min_Price``, ``Predicted_Max_Price``, ``Is_Forecast``.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if seed_data.empty:
        raise ValueError("seed_data must contain at least one historical row.")

    required_seed_cols = {"Arrival_Date", "Modal_Price"}
    missing_cols = required_seed_cols.difference(seed_data.columns)
    if missing_cols:
        raise ValueError(
            "seed_data is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    logger.info("Starting recursive %d-day forecast …", forecast_days)

    # Build rolling price history from seed data
    seed_data = seed_data.sort_values("Arrival_Date").reset_index(drop=True)
    price_history: List[float] = list(seed_data["Modal_Price"].values)

    # Compute price spread ratio for Min/Max estimation
    if {"Min_Price", "Max_Price"}.issubset(seed_data.columns):
        price_spread = float(
            (seed_data["Max_Price"] - seed_data["Min_Price"]).mean()
        )
    else:
        price_spread = float(seed_data["Modal_Price"].std() or 0.0)
    last_date    = pd.to_datetime(seed_data["Arrival_Date"].max())

    # VectorAssembler for single-row inference
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=FEATURES_VEC,
        handleInvalid="skip",
    )

    forecast_records: List[Dict] = []

    for day_offset in range(1, forecast_days + 1):
        n = len(price_history)

        # Guard: need at least 30 days of history
        if n < 30:
            logger.warning("Insufficient history (%d rows) for lag-30. Padding.", n)
            price_history = [price_history[0]] * (30 - n) + price_history
            n = len(price_history)

        # ── Compute lag features ─────────────────────────────────────
        lag_7  = price_history[-7]
        lag_1  = price_history[-1]
        lag_14 = price_history[-14]
        lag_30 = price_history[-30]
        avg_7  = float(np.mean(price_history[-7:]))
        avg_30 = float(np.mean(price_history[-30:]))
        std_7 = float(np.std(price_history[-7:]))
        std_30 = float(np.std(price_history[-30:]))
        change_7 = (lag_1 - lag_7) / lag_7 if lag_7 else 0.0
        change_30 = (lag_1 - lag_30) / lag_30 if lag_30 else 0.0

        # Estimated Min/Max from average spread
        pred_min = max(0.0, lag_7 - price_spread / 2)
        pred_max = lag_7 + price_spread / 2

        # ── Build single-row Spark DataFrame ─────────────────────────
        feature_row = {
            "Modal_Price_lag_1":   float(lag_1),
            "Modal_Price_lag_7":   float(lag_7),
            "Modal_Price_lag_14":  float(lag_14),
            "Modal_Price_lag_30":  float(lag_30),
            "Price_Rolling_Avg_7": float(avg_7),
            "Price_Rolling_Avg_30": float(avg_30),
            "Price_Change_7D":      float(change_7),
            "Price_Change_30D":     float(change_30),
            "Price_Volatility_7D":  float(std_7),
            "Price_Volatility_30D": float(std_30),
        }

        row_df  = spark.createDataFrame([feature_row])
        row_vec = assembler.transform(row_df)

        # ── Run inference ─────────────────────────────────────────────
        pred_price = float(
            model.transform(row_vec).select("prediction").collect()[0]["prediction"]
        )

        # Append to history for next iteration
        price_history.append(pred_price)

        forecast_date = last_date + timedelta(days=day_offset)
        forecast_records.append(
            {
                "Day":                    day_offset,
                "Forecast_Date":          forecast_date.date(),
                "Predicted_Modal_Price":  round(pred_price, 2),
                "Predicted_Min_Price":    round(pred_min, 2),
                "Predicted_Max_Price":    round(pred_max, 2),
                "Is_Forecast":            True,
            }
        )

        logger.debug("Day %2d (%s): ₹%.2f", day_offset, forecast_date.date(), pred_price)

    logger.info(
        "✔ Forecast complete. Day-1: ₹%.2f | Day-30: ₹%.2f",
        forecast_records[0]["Predicted_Modal_Price"],
        forecast_records[-1]["Predicted_Modal_Price"],
    )
    return pd.DataFrame(forecast_records)


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_model(model: RandomForestRegressionModel, path: str) -> None:
    """
    Saves the trained model to disk (local or HDFS path).

    Parameters
    ----------
    model : RandomForestRegressionModel
        Trained model to persist.
    path : str
        Destination path. Supports local paths and HDFS URIs.
    """
    model.write().overwrite().save(path)
    logger.info("✔ Model saved to %s", path)


def load_model(path: str) -> RandomForestRegressionModel:
    """
    Loads a previously saved RandomForestRegressionModel.

    Parameters
    ----------
    path : str
        Path to the saved model directory.

    Returns
    -------
    RandomForestRegressionModel
    """
    model = RandomForestRegressionModel.load(path)
    logger.info("✔ Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# High-Level Training Orchestrator
# ---------------------------------------------------------------------------

def train_and_evaluate(
    spark: SparkSession,
    train_df: DataFrame,
    test_df: DataFrame,
    feature_cols: Optional[List[str]] = None,
    num_trees: int = 100,
    max_depth: int = 10,
    model_save_path: Optional[str] = None,
) -> Tuple[RandomForestRegressionModel, Dict[str, float], List[Dict]]:
    """
    Convenience orchestrator: trains, evaluates, and extracts importances.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    train_df : DataFrame
        Training set (with features vector).
    test_df : DataFrame
        Test set (with features vector).
    feature_cols : list of str, optional
        Feature names matching VectorAssembler order.
    num_trees : int
        Number of Random Forest trees.
    max_depth : int
        Maximum tree depth.
    model_save_path : str, optional
        If provided, saves the model to this path after training.

    Returns
    -------
    tuple (RandomForestRegressionModel, dict, list)
        (model, metrics_dict, feature_importances_list)
    """
    model      = train_model(train_df, num_trees=num_trees, max_depth=max_depth)
    metrics    = evaluate_model(model, test_df)
    importance = get_feature_importances(model, feature_cols=feature_cols)

    if model_save_path:
        save_model(model, model_save_path)

    return model, metrics, importance
