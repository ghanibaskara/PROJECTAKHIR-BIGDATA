"""
spark_pipeline.py
=================
Processing & Feature Engineering Layer for Agricultural Commodity Price Analytics.

Responsibilities:
    1. SparkSession initialisation (local + HDFS-aware configuration).
    2. Data Cleansing — null removal, deduplication, outlier filtering.
    3. Feature Engineering — PySpark Window time-lagging (lag-7, lag-14, lag-30)
       and 7-day rolling average.
    4. VectorAssembler transformation for Spark MLlib.

Author: Agricultural Analytics Team
Version: 1.0.0
"""

import logging
import os
import sys
from datetime import date, timedelta
from typing import List, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SparkPipeline")

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

# Outlier thresholds (domain knowledge: Indian agricultural price range ₹)
MIN_PRICE_THRESHOLD =    1.0   # prices below ₹1 are considered erroneous
MAX_PRICE_THRESHOLD = 9_999_999.0  # extreme upper bound


# ---------------------------------------------------------------------------
# SparkSession Factory
# ---------------------------------------------------------------------------

def create_spark_session(
    app_name: str = "AgriPriceAnalytics",
    master: str = "local[*]",
    executor_memory: str = "2g",
    driver_memory: str = "2g",
    enable_hive: bool = False,
) -> SparkSession:
    """
    Creates and returns a configured SparkSession.

    Configuration is tuned for a Windows local development environment
    while remaining compatible with a YARN/standalone cluster when deployed.

    Parameters
    ----------
    app_name : str
        Spark UI application name.
    master : str
        Spark master URL. Use 'local[*]' for local mode.
    executor_memory : str
        Memory per executor (e.g., '2g', '4g').
    driver_memory : str
        Driver memory allocation.
    enable_hive : bool
        Enable HiveContext (requires Hive metastore; default False).

    Returns
    -------
    SparkSession
        Fully configured SparkSession.
    """
    logger.info("Initialising SparkSession '%s' (master=%s) …", app_name, master)

    python_executable = os.environ.get("PYSPARK_PYTHON", sys.executable)
    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        # Memory settings
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory",   driver_memory)
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", python_executable)
        # Serialisation & performance
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.shuffle.partitions", "8")   # reduce for local dev
        .config("spark.default.parallelism",    "8")
        # Parquet optimisations
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.parquet.mergeSchema",       "false")
        .config("spark.sql.parquet.filterPushdown",    "true")
        # Windows-specific: avoid IPv6 issues
        .config("spark.driver.host", "localhost")
        # Adaptive Query Execution (Spark 3.x)
        .config("spark.sql.adaptive.enabled",        "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    if enable_hive:
        builder = builder.enableHiveSupport()

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")  # reduce console noise

    logger.info(
        "✔ SparkSession ready. Version: %s | App: %s",
        spark.version, spark.sparkContext.appName,
    )
    return spark


# ---------------------------------------------------------------------------
# Data Cleansing
# ---------------------------------------------------------------------------

def cleanse_data(df: DataFrame) -> DataFrame:
    """
    Applies a multi-stage data cleansing pipeline to the raw DataFrame.

    Stages
    ------
    1. **Null removal** — drops rows where essential columns are null.
    2. **Deduplication** — removes exact duplicate rows.
    3. **Outlier filtering** — removes rows with physically implausible prices.
    4. **Negative price guard** — ensures all price columns are non-negative.

    Parameters
    ----------
    df : DataFrame
        Raw or lightly-typed Spark DataFrame.

    Returns
    -------
    DataFrame
        Cleansed DataFrame ready for feature engineering.
    """
    essential_cols = ["State", "Market", "Commodity", "Arrival_Date",
                      "Min_Price", "Max_Price", "Modal_Price"]

    initial_count = df.count()
    logger.info("Cleansing started. Initial row count: %s", f"{initial_count:,}")

    # ── Stage 1: Drop rows with null essentials ──────────────────────────
    df = df.dropna(subset=essential_cols)
    after_null = df.count()
    logger.info("After null drop: %s rows (removed %s)", f"{after_null:,}", f"{initial_count - after_null:,}")

    # ── Stage 2: Deduplicate ─────────────────────────────────────────────
    df = df.dropDuplicates(subset=essential_cols)
    after_dedup = df.count()
    logger.info("After deduplication: %s rows (removed %s)", f"{after_dedup:,}", f"{after_null - after_dedup:,}")

    # ── Stage 3: Outlier filter on Modal_Price ───────────────────────────
    df = df.filter(
        (F.col("Modal_Price") >= MIN_PRICE_THRESHOLD) &
        (F.col("Modal_Price") <= MAX_PRICE_THRESHOLD)
    )

    # ── Stage 4: Price consistency guard (Min ≤ Modal ≤ Max) ────────────
    df = df.filter(
        (F.col("Min_Price") <= F.col("Modal_Price")) &
        (F.col("Modal_Price") <= F.col("Max_Price"))
    )

    after_outlier = df.count()
    logger.info(
        "After outlier filtering: %s rows (removed %s)",
        f"{after_outlier:,}",
        f"{after_dedup - after_outlier:,}",
    )
    logger.info("✔ Cleansing complete. Final row count: %s", f"{after_outlier:,}")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering — Window Time-Lagging
# ---------------------------------------------------------------------------

def aggregate_daily_prices(df: DataFrame) -> DataFrame:
    """Create one deterministic daily observation per State/Market/Commodity."""
    group_cols = ["State", "Market", "Commodity", "Arrival_Date"]
    aggregations = [
        F.avg("Min_Price").alias("Min_Price"),
        F.avg("Max_Price").alias("Max_Price"),
        F.avg("Modal_Price").alias("Modal_Price"),
    ]
    for column_name in ("District", "Variety", "Grade", "Commodity_Code"):
        if column_name in df.columns:
            aggregations.append(
                F.first(column_name, ignorenulls=True).alias(column_name)
            )
    return df.groupBy(*group_cols).agg(*aggregations)

def engineer_features(df: DataFrame) -> DataFrame:
    """
    Applies PySpark Window-based time-lagging and rolling average features.

    Window Specification
    --------------------
    - Partitioned by: ``State``, ``Market``, ``Commodity``
    - Ordered by:     ``Arrival_Date`` (ascending)

    Engineered Features
    -------------------
    +----------------------+----------------------------------------------------+
    | Column               | Description                                        |
    +======================+====================================================+
    | Modal_Price_lag_7    | Modal_Price 7 days prior (t-7)                     |
    | Modal_Price_lag_14   | Modal_Price 14 days prior (t-14)                   |
    | Modal_Price_lag_30   | Modal_Price 30 days prior (t-30)                   |
    | Price_Rolling_Avg_7  | 7-day rolling mean of Modal_Price (t-6 to t)       |
    +----------------------+----------------------------------------------------+

    After computing lag features, rows where any lag feature is null (i.e.,
    the first 30 trading days per group) are dropped via ``df.na.drop()``.

    Parameters
    ----------
    df : DataFrame
        Cleansed DataFrame containing at minimum:
        State, Market, Commodity, Arrival_Date, Modal_Price.

    Returns
    -------
    DataFrame
        DataFrame enriched with lag and rolling-average columns,
        with null-lag rows removed.
    """
    logger.info("Engineering time-lagging features …")

    # ── Window Specification ─────────────────────────────────────────────
    windowSpec = (
        Window
        .partitionBy("State", "Market", "Commodity")
        .orderBy("Arrival_Date")
    )

    # Each row is one calendar day after daily aggregation.
    rolling7 = windowSpec.rowsBetween(-6, 0)
    rolling30 = windowSpec.rowsBetween(-29, 0)

    # ── Apply Lag Features ───────────────────────────────────────────────
    df = (
        df
        .withColumn("Modal_Price_lag_1",   F.lag("Modal_Price", 1).over(windowSpec))
        .withColumn("Modal_Price_lag_7",   F.lag("Modal_Price",  7).over(windowSpec))
        .withColumn("Modal_Price_lag_14",  F.lag("Modal_Price", 14).over(windowSpec))
        .withColumn("Modal_Price_lag_30",  F.lag("Modal_Price", 30).over(windowSpec))
        .withColumn("Price_Rolling_Avg_7", F.avg("Modal_Price").over(rolling7))
        .withColumn("Price_Rolling_Avg_30", F.avg("Modal_Price").over(rolling30))
        .withColumn(
            "Price_Change_7D",
            (F.col("Modal_Price") - F.col("Modal_Price_lag_7"))
            / F.when(F.col("Modal_Price_lag_7") != 0, F.col("Modal_Price_lag_7")),
        )
        .withColumn(
            "Price_Change_30D",
            (F.col("Modal_Price") - F.col("Modal_Price_lag_30"))
            / F.when(F.col("Modal_Price_lag_30") != 0, F.col("Modal_Price_lag_30")),
        )
        .withColumn("Price_Volatility_7D", F.stddev("Modal_Price").over(rolling7))
        .withColumn("Price_Volatility_30D", F.stddev("Modal_Price").over(rolling30))
    )

    # ── Drop Rows With Null Lag Features ─────────────────────────────────
    # This removes the initial window of each (State, Market, Commodity) group
    # where insufficient history exists to compute all lag features.
    feature_cols = FEATURE_COLS
    df = df.na.drop(subset=feature_cols)

    count_after = df.count()
    logger.info("✔ Feature engineering complete. Rows after lag-null drop: %s", f"{count_after:,}")
    return df


# ---------------------------------------------------------------------------
# VectorAssembler — MLlib Input Preparation
# ---------------------------------------------------------------------------

def assemble_features(
    df: DataFrame,
    feature_cols: Optional[List[str]] = None,
    output_col: str = FEATURES_VEC,
) -> Tuple[DataFrame, VectorAssembler]:
    """
    Transforms selected feature columns into a single ``features`` vector
    using Spark MLlib's VectorAssembler.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with all required feature columns present.
    feature_cols : list of str, optional
        Feature column names to include (default: FEATURE_COLS constant).
    output_col : str
        Name of the assembled feature vector column.

    Returns
    -------
    tuple (DataFrame, VectorAssembler)
        - Transformed DataFrame with the ``features`` vector column appended.
        - The fitted VectorAssembler (useful for later inference).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    logger.info("Assembling feature vector from columns: %s", feature_cols)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=output_col,
        handleInvalid="skip",  # skip rows with remaining nulls
    )

    df_assembled = assembler.transform(df)
    logger.info("✔ Features assembled into column '%s'.", output_col)
    return df_assembled, assembler


# ---------------------------------------------------------------------------
# Time-Series Train / Test Split
# ---------------------------------------------------------------------------

def time_series_split(
    df: DataFrame,
    test_year: Optional[int] = 2026,
    date_col: str = "Arrival_Date",
) -> Tuple[DataFrame, DataFrame]:
    """
    Splits the dataset into training and test sets based on calendar year.

    Rationale
    ---------
    Using random splits (``randomSplit``) on time-series data causes
    **data leakage**: future values can appear in the training set, inflating
    model performance metrics. A strict temporal split ensures the model
    is evaluated on truly unseen future data.

    Split Rule
    ----------
    - **Train**: dates before the requested ``test_year`` cutoff
    - **Test**: dates on or after the requested ``test_year`` cutoff

    If the requested cutoff leaves either side empty for the selected scope,
    the split falls back to an 80/20 chronological split over that scope's
    available date range. This is important for detail filters whose data
    starts in 2026 or has no observations in 2026. The fallback is still
    strictly temporal and never uses ``randomSplit``.

    Parameters
    ----------
    df : DataFrame
        Feature-engineered DataFrame (post VectorAssembler).
    test_year : int, optional
        Year marking the start of the test set (default: 2026). Pass ``None``
        to use the adaptive chronological split directly.
    date_col : str
        Column name of the date field.

    Returns
    -------
    tuple (DataFrame, DataFrame)
        (train_df, test_df)
    """
    bounds = df.select(
        F.min(F.col(date_col)).alias("date_min"),
        F.max(F.col(date_col)).alias("date_max"),
    ).first()
    if bounds is None or bounds["date_min"] is None or bounds["date_max"] is None:
        raise ValueError("Time-series split cannot run because the date range is empty.")

    date_min = bounds["date_min"]
    date_max = bounds["date_max"]
    requested_cutoff = date(test_year, 1, 1) if test_year is not None else None

    def split_at(cutoff_date: date) -> Tuple[DataFrame, DataFrame, int, int]:
        candidate_train = df.filter(F.col(date_col) < F.lit(cutoff_date))
        candidate_test = df.filter(F.col(date_col) >= F.lit(cutoff_date))
        return (
            candidate_train,
            candidate_test,
            candidate_train.count(),
            candidate_test.count(),
        )

    cutoff = requested_cutoff
    train_df = test_df = None
    train_count = test_count = 0
    if cutoff is not None:
        train_df, test_df, train_count, test_count = split_at(cutoff)

    if train_count == 0 or test_count == 0:
        requested_train_count = train_count
        requested_test_count = test_count
        if date_min == date_max:
            raise ValueError(
                "Time-series split cannot create train and test sets because "
                "the selected filter contains only one date."
            )
        span_days = (date_max - date_min).days
        adaptive_cutoff = date_min + timedelta(days=max(1, int(span_days * 0.8)))
        train_df, test_df, train_count, test_count = split_at(adaptive_cutoff)
        cutoff = adaptive_cutoff
        logger.warning(
            "Requested cutoff %s was unusable for the selected scope "
            "(train=%d, test=%d). Using adaptive chronological cutoff %s "
            "over %s to %s.",
            requested_cutoff or "none",
            requested_train_count,
            requested_test_count,
            cutoff,
            date_min,
            date_max,
        )

    logger.info("Time-series split: train < %s | test >= %s", cutoff, cutoff)

    logger.info(
        "✔ Split complete. Train: %s rows | Test: %s rows (%.1f%% test)",
        f"{train_count:,}",
        f"{test_count:,}",
        100 * test_count / max(train_count + test_count, 1),
    )
    if train_count == 0:
        raise ValueError(
            f"Time-series split produced no training rows before {cutoff}. "
            "The selected filter does not contain enough historical dates."
        )
    if test_count == 0:
        raise ValueError(
            f"Time-series split produced no test rows on or after {cutoff}. "
            "The selected filter does not contain enough dates for evaluation."
        )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Filter by Scope (State / Market / Commodity)
# ---------------------------------------------------------------------------

def filter_scope(
    df: DataFrame,
    state: Optional[str] = None,
    market: Optional[str] = None,
    commodity: Optional[str] = None,
) -> DataFrame:
    """
    Filters the DataFrame to a specific geographic / commodity scope.

    Parameters
    ----------
    df : DataFrame
        Full dataset.
    state : str, optional
        State name filter (case-sensitive).
    market : str, optional
        Market name filter.
    commodity : str, optional
        Commodity name filter.

    Returns
    -------
    DataFrame
        Filtered DataFrame. Returns the full DataFrame if all args are None.
    """
    if state:
        df = df.filter(F.col("State") == state)
    if market:
        df = df.filter(F.col("Market") == market)
    if commodity:
        df = df.filter(F.col("Commodity") == commodity)
    return df


# ---------------------------------------------------------------------------
# Full Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_full_pipeline(
    spark: SparkSession,
    parquet_path: str,
    state: Optional[str] = None,
    market: Optional[str] = None,
    commodity: Optional[str] = None,
    test_year: int = 2026,
) -> Tuple[DataFrame, DataFrame, DataFrame, VectorAssembler]:
    """
    Orchestrates the complete processing pipeline end-to-end.

    Pipeline Steps
    --------------
    1. Load Parquet data.
    2. Optionally filter by scope (State / Market / Commodity).
    3. Cleanse data (nulls, duplicates, outliers).
    4. Engineer lag & rolling-average features.
    5. Assemble MLlib feature vector.
    6. Time-series train/test split.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    parquet_path : str
        Path to the ingested Parquet dataset.
    state, market, commodity : str, optional
        Scope filters applied before processing.
    test_year : int
        Year used as the train/test cutoff boundary.

    Returns
    -------
    tuple (DataFrame, DataFrame, DataFrame, VectorAssembler)
        (full_featured_df, train_df, test_df, assembler)
    """
    logger.info("=== Starting Full Pipeline ===")

    # Step 1 — Load
    from hdfs_manager import to_spark_path

    df = spark.read.parquet(to_spark_path(parquet_path, require_exists=True))

    # Step 2 — Scope filter
    df = filter_scope(df, state=state, market=market, commodity=commodity)

    # Step 3 — Cleanse
    df = cleanse_data(df)

    # Step 4 — One observation per series and calendar day
    df = aggregate_daily_prices(df)

    # Step 5 — Feature engineering
    df = engineer_features(df)

    # Step 6 — Vector assembly
    df, assembler = assemble_features(df)

    # Keep the large feature DataFrame off the Java heap. The ingested dataset
    # can contain tens of millions of rows, so MEMORY_ONLY caching causes an
    # OutOfMemoryError while Spark materialises cached column batches.
    df.persist(StorageLevel.DISK_ONLY)

    # Step 7 — Temporal split
    train_df, test_df = time_series_split(df, test_year=test_year)

    logger.info("=== Pipeline Complete ===")
    return df, train_df, test_df, assembler
