"""
hdfs_manager.py
===============
Storage & Fallback Layer for Agricultural Commodity Price Analytics System.

Manages connections to Hadoop HDFS. If HDFS daemons are offline or unavailable
(common in local Windows dev environments), automatically falls back to a local
directory storage mode using ./data/parquet/.

Author: Agricultural Analytics Team
Version: 1.0.0
"""

import os
import glob
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    DateType, IntegerType, LongType
)
from pyspark.errors.exceptions.captured import IllegalArgumentException
from py4j.protocol import Py4JJavaError

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HDFSManager")

# ---------------------------------------------------------------------------
# Constants & Default Paths
# ---------------------------------------------------------------------------
DEFAULT_DATASET_PATH = r"D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR\Dataset"
DEFAULT_LOCAL_PARQUET_PATH = "./data/parquet"
HDFS_NAMENODE_URL = os.getenv("AGRI_HDFS_NAMENODE", "hdfs://localhost:9000").rstrip("/")
DEFAULT_HDFS_RAW_PATH = f"{HDFS_NAMENODE_URL}/agri_data/raw"
DEFAULT_HDFS_PARQUET_PATH = f"{HDFS_NAMENODE_URL}/agri_data/parquet"
HDFS_CHECK_TIMEOUT = 5  # seconds

# Raw CSV schema
RAW_CSV_SCHEMA = StructType([
    StructField("State", StringType(), True),
    StructField("District", StringType(), True),
    StructField("Market", StringType(), True),
    StructField("Commodity", StringType(), True),
    StructField("Variety", StringType(), True),
    StructField("Grade", StringType(), True),
    StructField("Arrival_Date", StringType(), True),   # parsed later
    StructField("Min_Price", StringType(), True),       # cast to Double
    StructField("Max_Price", StringType(), True),       # cast to Double
    StructField("Modal_Price", StringType(), True),     # cast to Double
    StructField("Commodity_Code", StringType(), True),  # cast to Long
])


class HDFSManager:
    """
    Manages data storage for the Agricultural Analytics pipeline.

    Priority:
        1. Hadoop HDFS (distributed storage) — preferred when available.
        2. Local filesystem fallback (./data/parquet/) — used when HDFS
           daemons are not running (typical Windows local dev).

    Parameters
    ----------
    spark : SparkSession
        Active Spark session used for all I/O operations.
    hdfs_path : str, optional
        HDFS base URI (default: hdfs://localhost:9000/agri_data/parquet).
    local_fallback_path : str, optional
        Local directory used when HDFS is unavailable.
    """

    def __init__(
        self,
        spark: SparkSession,
        hdfs_path: str = DEFAULT_HDFS_PARQUET_PATH,
        local_fallback_path: str = DEFAULT_LOCAL_PARQUET_PATH,
    ) -> None:
        self.spark = spark
        self.hdfs_path = hdfs_path
        self.local_fallback_path = Path(local_fallback_path).resolve()
        self._hdfs_available: Optional[bool] = None  # lazy-checked

        # Ensure local fallback directory exists
        self.local_fallback_path.mkdir(parents=True, exist_ok=True)
        logger.info("HDFSManager initialised. Local fallback: %s", self.local_fallback_path)

    # ------------------------------------------------------------------
    # HDFS Availability Check
    # ------------------------------------------------------------------

    def is_hdfs_available(self) -> bool:
        """
        Checks whether the Hadoop HDFS NameNode is reachable.

        Uses Spark's Hadoop ``FileSystem`` API to probe the NameNode. This
        works when Hadoop is installed inside WSL and the Windows process
        does not have the ``hdfs`` CLI on its PATH. Result is cached for the
        lifetime of this instance.

        Returns
        -------
        bool
            True if HDFS is running and accessible, False otherwise.
        """
        if self._hdfs_available is not None:
            return self._hdfs_available

        logger.info("Probing HDFS availability at %s …", HDFS_NAMENODE_URL)
        try:
            hadoop_conf = self.spark.sparkContext._jsc.hadoopConfiguration()
            hadoop_path = self.spark._jvm.org.apache.hadoop.fs.Path(
                f"{HDFS_NAMENODE_URL}/"
            )
            filesystem = hadoop_path.getFileSystem(hadoop_conf)
            self._hdfs_available = bool(filesystem.exists(hadoop_path))
        except (OSError, RuntimeError, ValueError, IllegalArgumentException, Py4JJavaError):
            self._hdfs_available = False

        status = "AVAILABLE ✔" if self._hdfs_available else "UNAVAILABLE — using local fallback ✘"
        logger.info("HDFS status: %s", status)
        return self._hdfs_available

    def get_storage_path(self, sub_dir: str = "") -> str:
        """
        Returns the active storage path (HDFS or local fallback).

        Parameters
        ----------
        sub_dir : str, optional
            Sub-directory appended to the base path.

        Returns
        -------
        str
            Fully-qualified storage path.
        """
        if self.is_hdfs_available():
            base = self.hdfs_path.rstrip("/")
            return f"{base}/{sub_dir}".rstrip("/") if sub_dir else base
        else:
            base = self.local_fallback_path
            return str(base / sub_dir) if sub_dir else str(base)

    def get_storage_mode(self) -> str:
        """Returns a human-readable label for the active storage mode."""
        return "HDFS (Distributed)" if self.is_hdfs_available() else "Local Filesystem (Fallback)"

    # ------------------------------------------------------------------
    # Write Operations
    # ------------------------------------------------------------------

    def write_parquet(
        self,
        df: DataFrame,
        sub_dir: str = "commodities",
        partition_cols: Optional[List[str]] = None,
        mode: str = "overwrite",
    ) -> str:
        """
        Writes a Spark DataFrame as partitioned Parquet to the active storage.

        Parameters
        ----------
        df : DataFrame
            The Spark DataFrame to persist.
        sub_dir : str
            Sub-directory inside the base storage path.
        partition_cols : list of str, optional
            Columns to partition by (default: ["State"]).
        mode : str
            Spark write mode: 'overwrite', 'append', 'ignore', 'error'.

        Returns
        -------
        str
            The path where data was written.
        """
        if partition_cols is None:
            partition_cols = ["State"]

        output_path = self.get_storage_path(sub_dir)
        spark_output_path = to_spark_path(output_path)
        logger.info(
            "Writing Parquet → %s (mode=%s, partitionBy=%s)",
            output_path, mode, partition_cols,
        )

        writer = df.write.mode(mode).format("parquet")
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.save(spark_output_path)

        logger.info("✔ Parquet written successfully to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Read Operations
    # ------------------------------------------------------------------

    def read_parquet(self, sub_dir: str = "commodities") -> DataFrame:
        """
        Reads Parquet data from the active storage location.

        Parameters
        ----------
        sub_dir : str
            Sub-directory inside the base storage path.

        Returns
        -------
        DataFrame
            Loaded Spark DataFrame.

        Raises
        ------
        FileNotFoundError
            If the target path does not exist in local fallback mode.
        """
        path = self.get_storage_path(sub_dir)
        logger.info("Reading Parquet from %s …", path)

        # Validate local path existence
        if not self.is_hdfs_available():
            local_path = Path(path)
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Parquet data not found at {local_path}. "
                    "Run ingestion first via the Streamlit dashboard."
                )

        df = self.spark.read.parquet(to_spark_path(path, require_exists=True))
        logger.info("✔ Parquet loaded. Estimated partitions: %d", df.rdd.getNumPartitions())
        return df

    # ------------------------------------------------------------------
    # HDFS Directory Utilities
    # ------------------------------------------------------------------

    def create_hdfs_dir(self, hdfs_dir: str) -> bool:
        """
        Creates a directory on HDFS (no-op if HDFS is unavailable).

        Parameters
        ----------
        hdfs_dir : str
            HDFS directory path to create (e.g., '/agri_data/parquet').

        Returns
        -------
        bool
            True if directory was created or already exists.
        """
        if not self.is_hdfs_available():
            logger.warning("HDFS unavailable — skipping mkdir %s", hdfs_dir)
            return False
        try:
            result = subprocess.run(
                ["hdfs", "dfs", "-mkdir", "-p", hdfs_dir],
                capture_output=True, text=True, timeout=15,
            )
            success = result.returncode == 0
            logger.info("HDFS mkdir %s → %s", hdfs_dir, "OK" if success else result.stderr)
            return success
        except Exception as exc:
            logger.error("HDFS mkdir failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Standalone Utility Functions
# ---------------------------------------------------------------------------

def discover_csv_files(dataset_dir: str = DEFAULT_DATASET_PATH) -> List[str]:
    """
    Discovers all CSV files (2020.csv – 2026.csv) in the dataset directory.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing the annual CSV files.

    Returns
    -------
    list of str
        Sorted list of absolute CSV file paths found.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no CSV files.
    """
    dataset_dir_path = Path(dataset_dir)
    if not dataset_dir_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Please verify the path or place your CSV files there."
        )

    csv_files = sorted(glob.glob(str(dataset_dir_path / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {dataset_dir}. "
            "Expected files like 2020.csv, 2021.csv, …, 2026.csv"
        )

    logger.info("Discovered %d CSV file(s): %s", len(csv_files), [Path(f).name for f in csv_files])
    return csv_files


def discover_hdfs_csv_files(
    spark: SparkSession,
    hdfs_raw_path: str = DEFAULT_HDFS_RAW_PATH,
) -> List[str]:
    """List CSV files placed in HDFS by the Apache NiFi ingestion flow."""
    raw_path = to_spark_path(hdfs_raw_path, require_exists=True)
    jvm = spark._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(raw_path)
    filesystem = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    statuses = filesystem.listStatus(hadoop_path)
    csv_paths = sorted(
        status.getPath().toString()
        for status in statuses
        if status.isFile() and status.getPath().getName().lower().endswith(".csv")
    )
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in NiFi HDFS raw directory: {hdfs_raw_path}"
        )
    logger.info("Discovered %d NiFi HDFS CSV file(s): %s", len(csv_paths), csv_paths)
    return csv_paths


def convert_csv_to_parquet(
    spark: SparkSession,
    input_csv_paths: List[str],
    output_parquet_path: str,
    partition_col: str = "State",
) -> Tuple[str, int]:
    """
    Reads raw CSV files, applies type casting, and writes as partitioned Parquet.

    Type Casting Rules
    ------------------
    - ``Arrival_Date``   → DateType  (format: yyyy-MM-dd)
    - ``Min_Price``      → DoubleType
    - ``Max_Price``      → DoubleType
    - ``Modal_Price``    → DoubleType
    - ``Commodity_Code`` → LongType

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    input_csv_paths : list of str
        Absolute paths to the CSV input files.
    output_parquet_path : str
        Destination directory for the Parquet output (HDFS or local).
    partition_col : str
        Column to partition the Parquet dataset by (default: "State").

    Returns
    -------
    tuple (str, int)
        (output_parquet_path, total_row_count)

    Raises
    ------
    ValueError
        If the resulting DataFrame is empty after loading.
    """
    logger.info("Starting CSV → Parquet conversion for %d file(s) …", len(input_csv_paths))

    if not input_csv_paths:
        raise ValueError("No CSV input paths were provided.")

    # Spark resolves scheme-less paths against ``fs.defaultFS``. When HDFS is
    # configured as the default filesystem, a Windows path such as
    # ``D:\data\2024.csv`` would incorrectly become
    # ``hdfs://localhost:9000/D:/data/2024.csv``. Explicitly mark local files
    # with the file:// scheme while preserving explicit distributed URIs.
    spark_csv_paths = [_to_spark_input_path(path) for path in input_csv_paths]

    # ── 1. Load raw CSVs ────────────────────────────────────────────────
    spark_output_path = to_spark_path(output_parquet_path)
    df_raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")   # use our manual schema
        .option("multiLine", "false")
        .option("encoding", "UTF-8")
        .schema(RAW_CSV_SCHEMA)
        .csv(spark_csv_paths)
    )

    if df_raw.head(1) == []:
        raise ValueError("Loaded DataFrame is empty. Check CSV file paths and content.")

    # ── 2. Type Casting ─────────────────────────────────────────────────
    df_typed = (
        df_raw
        .withColumn(
            "Arrival_Date",
            F.to_date(F.col("Arrival_Date"), "yyyy-MM-dd"),
        )
        .withColumn("Min_Price",      F.col("Min_Price").cast(DoubleType()))
        .withColumn("Max_Price",      F.col("Max_Price").cast(DoubleType()))
        .withColumn("Modal_Price",    F.col("Modal_Price").cast(DoubleType()))
        .withColumn("Commodity_Code", F.col("Commodity_Code").cast(LongType()))
        # Strip extra whitespace from string columns
        .withColumn("State",     F.trim(F.col("State")))
        .withColumn("District",  F.trim(F.col("District")))
        .withColumn("Market",    F.trim(F.col("Market")))
        .withColumn("Commodity", F.trim(F.col("Commodity")))
        .withColumn("Variety",   F.trim(F.col("Variety")))
        .withColumn("Grade",     F.trim(F.col("Grade")))
    )

    # ── 3. Basic Filtering (drop rows with null target) ──────────────────
    df_clean = df_typed.filter(
        F.col("Arrival_Date").isNotNull() & F.col("Modal_Price").isNotNull()
    )

    logger.info("Writing Parquet (partitionBy='%s') → %s", partition_col, output_parquet_path)

    # ── 4. Write Parquet ────────────────────────────────────────────────
    df_clean.write.mode("overwrite").partitionBy(partition_col).parquet(spark_output_path)

    # ── 5. Count rows (triggers action) ─────────────────────────────────
    total_rows = df_clean.count()
    logger.info("✔ Conversion complete. Total rows written: %s", f"{total_rows:,}")

    return output_parquet_path, total_rows


def to_spark_path(path: str, require_exists: bool = False) -> str:
    """Return a Spark-safe URI for a local path or preserve an explicit URI."""
    # Do not use ``urlparse(path).scheme`` alone: on Windows it interprets
    # the drive letter in ``D:\data\file.csv`` as a URI scheme.
    if "://" in path or path.lower().startswith("file:"):
        return path

    local_path = Path(path)
    if require_exists and not local_path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    return local_path.resolve().as_uri()


def _to_spark_input_path(path: str) -> str:
    """Return a Spark-safe URI for an existing local CSV or explicit URI."""
    return to_spark_path(path, require_exists=True)


def get_dataset_summary(
    spark: SparkSession,
    parquet_path: str,
) -> Dict[str, Any]:
    """
    Computes summary metadata for the stored Parquet dataset.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    parquet_path : str
        Path to the Parquet data directory (HDFS or local).

    Returns
    -------
    dict
        Keys:
        - ``total_rows``       (int)   Total record count.
        - ``date_min``         (str)   Earliest Arrival_Date (YYYY-MM-DD).
        - ``date_max``         (str)   Latest Arrival_Date (YYYY-MM-DD).
        - ``unique_states``    (list)  Sorted list of unique State values.
        - ``unique_commodities`` (list) Sorted list of unique Commodity values.
        - ``unique_markets``   (int)   Count of unique Market values.
        - ``parquet_size_mb``  (float) Storage size in MB from HDFS or local files.
        - ``storage_path``     (str)   Path used to read data.
    """
    logger.info("Computing dataset summary from %s …", parquet_path)
    df = spark.read.parquet(to_spark_path(parquet_path, require_exists=True))

    # Aggregate statistics in a single pass
    stats = df.agg(
        F.count("*").alias("total_rows"),
        F.min("Arrival_Date").alias("date_min"),
        F.max("Arrival_Date").alias("date_max"),
        F.countDistinct("Market").alias("unique_markets"),
    ).collect()[0]

    states      = sorted([r["State"]     for r in df.select("State").distinct().collect()])
    commodities = sorted([r["Commodity"] for r in df.select("Commodity").distinct().collect()])

    parquet_size_mb = get_parquet_size_mb(spark, parquet_path)

    summary = {
        "total_rows":         int(stats["total_rows"]),
        "date_min":           str(stats["date_min"]),
        "date_max":           str(stats["date_max"]),
        "unique_states":      states,
        "unique_commodities": commodities,
        "unique_markets":     int(stats["unique_markets"]),
        "parquet_size_mb":    parquet_size_mb,
        "storage_path":       parquet_path,
    }

    logger.info(
        "Summary: %s rows | %s – %s | %d states | %d commodities",
        f"{summary['total_rows']:,}",
        summary["date_min"],
        summary["date_max"],
        len(states),
        len(commodities),
    )
    return summary


def get_local_parquet_size_mb(parquet_dir: str) -> float:
    """
    Calculates the total size of all Parquet files under a local directory.

    Parameters
    ----------
    parquet_dir : str
        Root directory of the Parquet dataset.

    Returns
    -------
    float
        Total size in megabytes, or 0.0 if the directory does not exist.
    """
    path = Path(parquet_dir)
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*.parquet") if f.is_file())
    return round(total_bytes / (1024 ** 2), 2)


def get_parquet_size_mb(spark: SparkSession, parquet_path: str) -> float:
    """Return Parquet size from HDFS or the local filesystem."""
    if parquet_path.startswith(("hdfs://", "viewfs://")):
        try:
            hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
            hadoop_path = spark._jvm.org.apache.hadoop.fs.Path(parquet_path)
            filesystem = hadoop_path.getFileSystem(hadoop_conf)
            size_bytes = filesystem.getContentSummary(hadoop_path).getLength()
            return round(float(size_bytes) / (1024 ** 2), 2)
        except (OSError, RuntimeError, ValueError, IllegalArgumentException, Py4JJavaError) as exc:
            logger.warning("Unable to calculate HDFS Parquet size for %s: %s", parquet_path, exc)
            return 0.0

    local_path = Path(parquet_path.replace("file:///", "")) if parquet_path.startswith("file:///") else Path(parquet_path)
    if not local_path.exists():
        return 0.0
    total_bytes = sum(
        file_path.stat().st_size
        for file_path in local_path.rglob("*.parquet")
        if file_path.is_file()
    )
    return round(total_bytes / (1024 ** 2), 2)
