"""
app.py
======
Streamlit Frontend Dashboard for Agricultural Commodity Price Analytics
& Lagged Volatility Predictor.

Views
-----
1. Ingestion View  — Upload/ingest CSV data, convert to Parquet, view summary.
2. Analytics View  — BI feature importance + 30-day price forecasting dashboard.

Run with:
    streamlit run app.py

Author: Agricultural Analytics Team
Version: 1.0.0
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriPrice Analytics | Big Data Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Default Paths
# ---------------------------------------------------------------------------
DEFAULT_DATASET_PATH  = r"D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR\Dataset"
DEFAULT_PARQUET_PATH  = "./data/parquet/commodities"
DEFAULT_NIFI_RAW_PATH = (
    f"{os.getenv('AGRI_HDFS_NAMENODE', 'hdfs://localhost:9000').rstrip('/')}"
    "/agri_data/raw"
)
DEFAULT_MODEL_PATH    = "./data/model/rf_agri"
PARQUET_PATH_STATE_KEY = "parquet_path"


# ---------------------------------------------------------------------------
# Lazy Imports (heavy PySpark imports deferred to avoid slow startup)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⚡ Initialising Spark Session…")
def get_spark():
    """Returns a cached SparkSession (singleton for the Streamlit process)."""
    from spark_pipeline import create_spark_session
    return create_spark_session(app_name="AgriPriceAnalytics_UI")


@st.cache_resource
def get_hdfs_manager(_spark):
    """Returns a cached HDFSManager instance."""
    from hdfs_manager import HDFSManager
    return HDFSManager(spark=_spark)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a472a 0%, #2d6a4f 100%); }
    [data-testid="stSidebar"] * { color: #f0f7f4 !important; }

    /* Metric cards */
    [data-testid="stMetric"] { background: #f0f7f4; border-radius: 10px; padding: 12px; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #1a472a !important; }

    /* Section headers */
    .section-header { color: #1a472a; border-bottom: 2px solid #2d6a4f; padding-bottom: 6px; margin-bottom: 16px; }

    /* Success banner */
    .success-banner { background: #d4edda; border-left: 4px solid #28a745; padding: 12px; border-radius: 6px; }

    /* Info banner */
    .info-banner { background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 12px; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Renders the sidebar and returns the selected navigation page."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/wheat.png",
            width=80,
        )
        st.title("🌾 AgriPrice\nAnalytics")
        st.caption("Big Data & ML Platform · v1.0")
        st.divider()

        page = st.radio(
            "Navigate to:",
            options=["📥 Data Ingestion", "📊 Analytics & Prediction"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**System Info**")
        st.caption(f"Python: {os.sys.version.split()[0]}")

        # HDFS status indicator
        try:
            spark = get_spark()
            mgr   = get_hdfs_manager(spark)
            mode  = mgr.get_storage_mode()
            if "HDFS" in mode:
                st.success("🟢 HDFS Connected")
            else:
                st.warning("🟡 Local Fallback Mode")
        except Exception:
            st.error("🔴 Spark Unavailable")

        st.divider()
        st.caption("© 2024 Agricultural Analytics Team")

    return page


# ===========================================================================
# VIEW 1: DATA INGESTION
# ===========================================================================

def render_ingestion_view():
    """Renders the Data Ingestion page."""
    st.markdown("<h2 class='section-header'>📥 Data Ingestion & Storage</h2>", unsafe_allow_html=True)
    st.markdown(
        "Ingest raw CSV files through a local source or Apache NiFi HDFS Raw, "
        "then convert them to optimised **Apache Parquet** partitioned by `State`."
    )

    st.divider()

    ingestion_source = st.radio(
        "📡 Ingestion Source",
        options=["Local Dataset Folder", "NiFi HDFS Raw"],
        horizontal=True,
        help=(
            "Pilih Local untuk membaca CSV langsung dari Windows. Pilih NiFi HDFS Raw "
            "setelah flow NiFi memindahkan CSV ke /agri_data/raw."
        ),
    )

    # ── Dataset Path Input ────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        dataset_path = st.text_input(
            "📂 Dataset Directory Path",
            value=DEFAULT_DATASET_PATH,
            help="Directory containing 2020.csv – 2026.csv files.",
            placeholder=r"D:\path\to\Dataset",
        )
    with col2:
        parquet_out = st.text_input(
            "💾 Parquet Output Path",
            value=DEFAULT_PARQUET_PATH,
            help="Destination for Parquet files (local or HDFS).",
        )

    nifi_raw_path = DEFAULT_NIFI_RAW_PATH
    if ingestion_source == "NiFi HDFS Raw":
        nifi_raw_path = st.text_input(
            "📥 NiFi HDFS Raw Path",
            value=DEFAULT_NIFI_RAW_PATH,
            help="HDFS directory populated by the NiFi GetFile → PutHDFS flow.",
        )

    # ── File Preview ──────────────────────────────────────────────────────
    if ingestion_source == "Local Dataset Folder" and dataset_path:
        path_obj = Path(dataset_path)
        if path_obj.exists():
            csv_files = sorted(path_obj.glob("*.csv"))
            if csv_files:
                st.success(f"✔ Found **{len(csv_files)}** CSV file(s): {', '.join(f.name for f in csv_files)}")
            else:
                st.warning("⚠ No CSV files found in the specified directory.")
        else:
            st.error(f"❌ Directory not found: `{dataset_path}`")

    st.divider()

    # ── Ingestion Trigger ─────────────────────────────────────────────────
    col_btn, col_opts = st.columns([2, 3])
    with col_opts:
        partition_col = st.selectbox(
            "Partition By",
            options=["State", "Commodity", "District"],
            index=0,
            help="Column used to partition the Parquet dataset.",
        )
    with col_btn:
        ingest_clicked = st.button(
            "🚀 Ingest to HDFS & Convert to Parquet",
            type="primary",
            use_container_width=True,
        )

    if ingest_clicked:
        _run_ingestion(
            dataset_path,
            parquet_out,
            partition_col,
            use_nifi_raw=ingestion_source == "NiFi HDFS Raw",
            nifi_raw_path=nifi_raw_path,
        )

    # ── Dataset Summary (if data exists) ──────────────────────────────────
    st.divider()
    st.markdown("<h3 class='section-header'>📋 Stored Dataset Summary</h3>", unsafe_allow_html=True)

    if st.button("🔄 Refresh Summary", use_container_width=False):
        _render_dataset_summary(parquet_out)
    else:
        _render_dataset_summary(parquet_out)


def _run_ingestion(
    dataset_path: str,
    parquet_out: str,
    partition_col: str,
    use_nifi_raw: bool = False,
    nifi_raw_path: str = DEFAULT_NIFI_RAW_PATH,
):
    """Executes the CSV → Parquet ingestion pipeline with live progress feedback."""
    progress = st.progress(0, text="Initialising Spark…")
    status   = st.empty()

    try:
        # Step 1 — Spark init
        status.info("⚡ Starting Spark Session…")
        spark = get_spark()
        progress.progress(10, text="Spark ready.")

        # Step 2 — Discover CSVs
        status.info("🔍 Discovering CSV files…")
        from hdfs_manager import (
            HDFSManager,
            discover_csv_files,
            discover_hdfs_csv_files,
            convert_csv_to_parquet,
        )
        if use_nifi_raw:
            csv_files = discover_hdfs_csv_files(spark, nifi_raw_path)
        else:
            csv_files = discover_csv_files(dataset_path)
        progress.progress(20, text=f"Found {len(csv_files)} file(s).")

        # Step 3 — Convert
        status.info(
            f"⚙ Converting {len(csv_files)} CSV file(s) from "
            f"{'NiFi HDFS Raw' if use_nifi_raw else 'Local Dataset'} to Parquet…"
        )
        start_time = time.time()
        manager = HDFSManager(spark=spark)
        if parquet_out == DEFAULT_PARQUET_PATH and manager.is_hdfs_available():
            parquet_out = manager.get_storage_path("commodities")

        out_path, total_rows = convert_csv_to_parquet(
            spark=spark,
            input_csv_paths=csv_files,
            output_parquet_path=parquet_out,
            partition_col=partition_col,
        )
        elapsed = time.time() - start_time
        progress.progress(90, text="Conversion complete. Computing summary…")

        # Step 4 — Summary
        from hdfs_manager import get_dataset_summary
        summary = get_dataset_summary(spark, out_path)
        progress.progress(100, text="✔ Done!")

        status.empty()
        st.session_state[PARQUET_PATH_STATE_KEY] = out_path
        st.success(
            f"✅ **Ingestion Complete!** "
            f"`{total_rows:,}` rows written in `{elapsed:.1f}s` → `{out_path}`"
        )

        # Display quick stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",    f"{summary['total_rows']:,}")
        c2.metric("Parquet Size",      f"{summary['parquet_size_mb']} MB")
        c3.metric("Unique States",     str(len(summary['unique_states'])))
        c4.metric("Unique Markets",    str(summary['unique_markets']))

        with st.expander("View Full Summary"):
            st.json(summary)

    except FileNotFoundError as exc:
        progress.empty()
        st.error(f"❌ File not found: {exc}")
    except Exception as exc:
        progress.empty()
        st.error(f"❌ Ingestion failed: {exc}")
        logger.exception("Ingestion error")


def _render_dataset_summary(parquet_path: str):
    """Renders a summary card for the stored Parquet dataset."""
    try:
        spark   = get_spark()
        from hdfs_manager import get_dataset_summary
        summary = get_dataset_summary(spark, parquet_path)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Total Records",    f"{summary['total_rows']:,}")
        c2.metric("💾 Storage Size",     f"{summary['parquet_size_mb']} MB")
        c3.metric("📅 Date Range",        f"{summary['date_min']} → {summary['date_max']}")
        c4.metric("🏪 Unique Markets",    str(summary['unique_markets']))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**States Indexed:**")
            st.write(", ".join(summary["unique_states"]) if summary["unique_states"] else "—")
        with col_b:
            st.markdown("**Top Commodities:**")
            st.write(", ".join(summary["unique_commodities"][:15]) if summary["unique_commodities"] else "—")

    except Exception:
        st.info("ℹ No Parquet data found yet. Run ingestion first.")


# ===========================================================================
# VIEW 2: ANALYTICS & PREDICTION
# ===========================================================================

def render_analytics_view():
    """Renders the Analytics & Prediction page."""
    st.markdown(
        "<h2 class='section-header'>📊 Analytics & Prediction Dashboard</h2>",
        unsafe_allow_html=True,
    )

    # ── Scope Filters ──────────────────────────────────────────────────────
    st.markdown("#### 🔍 Filter Scope")
    f1, f2, f3 = st.columns(3)

    try:
        spark   = get_spark()
        parquet_path = st.session_state.get(
            PARQUET_PATH_STATE_KEY,
            DEFAULT_PARQUET_PATH,
        )
        from hdfs_manager import to_spark_path
        df_meta = spark.read.parquet(to_spark_path(parquet_path, require_exists=True))

        states      = sorted([r["State"]     for r in df_meta.select("State").distinct().collect()])
        selected_state = f1.selectbox("🗺 State",     ["(All)"] + states)

        if selected_state != "(All)":
            df_m = df_meta.filter(df_meta["State"] == selected_state)
        else:
            df_m = df_meta

        markets    = sorted([r["Market"]    for r in df_m.select("Market").distinct().collect()])
        selected_market = f2.selectbox("🏪 Market",    ["(All)"] + markets)

        if selected_market != "(All)":
            df_m2 = df_m.filter(df_m["Market"] == selected_market)
        else:
            df_m2 = df_m

        commodities = sorted([r["Commodity"] for r in df_m2.select("Commodity").distinct().collect()])
        selected_commodity = f3.selectbox("🌾 Commodity", ["(All)"] + commodities)

    except Exception:
        st.warning("⚠ No ingested data found. Please run Data Ingestion first.")
        return

    st.divider()

    analysis_mode = st.radio(
        "Analysis Mode",
        options=["🌐 Overview BI (All / Partial Scope)", "🎯 Detail Forecast (One Series)"],
        horizontal=True,
        help=(
            "Overview BI boleh memakai filter All atau sebagian filter. "
            "Forecast hanya dijalankan pada satu State, Market, dan Commodity."
        ),
    )
    overview_mode = analysis_mode.startswith("🌐")
    detail_scope_selected = all(
        value != "(All)"
        for value in (selected_state, selected_market, selected_commodity)
    )
    if overview_mode:
        st.info(
            "Overview BI menganalisis pola agregat tanpa mencampur banyak series "
            "menjadi satu forecast. Pilih filter sebagian jika ingin mempersempit insight."
        )
    elif not detail_scope_selected:
        st.info(
            "Pilih satu State, Market, dan Commodity untuk menjalankan "
            "historical trend dan forecast pada satu time series yang konsisten."
        )

    run_pipeline = st.button(
        "⚡ Run BI Overview" if overview_mode else "⚡ Run Analysis & Train Model",
        type="primary",
        use_container_width=True,
        disabled=not overview_mode and not detail_scope_selected,
    )

    # Use session state to persist results across rerenders
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None

    if run_pipeline:
        scope = {
            "state": None if selected_state == "(All)" else selected_state,
            "market": None if selected_market == "(All)" else selected_market,
            "commodity": None if selected_commodity == "(All)" else selected_commodity,
        }
        if overview_mode:
            _execute_overview(**scope)
        else:
            _execute_pipeline(**scope)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_bi, tab_pred = st.tabs(["📈 Business Intelligence", "🔮 30-Day Forecast"])

    with tab_bi:
        _render_bi_tab()

    with tab_pred:
        _render_prediction_tab()


def _execute_pipeline(
    state: Optional[str],
    market: Optional[str],
    commodity: Optional[str],
):
    """Runs the full ML pipeline and stores results in session state."""
    with st.spinner("🔄 Running pipeline… (this may take a few minutes)"):
        try:
            spark = get_spark()
            from spark_pipeline import run_full_pipeline
            from ml_model import train_and_evaluate, recursive_forecast

            # ── Pipeline ──
            full_df, train_df, test_df, _assembler = run_full_pipeline(
                spark=spark,
                parquet_path=st.session_state.get(
                    PARQUET_PATH_STATE_KEY,
                    DEFAULT_PARQUET_PATH,
                ),
                state=state,
                market=market,
                commodity=commodity,
                test_year=2026,
            )

            # ── Train & Evaluate ──
            model, metrics, importances = train_and_evaluate(
                spark=spark,
                train_df=train_df,
                test_df=test_df,
            )

            # ── Seed data for forecasting (last 60 actual rows) ──
            seed_pd = (
                full_df
                .orderBy("Arrival_Date", ascending=False)
                .limit(60)
                .select("Arrival_Date", "Modal_Price", "Min_Price", "Max_Price")
                .toPandas()
            )

            # ── 30-day Forecast ──
            forecast_df = recursive_forecast(
                model=model,
                spark=spark,
                seed_data=seed_pd,
                forecast_days=30,
            )

            # ── Historical trend (last 90 days) ──
            hist_pd = (
                full_df
                .orderBy("Arrival_Date", ascending=False)
                .limit(90)
                .select("Arrival_Date", "Modal_Price",
                        "Modal_Price_lag_7", "Modal_Price_lag_14", "Modal_Price_lag_30",
                        "Price_Rolling_Avg_7", "Price_Rolling_Avg_30",
                        "Price_Change_7D", "Price_Change_30D",
                        "Price_Volatility_30D")
                .toPandas()
                .sort_values("Arrival_Date")
            )

            st.session_state.pipeline_results = {
                "metrics":     metrics,
                "importances": importances,
                "forecast_df": forecast_df,
                "hist_pd":     hist_pd,
                "scope":       {"state": state, "market": market, "commodity": commodity},
            }
            st.success("✅ Pipeline complete!")

        except ValueError as exc:
            st.warning(f"⚠ Analysis cannot run for this filter: {exc}")
            logger.warning("Pipeline validation failed: %s", exc)
        except Exception as exc:
            st.error(f"❌ Pipeline error: {exc}")
            logger.exception("Pipeline error")


def _execute_overview(
    state: Optional[str],
    market: Optional[str],
    commodity: Optional[str],
):
    """Builds aggregate BI results without pretending multiple series are one forecast."""
    with st.spinner("🔄 Building aggregate BI overview…"):
        try:
            spark = get_spark()
            from hdfs_manager import to_spark_path
            from spark_pipeline import aggregate_daily_prices, cleanse_data

            parquet_path = st.session_state.get(
                PARQUET_PATH_STATE_KEY,
                DEFAULT_PARQUET_PATH,
            )
            df = spark.read.parquet(to_spark_path(parquet_path, require_exists=True))
            if state:
                df = df.filter(df["State"] == state)
            if market:
                df = df.filter(df["Market"] == market)
            if commodity:
                df = df.filter(df["Commodity"] == commodity)

            df = aggregate_daily_prices(cleanse_data(df))
            from pyspark.sql import functions as F
            from pyspark.sql.window import Window

            daily = (
                df.groupBy("Arrival_Date")
                .agg(
                    F.avg("Modal_Price").alias("Modal_Price"),
                    F.countDistinct("Commodity").alias("Commodity_Count"),
                    F.countDistinct("Market").alias("Market_Count"),
                )
            )
            window = Window.orderBy("Arrival_Date")
            daily = (
                daily.withColumn("Rolling_Avg_7", F.avg("Modal_Price").over(window.rowsBetween(-6, 0)))
                .withColumn("Rolling_Avg_30", F.avg("Modal_Price").over(window.rowsBetween(-29, 0)))
                .withColumn("Change_7D", F.col("Modal_Price") / F.lag("Modal_Price", 7).over(window) - 1)
                .withColumn("Change_30D", F.col("Modal_Price") / F.lag("Modal_Price", 30).over(window) - 1)
                .withColumn("Volatility_30D", F.stddev("Modal_Price").over(window.rowsBetween(-29, 0)))
            )

            # Build the same lagged feature contract as the detail model, but
            # on one aggregate daily series for an overview-level forecast.
            aggregate_features = (
                daily
                .withColumn("Modal_Price_lag_1", F.lag("Modal_Price", 1).over(window))
                .withColumn("Modal_Price_lag_7", F.lag("Modal_Price", 7).over(window))
                .withColumn("Modal_Price_lag_14", F.lag("Modal_Price", 14).over(window))
                .withColumn("Modal_Price_lag_30", F.lag("Modal_Price", 30).over(window))
                .withColumn("Price_Rolling_Avg_7", F.avg("Modal_Price").over(window.rowsBetween(-6, 0)))
                .withColumn("Price_Rolling_Avg_30", F.avg("Modal_Price").over(window.rowsBetween(-29, 0)))
                .withColumn(
                    "Price_Change_7D",
                    F.col("Modal_Price") / F.lag("Modal_Price", 7).over(window) - 1,
                )
                .withColumn(
                    "Price_Change_30D",
                    F.col("Modal_Price") / F.lag("Modal_Price", 30).over(window) - 1,
                )
                .withColumn(
                    "Price_Volatility_7D",
                    F.stddev("Modal_Price").over(window.rowsBetween(-6, 0)),
                )
                .withColumn(
                    "Price_Volatility_30D",
                    F.stddev("Modal_Price").over(window.rowsBetween(-29, 0)),
                )
                .na.drop(subset=[
                    "Modal_Price_lag_1", "Modal_Price_lag_7",
                    "Modal_Price_lag_14", "Modal_Price_lag_30",
                    "Price_Rolling_Avg_7", "Price_Rolling_Avg_30",
                    "Price_Change_7D", "Price_Change_30D",
                    "Price_Volatility_7D", "Price_Volatility_30D",
                ])
            )
            from spark_pipeline import assemble_features, time_series_split
            from ml_model import train_and_evaluate, recursive_forecast

            model_df, _ = assemble_features(aggregate_features)
            train_df, test_df = time_series_split(model_df, test_year=2026)
            model, metrics, _ = train_and_evaluate(
                spark=spark,
                train_df=train_df,
                test_df=test_df,
            )
            seed_pd = (
                aggregate_features
                .orderBy("Arrival_Date", ascending=False)
                .limit(60)
                .select("Arrival_Date", "Modal_Price")
                .toPandas()
            )
            forecast_pd = recursive_forecast(
                model=model,
                spark=spark,
                seed_data=seed_pd,
                forecast_days=30,
            )

            trend_pd = daily.orderBy("Arrival_Date").tail(180)
            trend_pd = pd.DataFrame([row.asDict() for row in trend_pd])
            ranking_pd = (
                df.groupBy("State", "Commodity")
                .agg(
                    F.avg("Modal_Price").alias("Average_Price"),
                    F.stddev("Modal_Price").alias("Volatility"),
                    F.count("*").alias("Observations"),
                )
                .filter(F.col("Observations") >= 7)
                .orderBy(F.desc("Average_Price"))
                .limit(15)
                .toPandas()
            )
            latest = daily.orderBy(F.desc("Arrival_Date")).limit(1).toPandas()

            st.session_state.pipeline_results = {
                "kind": "overview",
                "trend_pd": trend_pd,
                "ranking_pd": ranking_pd,
                "latest_pd": latest,
                "forecast_df": forecast_pd,
                "metrics": metrics,
                "scope": {"state": state, "market": market, "commodity": commodity},
            }
            st.success("✅ BI Overview complete!")
        except Exception as exc:
            st.error(f"❌ BI Overview error: {exc}")
            logger.exception("BI Overview error")


def _render_bi_tab():
    """Renders the Business Intelligence tab content."""
    res = st.session_state.get("pipeline_results")
    if not res:
        st.markdown("### 📈 Business Intelligence")
        st.info("ℹ Run the pipeline above to view Business Intelligence results.")
        return

    if res.get("kind") == "overview":
        _render_overview_bi(res)
        return

    st.markdown("### 📈 Business Intelligence — Lag Feature Importance")
    importances = res["importances"]
    hist_pd     = res["hist_pd"]
    scope       = res["scope"]

    scope_label = " | ".join(f"{k}: **{v}**" for k, v in scope.items() if v)
    if scope_label:
        st.caption(f"Scope: {scope_label}")

    latest = hist_pd.iloc[-1]
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Latest Price", f"₹{latest['Modal_Price']:,.2f}")
    metric_b.metric("Change 7D", f"{latest['Price_Change_7D'] * 100:+.2f}%")
    metric_c.metric("Change 30D", f"{latest['Price_Change_30D'] * 100:+.2f}%")
    metric_d.metric("Volatility 30D", f"₹{latest['Price_Volatility_30D']:,.2f}")

    # ── Feature Importance Chart ──────────────────────────────────────────
    imp_df = pd.DataFrame(importances)

    # Friendly display names
    name_map = {
        "Modal_Price_lag_1":   "Lag 1 Day (t-1)",
        "Modal_Price_lag_7":   "Lag 7 Days (t-7)",
        "Modal_Price_lag_14":  "Lag 14 Days (t-14)",
        "Modal_Price_lag_30":  "Lag 30 Days (t-30)",
        "Price_Rolling_Avg_7": "Rolling Avg 7D",
        "Price_Rolling_Avg_30": "Rolling Avg 30D",
        "Price_Change_7D":      "Change 7D",
        "Price_Change_30D":     "Change 30D",
        "Price_Volatility_7D":  "Volatility 7D",
        "Price_Volatility_30D": "Volatility 30D",
    }
    imp_df["label"] = imp_df["feature"].map(name_map).fillna(imp_df["feature"])

    fig_imp = px.bar(
        imp_df.sort_values("importance"),
        x="importance_pct",
        y="label",
        orientation="h",
        title="Feature Importance — Contribution to Modal Price Prediction (%)",
        labels={"importance_pct": "Importance (%)", "label": "Feature"},
        color="importance_pct",
        color_continuous_scale="Greens",
        text="importance_pct",
    )
    fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_imp.update_layout(
        height=380,
        coloraxis_showscale=False,
        plot_bgcolor="white",
        xaxis_title="Importance (%)",
        yaxis_title="",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # ── Historical Trend Chart ────────────────────────────────────────────
    st.markdown("### 📉 Historical Price Trend — Actual vs. Lagged Signals")

    hist_pd["Arrival_Date"] = pd.to_datetime(hist_pd["Arrival_Date"])

    fig_hist = go.Figure()

    fig_hist.add_trace(go.Scatter(
        x=hist_pd["Arrival_Date"], y=hist_pd["Modal_Price"],
        name="Actual Modal Price",
        line=dict(color="#1a472a", width=2.5),
        mode="lines",
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist_pd["Arrival_Date"], y=hist_pd["Modal_Price_lag_7"],
        name="Lag 7d (t-7)",
        line=dict(color="#52b788", width=1.5, dash="dash"),
        mode="lines",
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist_pd["Arrival_Date"], y=hist_pd["Modal_Price_lag_14"],
        name="Lag 14d (t-14)",
        line=dict(color="#f4a261", width=1.5, dash="dot"),
        mode="lines",
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist_pd["Arrival_Date"], y=hist_pd["Modal_Price_lag_30"],
        name="Lag 30d (t-30)",
        line=dict(color="#e76f51", width=1.5, dash="dashdot"),
        mode="lines",
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist_pd["Arrival_Date"], y=hist_pd["Price_Rolling_Avg_7"],
        name="Rolling Avg 7d",
        line=dict(color="#264653", width=2, dash="longdash"),
        mode="lines",
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist_pd["Arrival_Date"], y=hist_pd["Price_Rolling_Avg_30"],
        name="Rolling Avg 30d",
        line=dict(color="#6c757d", width=2, dash="dashdot"),
        mode="lines",
    ))

    fig_hist.update_layout(
        title="Historical Modal Price vs. Lagged Signals (Last 90 Days)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=420,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig_hist.update_xaxes(showgrid=True, gridcolor="#e8f5e9")
    fig_hist.update_yaxes(showgrid=True, gridcolor="#e8f5e9")
    st.plotly_chart(fig_hist, use_container_width=True)


def _render_overview_bi(res):
    """Renders aggregate BI metrics, rankings, and aggregate forecast."""
    trend_pd = res["trend_pd"].copy()
    ranking_pd = res["ranking_pd"].copy()
    latest_pd = res["latest_pd"]
    scope = res["scope"]

    scope_label = " | ".join(f"{key}: **{value}**" for key, value in scope.items() if value)
    st.caption(f"Overview scope: {scope_label or 'All data'}")
    if trend_pd.empty:
        st.warning("Tidak ada observasi yang cukup untuk overview pada filter ini.")
        return

    trend_pd["Arrival_Date"] = pd.to_datetime(trend_pd["Arrival_Date"])
    latest = trend_pd.iloc[-1]
    previous_7 = trend_pd.iloc[-8]["Modal_Price"] if len(trend_pd) >= 8 else None
    change_7 = ((latest["Modal_Price"] / previous_7) - 1) if previous_7 else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Aggregate Price", f"₹{latest['Modal_Price']:,.2f}")
    c2.metric("7D Aggregate Change", f"{change_7 * 100:+.2f}%" if change_7 is not None else "N/A")
    c3.metric(
        "Markets on Latest Day",
        f"{int(latest['Market_Count']):,}",
        help="Distinct markets contributing data on the latest available date.",
    )
    c4.metric(
        "Commodities on Latest Day",
        f"{int(latest['Commodity_Count']):,}",
        help="Distinct commodities contributing data on the latest available date.",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_pd["Arrival_Date"], y=trend_pd["Modal_Price"],
        name="Aggregate Modal Price", mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=trend_pd["Arrival_Date"], y=trend_pd["Rolling_Avg_30"],
        name="30D Rolling Average", mode="lines",
        line=dict(dash="dash"),
    ))
    fig.update_layout(
        title="Aggregate Historical Price Trend (Last 180 Days)",
        xaxis_title="Date", yaxis_title="Average Modal Price (₹)",
        height=400, plot_bgcolor="white", hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🏆 Highest Average Price by State and Commodity")
    if not ranking_pd.empty:
        display = ranking_pd.rename(columns={
            "State": "State", "Commodity": "Commodity",
            "Average_Price": "Average Price (₹)",
            "Volatility": "Volatility (₹)", "Observations": "Observations",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("#### 🔮 Aggregate 30-Day Forecast")
    st.caption(
        "Forecast ini memodelkan satu seri rata-rata harian dari scope yang dipilih. "
        "Gunakan Detail Forecast untuk prediksi satu komoditas pada satu market."
    )
    forecast_df = res["forecast_df"].copy()
    forecast_df["Forecast_Date"] = pd.to_datetime(forecast_df["Forecast_Date"])
    historical_tail = trend_pd.tail(30)
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=historical_tail["Arrival_Date"], y=historical_tail["Modal_Price"],
        name="Aggregate Actual", mode="lines",
    ))
    forecast_fig.add_trace(go.Scatter(
        x=forecast_df["Forecast_Date"], y=forecast_df["Predicted_Modal_Price"],
        name="Aggregate Forecast", mode="lines+markers",
        line=dict(dash="dash"),
    ))
    forecast_fig.update_layout(
        title="Aggregate Modal Price: Historical and Next 30 Days",
        xaxis_title="Date", yaxis_title="Average Modal Price (₹)",
        height=420, plot_bgcolor="white", hovermode="x unified",
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    fm1, fm2, fm3 = st.columns(3)
    fm1.metric("Aggregate RMSE", f"₹{res['metrics']['rmse']:,.2f}")
    fm2.metric("Aggregate MAE", f"₹{res['metrics']['mae']:,.2f}")
    fm3.metric("Aggregate R²", f"{res['metrics']['r2']:.4f}")


def _render_prediction_tab():
    """Renders the 30-Day Forecast tab content."""
    st.markdown("### 🔮 30-Day Price Projection")

    res = st.session_state.get("pipeline_results")
    if not res:
        st.info("ℹ Run the pipeline above to view the 30-day forecast.")
        return
    if res.get("kind") == "overview":
        st.info(
            "Mode Overview menampilkan insight agregat. Untuk forecast 30 hari, "
            "pilih mode Detail Forecast lalu pilih satu State, Market, dan Commodity."
        )
        return

    metrics     = res["metrics"]
    forecast_df = res["forecast_df"]
    hist_pd     = res["hist_pd"]

    # ── Model Performance Cards ───────────────────────────────────────────
    st.markdown("#### 🏆 Model Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE",  f"₹{metrics['rmse']:,.2f}",  help="Root Mean Squared Error (lower is better)")
    m2.metric("MAE",   f"₹{metrics['mae']:,.2f}",   help="Mean Absolute Error (lower is better)")
    m3.metric("R² Score", f"{metrics['r2']:.4f}",   help="Coefficient of determination (closer to 1 is better)")
    m4.metric(
        "Forecast Horizon",
        f"{len(forecast_df)} days",
        help="Number of projected days",
    )

    st.divider()

    # ── Forecast Chart ────────────────────────────────────────────────────
    st.markdown("#### 📅 Price Forecast Timeline")

    # Combine historical tail + forecast for continuous chart
    hist_tail = hist_pd.tail(30).copy()
    hist_tail["Arrival_Date"] = pd.to_datetime(hist_tail["Arrival_Date"])

    forecast_df["Forecast_Date"] = pd.to_datetime(forecast_df["Forecast_Date"])

    fig_fc = go.Figure()

    # Actual historical prices
    fig_fc.add_trace(go.Scatter(
        x=hist_tail["Arrival_Date"],
        y=hist_tail["Modal_Price"],
        name="Historical (Actual)",
        line=dict(color="#1a472a", width=2.5),
        mode="lines+markers",
        marker=dict(size=5),
    ))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["Forecast_Date"],
        y=forecast_df["Predicted_Modal_Price"],
        name="Forecast (Predicted)",
        line=dict(color="#f4a261", width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=6, symbol="diamond"),
    ))

    # Confidence band (Min/Max)
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast_df["Forecast_Date"], forecast_df["Forecast_Date"].iloc[::-1]]),
        y=pd.concat([forecast_df["Predicted_Max_Price"], forecast_df["Predicted_Min_Price"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(244, 162, 97, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Forecast Band (Min–Max)",
    ))

    # Vertical separator: actual vs forecast
    if not hist_tail.empty and not forecast_df.empty:
        split_date = hist_tail["Arrival_Date"].max()
        fig_fc.add_vline(
            x=split_date,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start",
            annotation_position="top right",
        )

    fig_fc.update_layout(
        title="30-Day Modal Price Forecast — Recursive RF Prediction",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=480,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig_fc.update_xaxes(showgrid=True, gridcolor="#e8f5e9")
    fig_fc.update_yaxes(showgrid=True, gridcolor="#e8f5e9")
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Forecast Table ─────────────────────────────────────────────────────
    with st.expander("📋 View Forecast Data Table"):
        display_df = forecast_df[["Day", "Forecast_Date", "Predicted_Modal_Price",
                                   "Predicted_Min_Price", "Predicted_Max_Price"]].copy()
        display_df.columns = ["Day", "Date", "Predicted Price (₹)", "Min Est. (₹)", "Max Est. (₹)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Key Forecast Insights ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### 💡 Forecast Insights")
    i1, i2, i3 = st.columns(3)

    day1_price  = forecast_df["Predicted_Modal_Price"].iloc[0]
    day30_price = forecast_df["Predicted_Modal_Price"].iloc[-1]
    trend_pct   = ((day30_price - day1_price) / max(day1_price, 1)) * 100
    trend_emoji = "📈" if trend_pct > 0 else "📉"

    i1.metric("Day 1 Estimate",   f"₹{day1_price:,.2f}")
    i2.metric("Day 30 Estimate",  f"₹{day30_price:,.2f}")
    i3.metric(
        f"{trend_emoji} 30-Day Trend",
        f"{trend_pct:+.1f}%",
        delta=f"{day30_price - day1_price:+.2f} ₹",
        delta_color="normal",
    )


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def main():
    """Main entry point for the Streamlit application."""
    page = render_sidebar()

    if "📥 Data Ingestion" in page:
        render_ingestion_view()
    elif "📊 Analytics & Prediction" in page:
        render_analytics_view()


if __name__ == "__main__":
    main()
