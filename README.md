# 🌾 Agricultural Commodity Price Analytics & Lagged Volatility Predictor

> **An End-to-End Big Data & Machine Learning platform** for analysing and predicting agricultural commodity price movements across Indian markets, powered by Apache Spark (PySpark), Hadoop HDFS, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

### Background
India's agricultural commodity markets exhibit high price volatility driven by seasonal supply-demand shifts, transportation costs, and regional demand differences. Traders, government agencies, and cooperatives need data-driven tools to:
- **Understand** how past price signals (7, 14, 30 days ago) influence today's market price.
- **Predict** future price movements to inform procurement, storage, and policy decisions.

### Dual Objectives

| Objective | Type | Methodology |
|-----------|------|-------------|
| How do historical prices (t-7, t-14, t-30) affect today's price? | **Business Intelligence** | PySpark Window Time-Lagging + Feature Importance Analysis |
| What will Modal_Price be 30 days from now? | **Machine Learning** | Distributed Random Forest + Recursive Forecasting |

---

## 🗂 Project Structure

```
PROJECTAKHIR/
│
├── 📄 app.py                  # Streamlit frontend dashboard (main entry point)
├── 📄 hdfs_manager.py         # Storage layer — HDFS + local fallback
├── 📄 spark_pipeline.py       # Processing layer — cleansing, feature engineering
├── 📄 ml_model.py             # ML layer — training, evaluation, forecasting
│
├── 📄 README.md               # Project documentation (this file)
├── 📄 SETUP_GUIDE.md          # Installation & environment setup guide
├── 📄 PRESENTATION_GUIDE.md   # Presentation script and demo guide
├── 📄 requirements.txt        # Python package dependencies
│
├── 📁 Dataset/                # Raw CSV data (2020.csv – 2026.csv)
│   ├── 2020.csv
│   ├── 2021.csv
│   ├── 2022.csv
│   ├── 2023.csv
│   ├── 2024.csv
│   ├── 2025.csv
│   └── 2026.csv
│
└── 📁 data/                   # Generated at runtime
    ├── parquet/               # Converted Parquet dataset (partitioned by State)
    └── model/                 # Saved RandomForest model artefacts
```

---

## 🧩 Module Descriptions

### `hdfs_manager.py` — Storage & Fallback Layer
Manages all data storage operations with intelligent fallback:

| Class / Function | Description |
|-----------------|-------------|
| `HDFSManager` | Main class managing HDFS vs local storage selection |
| `HDFSManager.is_hdfs_available()` | Probes HDFS NameNode through Spark's Hadoop FileSystem API (WSL-compatible) |
| `HDFSManager.get_storage_path()` | Returns active storage path (HDFS or local) |
| `HDFSManager.write_parquet()` | Writes Spark DataFrame as partitioned Parquet |
| `HDFSManager.read_parquet()` | Reads Parquet from active storage |
| `discover_csv_files()` | Scans dataset directory for CSV files |
| `convert_csv_to_parquet()` | Full CSV → Parquet conversion with type casting |
| `get_dataset_summary()` | Returns metadata: row count, date range, unique values |

**Fallback Logic:**
```
HDFS Available? ──YES──▶ Write/Read from hdfs://localhost:9000/agri_data/parquet/commodities
      │
      NO
      │
      └──────────────▶ Write/Read from ./data/parquet/commodities (local filesystem)
```

### Apache NiFi Ingestion

Apache NiFi is the ingestion layer for the recommended Big Data flow:

```text
GetFile → UpdateAttribute → PutHDFS
```

NiFi moves raw CSV files into:

```text
hdfs://localhost:9000/agri_data/raw/
```

The Streamlit Ingestion view supports `Local Dataset Folder` for direct
Windows CSV ingestion and `NiFi HDFS Raw` for CSV files already placed in HDFS
by NiFi. In both cases, PySpark performs type casting and writes the cleaned
Parquet dataset. NiFi does not perform feature engineering or Machine Learning.

---

### `spark_pipeline.py` — Processing & Feature Engineering Layer

| Function | Description |
|----------|-------------|
| `create_spark_session()` | Initialises configured SparkSession (local or cluster) |
| `cleanse_data()` | 4-stage cleansing: nulls → dedup → outliers → price consistency |
| `engineer_features()` | Daily series aggregation, calendar lagging, rolling averages, price changes, and volatility |
| `assemble_features()` | VectorAssembler transformation for MLlib |
| `time_series_split()` | Temporal train/test split (no data leakage) |
| `filter_scope()` | Filters by State, Market, Commodity; detail forecasting requires one complete series selection |
| `run_full_pipeline()` | End-to-end pipeline orchestrator |

---

### `ml_model.py` — Machine Learning & Analytics Layer

| Function | Description |
|----------|-------------|
| `train_model()` | Trains distributed RandomForestRegressor |
| `evaluate_model()` | Computes RMSE, MAE, R² on test set |
| `get_feature_importances()` | Extracts and ranks feature importance scores |
| `recursive_forecast()` | Generates 30-day recursive price forecast |

The Analytics view has two complementary modes:

- **Overview BI** accepts `All` or partial filters and summarizes aggregate daily
  price trends, market/commodity coverage, average-price rankings, volatility,
  and an aggregate 30-day forecast of the scope's daily average price. The
  overview does not display templated "new insight" or "future recommendation"
  narratives.
- **Detail Forecast** requires one State + Market + Commodity combination so
  lag features and the recursive 30-day forecast refer to one consistent time
  series.

The Overview forecast is trained on one derived daily series: the average
`Modal_Price` for the currently selected scope. It must not be interpreted as
a forecast for every individual commodity or market in that scope. Use Detail
Forecast when a series-specific prediction is required.
| `save_model()` / `load_model()` | Model persistence to HDFS or local path |
| `train_and_evaluate()` | Convenience orchestrator for all ML steps |

---

### `app.py` — Streamlit Frontend Dashboard

Two main views accessible from the sidebar:

#### View 1: 📥 Data Ingestion
- Select `Local Dataset Folder` or `NiFi HDFS Raw`
- Use `hdfs://localhost:9000/agri_data/raw` after NiFi has placed CSV files in HDFS
- One-click ingestion with visual progress bar
- Real-time status updates during Spark processing
- Post-ingestion summary metrics (total rows, storage size, commodities, markets)

#### View 2: 📊 Analytics & Prediction
- Dropdown filters: State → Market → Commodity (cascading)
- The analytics view automatically reuses the Parquet path selected during ingestion; if no custom path was selected, it uses `./data/parquet/commodities`
- **Overview BI:** Aggregate trend, rolling average, State/Commodity ranking, aggregate 30-day forecast, and RMSE/MAE/R².
- **Detail Forecast:** Feature importance, historical lag trend, and series-specific recursive forecast when one State + Market + Commodity is selected.

---

## 🔄 Data Lifecycle (End-to-End Flow)

```
Raw CSV Files                 Apache NiFi                 HDFS Raw
(2024–2026.csv)    ──────▶   GetFile → PutHDFS   ──────▶ /agri_data/raw/
                                  │                           │
                                  │                           ▼
                                  │                  Streamlit / PySpark
                                  │                           │
                                  │                           ▼
                                  └──────────────▶ HDFS / Local Parquet
                                                   (partitioned by State)
                                                           │
                                                           ▼
                                                      spark_pipeline.py
                                                     cleanse_data()
                                                     engineer_features()
                                                           │
                                                           │ lag_7, lag_14, lag_30
                                                           │ rolling_avg_7
                                                           ▼
                                                      ml_model.py
                                                    train_model() ──▶ RandomForestRegressor
                                                    evaluate_model() ──▶ RMSE / MAE / R²
                                                    get_feature_importances()
                                                    recursive_forecast()
                                                           │
                                                           ▼
                                                   Streamlit Dashboard
                                                  Analytics & Prediction View
                                                  (Plotly Charts + Metric Cards)
```

---

## 🔬 Key Technical Features

### 1. PySpark Window Time-Lagging
The core Business Intelligence feature uses PySpark's `Window` function to compute lagged price signals:

```python
windowSpec = Window.partitionBy("State", "Market", "Commodity").orderBy("Arrival_Date")

df = df.withColumn("Modal_Price_lag_7",   F.lag("Modal_Price",  7).over(windowSpec))
       .withColumn("Modal_Price_lag_14",  F.lag("Modal_Price", 14).over(windowSpec))
       .withColumn("Modal_Price_lag_30",  F.lag("Modal_Price", 30).over(windowSpec))
       .withColumn("Price_Rolling_Avg_7", F.avg("Modal_Price").over(windowSpec.rowsBetween(-6, 0)))
```

**Why partition by State + Market + Commodity?** Each commodity in each market has its own price dynamics. Partitioning ensures lag values are computed *within* the same product-market context, not across unrelated records.

---

### 2. Time-Series Split (No Data Leakage)
```
Preferred split:
Training Set              Test Set
(before 2026)     |       (2026 onward)
──────────────────┼──────────────────────▶ time
                 Jan 1, 2026
```

Using `randomSplit()` would allow future data to appear in training, giving the model
"knowledge of the future" and producing inflated, unreliable metrics. The pipeline
therefore uses a chronological split. If a selected State + Market + Commodity series
does not contain dates on both sides of the preferred 2026 cutoff, the pipeline
automatically uses an 80/20 chronological split over that series' available date
range. This keeps the evaluation leakage-free even when a detail series starts in
2026 or has no observations in 2026.

---

### 3. Feature Importance Analysis
The Random Forest model computes mean decrease in variance (impurity) for each feature across all trees. This answers:

> *"Among the price signals from 7 days ago, 14 days ago, and 30 days ago — which has the strongest causal influence on today's price?"*

Results are displayed as a **Plotly horizontal bar chart** in the Business Intelligence tab.

---

### 4. Recursive 30-Day Forecasting
```
Day 1:  [lag_7=D-7, lag_14=D-14, lag_30=D-30] ──▶ RF ──▶ Predicted_Day_1
Day 2:  [lag_7=Predicted_Day_1, lag_14=D-13, lag_30=D-29] ──▶ RF ──▶ Predicted_Day_2
...
Day 30: [all lags = previous predictions] ──▶ RF ──▶ Predicted_Day_30
```

Each predicted day's price is fed back as the lag feature for the next iteration, enabling true multi-step ahead forecasting without requiring future ground-truth labels.

---

## 📊 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `State` | String | Indian state name |
| `District` | String | District within the state |
| `Market` | String | Market/mandi name |
| `Commodity` | String | Agricultural commodity name |
| `Variety` | String | Commodity variety/grade description |
| `Grade` | String | Quality grade |
| `Arrival_Date` | Date (YYYY-MM-DD) | Date of price recording |
| `Min_Price` | Double (₹) | Minimum price recorded that day |
| `Max_Price` | Double (₹) | Maximum price recorded that day |
| `Modal_Price` | Double (₹) | Most common/modal price (**prediction target**) |
| `Commodity_Code` | Long | Numeric commodity identifier |

---

## 🛠 Technology Stack

| Layer | Technology |
|-------|-----------|
| Distributed Processing | Apache Spark 3.x (PySpark) |
| Distributed Storage | Hadoop HDFS / Local Parquet |
| ML Algorithm | Spark MLlib RandomForestRegressor |
| Dashboard UI | Streamlit |
| Data Visualisation | Plotly Express & Graph Objects |
| Data Manipulation | Pandas, NumPy |

---

## 📬 Contact
Agricultural Analytics Team — Big Data Course Project  
Dataset source: *Daily Market Prices of Agricultural Commodities in India (Agmarknet)*
