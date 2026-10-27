# ⚙ SETUP_GUIDE.md — Environment Setup & Configuration Guide

> **Agricultural Commodity Price Analytics & Lagged Volatility Predictor**  
> Complete step-by-step guide to configure the Big Data environment on Windows.

---

## 📋 Table of Contents
1. [System Prerequisites](#1-system-prerequisites)
2. [Java JDK Setup (Required for PySpark)](#2-java-jdk-setup)
3. [Apache Spark Installation](#3-apache-spark-installation)
4. [Hadoop Setup & winutils.exe (Windows)](#4-hadoop-setup--winutilsexe-windows)
5. [Python Virtual Environment](#5-python-virtual-environment)
6. [Package Installation](#6-package-installation)
7. [Hadoop HDFS Configuration & Startup](#7-hadoop-hdfs-configuration--startup)
8. [Apache NiFi Ingestion](#8-apache-nifi-ingestion)
9. [Fallback Mode (Local — No HDFS)](#9-fallback-mode-local--no-hdfs)
10. [Running the Application](#10-running-the-application)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Prerequisites

| Component | Minimum Version | Recommended | Notes |
|-----------|----------------|-------------|-------|
| **Operating System** | Windows 10 64-bit | Windows 11 | WSL2 optional |
| **Python** | 3.9 | **3.10 / 3.11** | Must be 64-bit |
| **Java JDK** | JDK 8 (1.8) | **JDK 11 (LTS)** | Required by Spark & Hadoop |
| **Apache Spark** | 3.3.x | **3.4.x / 3.5.x** | With Hadoop 3 binaries |
| **Hadoop** | 3.2.x | **3.3.x** | Only winutils needed for local |
| **RAM** | 8 GB | **16 GB+** | Spark driver + executor memory |
| **Disk Space** | 10 GB | 20 GB+ | Dataset + Parquet + models |

> **⚠ Important:** Always use **64-bit** versions of Java and Python. PySpark will fail silently with 32-bit installations.

---

## 2. Java JDK Setup

### 2.1 Download & Install JDK 11
1. Download **JDK 11 LTS** from:
   - [Adoptium (Temurin) — Recommended](https://adoptium.net/temurin/releases/?version=11)
   - [Oracle JDK 11](https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html)
2. Run the installer. Default installation path: `C:\Program Files\Eclipse Adoptium\jdk-11.x.x.x-hotspot\`

### 2.2 Set JAVA_HOME Environment Variable
Open **PowerShell as Administrator**:

```powershell
# Set JAVA_HOME (adjust path to match your JDK install location)
[System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Eclipse Adoptium\jdk-11.0.21.9-hotspot", "Machine")

# Add Java bin to PATH
$path = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
[System.Environment]::SetEnvironmentVariable("PATH", "$path;C:\Program Files\Eclipse Adoptium\jdk-11.0.21.9-hotspot\bin", "Machine")
```

Or set manually via **System Properties → Environment Variables → System Variables**.

### 2.3 Verify Installation
```powershell
# Open a NEW PowerShell window after setting env vars
java -version
# Expected: openjdk version "11.x.x" ...

echo $env:JAVA_HOME
# Expected: C:\Program Files\Eclipse Adoptium\jdk-11.0.21.9-hotspot
```

---

## 3. Apache Spark Installation

### 3.1 Download Spark
1. Visit [https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html)
2. Select: **Spark 3.4.x** | **Package type: Pre-built for Apache Hadoop 3.3 and later**
3. Download the `.tgz` file (e.g., `spark-3.4.1-bin-hadoop3.tgz`)

### 3.2 Extract & Install
```powershell
# Create directory
New-Item -ItemType Directory -Path "C:\spark" -Force

# Extract (using 7-Zip or tar in PowerShell 5+)
tar -xvzf spark-3.4.1-bin-hadoop3.tgz -C "C:\spark"

# Rename for convenience
Rename-Item "C:\spark\spark-3.4.1-bin-hadoop3" "C:\spark\spark-3.4.1"
```

### 3.3 Set SPARK_HOME
```powershell
[System.Environment]::SetEnvironmentVariable("SPARK_HOME", "C:\spark\spark-3.4.1", "Machine")

$path = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
[System.Environment]::SetEnvironmentVariable("PATH", "$path;C:\spark\spark-3.4.1\bin", "Machine")
```

### 3.4 Verify Spark
```powershell
# Open NEW PowerShell window
spark-shell --version
# Expected: version 3.4.x
```

---

## 4. Hadoop Setup & winutils.exe (Windows)

> **Why winutils?** PySpark on Windows requires Hadoop's native binaries (`winutils.exe` and `hadoop.dll`) to handle file system operations. Without them, you'll encounter `NullPointerException` errors.

### 4.1 Download winutils
Download `winutils.exe` and `hadoop.dll` matching your Hadoop version from:
- [https://github.com/cdarlint/winutils](https://github.com/cdarlint/winutils)
- Navigate to the folder matching your Hadoop version (e.g., `hadoop-3.3.5/bin/`)

### 4.2 Install winutils
```powershell
# Create Hadoop bin directory
New-Item -ItemType Directory -Path "C:\hadoop\bin" -Force

# Copy downloaded files
Copy-Item "path\to\downloaded\winutils.exe" "C:\hadoop\bin\"
Copy-Item "path\to\downloaded\hadoop.dll"   "C:\hadoop\bin\"
```

### 4.3 Set HADOOP_HOME
```powershell
[System.Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "Machine")

$path = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
[System.Environment]::SetEnvironmentVariable("PATH", "$path;C:\hadoop\bin", "Machine")
```

### 4.4 Fix Temp Directory Permissions
```powershell
# winutils requires the C:\tmp\hive directory with specific permissions
C:\hadoop\bin\winutils.exe chmod 777 C:\tmp\hive
# If C:\tmp\hive doesn't exist, create it first:
New-Item -ItemType Directory -Path "C:\tmp\hive" -Force
C:\hadoop\bin\winutils.exe chmod 777 C:\tmp\hive
```

### 4.5 Verify
```powershell
echo $env:HADOOP_HOME
# Expected: C:\hadoop

C:\hadoop\bin\winutils.exe ls C:\
# Should list C:\ directory without errors
```

---

## 5. Python Virtual Environment

### 5.1 Verify Python Version
```powershell
python --version
# Expected: Python 3.10.x or 3.11.x (64-bit)
```

### 5.2 Create Virtual Environment
```powershell
# Navigate to the project directory
cd "D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR"

# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# If execution policy blocks activation:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### 5.3 Verify Activation
```powershell
# Your prompt should show (.venv) prefix
(.venv) PS D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR>

# Confirm Python path
python -c "import sys; print(sys.executable)"
# Expected: D:\...\PROJECTAKHIR\.venv\Scripts\python.exe
```

---

## 6. Package Installation

```powershell
# Ensure virtual environment is active (see step 5.2)
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### Manual Installation (if requirements.txt fails)
```powershell
pip install pyspark==3.4.1
pip install pyarrow>=12.0.0
pip install streamlit>=1.28.0
pip install plotly>=5.17.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install python-dotenv>=1.0.0
```

### Verify PySpark Installation
```powershell
python -c "import pyspark; print(pyspark.__version__)"
# Expected: 3.4.x
```

---

## 7. Hadoop HDFS Configuration & Startup

### 7.0 HDFS on Ubuntu WSL with the Windows Streamlit App

The application can run Streamlit and PySpark on Windows while the HDFS
NameNode/DataNode run inside Ubuntu WSL. The default endpoint is:

```text
hdfs://localhost:9000
```

In Ubuntu WSL, verify that HDFS is running and the application directory exists:

```bash
jps
hdfs dfs -mkdir -p /agri_data/parquet
hdfs dfs -chmod 755 /agri_data
hdfs dfs -ls /
```

If the NameNode is bound to a different host or port, configure it before
starting Streamlit:

```powershell
$env:AGRI_HDFS_NAMENODE = "hdfs://localhost:9000"
streamlit run app.py
```

The application checks HDFS through Spark's Hadoop `FileSystem` API, not
through the Windows `hdfs` command. This is required when Hadoop is installed
only inside WSL. CSV files remain readable from the Windows dataset path, while
the default Parquet output is redirected to HDFS when the WSL NameNode is
reachable. If it is unavailable, local fallback storage is used.

> **Optional:** Skip this section if running in **Fallback Mode** (local filesystem). The application auto-detects HDFS availability.

### 7.1 Download Full Hadoop
Download Apache Hadoop 3.3.x from [https://hadoop.apache.org/releases.html](https://hadoop.apache.org/releases.html)

```powershell
# Extract to C:\hadoop-full (separate from the winutils-only C:\hadoop)
tar -xvzf hadoop-3.3.5.tar.gz -C "C:\"
```

### 7.2 Configure core-site.xml
Edit `C:\hadoop-3.3.5\etc\hadoop\core-site.xml`:
```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>
```

### 7.3 Configure hdfs-site.xml
Edit `C:\hadoop-3.3.5\etc\hadoop\hdfs-site.xml`:
```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file:///C:/hadoop-data/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:///C:/hadoop-data/datanode</value>
  </property>
</configuration>
```

### 7.4 Set JAVA_HOME in hadoop-env.cmd
Edit `C:\hadoop-3.3.5\etc\hadoop\hadoop-env.cmd`:
```batch
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.0.21.9-hotspot
```

### 7.5 Format NameNode (First Time Only)
```powershell
C:\hadoop-3.3.5\bin\hdfs namenode -format
```

### 7.6 Start HDFS Daemons
```powershell
# Start NameNode and DataNode
C:\hadoop-3.3.5\sbin\start-dfs.cmd

# Verify daemons are running (should show NameNode, DataNode)
jps
```

### 7.7 Create HDFS Application Directories
```powershell
hdfs dfs -mkdir -p /agri_data/raw
hdfs dfs -mkdir -p /agri_data/parquet/commodities
hdfs dfs -chmod 755 /agri_data
hdfs dfs -ls /
```

### 7.8 Access HDFS Web UI
Open browser: [http://localhost:9870](http://localhost:9870)

---

## 8. Apache NiFi Ingestion

NiFi is the official ingestion layer in the recommended project flow. It moves
raw CSV files into HDFS; PySpark then performs type casting and writes the
clean Parquet dataset. NiFi is not used for feature engineering or Machine
Learning.

### 8.1 Where to Run NiFi

Because HDFS runs in Ubuntu WSL, the most reliable setup is to run NiFi in the
same Ubuntu WSL environment. Running NiFi on Windows is possible, but requires
the Windows Hadoop client configuration and compatible Java/Hadoop libraries.

Use the same Java major version for Hadoop and NiFi. Keep the Windows Streamlit
application and the WSL HDFS endpoint reachable at `hdfs://localhost:9000`.

### 8.2 Install and Start NiFi in Ubuntu WSL

Download a supported Apache NiFi binary from
[nifi.apache.org/download](https://nifi.apache.org/download.html), extract it
under WSL, and configure Java:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export NIFI_HOME=$HOME/nifi
export PATH=$JAVA_HOME/bin:$PATH
$NIFI_HOME/bin/nifi.sh set-single-user-credentials admin ChangeThisPassword123!
$NIFI_HOME/bin/nifi.sh start
$NIFI_HOME/bin/nifi.sh status
```

Open the NiFi UI using the HTTPS URL printed by NiFi, commonly:

```text
https://localhost:8443/nifi
```

Do not commit the generated credentials. Use a strong local password.

### 8.3 Create the NiFi Flow

Create this flow in the NiFi canvas:

```text
GetFile → UpdateAttribute → PutHDFS
```

Configure `GetFile`:

```text
Input Directory:
/mnt/d/KULIAH/TUGAS-KELAS/Big-Data/PROJECTAKHIR/Dataset
Keep Source File:
true
```

Configure `UpdateAttribute` with optional metadata:

```text
source_system = agricultural_dataset
ingestion_date = ${now():format("yyyy-MM-dd")}
```

Configure `PutHDFS`:

```text
Hadoop Configuration Resources:
/etc/hadoop/conf/core-site.xml,/etc/hadoop/conf/hdfs-site.xml
Directory:
/agri_data/raw
Conflict Resolution Strategy:
replace
```

If Hadoop configuration files are stored elsewhere, use their actual WSL paths.
Do not use a Windows `D:\...` path in a processor running inside WSL.

Connect the `success` relationship from each processor to the next processor.
Connect `failure` relationships to `LogAttribute` or a local failure queue.
Start the processors and confirm that the CSV files appear in HDFS:

```bash
hdfs dfs -ls /agri_data/raw
```

### 8.4 Run PySpark Ingestion from NiFi Raw

Start the Windows application with HDFS configured:

```powershell
cd "D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR"
.\.venv\Scripts\Activate.ps1
$env:AGRI_HDFS_NAMENODE = "hdfs://localhost:9000"
streamlit run app.py
```

In the Streamlit Data Ingestion view:

1. Select **NiFi HDFS Raw** as the ingestion source.
2. Keep the raw path as `hdfs://localhost:9000/agri_data/raw`.
3. Keep the output path as the default path.
4. Choose the partition column `State`.
5. Click the ingestion button.

The application lists CSV files from HDFS, reads them with PySpark, casts the
schema, and writes Parquet to:

```text
hdfs://localhost:9000/agri_data/parquet/commodities
```

The Analytics view then reads that Parquet path.

### 8.5 Verify and Re-run Safely

Verify raw and processed data:

```bash
hdfs dfs -ls /agri_data/raw
hdfs dfs -ls /agri_data/parquet/commodities
```

Before a clean re-ingestion, remove only the project data directories:

```bash
hdfs dfs -rm -r -f /agri_data/parquet/commodities
hdfs dfs -rm -r -f /agri_data/raw
hdfs dfs -mkdir -p /agri_data/raw
hdfs dfs -mkdir -p /agri_data/parquet/commodities
```

Do not remove the HDFS root or unrelated directories.

---

## 9. Fallback Mode (Local — No HDFS)

If HDFS is not running, the application automatically falls back to local filesystem storage. No configuration is required.

### How Fallback Works
```
Application start → HDFSManager.is_hdfs_available() 
    → probes: Spark Hadoop FileSystem API against AGRI_HDFS_NAMENODE
    → FAILS (HDFS not running)
    → ✔ Uses: ./data/parquet/commodities (local directory)
```

### Testing Fallback Mode
```powershell
# Navigate to project directory (venv active)
cd "D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR"

# Run directly — no HDFS setup needed
streamlit run app.py

# Data will be stored locally at:
# D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR\data\parquet\commodities\
```

### Verify Fallback in Python
```python
from pyspark.sql import SparkSession
from hdfs_manager import HDFSManager

spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
mgr = HDFSManager(spark)

print(mgr.is_hdfs_available())    # → False
print(mgr.get_storage_mode())     # → "Local Filesystem (Fallback)"
print(mgr.get_storage_path())     # → "<absolute path>/data/parquet"
```

---

## 10. Running the Application

### Pre-Flight Checklist
- [ ] Java JDK 11 installed and `JAVA_HOME` set
- [ ] Spark installed and `SPARK_HOME` set
- [ ] `winutils.exe` in `C:\hadoop\bin` and `HADOOP_HOME` set
- [ ] Python virtual environment activated
- [ ] All packages from `requirements.txt` installed
- [ ] Dataset CSV files present in `D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR\Dataset\`

### Launch Command
```powershell
# 1. Navigate to project directory
cd "D:\KULIAH\TUGAS-KELAS\Big-Data\PROJECTAKHIR"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Launch Streamlit app
streamlit run app.py
```

### Expected Output
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Application Workflow
```
Browser → localhost:8501
    │
    ├── Sidebar: "📥 Data Ingestion"
    │       │
    │       ├── 1. Select Local Dataset Folder or NiFi HDFS Raw
    │       ├── 2. Click the ingestion button
    │       └── 3. View summary statistics after ingestion
    │
    └── Sidebar: "📊 Analytics & Prediction"
            │
            ├── 1. Select State → Market → Commodity filters
            ├── 2. Click "⚡ Run Analysis & Train Model"
            ├── Tab A: View Feature Importance + Historical Trend
            └── Tab B: View 30-Day Forecast + Model Metrics
```

---

## 11. Troubleshooting

### ❌ `JAVA_HOME is not set`
```powershell
# Verify JAVA_HOME
echo $env:JAVA_HOME

# If empty, set it (restart shell after):
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-11.0.21.9-hotspot"
```

### ❌ `winutils.exe: Could not locate executable`
```powershell
# Verify winutils exists
Test-Path "C:\hadoop\bin\winutils.exe"  # should return True

# Verify HADOOP_HOME
echo $env:HADOOP_HOME  # should return C:\hadoop
```

### ❌ `java.io.IOException: Failed to create a temp dir`
```powershell
# Create and permission the temp directory
New-Item -ItemType Directory -Path "C:\tmp\hive" -Force
C:\hadoop\bin\winutils.exe chmod 777 C:\tmp\hive
```

### ❌ PySpark `OutOfMemoryError`
The pipeline uses disk-only persistence for the large feature DataFrame.
Restart Streamlit after applying the latest code. For a full-dataset run,
reduce data scope using State/Market/Commodity filters, or increase driver
memory in `spark_pipeline.py`:
```python
# In create_spark_session(), increase memory:
.config("spark.driver.memory", "4g")
.config("spark.executor.memory", "4g")
```

### ❌ `ModuleNotFoundError: No module named 'pyspark'`
```powershell
# Confirm virtual environment is active
pip show pyspark  # should show pyspark info

# If not installed:
pip install pyspark==3.4.1
```

### ❌ Streamlit port already in use
```powershell
# Run on a different port
streamlit run app.py --server.port 8502
```

### ❌ HDFS `Connection refused`
This is expected on Windows local dev. The app will use Fallback Mode automatically. Check the sidebar for `🟡 Local Fallback Mode` indicator.

### ❌ `py4j.protocol.Py4JJavaError` during Window functions
Ensure Spark's shuffle partitions are configured for small datasets:
```python
.config("spark.sql.shuffle.partitions", "8")
```

---

## 📁 Environment Variables Summary

| Variable | Example Value | Required |
|----------|--------------|----------|
| `JAVA_HOME` | `C:\Program Files\Eclipse Adoptium\jdk-11.0.21.9-hotspot` | ✅ Yes |
| `SPARK_HOME` | `C:\spark\spark-3.4.1` | ✅ Yes |
| `HADOOP_HOME` | `C:\hadoop` | ✅ Yes (winutils) |
| `PATH` | Include `%JAVA_HOME%\bin`, `%SPARK_HOME%\bin`, `%HADOOP_HOME%\bin` | ✅ Yes |
| `PYSPARK_PYTHON` | `D:\...\PROJECTAKHIR\.venv\Scripts\python.exe` | ⚠ Optional |

---

*Last updated: October 2026 | Agricultural Analytics Team — Big Data Course Project*
