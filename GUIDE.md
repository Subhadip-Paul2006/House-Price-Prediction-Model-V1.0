# 📘 GUIDE.md — The Complete Project Guide

> **House Price Prediction App** — A single, friendly document that explains *every file, every folder, how they connect, and how the model is trained.*
> Built for a **new user** who is making or modifying the model. No prior knowledge of this codebase assumed.

---

## 📑 Table of Contents

1. [Introduction — What Is This Project?](#-introduction--what-is-this-project)
2. [File Structure — Treeview](#-file-structure--treeview)
3. [What Every File & Folder Does](#-what-every-file--folder-does)
4. [How Files Connect to Each Other](#-how-files-connect-to-each-other)
5. [How the Model Is Trained (End-to-End)](#-how-the-model-is-trained-end-to-end)
6. [Architecture Diagrams](#-architecture-diagrams)
7. [C4 Diagrams (Context · Container · Component)](#-c4-diagrams)
8. [Sequence Diagrams](#-sequence-diagrams)
9. [Class Diagram](#-class-diagram)
10. [Block Diagram](#-block-diagram)
11. [Large Advanced Flowchart](#-large-advanced-flowchart)
12. [Entity Relationship Diagram](#-entity-relationship-diagram)
13. [User Journey Diagram](#-user-journey-diagram)
14. [Multi-Layer Event Modeling](#-multi-layer-event-modeling)
15. [Mindmaps](#-mindmaps)
16. [Kanban Diagram](#-kanban-diagram)
17. [Gantt Diagram](#-gantt-diagram)
18. [Git Graph](#-git-graph)
19. [XY Charts](#-xy-charts)
20. [Pie Charts](#-pie-charts)
21. [Quick Start — 3 Commands](#-quick-start--3-commands)
22. [Common Modifications & Experiments](#-common-modifications--experiments)
23. [Troubleshooting Cheat-Sheet](#-troubleshooting-cheat-sheet)

---

## 🌟 Introduction — What Is This Project?

The **House Price Prediction App** predicts the price of a house (in **Lakhs ₹**) from 16 property features, using two Machine Learning models:

- **Linear Regression** — a fast, interpretable baseline (current R² = **0.9442**)
- **Random Forest** — 200 decision trees voting on the price (current R² = **0.9257**)

The whole system is a pipeline: **data → cleaning → encoding/scaling → training → evaluation → saved artifact → Streamlit web app**. A user opens a web page, enters house details, clicks *Predict*, and gets an instant price estimate with a 95% confidence band and a downloadable PDF report.

```mermaid
flowchart LR
    A["📊 Data<br/>1000 houses"] --> B["🧹 Clean"]
    B --> C["🔀 Split<br/>80/20"]
    C --> D["🤖 Train<br/>Model"]
    D --> E["📏 Evaluate<br/>R² · MAE · RMSE"]
    E --> F["💾 Save<br/>model.joblib"]
    F --> G["🌐 Streamlit<br/>Web App"]
    G --> H["🔮 Price<br/>Estimate"]

    style A fill:#4CAF50,color:#fff
    style D fill:#2196F3,color:#fff
    style G fill:#FF4B4B,color:#fff
    style H fill:#FF9800,color:#fff
```

---

## 🌳 File Structure — Treeview

The exact layout of the repository (folders marked 📁, files marked 📄):

```mermaid
flowchart LR
    ROOT["📁 House Price Predictor/"]:::dir

    ROOT --> APP["📄 app.py<br/>Streamlit web UI"]
    ROOT --> TRAIN["📄 train.py<br/>Training pipeline"]
    ROOT --> PRE["📄 preprocess.py<br/>Clean · encode · scale"]
    ROOT --> MDL["📄 model.py<br/>Save/load/predict"]
    ROOT --> EVAL["📄 evaluate.py<br/>MAE · RMSE · R²"]
    ROOT --> GEN["📄 generate_data.py<br/>Create dataset"]

    ROOT --> DATA["📁 data/"]:::dir
    DATA --> CSV["📄 house_data.csv<br/>1000 rows × 17 cols"]
    DATA --> RAW["📁 raw/ (empty)"]
    DATA --> PROC["📁 processed/ (empty)"]

    ROOT --> MODELS["📁 models/"]:::dir
    MODELS --> LR["📁 linear_regression/"]
    MODELS --> RF["📁 random_forest/"]

    ROOT --> NB["📁 notebooks/ (empty)"]
    ROOT --> SS["📁 screenshots/"]
    SS --> IMG["🖼️ UI captures"]

    ROOT --> TD1["📁 test_dataset01/<br/>Ames Housing CSV"]
    ROOT --> TD2["📁 test_dataset02/<br/>Kaggle House Prices CSV"]

    ROOT --> REQ["📄 requirements.txt"]
    ROOT --> GI["📄 .gitignore"]
    ROOT --> DOCS["📄 Docs: README · USAGE · PRD ·<br/>TRD · ARCHITECTURE · GUIDE.md"]

    classDef dir fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px;
    classDef file fill:#e9ecef,color:#000,stroke:#adb5bd;
    class DATA,MODELS,NB,SS,TD1,TD2 dir;
```

**Plain-text tree (copy-paste friendly):**

```text
House Price Predictor/
│
├── 📄 app.py                  # 🌐 Streamlit web UI (entry point for users)
├── 📄 train.py                # 🤖 Training pipeline orchestrator
├── 📄 preprocess.py           # 🧹 Data loading, cleaning, encoding, scaling
├── 📄 model.py                # 💾 Artifact save/load + Predictor wrapper
├── 📄 evaluate.py             # 📏 Regression metrics (MAE, RMSE, R²…)
├── 📄 generate_data.py        # 🎲 One-time synthetic dataset generator
├── 📄 requirements.txt        # 📦 Pinned Python dependencies
├── 📄 .gitignore              # 🚫 Files excluded from git
│
├── 📁 data/
│   ├── 📄 house_data.csv      # ← The 17-column dataset (1000 rows)
│   ├── 📁 raw/                # (reserved for original/raw data)
│   └── 📁 processed/          # (reserved for cleaned/saved data)
│
├── 📁 models/                 # ← Trained artifacts live here
│   ├── 📁 linear_regression/
│   │   └── 📁 <timestamp>/
│   │       ├── model.joblib   # ← Serialized model + preprocessor
│   │       └── metrics.json   # ← MAE/RMSE/R² scores
│   └── 📁 random_forest/
│       └── 📁 <timestamp>/
│           ├── model.joblib
│           └── metrics.json
│
├── 📁 notebooks/              # (reserved for EDA Jupyter notebooks)
│
├── 📁 screenshots/            # 🖼️ Browser captures of the running app
│
├── 📁 test_dataset01/         # 📊 Alternate dataset: Ames Housing
│   ├── 📄 AmesHousing.csv
│   └── 📄 test.py
│
├── 📁 test_dataset02/         # 📊 Alternate dataset: Kaggle House Prices
│   ├── 📄 train.csv
│   ├── 📄 data_description.txt
│   └── 📄 test.py
│
└── 📄 Docs
    ├── 📄 README.md           # Project overview & badges
    ├── 📄 USAGE.md            # Setup & contribution guide
    ├── 📄 PRD.md              # Product Requirements Document
    ├── 📄 TRD.md              # Technical Requirements Document
    ├── 📄 ARCHITECTURE.md     # System design reference
    ├── 📄 Idea.md             # Original project idea / plan
    ├── 📄 summary.md          # Private learning cheat-sheet (gitignored)
    └── 📄 GUIDE.md            # ← THIS FILE
```

---

## 📂 What Every File & Folder Does

### 🐍 Python Source Files

| # | File | Role | Key contents |
|---|------|------|--------------|
| 1 | **`generate_data.py`** | 🎲 **Dataset creator** | Builds 1000 realistic house rows with NumPy, computes Price from a formula, injects ~2% missing values so the cleaner has work to do. Writes `data/house_data.csv`. **Run once.** |
| 2 | **`preprocess.py`** | 🧹 **Data prep** | `HouseData` (loads CSV), `Preprocessor` (imputes NaN, one-hot encodes categoricals, scales numerics, 80/20 split). Defines which columns are numeric vs categorical. |
| 3 | **`evaluate.py`** | 📏 **Metrics** | `Evaluator` class: `mae`, `mse`, `rmse`, `r2`, `residual_std`, `report()`, `compare_models()`. |
| 4 | **`train.py`** | 🤖 **Orchestrator** | Wires everything together: load → clean → split → preprocess → train → cross-validate → evaluate → save. CLI: `python train.py --model all` |
| 5 | **`model.py`** | 💾 **Persistence + Inference** | `ArtifactStore` (save/load/version `.joblib` + `metrics.json`), `Predictor` (wraps model + preprocessor; `predict()` and `confidence()`). |
| 6 | **`app.py`** | 🌐 **Streamlit UI** | 3 tabs — **Predict**, **Compare**, **Charts** — plus sidebar inputs, PDF/text report download. The only file users interact with. |

### 📁 Folders

| Folder | Purpose |
|--------|---------|
| `data/` | Holds the dataset. `house_data.csv` is the live file; `raw/` & `processed/` are reserved. |
| `models/` | Each training run saves a timestamped subfolder (`model.joblib` + `metrics.json`). The app always loads the **latest** run. |
| `notebooks/` | Reserved for exploratory Jupyter notebooks (currently empty). |
| `screenshots/` | PNG captures of the running app for the README. |
| `test_dataset01/` | **Ames Housing** — an alternate real-world dataset (Iowa home sales) for experimentation. |
| `test_dataset02/` | **Kaggle House Prices** — the famous advanced-regression competition dataset. |

### 📄 The 16 Model Features (the CSV's 17 columns = 16 features + 1 target)

| # | Column | Type | Example | Role |
|---|--------|------|---------|------|
| 1 | `Area` | Numeric | 1800 | Built-up area (sq ft) |
| 2 | `Bedrooms` | Numeric | 3 | Bedroom count |
| 3 | `Bathrooms` | Numeric | 2 | Bathroom count |
| 4 | `Age` | Numeric | 5 | Property age (years) |
| 5 | `Location` | Categorical | Urban | Downtown / Urban / Suburban / Rural |
| 6 | **`Price`** | Numeric | 162.5 | 🎯 **TARGET (what we predict)** |
| 7 | `Lot Area` | Numeric | 9500 | Plot area (sq ft) |
| 8 | `Overall Quality` | Numeric | 5 | Material & finish (1–10) |
| 9 | `Overall Condition` | Numeric | 5 | Condition (1–10) |
| 10 | `Garage Cars` | Numeric | 1 | Garage capacity (cars) |
| 11 | `Garage Area` | Numeric | 480 | Garage size (sq ft) |
| 12 | `Total Basement SF` | Numeric | 800 | Basement area (sq ft) |
| 13 | `Fireplaces` | Numeric | 0 | Fireplace count |
| 14 | `Neighborhood` | Categorical | West Side | 28 named areas |
| 15 | `House Style` | Categorical | 1Story | 1Story / 2Story / SLvl / … |
| 16 | `Central Air` | Categorical | Yes | Yes / No |
| 17 | `Kitchen Quality` | Categorical | TA | Po / Fa / TA / Gd / Ex |

---

## 🔗 How Files Connect to Each Other

Think of the project as a **one-way assembly line** with a feedback loop at the end (the web app calls back into the saved model):

```mermaid
flowchart TB
    GEN["📄 generate_data.py"] -->|"writes"| CSV[("📂 data/house_data.csv")]
    CSV -->|"read by"| PRE["📄 preprocess.py<br/>HouseData.load()"]
    PRE -->|"cleaned df"| TRN["📄 train.py<br/>run_pipeline()"]
    EVAL["📄 evaluate.py"] -->|"metrics"| TRN
    TRN -->|"saves"| MDL[("💾 models/<br/>model.joblib<br/>metrics.json")]
    MDL -->|"loaded by"| MDLFN["📄 model.py<br/>ArtifactStore + Predictor"]
    MDLFN -->|"called by"| APP["📄 app.py<br/>Streamlit UI"]
    PRE -.->|"transform() at inference"| MDLFN

    style GEN fill:#66bb6a,color:#fff
    style CSV fill:#42a5f5,color:#fff
    style PRE fill:#26c6da,color:#fff
    style TRN fill:#ffa726,color:#fff
    style EVAL fill:#ab47bc,color:#fff
    style MDL fill:#ef5350,color:#fff
    style APP fill:#ec407a,color:#fff
```

### The call graph in words

1. `generate_data.py` **writes** the CSV → consumed by `train.py` (via `preprocess.py`).
2. `train.py` **imports** `HouseData`, `Preprocessor` (from `preprocess.py`), `ModelTrainer` (itself), `Evaluator` (from `evaluate.py`), and `ArtifactStore` (from `model.py`).
3. `train.py` **saves** the model using `ArtifactStore.save()`.
4. `app.py` **imports** `ArtifactStore` + `Predictor` (from `model.py`) and **loads** the saved artifact.
5. At prediction time, `Predictor` re-uses the fitted `Preprocessor.transform()` that was **bundled** inside `model.joblib` — this guarantees the user's input is encoded the *exact same way* the training data was.

> ⚠️ **Critical insight:** The fitted `ColumnTransformer` is saved *inside* `model.joblib`. That's why the web app produces consistent predictions — it never re-fits; it only `transform()`s new input.

---

## 🏗️ How the Model Is Trained (End-to-End)

This is the heart of the project. Follow these 9 stages in order — they map 1:1 to the log output of `python train.py`.

```mermaid
flowchart TD
    S1["STEP 1 — LOAD<br/>HouseData.load() reads<br/>data/house_data.csv → DataFrame"]
    S2["STEP 2 — CLEAN<br/>• Drop rows missing Price<br/>• Impute numeric NaN → median<br/>• Impute categorical NaN → mode"]
    S3["STEP 3 — SELECT FEATURES<br/>X = 16 feature columns<br/>y = Price"]
    S4["STEP 4 — SPLIT<br/>train_test_split(seed=42)<br/>800 train / 200 test"]
    S5["STEP 5 — ENCODE & SCALE<br/>ColumnTransformer:<br/>• Numeric → median + StandardScaler<br/>• Categorical → OneHotEncoder"]
    S6["STEP 6 — TRAIN<br/>model.fit(X_train, y_train)"]
    S7["STEP 7 — CROSS-VALIDATE<br/>5-fold CV on train set"]
    S8["STEP 8 — EVALUATE<br/>Predict on test set<br/>→ MAE, RMSE, R², residual_std"]
    S9["STEP 9 — PERSIST<br/>ArtifactStore.save() →<br/>model.joblib + metrics.json"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9

    style S1 fill:#4CAF50,color:#fff
    style S2 fill:#4CAF50,color:#fff
    style S3 fill:#4CAF50,color:#fff
    style S4 fill:#2196F3,color:#fff
    style S5 fill:#2196F3,color:#fff
    style S6 fill:#FF9800,color:#fff
    style S7 fill:#FF9800,color:#fff
    style S8 fill:#9C27B0,color:#fff
    style S9 fill:#9C27B0,color:#fff
```

### Detailed stage explanations

| Stage | What happens | Where in code |
|-------|--------------|---------------|
| **1. Load** | `pd.read_csv()` → DataFrame (1000 × 17) | `HouseData.load()` in `preprocess.py` |
| **2. Clean** | Drop NaN-Price rows; impute numeric → median, categorical → mode | `Preprocessor.clean()` in `preprocess.py` |
| **3. Select** | Split DataFrame into `X` (16 cols) and `y` (Price) | `train.py` `run_pipeline()` |
| **4. Split** | `train_test_split(test_size=0.2, seed=42)` → 800 train / 200 test | `Preprocessor.split()` |
| **5. Encode/Scale** | `ColumnTransformer.fit_transform()` builds 55 encoded columns (11 scaled + 44 one-hot) | `Preprocessor.fit_transform()` |
| **6. Train** | `model.fit(X_train_enc, y_train)` — the model *learns* the price formula | `ModelTrainer.train()` in `train.py` |
| **7. Cross-validate** | 5-fold CV checks for overfitting | `ModelTrainer.cross_validate()` |
| **8. Evaluate** | Predict on the held-out test set → compute metrics | `Evaluator.report()` in `evaluate.py` |
| **9. Persist** | `joblib.dump({model, preprocessor})` + `metrics.json` to timestamped folder | `ArtifactStore.save()` in `model.py` |

---

## 🏛️ Architecture Diagrams

### High-level layered architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        USER([👤 End User])
    end
    subgraph Presentation["Presentation Layer (app.py)"]
        UI["Streamlit UI<br/>Sidebar · Predict · Compare · Charts"]
    end
    subgraph Logic["Business Logic Layer"]
        PRE["Preprocessor"]
        MODEL["Predictor"]
        METRICS["Evaluator"]
    end
    subgraph Data["Data Layer"]
        CSV[("house_data.csv")]
        ART[("models/*.joblib")]
    end

    USER -->|browser| UI
    UI -->|features dict| MODEL
    MODEL -->|scaled X| PRE
    MODEL -->|price + CI| UI
    CSV -->|load| PRE
    METRICS -->|MAE/RMSE/R²| ART
    ART <-->|save / load| MODEL

    classDef client fill:#ffd166,color:#000
    classDef ui fill:#ef476f,color:#fff
    classDef logic fill:#06d6a0,color:#000
    classDef data fill:#118ab2,color:#fff
    class USER client
    class UI ui
    class PRE,MODEL,METRICS logic
    class CSV,ART data
```

### Design principles applied

| Principle | How it's applied |
|-----------|------------------|
| **Separation of concerns** | Data (preprocess), modeling (train/model), UI (app) are isolated modules |
| **Reproducibility** | Every random source is seeded (`random_state=42`); artifacts versioned by timestamp |
| **Thin UI, fat model** | Streamlit only renders; all logic lives in Python classes |
| **Fail loud, fail early** | Invalid inputs raise typed exceptions surfaced as user errors |

---

## 🧱 C4 Diagrams

### Level 1 — System Context

```mermaid
C4Context
    title House Price Prediction — System Context
    Person(user, "End User", "Wants an instant house-price estimate")
    System_Boundary(hpp, "House Price Prediction") {
        Container(app, "Streamlit Web App", "Python / Streamlit", "Captures inputs, returns predictions")
    }
    System_Ext(data_src, "Public Dataset", "Kaggle / synthetic CSV")
    System_Ext(github, "GitHub", "Source control & CI")
    Rel(user, app, "Uses", "HTTPS / browser")
    Rel(app, data_src, "Sources training data", "manual / generate_data.py")
    Rel(github, app, "Deploys via", "git pull")
```

### Level 2 — Container

```mermaid
C4Context
    title House Price Prediction — Container View
    Person(user, "End User", "Buyer / enthusiast / learner")
    System_Boundary(hpp, "House Price Prediction") {
        Container(web, "Streamlit UI", "Python · Streamlit", "Form inputs, charts, PDF report")
        Container(pipeline, "Training Pipeline", "Python · scikit-learn", "Ingest, clean, train, evaluate")
        ContainerDb(artifacts, "Model Store", "filesystem · .joblib", "Serialized models + metrics")
        ContainerDb(data, "Data Store", "filesystem · CSV", "house_data.csv")
    }
    Rel(user, web, "Opens in browser", "HTTP")
    Rel(web, artifacts, "Loads model for inference", "file read")
    Rel(pipeline, data, "Reads raw", "file I/O")
    Rel(pipeline, artifacts, "Writes trained model", "file write")
```

### Level 3 — Component

```mermaid
flowchart LR
    subgraph Web["Streamlit UI Container (app.py)"]
        Inputs["InputForm"]
        Result["ResultPanel"]
        Charts["ChartPanel"]
        Report["ReportExporter"]
    end
    subgraph Pipeline["Training Pipeline (train.py)"]
        Ingest["Ingestor"]
        Clean["Cleaner"]
        Train["Trainer"]
        Eval["Evaluator"]
        Persist["ArtifactStore"]
    end
    Inputs --> Result --> Charts
    Result --> Report
    Ingest --> Clean --> Train --> Eval --> Persist
    Train <-->|save/load| Persist

    classDef web fill:#ef476f,color:#fff
    classDef pipe fill:#06d6a0,color:#000
    class Inputs,Result,Charts,Report web
    class Ingest,Clean,Train,Eval,Persist pipe
```

---

## 🔁 Sequence Diagrams

### Prediction flow (user → result)

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Streamlit App (app.py)
    participant P as Predictor (model.py)
    participant AS as ArtifactStore
    participant PP as Preprocessor

    U->>UI: Open app, fill 16 features in sidebar
    UI->>UI: Validate inputs (bounds, types)
    UI->>P: predict(features_dict)
    P->>AS: load("models/random_forest/latest")
    AS-->>P: (model, fitted Preprocessor)
    P->>PP: transform(features_df)
    PP->>PP: impute → one-hot encode → scale
    PP-->>P: X_vec (1 × 55 ndarray)
    P->>P: model.predict(X_vec)
    P-->>UI: price + 95% confidence interval
    UI->>UI: render cards + charts + report
    UI-->>U: "Estimated ₹75.0 L (±4 L)"
```

### Training flow (developer → saved artifact)

```mermaid
sequenceDiagram
    autonumber
    participant Dev as Developer
    participant TP as train.py
    participant HD as HouseData
    participant PP as Preprocessor
    participant MT as ModelTrainer
    participant EV as Evaluator
    participant AS as ArtifactStore

    Dev->>TP: python train.py --model random_forest
    TP->>HD: load("data/house_data.csv")
    HD-->>TP: DataFrame (1000 × 17)
    TP->>PP: clean(df)
    PP-->>TP: cleaned DataFrame
    TP->>PP: split(X, y, seed=42)
    PP-->>TP: X_train, X_test, y_train, y_test
    TP->>PP: fit_transform(X_train)
    PP-->>TP: encoded X_train
    TP->>MT: train(X_train, y_train)
    MT-->>TP: fitted model
    TP->>MT: cross_validate(5-fold)
    MT-->>TP: CV R² mean
    TP->>EV: report(y_test, y_pred)
    EV-->>TP: {mae, rmse, r2, residual_std}
    TP->>AS: save(model, preprocessor, metrics)
    AS-->>TP: artifact path
    TP-->>Dev: "Training complete · R²=0.9257"
```

---

## 🧬 Class Diagram

```mermaid
classDiagram
    class HouseData {
        +str path
        +DataFrame df
        +load() DataFrame
        +info() void
        +describe() DataFrame
    }

    class Preprocessor {
        +StandardScaler scaler
        +OneHotEncoder encoder
        +ColumnTransformer column_transformer
        +clean(df) DataFrame
        +split(X, y) tuple
        +fit_transform(X) ndarray
        +transform(X) ndarray
    }

    class ModelTrainer {
        +str algorithm
        +int seed
        +RegressorMixin model
        +train(X, y) ModelTrainer
        +cross_validate(X, y, k) dict
    }

    class Evaluator {
        <<static>>
        +mae(y_true, y_pred) float
        +mse(y_true, y_pred) float
        +rmse(y_true, y_pred) float
        +r2(y_true, y_pred) float
        +residual_std(y_true, y_pred) float
        +report(y_true, y_pred) dict
        +compare_models(results) str
    }

    class ArtifactStore {
        <<static>>
        +save(model, preprocessor, metrics) str
        +load(algorithm_name) tuple
        +version(algorithm_name) str
    }

    class Predictor {
        +RegressorMixin model
        +Preprocessor preprocessor
        +float residual_std
        +predict(features) float
        +confidence(features) tuple
    }

    HouseData --> Preprocessor : provides df
    Preprocessor --> ModelTrainer : provides X/y
    ModelTrainer --> Evaluator : provides y_pred
    ModelTrainer --> ArtifactStore : persists model
    ArtifactStore --> Predictor : supplies model
    Preprocessor --> Predictor : provides transform
    Predictor --> StreamlitApp : serves inference
```

---

## 🧱 Block Diagram

A black-box view — each block is a module; arrows show data handoff.

```mermaid
flowchart LR
    subgraph Input["INPUT"]
        CSV[("house_data.csv")]
        UserInput["User Form<br/>16 features"]
    end

    subgraph Processing["PROCESSING"]
        GEN["generate_data.py<br/>(create data)"]
        CLEAN["preprocess.py<br/>Clean + Encode + Scale"]
        TRAIN["train.py<br/>Train + Evaluate"]
    end

    subgraph Storage["STORAGE"]
        ART[("models/*.joblib<br/>+ metrics.json")]
    end

    subgraph Output["OUTPUT"]
        APP["app.py<br/>Streamlit UI"]
        PDF["📄 PDF Report"]
        PRED["💰 Price + CI"]
    end

    GEN --> CSV
    CSV --> CLEAN
    CLEAN --> TRAIN
    TRAIN --> ART
    ART --> APP
    UserInput --> APP
    APP --> PRED
    APP --> PDF

    style Input fill:#e3f2fd
    style Processing fill:#fff3e0
    style Storage fill:#fce4ec
    style Output fill:#e8f5e9
```

---

## 🗺️ Large Advanced Flowchart

The full decision tree covering **both** the training path and the inference path, including error branches.

```mermaid
flowchart TD
    Start([🚀 Start]) --> Q1{Running<br/>for the first<br/>time?}

    Q1 -->|Yes| GEN["generate_data.py<br/>build 1000 rows"]
    Q1 -->|No| Q2

    GEN --> CSV[("data/house_data.csv")]
    CSV --> Q2{Train or<br/>Predict?}

    %% ─── TRAINING PATH ───
    Q2 -->|Train| T1["Load CSV → DataFrame"]
    T1 --> T2["Drop NaN-Price rows"]
    T2 --> T3["Impute numeric → median<br/>Impute categorical → mode"]
    T3 --> T4["Split 80/20 (seed=42)"]
    T4 --> T5["Fit ColumnTransformer:<br/>Scale + OneHotEncode"]
    T5 --> T6["Train model.fit()"]
    T6 --> T7["5-fold cross-validate"]
    T7 --> T8{CV R² ≥ 0.85?}
    T8 -->|No| ALERT["⚠️ Investigate:<br/>add features / tune"]
    ALERT --> T6
    T8 -->|Yes| T9["Evaluate on test set"]
    T9 --> T10{Test R² ≥ 0.90?}
    T10 -->|No| ALERT
    T10 -->|Yes| T11["Save model.joblib<br/>+ metrics.json"]
    T11 --> DONE1([✅ Training complete])

    %% ─── INFERENCE PATH ───
    Q2 -->|Predict| P1["User opens Streamlit"]
    P1 --> P2["Fill sidebar: 16 features"]
    P2 --> P3["Click Predict"]
    P3 --> P4["Load latest model.joblib"]
    P4 --> P5["Predictor.predict(features)"]
    P5 --> P6["transform → model.predict"]
    P6 --> P7["Compute 95% CI<br/>(±1.96 × residual_std)"]
    P7 --> P8["Render cards + charts"]
    P8 --> P9{Download<br/>report?}
    P9 -->|Yes| P10["Generate PDF + TXT"]
    P9 -->|No| DONE2([✅ Result shown])
    P10 --> DONE2

    style Start fill:#4CAF50,color:#fff
    style DONE1 fill:#4CAF50,color:#fff
    style DONE2 fill:#4CAF50,color:#fff
    style ALERT fill:#ff5252,color:#fff
    style T8 fill:#ffd166,color:#000
    style T10 fill:#ffd166,color:#000
```

---

## 🗃️ Entity Relationship Diagram

The data model: how the **dataset**, **features**, **model**, and **metrics** entities relate.

```mermaid
erDiagram
    DATASET ||--o{ SAMPLE : contains
    SAMPLE ||--|| FEATURES : has
    SAMPLE ||--|| TARGET : has
    FEATURES ||--o{ NUMERIC_FEATURE : includes
    FEATURES ||--o{ CATEGORICAL_FEATURE : includes
    MODEL ||--|| METRICS : evaluated_by
    MODEL ||--|| PREPROCESSOR : bundles
    MODEL ||--o{ SAMPLE : trained_on

    DATASET {
        string filename PK
        int n_rows
        int n_cols
        date created
    }
    SAMPLE {
        int row_id PK
    }
    FEATURES {
        int sample_id FK
    }
    NUMERIC_FEATURE {
        string name PK
        float value
        string impute_strategy "median"
    }
    CATEGORICAL_FEATURE {
        string name PK
        string value
        string impute_strategy "mode"
        int onehot_columns
    }
    TARGET {
        int sample_id FK
        float Price
    }
    MODEL {
        string algorithm PK "LR or RF"
        string run_id PK "timestamp"
        string joblib_path
    }
    METRICS {
        string run_id FK
        float mae
        float rmse
        float r2
        float cv_mean_r2
    }
    PREPROCESSOR {
        string run_id FK
        int n_encoded_features "55"
    }
```

---

## 🛣️ User Journey Diagram

```mermaid
journey
    title New User's Journey — From Clone to Prediction
    section Discover
      Read README & GUIDE: 5: You
      Clone the repository: 5: You
    section Setup
      Create venv: 4: You
      pip install -r requirements.txt: 4: You
      Verify install: 4: You
    section Data
      Run generate_data.py: 5: You
      Inspect house_data.csv: 4: You
    section Train
      Run python train.py: 4: You
      Check R² score in console: 5: You
    section Launch
      Run streamlit run app.py: 5: You
      Browser opens localhost:8501: 5: You
    section Predict
      Fill 16 features in sidebar: 4: You
      Click Predict: 5: You
      See price + confidence band: 5: You
    section Export
      Download PDF report: 4: You
      Compare two properties: 4: You
```

---

## 🎭 Multi-Layer Event Modeling

Shows events across **UI · Application · Model · Storage** layers simultaneously.

```mermaid
flowchart TB
    subgraph UI["🎨 UI Layer (Streamlit)"]
        E1["User opens app"]
        E2["Sidebar renders inputs"]
        E3["Predict button clicked"]
        E4["Result cards render"]
        E5["Report downloaded"]
    end
    subgraph APP["⚙️ Application Layer (app.py)"]
        A1["Load Predictor via cache"]
        A2["Collect features dict"]
        A3["Call Predictor.predict()"]
        A4["Format price + CI"]
        A5["Generate PDF bytes"]
    end
    subgraph MODEL["🤖 Model Layer (model.py)"]
        M1["ArtifactStore.load()"]
        M2["Preprocessor.transform()"]
        M3["model.predict(X_vec)"]
        M4["Compute ±1.96 × std"]
    end
    subgraph STORAGE["💾 Storage Layer"]
        S1[("read model.joblib")]
        S2[("read metrics.json")]
    end

    E1 --> A1 --> M1 --> S1
    E2 --> A2
    E3 --> A3 --> M2 --> M3 --> M4 --> A4 --> E4
    A4 --> S2
    E3 --> A5 --> E5

    style UI fill:#fce4ec
    style APP fill:#e8f5e9
    style MODEL fill:#e3f2fd
    style STORAGE fill:#fff3e0
```

---

## 🧠 Mindmaps

### Project scope mindmap

```mermaid
mindmap
  root((House Price<br/>Predictor))
    Data
      generate_data.py
      16 features
      1000 rows
      2% missing values
    Preprocessing
      HouseData loader
      Impute NaN
      OneHotEncoder
      StandardScaler
      ColumnTransformer
    Modeling
      Linear Regression
      Random Forest
      80/20 split
      5-fold CV
    Evaluation
      MAE
      RMSE
      R2
      residual_std
    App
      Predict tab
      Compare tab
      Charts tab
      PDF report
    Deployment
      Streamlit Cloud
      localhost:8501
```

### Dataset feature mindmap

```mermaid
mindmap
  root((16 Features))
    Numeric 11
      Area
      Bedrooms
      Bathrooms
      Age
      Lot Area
      Overall Quality
      Overall Condition
      Garage Cars
      Garage Area
      Total Basement SF
      Fireplaces
    Categorical 5
      Location 4 values
      Neighborhood 28 values
      House Style 6 values
      Central Air 2 values
      Kitchen Quality 5 values
```

---

## 📋 Kanban Diagram

```mermaid
flowchart LR
    BACKLOG["📥 Backlog"]:::col
    TODO["🗓️ To Do"]:::col
    DOING["🔨 In Progress"]:::col
    REVIEW["👀 Review"]:::col
    DONE["✅ Done"]:::col

    BACKLOG --> TODO --> DOING --> REVIEW --> DONE

    BACKLOG -.- B1["Map integration"]
    BACKLOG -.- B2["Analytics dashboard"]
    BACKLOG -.- B3["Multi-model ensemble"]

    TODO -.- T1["Add XGBoost model"]
    TODO -.- T2["Write pytest unit tests"]

    DOING -.- D1["17-column dataset refactor"]

    REVIEW -.- R1["Bug-fix PR: dead code removal"]
    REVIEW -.- R2["GUIDE.md documentation"]

    DONE -.- Z1["Linear Regression baseline"]
    DONE -.- Z2["Random Forest + toggle"]
    DONE -.- Z3["16-feature preprocessing"]
    DONE -.- Z4["Confidence band UI"]
    DONE -.- Z5["PDF report download"]

    classDef col fill:#1d3557,color:#fff,stroke:#0d1b2a,stroke-width:2px;
```

---

## 📅 Gantt Diagram

```mermaid
gantt
    title House Price Predictor — Build Timeline (8 weeks)
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Discovery
    Research & idea (Idea.md)        :a1, 2026-06-23, 4d
    PRD + TRD draft                  :a2, after a1, 4d

    section Data
    generate_data.py (5 features)    :b1, after a2, 3d
    EDA + preprocess                 :b2, after b1, 4d
    Expand to 16 features            :b3, after b2, 3d

    section Modeling
    Linear Regression baseline       :c1, after b2, 3d
    Evaluation harness               :c2, after c1, 2d
    Random Forest + model toggle     :c3, after c2, 4d
    Retrain on 16 features           :c4, after c3, 2d

    section App
    Streamlit UI scaffold            :d1, after c2, 3d
    Confidence band + charts         :d2, after d1, 3d
    PDF report                       :d3, after d2, 2d
    Compare tab                      :d4, after d3, 2d

    section Hardening
    Bug sweep + optimizations        :e1, after d4, 3d
    Docs: README/USAGE/ARCH/GUIDE    :e2, after e1, 3d
    Screenshots                      :e3, after e2, 1d

    section Launch
    Deploy to Streamlit Cloud        :f1, after e3, 2d
    v1.0 release                     :milestone, f2, after f1, 0d
```

---

## 🌿 Git Graph

```mermaid
gitGraph
    commit id: "init"
    commit id: "scaffold"
    commit id: "5-feature model"
    branch develop
    checkout develop
    commit id: "preprocessing"
    commit id: "LR + RF"
    commit id: "Streamlit UI"
    commit id: "16-feature refactor"
    commit id: "bug-fix sweep"
    branch docs/guide
    commit id: "GUIDE.md"
    checkout develop
    merge docs/guide
    checkout main
    merge develop tag: "v1.0"
```

**Branching rules**
- `main` — always deployable; tagged releases only.
- `develop` — integration branch.
- `feat/*`, `fix/*`, `docs/*` — short-lived feature branches → PR into `develop`.
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`.

---

## 📈 XY Charts

### Model performance comparison (actual measured values)

```mermaid
xychart-beta
    title "R² Score — Higher is Better"
    x-axis ["Linear Regression", "Random Forest"]
    y-axis "R² Score" 0.85 --> 0.96
    bar [0.9442, 0.9257]
```

```mermaid
xychart-beta
    title "MAE (Lakhs ₹) — Lower is Better"
    x-axis ["Linear Regression", "Random Forest"]
    y-axis "MAE" 0 --> 20
    bar [14.42, 16.07]
```

```mermaid
xychart-beta
    title "RMSE (Lakhs ₹) — Lower is Better"
    x-axis ["Linear Regression", "Random Forest"]
    y-axis "RMSE" 0 --> 25
    bar [18.71, 21.59]
```

### R² vs. dataset size (illustrative scaling curve)

```mermaid
xychart-beta
    title "R² vs. Training Data Size"
    x-axis ["100", "500", "1k", "5k", "10k", "50k"]
    y-axis "R² Score" 0 --> 1
    line [0.65, 0.82, 0.94, 0.96, 0.97, 0.98]
```

### Inference latency vs. concurrency

```mermaid
xychart-beta
    title "Inference Latency vs. Concurrency"
    x-axis ["1", "5", "10", "25", "50", "100"]
    y-axis "Latency (ms)" 0 --> 1000
    line [45, 60, 110, 320, 720, 950]
```

---

## 🥧 Pie Charts

### Feature type distribution (16 features)

```mermaid
pie showData
    title Feature Types (16 total)
    "Numeric (11)" : 11
    "Categorical (5)" : 5
```

### Encoded column budget (55 columns after preprocessing)

```mermaid
pie showData
    title Columns After ColumnTransformer (55 total)
    "Numeric (scaled)" : 11
    "Location (one-hot)" : 4
    "Neighborhood (one-hot)" : 28
    "House Style (one-hot)" : 6
    "Central Air (one-hot)" : 2
    "Kitchen Quality (one-hot)" : 5
```

### Effort allocation by phase (estimated)

```mermaid
pie showData
    title Development Effort by Phase
    "Data & Preprocessing" : 25
    "Modeling & Training" : 25
    "Streamlit UI" : 25
    "Evaluation & Reports" : 15
    "Documentation" : 10
```

---

## 🚀 Quick Start — 3 Commands

```mermaid
flowchart LR
    CMD1["① python generate_data.py"] --> CMD2["② python train.py"]
    CMD2 --> CMD3["③ streamlit run app.py"]
    CMD3 --> LIVE["🌐 http://localhost:8501"]

    style CMD1 fill:#4CAF50,color:#fff
    style CMD2 fill:#2196F3,color:#fff
    style CMD3 fill:#FF4B4B,color:#fff
    style LIVE fill:#FF9800,color:#fff
```

```bash
# 0. One-time setup
python -m venv venv
venv\Scripts\activate              # Windows CMD
source venv/bin/activate           # macOS / Linux
pip install -r requirements.txt

# 1. Generate the dataset (skip if data/house_data.csv exists)
python generate_data.py

# 2. Train BOTH models (Linear Regression + Random Forest)
python train.py
# Or train one:  python train.py --model random_forest

# 3. Launch the web app
streamlit run app.py
```

The app opens at **`http://localhost:8501`**. Stop it with `Ctrl+C`.

---

## 🧪 Common Modifications & Experiments

| I want to… | Edit this | Then run |
|------------|-----------|----------|
| Use more/fewer trees | `train.py` → `RandomForestRegressor(n_estimators=500)` | `python train.py --model random_forest` |
| Generate more data | `generate_data.py` → `N_SAMPLES = 5000` | `python generate_data.py` → `python train.py` |
| Change the 80/20 split | `preprocess.py` → `test_size=0.3` | `python train.py` |
| Add a new feature | Add column in `generate_data.py`, add to `NUMERIC_COLS`/`CATEGORICAL_COLS` in `preprocess.py`, add input in `app.py` | regenerate → retrain |
| Switch default model | `app.py` → `index=0` in model radio | restart app |
| Try an alternate dataset | Swap `data/house_data.csv` (e.g. Ames from `test_dataset01/`) | `python train.py` |

---

## 🔧 Troubleshooting Cheat-Sheet

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate venv, then `pip install -r requirements.txt` |
| `FileNotFoundError: model.joblib` | Run `python train.py` first |
| `FileNotFoundError: house_data.csv` | Run `python generate_data.py` first |
| Streamlit port 8501 in use | `streamlit run app.py --server.port 8502` |
| PowerShell execution disabled | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Predictions look wrong | Ensure you retrained after changing features: the saved `ColumnTransformer` must match the new schema |
| `venv\Scripts\activate` not recognized | Use `.\venv\Scripts\Activate.ps1` in PowerShell |
| Negative price estimate | Clamped to 0 in `Predictor.predict()` (post-bug-fix) |

---

## 📌 Final Mental Model

> **One sentence:** *Raw CSV → `preprocess.py` cleans & encodes it → `train.py` fits a model and saves it via `model.py` → `app.py` loads that artifact and serves predictions to the user.*
>
> **Three numbers to remember:** 16 features · 1000 rows · R² ≈ 0.94 (Linear) / 0.93 (Random Forest).
>
> **The single most important file to read first:** `train.py` — it shows the entire pipeline wired together top-to-bottom.

---

_GUIDE.md v1.0 — Companion to README · USAGE · PRD · TRD · ARCHITECTURE · Idea · summary._
