# Machine-learning-based-Buyer-Segmentation-and-Investment-Profiling-for-Real-Estate-Market
# 🏢 Parcl Co. Limited — ML-Based Buyer Segmentation & Investment Profiling

> **Unified Mentor Internship Project**
> Real Estate Market Intelligence using Unsupervised Machine Learning

---

## 📌 Project Overview

This project applies **K-Means and Hierarchical Clustering** to Parcl's client dataset of 2,000 records to discover hidden buyer segments. The system identifies four distinct buyer profiles that enable smarter marketing, better property recommendations, and data-driven investor targeting.

### What it does
- Cleans and encodes client demographic data
- Runs K-Means clustering (k=4) validated by Silhouette and Elbow methods
- Generates 14 EDA + clustering visualisation charts
- Produces an interactive Streamlit analytics dashboard
- Outputs a full research paper (Word document)

---

## 📁 Project Structure

```
ML Buyer Segmentation/files/
│
├── buyer_segmentation_pipeline.py   # Full ML pipeline (run this first)
├── streamlit_dashboard.py           # Interactive dashboard
├── README.md                        # This file
│
├── clients.csv                      # Input: 2,000 client records
├── properties.csv                   # Input: property transactions

## ⚙️ Requirements

**Python version:** 3.8 or higher

Install all dependencies with one command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy streamlit plotly
```

---

## 🚀 How to Run

### Step 1 — Run the ML Pipeline

```bash
python buyer_segmentation_pipeline.py
```

This will:
- Clean and encode the client data
- Run K-Means (k=4) and Hierarchical clustering
- Save all 14 charts to `parcl_output/figures/`
- Save cluster results to `parcl_output/`
- Print a full summary to the terminal

**Expected output in terminal:**
```
======================================================================
  STEP 1 — DATA CLEANING
======================================================================
  Loaded 2,000 client records, 12 columns.
  ...
  PIPELINE COMPLETE
  Silhouette score: 0.1619
  Total clients:    2,000
  Figures saved:    14
```

### Step 2 — Launch the Dashboard

```bash
streamlit run streamlit_dashboard.py
```

Your browser will open at **http://localhost:8501**

> If it doesn't open automatically, go to http://localhost:8501 manually.

---

## 📊 Buyer Segments Discovered

| Cluster | Segment | Size | Key Characteristic |
|---------|---------|------|-------------------|
| C0 | 🌍 Global Investors | 254 (12.7%) | All non-USA, 29.9% loan rate |
| C1 | 🏠 First-Time / Loan Buyers | 617 (30.9%) | **100% loan rate**, USA-dominant |
| C2 | 💰 Cash / Corporate Buyers | 1,026 (51.3%) | **0% loan rate**, largest segment |
| C3 | 🏢 Luxury / High-Satisfaction | 103 (5.1%) | All Company entities, 35% investment |

**Model:** K-Means, k=4 | **Silhouette Score:** 0.1619 | **Validated by:** Hierarchical Clustering

---

## 🖥️ Dashboard Features

The Streamlit dashboard has 4 tabs:

| Tab | Contents |
|-----|----------|
| 📊 Overview | Segment donut chart, PCA scatter, segment profile cards |
| 📈 Investor Behaviour | Age vs loan bar chart, acquisition purpose, satisfaction lines, referral heatmap |
| 🌍 Geographic Analysis | Country bars, segment mix per country, top 15 regions |
| 🔍 Segment Deep Dive | Drill into any segment — country, age, satisfaction, referral charts + CSV export |

**Sidebar filters:** Country, Acquisition Purpose, Client Type, Segment

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install <module_name>` |
| `FileNotFoundError: clients.csv` | Make sure `clients.csv` is in the **same folder** as the scripts |
| Streamlit not recognised | Try `python -m streamlit run streamlit_dashboard.py` |
| Port already in use | Run `streamlit run streamlit_dashboard.py --server.port 8502` |
| Blank browser tab | Wait 10 seconds and refresh |

---

## 📋 Data Dictionary

| Feature | Description | Type |
|---------|-------------|------|
| `client_id` | Unique client identifier | String |
| `client_type` | Individual or Company | Categorical |
| `gender` | M / F | Binary |
| `country` | Country of residence | Categorical (10 values) |
| `region` | Geographic region | Categorical (55 values) |
| `date_of_birth` | Used to derive age | Date (mixed formats) |
| `acquisition_purpose` | Home / Investment | Binary |
| `loan_applied` | Yes / No | Binary |
| `referral_channel` | Website / Agency / Client | Categorical |
| `satisfaction_score` | 1–5 rating | Integer |

---

## 🧠 Methodology Summary

```
clients.csv
    │
    ▼
Step 1: Data Cleaning
    → Parse mixed DOB formats → derive age
    → Standardise labels, remove duplicates
    │
    ▼
Step 2: Feature Encoding
    → Binary encoding: loan_applied, gender
    → Label encoding: client_type, purpose, channel, country, region
    │
    ▼
Step 3: Feature Scaling
    → StandardScaler on all 9 features
    │
    ▼
Step 4: Clustering
    → K-Means (k=4, k-means++, 20 restarts)
    → Hierarchical (Ward linkage, n=200 sample) for validation
    │
    ▼
Step 5: Evaluation
    → Elbow Method (WCSS vs k)
    → Silhouette Score: 0.1619
    │
    ▼
Step 6: Interpretation
    → 4 labelled buyer segments
    → 14 visualisation charts
    → cluster_results.csv with all client assignments
```

---

## 📄 Deliverables

| File | Description |
|------|-------------|
| `buyer_segmentation_pipeline.py` | End-to-end ML script |
| `streamlit_dashboard.py` | Interactive analytics dashboard |
| `Parcl_Buyer_Segmentation_Report.docx` | 10-section research paper |
| `parcl_output/cluster_results.csv` | 2,000 clients with cluster labels |
| `parcl_output/segment_readable_summary.csv` | Per-segment summary stats |
| `parcl_output/figures/*.png` | 14 EDA + clustering charts |

---

## 👩‍💻 Built With

- **Python 3.11**
- **pandas, numpy** — data manipulation
- **scikit-learn** — KMeans, PCA, StandardScaler, silhouette_score
- **scipy** — hierarchical clustering / dendrogram
- **matplotlib, seaborn** — pipeline charts
- **plotly** — interactive dashboard charts
- **streamlit** — web dashboard framework

---

*Prepared for Unified Mentor Internship Program — Parcl Co. Limited*
