"""
=============================================================================
Parcl Co. Limited — ML-Based Buyer Segmentation & Investment Profiling
=============================================================================
Pipeline:   Step 1  Data Cleaning
            Step 2  Feature Encoding
            Step 3  Feature Scaling
            Step 4  Clustering (K-Means + Hierarchical)
            Step 5  Optimal Cluster Selection (Elbow + Silhouette)
            Step 6  Cluster Interpretation & Labelling

Usage:
    python buyer_segmentation_pipeline.py
    (clients.csv and properties.csv must be in the same folder, or edit PATHS below)

Outputs (written to ./parcl_output/):
    • figures/  – all plots (EDA, Elbow, Silhouette, Dendrogram, Cluster profiles)
    • cluster_results.csv  – clients with assigned cluster labels
    • cluster_summary.csv  – per-cluster descriptive statistics
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR)          # expects csvs alongside script
OUT_DIR    = os.path.join(BASE_DIR, "parcl_output")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Override paths if csvs are elsewhere
CLIENTS_CSV    = "clients.csv"
PROPERTIES_CSV = "properties.csv"

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = ["#2D6A4F", "#40916C", "#74C69D", "#B7E4C7",
           "#1B4332", "#95D5B2", "#D8F3DC", "#52B788"]
CLUSTER_COLOURS = {
    0: "#E63946", 1: "#2196F3", 2: "#FF9800", 3: "#4CAF50"
}
CLUSTER_LABELS = {
    0: "C1 – Global Investors",
    1: "C2 – First-Time / Loan Buyers",
    2: "C3 – Corporate Buyers",
    3: "C4 – Luxury / High-Satisfaction",
}

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)

# =============================================================================
# STEP 1 – DATA CLEANING
# =============================================================================
print("\n" + "="*70)
print("  STEP 1 — DATA CLEANING")
print("="*70)

clients = pd.read_csv(CLIENTS_CSV)
print(f"  Loaded {len(clients):,} client records, {clients.shape[1]} columns.")

# --- 1a. Parse mixed-format dates of birth ---------------------------------
def smart_parse_dob(dob_str):
    """Handles MM/DD/YYYY, MM-DD-YYYY, DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD."""
    if pd.isna(dob_str):
        return pd.NaT
    dob_str = str(dob_str).strip()
    for fmt in ["%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y",
                "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d"]:
        try:
            return pd.to_datetime(dob_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT

clients["dob_parsed"] = clients["date_of_birth"].apply(smart_parse_dob)
null_dob = clients["dob_parsed"].isnull().sum()
print(f"  DOB parse failures: {null_dob}  (filled with median age if any)")

if null_dob > 0:
    median_dob = clients["dob_parsed"].median()
    clients["dob_parsed"].fillna(median_dob, inplace=True)

today = pd.Timestamp.today()
clients["age"] = ((today - clients["dob_parsed"]).dt.days / 365.25).round(1)

# --- 1b. Age-group bins ----------------------------------------------------
clients["age_group"] = pd.cut(
    clients["age"],
    bins=[0, 30, 40, 50, 60, 70, 120],
    labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
)

# --- 1c. Standardise labels ------------------------------------------------
for col in ["client_type", "gender", "country", "region",
            "acquisition_purpose", "loan_applied", "referral_channel"]:
    clients[col] = clients[col].str.strip().str.title()

# --- 1d. Drop duplicates ---------------------------------------------------
before = len(clients)
clients.drop_duplicates(subset="client_id", inplace=True)
print(f"  Duplicate rows removed: {before - len(clients)}")
print(f"  Missing values after cleaning:\n{clients.isnull().sum()[clients.isnull().sum()>0]}")
print(f"  Final clean dataset: {len(clients):,} clients")

# =============================================================================
# EDA — EXPLORATORY DATA ANALYSIS  (figures saved to FIG_DIR)
# =============================================================================
print("\n" + "="*70)
print("  EDA — EXPLORATORY DATA ANALYSIS")
print("="*70)

# -- Figure 1 : Client-type & acquisition purpose breakdown -----------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Parcl — Client Overview", fontsize=15, fontweight="bold", y=1.01)

# Pie: client type
ct = clients["client_type"].value_counts()
axes[0].pie(ct, labels=ct.index, autopct="%1.1f%%",
            colors=[PALETTE[0], PALETTE[2]], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Client Type")

# Bar: acquisition purpose
ap = clients["acquisition_purpose"].value_counts()
bars = axes[1].bar(ap.index, ap.values,
                   color=[PALETTE[0], PALETTE[2]], edgecolor="white", width=0.5)
for bar, val in zip(bars, ap.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val:,}", ha="center", fontsize=10, fontweight="bold")
axes[1].set_title("Acquisition Purpose")
axes[1].set_ylabel("Count")

# Bar: referral channel
rc = clients["referral_channel"].value_counts()
bars2 = axes[2].bar(rc.index, rc.values,
                    color=PALETTE[:3], edgecolor="white", width=0.5)
for bar, val in zip(bars2, rc.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{val:,}", ha="center", fontsize=10, fontweight="bold")
axes[2].set_title("Referral Channel")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_01_client_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_01_client_overview.png")

# -- Figure 2 : Age distribution & loan behaviour ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Age Distribution & Loan Behaviour", fontsize=14, fontweight="bold")

axes[0].hist(clients["age"], bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
axes[0].axvline(clients["age"].mean(), color="#E63946", linestyle="--",
                linewidth=1.8, label=f"Mean: {clients['age'].mean():.1f} yrs")
axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel("Count")
axes[0].set_title("Client Age Distribution"); axes[0].legend()

la_ct = clients.groupby(["age_group", "loan_applied"],
                         observed=True).size().unstack(fill_value=0)
la_ct.plot(kind="bar", ax=axes[1], color=[PALETTE[0], PALETTE[2]],
           edgecolor="white", width=0.65)
axes[1].set_title("Loan Applied by Age Group")
axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(title="Loan Applied")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_02_age_loan.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_02_age_loan.png")

# -- Figure 3 : Geographic distribution ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Geographic Distribution of Buyers", fontsize=14, fontweight="bold")

country_ct = clients["country"].value_counts()
sns.barplot(x=country_ct.values, y=country_ct.index,
            palette=PALETTE[:len(country_ct)], ax=axes[0])
axes[0].set_title("Buyers by Country"); axes[0].set_xlabel("Count")
for i, v in enumerate(country_ct.values):
    axes[0].text(v + 5, i, str(v), va="center", fontsize=9)

top_regions = clients["region"].value_counts().head(15)
sns.barplot(x=top_regions.values, y=top_regions.index,
            palette=PALETTE[:len(top_regions)], ax=axes[1])
axes[1].set_title("Top 15 Regions"); axes[1].set_xlabel("Count")
for i, v in enumerate(top_regions.values):
    axes[1].text(v + 3, i, str(v), va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_03_geography.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_03_geography.png")

# -- Figure 4 : Satisfaction score distribution ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Satisfaction Score Analysis", fontsize=14, fontweight="bold")

ss_ct = clients["satisfaction_score"].value_counts().sort_index()
axes[0].bar(ss_ct.index.astype(str), ss_ct.values,
            color=PALETTE[:len(ss_ct)], edgecolor="white", width=0.6)
for i, v in enumerate(ss_ct.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Score"); axes[0].set_ylabel("Count")
axes[0].set_title("Overall Satisfaction Distribution")

ss_aq = clients.groupby(["acquisition_purpose", "satisfaction_score"],
                         observed=True)["client_id"].count().unstack(fill_value=0)
ss_aq.T.plot(kind="bar", ax=axes[1], color=[PALETTE[0], PALETTE[2]],
             edgecolor="white", width=0.65)
axes[1].set_title("Satisfaction Score by Acquisition Purpose")
axes[1].set_xlabel("Satisfaction Score"); axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Purpose")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_04_satisfaction.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_04_satisfaction.png")

# -- Figure 5 : Cross-tab heatmap: country × purpose ----------------------
ct_hm = pd.crosstab(clients["country"], clients["acquisition_purpose"])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(ct_hm, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_title("Country × Acquisition Purpose Heatmap", fontsize=13, fontweight="bold")
ax.set_xlabel("Purpose"); ax.set_ylabel("Country")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_05_country_purpose_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_05_country_purpose_heatmap.png")

# =============================================================================
# STEP 2 — FEATURE ENCODING
# =============================================================================
print("\n" + "="*70)
print("  STEP 2 — FEATURE ENCODING")
print("="*70)

df = clients.copy()

# Binary label encoding (Yes/No → 1/0)
df["loan_applied_enc"]        = (df["loan_applied"] == "Yes").astype(int)
df["gender_enc"]              = (df["gender"] == "M").astype(int)

# Label encoding for ordinal-ish or low-cardinality fields
le_client_type  = LabelEncoder()
le_purpose      = LabelEncoder()
le_channel      = LabelEncoder()
le_country      = LabelEncoder()
le_region       = LabelEncoder()

df["client_type_enc"]        = le_client_type.fit_transform(df["client_type"])
df["acquisition_purpose_enc"]= le_purpose.fit_transform(df["acquisition_purpose"])
df["referral_channel_enc"]   = le_channel.fit_transform(df["referral_channel"])
df["country_enc"]            = le_country.fit_transform(df["country"])
df["region_enc"]             = le_region.fit_transform(df["region"])

# One-hot for country (top representation) — kept as supplement
country_dummies = pd.get_dummies(df["country"], prefix="country", drop_first=False)
df = pd.concat([df, country_dummies], axis=1)

print("  Encoded features created:")
enc_cols = ["loan_applied_enc", "gender_enc", "client_type_enc",
            "acquisition_purpose_enc", "referral_channel_enc",
            "country_enc", "region_enc"]
for c in enc_cols:
    print(f"    {c}: {df[c].unique()[:8]}")

# =============================================================================
# STEP 3 — FEATURE SCALING
# =============================================================================
print("\n" + "="*70)
print("  STEP 3 — FEATURE SCALING")
print("="*70)

# Feature matrix — strictly client.csv fields only
FEATURE_COLS = [
    "age",                     # numeric
    "satisfaction_score",      # numeric
    "loan_applied_enc",        # binary
    "gender_enc",              # binary
    "client_type_enc",         # label encoded
    "acquisition_purpose_enc", # label encoded
    "referral_channel_enc",    # label encoded
    "country_enc",             # label encoded
    "region_enc",              # label encoded
]

X_raw = df[FEATURE_COLS].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"  Feature matrix shape: {X_scaled.shape}")
print(f"  Features used: {FEATURE_COLS}")
print(f"  Scaling: StandardScaler — mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

# =============================================================================
# STEP 4 & 5 — OPTIMAL CLUSTER SELECTION: ELBOW + SILHOUETTE
# =============================================================================
print("\n" + "="*70)
print("  STEP 4 & 5 — ELBOW METHOD + SILHOUETTE ANALYSIS")
print("="*70)

K_RANGE = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=15,
                max_iter=400, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, random_state=42)
    silhouette_scores.append(sil)
    print(f"  k={k:2d}  inertia={km.inertia_:>10.1f}  silhouette={sil:.4f}")

# -- Figure 6 : Elbow + Silhouette side by side ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optimal Cluster Selection", fontsize=14, fontweight="bold")

# Elbow
axes[0].plot(list(K_RANGE), inertias, "o-", color=PALETTE[0],
             linewidth=2.2, markersize=7)
axes[0].axvline(4, color="#E63946", linestyle="--", linewidth=1.8,
                label="k = 4 (optimal)")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].legend()

# Silhouette
axes[1].plot(list(K_RANGE), silhouette_scores, "s-", color=PALETTE[2],
             linewidth=2.2, markersize=7)
best_k = list(K_RANGE)[np.argmax(silhouette_scores)]
axes[1].axvline(best_k, color="#E63946", linestyle="--", linewidth=1.8,
                label=f"Best k = {best_k}")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_01_elbow_silhouette.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Optimal k from silhouette: {best_k}")
print("  Saved: cluster_01_elbow_silhouette.png")

# =============================================================================
# STEP 4 — K-MEANS (k = 4)
# =============================================================================
K_FINAL = 4
print("\n" + "="*70)
print(f"  STEP 4A — K-MEANS CLUSTERING  (k={K_FINAL})")
print("="*70)

kmeans = KMeans(n_clusters=K_FINAL, init="k-means++", n_init=20,
                max_iter=500, random_state=42)
km_labels = kmeans.fit_predict(X_scaled)
df["kmeans_cluster"] = km_labels

km_sil = silhouette_score(X_scaled, km_labels)
print(f"  K-Means silhouette score: {km_sil:.4f}")
print(f"  Cluster sizes:\n{pd.Series(km_labels).value_counts().sort_index()}")

# -- Figure 7 : Silhouette plot per cluster ---------------------------------
sample_vals  = silhouette_samples(X_scaled, km_labels)
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
for i in range(K_FINAL):
    ith = np.sort(sample_vals[km_labels == i])
    size = ith.shape[0]
    y_upper = y_lower + size
    color = list(CLUSTER_COLOURS.values())[i]
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith,
                     facecolor=color, edgecolor=color, alpha=0.75)
    ax.text(-0.05, y_lower + 0.5 * size,
            CLUSTER_LABELS[i].split("–")[0].strip(), fontsize=9)
    y_lower = y_upper + 10

ax.axvline(km_sil, color="#E63946", linestyle="--", linewidth=1.8,
           label=f"Mean = {km_sil:.3f}")
ax.set_title("Silhouette Plot — K-Means (k=4)", fontsize=13, fontweight="bold")
ax.set_xlabel("Silhouette Coefficient"); ax.set_ylabel("Cluster")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_02_silhouette_plot.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_02_silhouette_plot.png")

# =============================================================================
# STEP 4B — HIERARCHICAL CLUSTERING (Dendrogram)
# =============================================================================
print("\n" + "="*70)
print("  STEP 4B — HIERARCHICAL CLUSTERING")
print("="*70)

# Sample 200 rows for dendrogram readability
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=200, replace=False)
X_sample   = X_scaled[sample_idx]

Z = linkage(X_sample, method="ward")

fig, ax = plt.subplots(figsize=(16, 6))
dend = dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]),
                  above_threshold_color="#AAAAAA", leaf_font_size=0,
                  show_leaf_counts=False, no_labels=True)
ax.axhline(y=sorted(Z[:, 2])[-3], color="#E63946",
           linestyle="--", linewidth=1.8, label="Cut line (4 clusters)")
ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage, n=200 sample)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Client Samples"); ax.set_ylabel("Linkage Distance")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_03_dendrogram.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_03_dendrogram.png")

# Validate hierarchical at k=4 -------------------------------------------------
hier_labels_sample = fcluster(Z, t=4, criterion="maxclust") - 1
hier_sil = silhouette_score(X_sample, hier_labels_sample)
print(f"  Hierarchical silhouette (n=200 sample, k=4): {hier_sil:.4f}")
print(f"  → K-Means ({km_sil:.4f}) vs Hierarchical ({hier_sil:.4f}) on sample")

# =============================================================================
# STEP 5 — PCA VISUALISATION
# =============================================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]
var_exp = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(10, 7))
for k_id in range(K_FINAL):
    mask = km_labels == k_id
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLOURS[k_id], label=CLUSTER_LABELS[k_id],
               alpha=0.65, s=45, edgecolors="white", linewidths=0.3)

# Plot centroids in PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c=list(CLUSTER_COLOURS.values()), marker="X",
           s=220, edgecolors="black", linewidths=1.2, zorder=5, label="Centroids")

ax.set_title("K-Means Clusters — PCA 2D Projection", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_04_pca_scatter.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_04_pca_scatter.png")

# =============================================================================
# STEP 6 — CLUSTER INTERPRETATION & LABELLING
# =============================================================================
print("\n" + "="*70)
print("  STEP 6 — CLUSTER INTERPRETATION")
print("="*70)

# Map cluster IDs to meaningful labels
# (determined by inspecting cluster characteristics below)
LABEL_MAP = {
    0: "Global Investors",
    1: "First-Time / Loan Buyers",
    2: "Corporate Buyers",
    3: "Luxury / High-Satisfaction",
}
df["cluster_label"] = df["kmeans_cluster"].map(LABEL_MAP)

# Descriptive stats per cluster
profile_cols = ["age", "satisfaction_score", "loan_applied_enc",
                "gender_enc", "client_type_enc",
                "acquisition_purpose_enc", "referral_channel_enc",
                "country_enc", "region_enc"]

cluster_summary = df.groupby("kmeans_cluster")[profile_cols].mean().round(3)
cluster_summary["count"] = df.groupby("kmeans_cluster")["client_id"].count()
cluster_summary["label"] = cluster_summary.index.map(LABEL_MAP)
cluster_summary = cluster_summary[["label", "count"] + profile_cols]

print("\n  Cluster Profile (means):")
print(cluster_summary.to_string())

# Readable breakdown
print("\n  Readable breakdown per cluster:")
for cid, label in LABEL_MAP.items():
    sub = df[df["kmeans_cluster"] == cid]
    print(f"\n  ── {label} (Cluster {cid}, n={len(sub)}) ──")
    print(f"    Age (mean):           {sub['age'].mean():.1f} yrs")
    print(f"    Satisfaction (mean):  {sub['satisfaction_score'].mean():.2f}")
    print(f"    Loan applied:         {(sub['loan_applied']=='Yes').mean()*100:.1f}%")
    print(f"    Acquisition purpose:  {sub['acquisition_purpose'].value_counts().to_dict()}")
    print(f"    Client type:          {sub['client_type'].value_counts().to_dict()}")
    print(f"    Top countries:        {sub['country'].value_counts().head(3).to_dict()}")
    print(f"    Top referral:         {sub['referral_channel'].value_counts().head(2).to_dict()}")

# =============================================================================
# CLUSTER PROFILE VISUALISATIONS
# =============================================================================

# -- Figure 8 : Cluster size donut -----------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
sizes  = [len(df[df["kmeans_cluster"] == k]) for k in range(K_FINAL)]
colors = list(CLUSTER_COLOURS.values())
labels = [f"{CLUSTER_LABELS[k]}\n({sizes[k]:,})" for k in range(K_FINAL)]

wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct="%1.1f%%", colors=colors,
    startangle=140, wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
    pctdistance=0.8, textprops={"fontsize": 9}
)
for at in autotexts:
    at.set_fontweight("bold")
ax.set_title("Buyer Segment Distribution", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_05_segment_donut.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: cluster_05_segment_donut.png")

# -- Figure 9 : Key metrics per cluster (grouped bar) ----------------------
metrics = {
    "Avg Age": df.groupby("kmeans_cluster")["age"].mean(),
    "Avg Satisfaction": df.groupby("kmeans_cluster")["satisfaction_score"].mean(),
    "Loan Rate (%)": df.groupby("kmeans_cluster")["loan_applied_enc"].mean() * 100,
}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Key Metrics by Buyer Segment", fontsize=14, fontweight="bold")

for ax, (metric_name, metric_vals) in zip(axes, metrics.items()):
    bars = ax.bar(
        [CLUSTER_LABELS[k].split("–")[1].strip() for k in range(K_FINAL)],
        [metric_vals[k] for k in range(K_FINAL)],
        color=list(CLUSTER_COLOURS.values()), edgecolor="white", width=0.6
    )
    for bar, val in zip(bars, [metric_vals[k] for k in range(K_FINAL)]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3, f"{val:.1f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=18)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_06_key_metrics.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_06_key_metrics.png")

# -- Figure 10: Acquisition purpose & Loan stacked bar per cluster ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Acquisition Purpose & Loan Behaviour by Cluster",
             fontsize=13, fontweight="bold")

# Acquisition purpose
ap_ct = df.groupby(["kmeans_cluster", "acquisition_purpose"],
                   observed=True).size().unstack(fill_value=0)
ap_pct = ap_ct.div(ap_ct.sum(axis=1), axis=0) * 100
ap_pct.plot(kind="bar", ax=axes[0], color=[PALETTE[0], PALETTE[2]],
            edgecolor="white", width=0.65, stacked=True)
axes[0].set_title("Acquisition Purpose %")
axes[0].set_xlabel("Cluster"); axes[0].set_ylabel("% of Segment")
axes[0].set_xticklabels([f"C{i}" for i in range(K_FINAL)], rotation=0)
axes[0].legend(title="Purpose", loc="upper right")

# Loan applied
la_ct = df.groupby(["kmeans_cluster", "loan_applied"],
                   observed=True).size().unstack(fill_value=0)
la_pct = la_ct.div(la_ct.sum(axis=1), axis=0) * 100
la_pct.plot(kind="bar", ax=axes[1], color=[PALETTE[0], PALETTE[2]],
            edgecolor="white", width=0.65, stacked=True)
axes[1].set_title("Loan Applied %")
axes[1].set_xlabel("Cluster"); axes[1].set_ylabel("% of Segment")
axes[1].set_xticklabels([f"C{i}" for i in range(K_FINAL)], rotation=0)
axes[1].legend(title="Loan Applied", loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_07_purpose_loan.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_07_purpose_loan.png")

# -- Figure 11: Referral channel heatmap by cluster -------------------------
ref_ct = df.groupby(["kmeans_cluster", "referral_channel"],
                    observed=True).size().unstack(fill_value=0)
ref_pct = ref_ct.div(ref_ct.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(ref_pct, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_title("Referral Channel Mix by Cluster (%)", fontsize=13, fontweight="bold")
ax.set_yticklabels([f"C{i}" for i in range(K_FINAL)], rotation=0)
ax.set_xlabel("Referral Channel"); ax.set_ylabel("Cluster")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_08_referral_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_08_referral_heatmap.png")

# -- Figure 12: Country mix per cluster (top 5 countries) ------------------
top5_countries = clients["country"].value_counts().head(5).index.tolist()
country_cluster = df[df["country"].isin(top5_countries)].groupby(
    ["kmeans_cluster", "country"], observed=True).size().unstack(fill_value=0)
country_pct = country_cluster.div(country_cluster.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(11, 5))
country_pct.plot(kind="bar", ax=ax,
                 color=PALETTE[:len(top5_countries)],
                 edgecolor="white", width=0.7)
ax.set_title("Country Mix per Buyer Segment (Top 5 Countries, %)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Cluster"); ax.set_ylabel("% of Segment")
ax.set_xticklabels([CLUSTER_LABELS[i].split("–")[1].strip()
                    for i in range(K_FINAL)], rotation=15)
ax.legend(title="Country", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_09_country_mix.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_09_country_mix.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("  SAVING RESULTS")
print("="*70)

output_cols = [
    "client_id", "client_type", "gender", "age", "age_group",
    "country", "region", "acquisition_purpose", "loan_applied",
    "satisfaction_score", "referral_channel",
    "kmeans_cluster", "cluster_label"
]
df[output_cols].to_csv(os.path.join(OUT_DIR, "cluster_results.csv"), index=False)
print(f"  Saved: cluster_results.csv  ({len(df):,} rows)")

cluster_summary.to_csv(os.path.join(OUT_DIR, "cluster_summary.csv"))
print(f"  Saved: cluster_summary.csv")

# Detailed readable summary
summary_rows = []
for cid, label in LABEL_MAP.items():
    sub = df[df["kmeans_cluster"] == cid]
    summary_rows.append({
        "Cluster ID":            cid,
        "Segment Label":         label,
        "Count":                 len(sub),
        "Pct of Total":          f"{len(sub)/len(df)*100:.1f}%",
        "Avg Age":               round(sub["age"].mean(), 1),
        "Avg Satisfaction":      round(sub["satisfaction_score"].mean(), 2),
        "Loan Rate %":           round((sub["loan_applied"] == "Yes").mean() * 100, 1),
        "Investment Purpose %":  round((sub["acquisition_purpose"] == "Investment").mean() * 100, 1),
        "Top Country":           sub["country"].value_counts().idxmax(),
        "Top Referral":          sub["referral_channel"].value_counts().idxmax(),
    })

readable_summary = pd.DataFrame(summary_rows)
readable_summary.to_csv(os.path.join(OUT_DIR, "segment_readable_summary.csv"), index=False)
print(f"  Saved: segment_readable_summary.csv")

print("\n" + "="*70)
print("  PIPELINE COMPLETE")
print(f"  All outputs in: {OUT_DIR}")
print("="*70)
print(f"\n  Final model:      K-Means, k={K_FINAL}")
print(f"  Silhouette score: {km_sil:.4f}")
print(f"  Total clients:    {len(df):,}")
print(f"  Figures saved:    {len(os.listdir(FIG_DIR))}")
