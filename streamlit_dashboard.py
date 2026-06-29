"""
=============================================================================
Parcl Co. Limited — Buyer Segmentation & Investment Profiling Dashboard
Professional Admin UI  |  Dark sidebar · Card grid · Plotly charts
=============================================================================
Run:  streamlit run streamlit_dashboard.py
Deps: pip install streamlit pandas numpy scikit-learn plotly
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parcl Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────
C = {
    "bg":        "#0F1624",   # page background
    "sidebar":   "#0A0F1E",   # sidebar
    "card":      "#151E2F",   # card surface
    "card2":     "#1A2540",   # card hover / alt
    "border":    "#1F2E4A",   # card border
    "accent1":   "#4F8EF7",   # blue  – C1
    "accent2":   "#34D399",   # green – C2
    "accent3":   "#F59E0B",   # amber – C3
    "accent4":   "#F43F5E",   # rose  – C4
    "text":      "#E2E8F0",   # primary text
    "muted":     "#64748B",   # secondary text
    "white":     "#FFFFFF",
}
SEG_COLORS  = [C["accent1"], C["accent2"], C["accent3"], C["accent4"]]
SEG_NAMES   = {0:"Global Investors", 1:"Loan Buyers", 2:"Cash Buyers", 3:"Corporate Buyers"}
SEG_ICONS   = {0:"🌍", 1:"🏠", 2:"💰", 3:"🏢"}
SEG_SHORT   = {0:"C1", 1:"C2", 2:"C3", 3:"C4"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=C["text"], size=12),
    margin=dict(l=16, r=16, t=36, b=16),
)

# Default axis style reused individually per chart
AXIS = dict(gridcolor=C["border"], linecolor=C["border"], tickcolor=C["muted"])

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"], .stApp {{
    background-color: {C["bg"]} !important;
    font-family: 'Inter', sans-serif !important;
    color: {C["text"]} !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {C["sidebar"]} !important;
    border-right: 1px solid {C["border"]};
}}
[data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {{
    background-color: {C["card"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px !important;
    color: {C["text"]} !important;
}}
[data-testid="stSidebar"] label {{
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: {C["muted"]} !important;
}}

/* ── Main area ── */
[data-testid="stMainBlockContainer"] {{
    background-color: {C["bg"]} !important;
    padding: 1.5rem 2rem !important;
}}
section[data-testid="stMain"] > div {{ background: {C["bg"]} !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C["card"]} !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {C["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 7px !important;
    color: {C["muted"]} !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 18px !important;
    border: none !important;
}}
.stTabs [aria-selected="true"] {{
    background: {C["accent1"]} !important;
    color: {C["white"]} !important;
    font-weight: 600 !important;
}}

/* ── Plotly charts ── */
.js-plotly-plot .plotly {{ border-radius: 12px; }}

/* ── Metrics ── */
[data-testid="stMetricValue"] {{ color: {C["white"]} !important; font-size: 1.8rem !important; font-weight: 700 !important; }}
[data-testid="stMetricLabel"] {{ color: {C["muted"]} !important; font-size: 0.8rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }}
[data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}
[data-testid="metric-container"] {{
    background: {C["card"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 12px !important;
    padding: 20px 22px !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    background: {C["card"]} !important;
    border-radius: 10px;
    border: 1px solid {C["border"]};
}}
[data-testid="stDataFrameResizable"] th {{
    background: {C["card2"]} !important;
    color: {C["muted"]} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {{
    background: {C["accent1"]} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {{
    background: {C["card"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px !important;
    color: {C["text"]} !important;
}}

/* ── Section divider ── */
hr {{ border-color: {C["border"]} !important; margin: 1.5rem 0 !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {C["border"]}; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)

# ── Helper: styled card HTML ───────────────────────────────────────────────
def card(content_html, border_color=None, padding="20px 22px"):
    border_left = f"border-left: 4px solid {border_color};" if border_color else ""
    return f"""
    <div style="background:{C['card']};border:1px solid {C['border']};
                {border_left}border-radius:12px;padding:{padding};
                margin-bottom:12px;box-shadow:0 2px 12px rgba(0,0,0,0.3);">
        {content_html}
    </div>"""

def section_header(title, subtitle=""):
    sub = f'<p style="color:{C["muted"]};font-size:0.82rem;margin:2px 0 0">{subtitle}</p>' if subtitle else ""
    return st.markdown(f"""
    <div style="margin-bottom:20px">
        <h2 style="color:{C['white']};font-size:1.15rem;font-weight:700;
                   margin:0;letter-spacing:-0.01em">{title}</h2>
        {sub}
    </div>""", unsafe_allow_html=True)

def badge(text, color):
    return f'<span style="background:{color}22;color:{color};border:1px solid {color}44;\
font-size:0.72rem;font-weight:600;padding:2px 10px;border-radius:20px;\
letter-spacing:0.04em">{text}</span>'

# ── Data pipeline ──────────────────────────────────────────────────────────
CLIENTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clients.csv")

@st.cache_data(show_spinner="Loading segmentation model…")
def run_pipeline(path):
    df = pd.read_csv(path)
    def parse_dob(d):
        if pd.isna(d): return pd.NaT
        for fmt in ["%m/%d/%Y","%m-%d-%Y","%d/%m/%Y","%d-%m-%Y","%Y-%m-%d"]:
            try: return pd.to_datetime(str(d).strip(), format=fmt)
            except: pass
        return pd.NaT
    df["dob_p"] = df["date_of_birth"].apply(parse_dob)
    df["dob_p"].fillna(df["dob_p"].median(), inplace=True)
    df["age"] = ((pd.Timestamp.today()-df["dob_p"]).dt.days/365.25).round(1)
    df["age_group"] = pd.cut(df["age"],[0,30,40,50,60,70,120],
                             labels=["<30","30-40","40-50","50-60","60-70","70+"])
    for c in ["client_type","gender","country","region",
              "acquisition_purpose","loan_applied","referral_channel"]:
        df[c] = df[c].str.strip().str.title()
    df["loan_enc"]  = (df["loan_applied"]=="Yes").astype(int)
    df["gender_enc"]= (df["gender"]=="M").astype(int)
    for c,n in [("client_type","ct_enc"),("acquisition_purpose","ap_enc"),
                ("referral_channel","rc_enc"),("country","co_enc"),("region","rg_enc")]:
        le=LabelEncoder(); df[n]=le.fit_transform(df[c])
    feats=["age","satisfaction_score","loan_enc","gender_enc",
           "ct_enc","ap_enc","rc_enc","co_enc","rg_enc"]
    X = StandardScaler().fit_transform(df[feats])
    km = KMeans(n_clusters=4,init="k-means++",n_init=20,max_iter=500,random_state=42)
    df["cluster"] = km.fit_predict(X)
    df["segment"] = df["cluster"].map(SEG_NAMES)
    pca = PCA(n_components=2,random_state=42)
    Xp  = pca.fit_transform(X)
    df["pca1"],df["pca2"] = Xp[:,0],Xp[:,1]
    sil = silhouette_score(X, df["cluster"])
    return df, sil

df, sil = run_pipeline(CLIENTS_CSV)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 24px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <div style="background:{C['accent1']};border-radius:8px;
                        padding:6px 9px;font-size:1.1rem">🏢</div>
            <div>
                <div style="font-weight:700;font-size:1rem;color:{C['white']}">Parcl</div>
                <div style="font-size:0.72rem;color:{C['muted']}">Buyer Intelligence</div>
            </div>
        </div>
    </div>
    <hr style="border-color:{C['border']};margin:0 0 20px">
    """, unsafe_allow_html=True)

    st.markdown(f'<p style="color:{C["muted"]};font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px">Filters</p>', unsafe_allow_html=True)

    sel_country = st.multiselect("Country", sorted(df["country"].unique()), default=list(df["country"].unique()))
    sel_purpose = st.multiselect("Acquisition Purpose", df["acquisition_purpose"].unique().tolist(), default=df["acquisition_purpose"].unique().tolist())
    sel_type    = st.multiselect("Client Type", df["client_type"].unique().tolist(), default=df["client_type"].unique().tolist())
    sel_seg     = st.multiselect("Segment", list(SEG_NAMES.values()), default=list(SEG_NAMES.values()))

    st.markdown(f'<hr style="border-color:{C["border"]};margin:20px 0">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{C['card']};border:1px solid {C['border']};
                border-radius:10px;padding:14px 16px">
        <p style="color:{C['muted']};font-size:0.7rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.08em;margin:0 0 10px">Model Info</p>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="color:{C['muted']};font-size:0.8rem">Algorithm</span>
            <span style="color:{C['text']};font-size:0.8rem;font-weight:600">K-Means k=4</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="color:{C['muted']};font-size:0.8rem">Silhouette</span>
            <span style="color:{C['accent2']};font-size:0.8rem;font-weight:600">{sil:.4f}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:{C['muted']};font-size:0.8rem">Total Clients</span>
            <span style="color:{C['text']};font-size:0.8rem;font-weight:600">{len(df):,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────
fdf = df[
    df["country"].isin(sel_country) &
    df["acquisition_purpose"].isin(sel_purpose) &
    df["client_type"].isin(sel_type) &
    df["segment"].isin(sel_seg)
]

# ── Top header ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start;
            margin-bottom:28px">
    <div>
        <h1 style="color:{C['white']};font-size:1.6rem;font-weight:700;
                   margin:0;letter-spacing:-0.02em">Buyer Segmentation Dashboard</h1>
        <p style="color:{C['muted']};font-size:0.85rem;margin:4px 0 0">
            Real Estate Market Intelligence — Parcl Co. Limited &nbsp;·&nbsp;
            <span style="color:{C['accent2']}">{len(fdf):,} clients</span> after filters
        </p>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
        {badge("LIVE", C['accent2'])}
        {badge("K-MEANS · k=4", C['accent1'])}
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────
kpi_cols = st.columns(4)
for i, k in enumerate(range(4)):
    n   = len(fdf[fdf["cluster"]==k])
    pct = n/max(len(fdf),1)*100
    avg_sat = fdf[fdf["cluster"]==k]["satisfaction_score"].mean()
    loan_r  = fdf[fdf["cluster"]==k]["loan_enc"].mean()*100
    kpi_cols[i].markdown(f"""
    <div style="background:{C['card']};border:1px solid {C['border']};
                border-top:3px solid {SEG_COLORS[k]};border-radius:12px;
                padding:20px 20px 16px;box-shadow:0 4px 16px rgba(0,0,0,0.25)">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div style="font-size:1.55rem">{SEG_ICONS[k]}</div>
            <div style="background:{SEG_COLORS[k]}22;color:{SEG_COLORS[k]};
                        font-size:0.7rem;font-weight:700;padding:3px 9px;
                        border-radius:20px;border:1px solid {SEG_COLORS[k]}44">
                {SEG_SHORT[k]}
            </div>
        </div>
        <div style="color:{C['white']};font-size:1.9rem;font-weight:700;
                    line-height:1;margin-bottom:2px">{n:,}</div>
        <div style="color:{C['muted']};font-size:0.78rem;font-weight:500;
                    margin-bottom:14px">{SEG_NAMES[k]}</div>
        <div style="display:flex;justify-content:space-between;
                    border-top:1px solid {C['border']};padding-top:10px">
            <div>
                <div style="color:{C['muted']};font-size:0.68rem;text-transform:uppercase;
                            letter-spacing:0.06em">Share</div>
                <div style="color:{SEG_COLORS[k]};font-size:0.88rem;font-weight:600">{pct:.1f}%</div>
            </div>
            <div>
                <div style="color:{C['muted']};font-size:0.68rem;text-transform:uppercase;
                            letter-spacing:0.06em">Avg Sat.</div>
                <div style="color:{C['text']};font-size:0.88rem;font-weight:600">{avg_sat:.2f}/5</div>
            </div>
            <div>
                <div style="color:{C['muted']};font-size:0.68rem;text-transform:uppercase;
                            letter-spacing:0.06em">Loan %</div>
                <div style="color:{C['text']};font-size:0.88rem;font-weight:600">{loan_r:.0f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  Overview  ",
    "  📈  Investor Behaviour  ",
    "  🌍  Geographic Analysis  ",
    "  🔍  Segment Deep Dive  ",
])

# ══════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.05, 0.95], gap="medium")

    with col_l:
        section_header("Segment Distribution", "Share of total filtered clients")
        sizes  = [len(fdf[fdf["cluster"]==k]) for k in range(4)]
        labels = [f"{SEG_ICONS[k]} {SEG_NAMES[k]}" for k in range(4)]
        fig = go.Figure(go.Pie(
            values=sizes, labels=labels,
            hole=0.62,
            marker=dict(colors=SEG_COLORS, line=dict(color=C["bg"], width=3)),
            textfont=dict(size=12, color=C["text"]),
            hovertemplate="<b>%{label}</b><br>%{value:,} clients<br>%{percent}<extra></extra>",
        ))
        total = sum(sizes)
        fig.add_annotation(text=f"<b>{total:,}</b>", font=dict(size=26, color=C["white"]),
                           showarrow=False, y=0.06)
        fig.add_annotation(text="Total Clients", font=dict(size=11, color=C["muted"]),
                           showarrow=False, y=-0.06)
        fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          legend=dict(orientation="v", x=1.0, y=0.5,
                                      font=dict(size=11, color=C["text"]),
                                      bgcolor="rgba(0,0,0,0)"))
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px;box-shadow:0 4px 20px rgba(0,0,0,0.3)">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        section_header("PCA Cluster Projection", "2D view of feature-space separation")
        fig2 = go.Figure()
        for k in range(4):
            mask = fdf["cluster"]==k
            fig2.add_trace(go.Scatter(
                x=fdf.loc[mask,"pca1"], y=fdf.loc[mask,"pca2"],
                mode="markers",
                marker=dict(color=SEG_COLORS[k], size=5, opacity=0.65,
                            line=dict(width=0)),
                name=f"{SEG_ICONS[k]} {SEG_NAMES[k]}",
                hovertemplate=f"<b>{SEG_NAMES[k]}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                           xaxis_title="PC1", yaxis_title="PC2",
                           legend=dict(orientation="v", x=1.0, y=1,
                                       font=dict(size=10)))
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px;box-shadow:0 4px 20px rgba(0,0,0,0.3)">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Segment summary cards
    section_header("Segment Profiles", "Key characteristics at a glance")
    seg_cols = st.columns(4, gap="small")
    profiles = {
        0: ("Non-USA individual buyers. Low loan rate, cash-flexible.", ["International","Cash-flexible","Website & Agency"]),
        1: ("100% loan-financed. Largest USA-domestic buyer group.", ["100% Loan rate","USA-dominant","Home-focused"]),
        2: ("Zero loan applications. Established cash buyers, all USA.", ["0% Loan rate","USA dominant","Largest segment"]),
        3: ("All corporate entities. Highest investment purpose.", ["All Corporate","Highest invest %","B2B referral"]),
    }
    for k in range(4):
        desc, tags = profiles[k]
        tag_html = "".join([f'<span style="background:{SEG_COLORS[k]}18;color:{SEG_COLORS[k]};'
                            f'font-size:0.68rem;font-weight:600;padding:2px 8px;border-radius:4px;'
                            f'margin-right:4px;margin-bottom:4px;display:inline-block">{t}</span>'
                            for t in tags])
        seg_cols[k].markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};
                    border-radius:12px;padding:16px;height:170px;
                    box-shadow:0 4px 16px rgba(0,0,0,0.25)">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
                <span style="font-size:1.3rem">{SEG_ICONS[k]}</span>
                <div>
                    <div style="color:{SEG_COLORS[k]};font-size:0.72rem;font-weight:700;
                                text-transform:uppercase;letter-spacing:0.06em">{SEG_SHORT[k]}</div>
                    <div style="color:{C['white']};font-size:0.88rem;font-weight:600">{SEG_NAMES[k]}</div>
                </div>
            </div>
            <p style="color:{C['muted']};font-size:0.78rem;line-height:1.5;margin-bottom:10px">{desc}</p>
            <div style="flex-wrap:wrap">{tag_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — INVESTOR BEHAVIOUR
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_header("Investor Behaviour Dashboard", "Financing, age, purpose and satisfaction by segment")

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        # Grouped bar: age & loan side by side
        age_m  = [fdf[fdf["cluster"]==k]["age"].mean() for k in range(4)]
        loan_r = [fdf[fdf["cluster"]==k]["loan_enc"].mean()*100 for k in range(4)]
        names  = [SEG_NAMES[k] for k in range(4)]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Avg Age (yrs)", x=names, y=age_m,
                             marker_color=C["accent1"], width=0.35,
                             hovertemplate="<b>%{x}</b><br>Avg Age: %{y:.1f} yrs<extra></extra>"))
        fig.add_trace(go.Bar(name="Loan Rate (%)", x=names, y=loan_r,
                             marker_color=C["accent3"], width=0.35,
                             hovertemplate="<b>%{x}</b><br>Loan Rate: %{y:.1f}%<extra></extra>"))
        fig.update_layout(**PLOTLY_LAYOUT, title="Age vs Loan Rate by Segment",
                          barmode="group", height=320,
                          xaxis_tickangle=-15)
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Stacked bar: acquisition purpose %
        home_pct = []; inv_pct = []
        for k in range(4):
            sub = fdf[fdf["cluster"]==k]
            t = max(len(sub),1)
            home_pct.append((sub["acquisition_purpose"]=="Home").sum()/t*100)
            inv_pct.append((sub["acquisition_purpose"]=="Investment").sum()/t*100)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Home", x=names, y=home_pct,
                              marker_color=C["accent2"],
                              hovertemplate="<b>%{x}</b><br>Home: %{y:.1f}%<extra></extra>"))
        fig2.add_trace(go.Bar(name="Investment", x=names, y=inv_pct,
                              marker_color=C["accent4"],
                              hovertemplate="<b>%{x}</b><br>Investment: %{y:.1f}%<extra></extra>"))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Acquisition Purpose by Segment (%)",
                           barmode="stack", height=320, xaxis_tickangle=-15)
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="medium")

    with c3:
        # Satisfaction radar / line
        scores = [1,2,3,4,5]
        fig3 = go.Figure()
        for k in range(4):
            sub  = fdf[fdf["cluster"]==k]
            vals = [len(sub[sub["satisfaction_score"]==s]) for s in scores]
            fig3.add_trace(go.Scatter(
                x=scores, y=vals, mode="lines+markers",
                name=SEG_NAMES[k],
                line=dict(color=SEG_COLORS[k], width=2.5),
                marker=dict(size=7),
                hovertemplate=f"Score %{{x}}: %{{y}} clients<extra>{SEG_NAMES[k]}</extra>",
            ))
        fig3.update_layout(**PLOTLY_LAYOUT, title="Satisfaction Score Distribution",
                           height=320,
                           xaxis=dict(tickvals=scores, title="Score", gridcolor=C["border"], linecolor=C["border"]),
                           yaxis=dict(title="Count", gridcolor=C["border"], linecolor=C["border"]))
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        # Referral channel heatmap
        rc_data = fdf.groupby(["cluster","referral_channel"], observed=True).size().unstack(fill_value=0)
        rc_pct  = rc_data.div(rc_data.sum(axis=1), axis=0) * 100
        fig4 = go.Figure(go.Heatmap(
            z=rc_pct.values,
            x=rc_pct.columns.tolist(),
            y=[SEG_NAMES[k] for k in rc_pct.index],
            colorscale=[[0,"#151E2F"],[0.5,C["accent1"]],[1,C["accent2"]]],
            text=rc_pct.values.round(1),
            texttemplate="%{text}%",
            textfont=dict(size=13, color=C["white"]),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
            showscale=True,
            colorbar=dict(tickfont=dict(color=C["muted"]), bgcolor="rgba(0,0,0,0)"),
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, title="Referral Channel Mix (%)", height=320)
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_header("Geographic Buyer Analysis", "Country and region distribution by segment")

    c1, c2 = st.columns([1,1], gap="medium")

    with c1:
        country_ct = fdf["country"].value_counts().reset_index()
        country_ct.columns = ["Country","Count"]
        fig = px.bar(country_ct, x="Count", y="Country", orientation="h",
                     color="Count",
                     color_continuous_scale=[[0,"rgba(79,142,247,0.15)"],[1,"rgba(79,142,247,1)"]],
                     labels={"Count":"Clients"},
                     title="Clients by Country")
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          coloraxis_showscale=False,
                          xaxis=AXIS,
                          yaxis=dict(autorange="reversed", gridcolor=C["border"], linecolor=C["border"]))
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} clients<extra></extra>")
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        top5 = df["country"].value_counts().head(5).index.tolist()
        cc   = fdf[fdf["country"].isin(top5)].groupby(["country","cluster"], observed=True).size().reset_index(name="n")
        cc["segment"] = cc["cluster"].map(SEG_NAMES)
        cc["color"]   = cc["cluster"].map(dict(enumerate(SEG_COLORS)))
        fig2 = px.bar(cc, x="country", y="n", color="segment",
                      color_discrete_map={SEG_NAMES[k]: SEG_COLORS[k] for k in range(4)},
                      barmode="stack", title="Segment Mix per Country",
                      labels={"n":"Clients","country":"Country","segment":"Segment"})
        fig2.update_layout(**PLOTLY_LAYOUT, height=380,
                           legend=dict(font=dict(size=10), orientation="h", y=-0.2))
        fig2.update_traces(hovertemplate="<b>%{x}</b><br>%{y} clients<extra></extra>")
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Top regions full width
    top_reg = fdf["region"].value_counts().head(15).reset_index()
    top_reg.columns = ["Region","Count"]
    fig3 = px.bar(top_reg, x="Region", y="Count",
                  color="Count",
                  color_continuous_scale=[[0,"rgba(52,211,153,0.12)"],[1,"rgba(52,211,153,1)"]],
                  title="Top 15 Regions by Client Volume")
    fig3.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_showscale=False)
    fig3.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} clients<extra></extra>")
    st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px;margin-top:12px">', unsafe_allow_html=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 4 — SEGMENT DEEP DIVE
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    sel_k = st.selectbox(
        "Select Segment",
        options=list(SEG_NAMES.values()),
        format_func=lambda x: f"{SEG_ICONS[list(SEG_NAMES.values()).index(x)]}  {x}",
    )
    k_id = [k for k,v in SEG_NAMES.items() if v==sel_k][0]
    sub  = fdf[fdf["cluster"]==k_id]

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # KPI strip
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Clients", f"{len(sub):,}")
    m2.metric("Avg Age", f"{sub['age'].mean():.1f} yrs")
    m3.metric("Avg Satisfaction", f"{sub['satisfaction_score'].mean():.2f} / 5")
    m4.metric("Loan Rate", f"{sub['loan_enc'].mean()*100:.1f}%")
    m5.metric("Investment %", f"{(sub['acquisition_purpose']=='Investment').mean()*100:.1f}%")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        cc = sub["country"].value_counts().head(7).reset_index()
        cc.columns = ["Country","Count"]
        fig = px.bar(cc, x="Count", y="Country", orientation="h",
                     title="Top Countries",
                     color_discrete_sequence=[SEG_COLORS[k_id]])
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis=AXIS,
                          yaxis=dict(autorange="reversed", gridcolor=C["border"], linecolor=C["border"]))
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        fig2 = px.histogram(sub, x="age", nbins=20, title="Age Distribution",
                            color_discrete_sequence=[SEG_COLORS[k_id]])
        fig2.add_vline(x=sub["age"].mean(), line_dash="dash",
                       line_color=C["accent4"],
                       annotation_text=f"Mean {sub['age'].mean():.1f}",
                       annotation_font_color=C["accent4"])
        fig2.update_layout(**PLOTLY_LAYOUT, height=300)
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        rc = sub["referral_channel"].value_counts().reset_index()
        rc.columns = ["Channel","Count"]
        fig3 = go.Figure(go.Pie(
            values=rc["Count"], labels=rc["Channel"], hole=0.55,
            marker=dict(colors=[SEG_COLORS[k_id], C["accent3"], C["accent2"]],
                        line=dict(color=C["bg"], width=2)),
            textfont=dict(size=12, color=C["text"]),
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, title="Referral Channel", height=300)
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};\
border-radius:12px;padding:16px">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent clients table
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section_header("Client Records", f"Showing up to 100 clients from {sel_k}")

    cols_show = ["client_id","client_type","gender","age","country","region",
                 "acquisition_purpose","loan_applied","satisfaction_score","referral_channel"]
    st.dataframe(
        sub[cols_show].head(100).reset_index(drop=True),
        use_container_width=True, height=320
    )

    csv_out = sub[cols_show].to_csv(index=False)
    st.download_button(
        f"⬇️  Export {sel_k} data",
        data=csv_out,
        file_name=f"parcl_{sel_k.replace(' ','_')}.csv",
        mime="text/csv"
    )
