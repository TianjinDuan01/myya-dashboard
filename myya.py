import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from scipy.stats import chi2_contingency


# =========================
# CONFIG
# =========================
TARGET_BRA = "Myya Bra"
TARGET_SIZE = "Myya Bra Size"
OTHER_BRA_COLS = ["Other Bra 1", "Other Bra 2"]

SEGMENT_FIELDS = [
    "Patient Surgery Type",
    "Breast Projection (Profile)",
    "Side",
    "Patient Age",
    "Bra Size (Band Size)",
    "Underbust Measurement (in)",
]

BUSINESS_DRIVER_CANDIDATES = [
    "Patient Surgery Type",
    "Breast Projection (Profile)",
    "Underbust Measurement (in)",
    "Bra Size (Band Size)",
    "Patient Age",
]

DRIVER_EXPLANATIONS_BUSINESS = {
    "Patient Surgery Type": "Different pathways create different constraints—standardization reduces exceptions and rework.",
    "Breast Projection (Profile)": "Key differentiator that helps avoid trial-and-error and improves repeatability.",
    "Underbust Measurement (in)": "Better measurement completeness improves consistency and reduces sizing ambiguity.",
    "Bra Size (Band Size)": "Familiar anchor that helps standardize across brand inconsistencies.",
    "Patient Age": "May reflect comfort/stability preferences; helps tailor without losing consistency.",
}

# Data URL config
DEFAULT_DATA_URL = ""          # optional fallback
DATA_URL_QUERY_KEY = "data"    # https://xxx.streamlit.app/?data=https://...csv
REQUEST_TIMEOUT = 30

# Brand accent (Myya Pink)
PINK = "#ff2d7a"
PINK_SOFT = "#ffe4ee"
INK = "#111827"
MUTED = "#6b7280"


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(page_title="Myya — Executive Story", layout="wide")

st.markdown(
    f"""
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
.block-container {{padding-top: 1.0rem; padding-bottom: 2.6rem; max-width: 1180px;}}

.smallcap {{font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: {MUTED};}}
.heroTitle {{font-size: 46px; font-weight: 950; line-height: 1.05; margin: 8px 0 10px; color:{INK};}}
.heroSub {{font-size: 18px; color: #374151; max-width: 980px; line-height: 1.55;}}

.storyBlock {{padding: 14px 0 6px;}}
.blockKicker {{font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; margin-bottom: 6px;}}
.blockHeadline {{font-size: 26px; font-weight: 950; margin: 0 0 6px; color: {INK};}}
.blockBody {{font-size: 14px; color: #4b5563; margin: 0 0 10px; line-height: 1.55;}}

.card {{border: 1px solid #ececec; border-radius: 16px; padding: 14px 16px; background: #fff;}}
.cardPink {{border: 1px solid {PINK_SOFT}; background: linear-gradient(180deg, #ffffff 0%, {PINK_SOFT} 160%);}}
.cardTitle {{font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; margin-bottom: 6px;}}
.cardValue {{font-size: 30px; font-weight: 950; color: {INK}; line-height: 1.1;}}
.cardNote {{font-size: 13px; color: #4b5563; margin-top: 6px; line-height: 1.35;}}

.divider {{border: none; border-top: 1px solid #f1f1f1; margin: 18px 0;}}
.pill {{
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  background: {PINK_SOFT}; color:{PINK}; font-size: 12px; font-weight: 800;
  margin-right: 6px; border: 1px solid {PINK_SOFT};
}}
</style>
    """,
    unsafe_allow_html=True
)

PLOTLY_LAYOUT_BASE = dict(
    margin=dict(l=10, r=10, t=55, b=10),
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial", size=13, color=INK),
    title=dict(font=dict(size=16, color=INK)),
)


# =========================
# RENDER HELPERS (compat across Streamlit versions)
# =========================
def plot(fig):
    """Plotly chart, uses stretch width if supported."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

def table(df, height=260):
    """Dataframe, uses stretch width if supported."""
    try:
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        st.dataframe(df, use_container_width=True, height=height)


# =========================
# DATA + STATS HELPERS
# =========================
def series_clean_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "—": np.nan, "–": np.nan})
    return s

def pct(x, digits=0):
    if pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"

def safe_nunique(s: pd.Series) -> int:
    return int(s.dropna().nunique()) if s is not None else 0

def interval_to_label(x):
    if isinstance(x, pd.Interval):
        return f"{x.left:g}–{x.right:g}"
    return str(x)

def cramers_v(cm: np.ndarray) -> float:
    if cm.size == 0:
        return np.nan
    chi2, _, _, _ = chi2_contingency(cm)
    n = cm.sum()
    if n <= 1:
        return np.nan
    r, k = cm.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt((chi2 / n) / denom))

def eta_squared_by_groups(num: pd.Series, grp: pd.Series) -> float:
    sub = pd.DataFrame({"num": num, "grp": grp}).dropna()
    if sub.empty or sub["grp"].nunique() < 2:
        return np.nan
    grand_mean = sub["num"].mean()
    ss_total = ((sub["num"] - grand_mean) ** 2).sum()
    if ss_total <= 0:
        return np.nan
    ss_between = sub.groupby("grp")["num"].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
    return float(ss_between / ss_total)

def driver_strength_internal(df, feature, target_cat):
    # method labels intentionally hidden; internal scoring only
    if feature not in df.columns or target_cat not in df.columns:
        return np.nan
    if pd.api.types.is_numeric_dtype(df[feature]):
        return eta_squared_by_groups(df[feature], df[target_cat])
    sub = df[[feature, target_cat]].dropna()
    if sub.empty:
        return np.nan
    ct = pd.crosstab(sub[feature], sub[target_cat]).values
    return cramers_v(ct)

def impact_level(v: float) -> str:
    if pd.isna(v): return "—"
    if v >= 0.30: return "High"
    if v >= 0.10: return "Medium"
    return "Low"

def hhi(shares: np.ndarray) -> float:
    if shares is None or len(shares) == 0:
        return np.nan
    return float(np.sum(np.square(shares)))

def card(title, value, note, pink=False):
    cls = "card cardPink" if pink else "card"
    st.markdown(
        f"""
<div class="{cls}">
  <div class="cardTitle">{title}</div>
  <div class="cardValue">{value}</div>
  <div class="cardNote">{note}</div>
</div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def load_and_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    key_col = "Patient ID (Actual - Primary Key)"
    if key_col in df.columns:
        df[key_col] = df[key_col].astype(str).str.strip()
        df = df[df[key_col] != "EHR Data"].copy()

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = series_clean_str(df[c])

    for c in ["Patient Surgery Type", "Side", "Breast Projection (Profile)"]:
        if c in df.columns:
            df[c] = series_clean_str(df[c]).str.replace(r"\s+", " ", regex=True).str.title()

    for c in ["Patient Age", "Underbust Measurement (in)", "Bra Size (Band Size)", "Sternum→Back (in)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def normalize_data_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    # Google Drive share link -> direct download
    if "drive.google.com" in url and "/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    url = normalize_data_url(url)
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

def get_data_url() -> str:
    # 1) Streamlit secrets (Cloud). Local run may have no secrets.toml -> must not crash.
    try:
        if "DATA_URL" in st.secrets and str(st.secrets["DATA_URL"]).strip():
            return str(st.secrets["DATA_URL"]).strip()
    except Exception:
        pass

    # 2) Query param (?data=...)
    try:
        qp = st.query_params
        if DATA_URL_QUERY_KEY in qp and str(qp[DATA_URL_QUERY_KEY]).strip():
            return str(qp[DATA_URL_QUERY_KEY]).strip()
    except Exception:
        pass

    # 3) Default fallback
    return DEFAULT_DATA_URL


# =========================
# SIDEBAR (URL + global controls FIRST)
# =========================
with st.sidebar:
    st.markdown("## Data source")
    data_url_input = st.text_input(
        "CSV URL (optional override)",
        value="",
        placeholder="https://.../your.csv",
        help="Leave empty to use Streamlit Secrets (DATA_URL) or ?data= query param."
    )

    st.markdown("## Controls")
    top_n = st.slider("Top N items", 5, 20, 10)
    show_details = st.toggle("Show detail tables", value=False)


# =========================
# LOAD DATA (MUST happen AFTER data_url_input exists)
# =========================
data_url = (data_url_input or "").strip() or get_data_url()
data_url = normalize_data_url(data_url)

if not data_url:
    st.error(
        "No data URL provided.\n\n"
        "Provide one of the following:\n"
        "• Streamlit Secrets: DATA_URL=\"https://...csv\"\n"
        "• Query param: ?data=https://...csv\n"
        "• Sidebar input: CSV URL"
    )
    st.stop()

try:
    raw_df = load_csv_from_url(data_url)
except Exception as e:
    st.error(f"Failed to load CSV from URL:\n{data_url}\n\nError: {e}")
    st.stop()

df = load_and_clean_df(raw_df)

# Validate required columns early
for req in [TARGET_BRA, TARGET_SIZE]:
    if req not in df.columns:
        st.error(f"Missing required column: {req}")
        st.stop()


# =========================
# SIDEBAR (filters AFTER data loaded)
# =========================
with st.sidebar:
    st.divider()
    st.markdown("## Filters")
    st.caption("Filters apply to the whole story.")

    def multiselect_filter(col, df_):
        if col not in df_.columns:
            return None
        opts = sorted(df_[col].dropna().astype(str).unique().tolist())
        if not opts:
            return None
        return st.multiselect(col, opts, default=opts)

    f_surg = multiselect_filter("Patient Surgery Type", df)
    f_proj = multiselect_filter("Breast Projection (Profile)", df)
    f_side = multiselect_filter("Side", df)

    age_range = None
    if "Patient Age" in df.columns and pd.api.types.is_numeric_dtype(df["Patient Age"]) and df["Patient Age"].notna().any():
        mn, mx = float(df["Patient Age"].min()), float(df["Patient Age"].max())
        age_range = st.slider("Age range", mn, mx, (mn, mx))

    st.divider()
    driver_options = [c for c in BUSINESS_DRIVER_CANDIDATES if c in df.columns]
    picked_driver = st.selectbox("Driver spotlight", driver_options, index=0 if driver_options else 0)


# =========================
# APPLY FILTERS
# =========================
df_f = df.copy()
if f_surg is not None:
    df_f = df_f[df_f["Patient Surgery Type"].astype(str).isin(f_surg)]
if f_proj is not None:
    df_f = df_f[df_f["Breast Projection (Profile)"].astype(str).isin(f_proj)]
if f_side is not None:
    df_f = df_f[df_f["Side"].astype(str).isin(f_side)]
if age_range is not None:
    df_f = df_f[df_f["Patient Age"].between(age_range[0], age_range[1], inclusive="both")]

if len(df_f) < 5:
    st.warning("Not enough rows after filters. Please broaden filters.")
    st.stop()


# =========================
# HERO
# =========================
st.markdown('<div class="smallcap">Myya • Executive Story</div>', unsafe_allow_html=True)
st.markdown('<div class="heroTitle">Myya wins against alternatives</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="heroSub">
This page is designed for CEOs and website visitors: it starts with replacement proof in real-world decisions,
explains what drives repeatability, and ends with where to scale next.
</div>
<hr class="divider"/>
    """,
    unsafe_allow_html=True
)


# =========================
# KPIs (replacement-first)
# =========================
cases = len(df_f)
bra_types = safe_nunique(df_f[TARGET_BRA])
size_types = safe_nunique(df_f[TARGET_SIZE])
underbust_missing = df_f["Underbust Measurement (in)"].isna().mean() if "Underbust Measurement (in)" in df_f.columns else np.nan

available_other = [c for c in OTHER_BRA_COLS if c in df_f.columns]
win_rate = None
considered = None
most_replaced = None

if available_other:
    tmp = df_f[[TARGET_BRA] + available_other].copy()
    tmp["competitor_mentioned"] = tmp[available_other].notna().any(axis=1)
    considered = int(tmp["competitor_mentioned"].sum())
    selected_anyway = int(tmp.loc[tmp["competitor_mentioned"], TARGET_BRA].notna().sum())
    win_rate = selected_anyway / max(considered, 1)

    myya_set = set(df_f[TARGET_BRA].dropna().astype(str).str.strip().unique().tolist())
    melted0 = tmp.melt(id_vars=[TARGET_BRA], value_vars=available_other, value_name="competitor").dropna(subset=["competitor", TARGET_BRA])
    melted0["competitor"] = melted0["competitor"].astype(str).str.strip()
    melted0[TARGET_BRA] = melted0[TARGET_BRA].astype(str).str.strip()
    melted0 = melted0[~melted0["competitor"].isin(myya_set)]
    if not melted0.empty:
        most_replaced = melted0["competitor"].value_counts().index[0]

k1, k2, k3, k4, k5 = st.columns(5)
with k1: card("Cases in view", f"{cases:,}", "Filtered scope")
with k2: card("Alternatives considered", f"{considered:,}" if considered is not None else "—", "Cases with a competitor listed", pink=True)
with k3: card("Myya selected anyway", pct(win_rate, 0) if win_rate is not None else "—", "Replacement evidence", pink=True)
with k4: card("Most replaced alternative", most_replaced if most_replaced is not None else "—", "Top competitor mentioned")
with k5: card("Measurement completeness", pct(1-underbust_missing, 0) if pd.notna(underbust_missing) else "—", "Underbust recorded")

st.caption(f"CSV URL: {data_url}")


# =========================
# STORY BLOCK 1 — Proof (Replacement)
# =========================
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.markdown('<div class="storyBlock">', unsafe_allow_html=True)
st.markdown('<div class="blockKicker">Proof</div>', unsafe_allow_html=True)

if not available_other:
    st.markdown('<div class="blockHeadline">Replacement proof requires competitor fields.</div>', unsafe_allow_html=True)
    st.info("Competitive evidence requires 'Other Bra 1/2' columns (not found in this dataset).")
else:
    st.markdown(
        f'<div class="blockHeadline">When alternatives appear, <span class="pill">Myya still gets selected</span> — measurable replacement.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="blockBody">CEO/website-friendly view: which alternatives are most often replaced, and which Myya products win.</div>',
        unsafe_allow_html=True
    )

    tmp = df_f[[TARGET_BRA] + available_other].copy()
    myya_set = set(df_f[TARGET_BRA].dropna().astype(str).str.strip().unique().tolist())

    melted = tmp.melt(id_vars=[TARGET_BRA], value_vars=available_other, value_name="competitor").dropna(subset=["competitor", TARGET_BRA])
    melted["competitor"] = melted["competitor"].astype(str).str.strip()
    melted[TARGET_BRA] = melted[TARGET_BRA].astype(str).str.strip()
    melted = melted[~melted["competitor"].isin(myya_set)]

    if melted.empty:
        st.info("Competitor evidence becomes empty after cleaning (competitor fields may contain Myya names).")
    else:
        left, right = st.columns([0.5, 0.5], gap="large")

        with left:
            comp_counts = (
                melted.groupby("competitor").size().reset_index(name="replaced_cases")
                .sort_values("replaced_cases", ascending=False)
            )
            comp_counts["share"] = comp_counts["replaced_cases"] / max(comp_counts["replaced_cases"].sum(), 1)

            fig = px.bar(
                comp_counts.head(top_n).iloc[::-1],
                x="replaced_cases",
                y="competitor",
                orientation="h",
                title="Most replaced alternatives (Leaderboard)",
                hover_data={"share": ":.1%"},
            )
            fig.update_layout(**PLOTLY_LAYOUT_BASE)
            plot(fig)

        with right:
            mat = melted.groupby(["competitor", TARGET_BRA]).size().reset_index(name="count")
            top_comp = mat.groupby("competitor")["count"].sum().sort_values(ascending=False).head(10).index.tolist()
            top_myya = mat.groupby(TARGET_BRA)["count"].sum().sort_values(ascending=False).head(8).index.tolist()
            mat2 = mat[mat["competitor"].isin(top_comp) & mat[TARGET_BRA].isin(top_myya)].copy()
            pivot = mat2.pivot_table(index="competitor", columns=TARGET_BRA, values="count", aggfunc="sum", fill_value=0)

            fig2 = px.imshow(pivot, aspect="auto", title="Replacement map (competitor → selected Myya)")
            fig2.update_layout(**PLOTLY_LAYOUT_BASE)
            plot(fig2)

        if show_details:
            table(melted.head(400), height=280)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# STORY BLOCK 2 — Why it wins (Drivers)
# =========================
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.markdown('<div class="storyBlock">', unsafe_allow_html=True)
st.markdown('<div class="blockKicker">Why it wins</div>', unsafe_allow_html=True)

drivers = [c for c in BUSINESS_DRIVER_CANDIDATES if c in df_f.columns]
drv_rows = []
for d in drivers:
    score = driver_strength_internal(df_f, d, TARGET_BRA)
    drv_rows.append({"Driver": d, "Strength": float(score) if pd.notna(score) else np.nan})

drv = pd.DataFrame(drv_rows).dropna(subset=["Strength"]).sort_values("Strength", ascending=False)

if drv.empty:
    st.info("Not enough data to compute driver ranking in this view.")
else:
    top_driver = drv.iloc[0]["Driver"]
    story_driver = picked_driver if (picked_driver in drv["Driver"].tolist()) else top_driver

    st.markdown(
        f'<div class="blockHeadline">Myya wins more consistently when <span class="pill">{top_driver}</span> is captured and used well.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="blockBody">We keep this buyer-friendly: plain language only (no method labels). Select a driver and see how the product mix shifts.</div>',
        unsafe_allow_html=True
    )

    L, R = st.columns([0.62, 0.38], gap="large")

    with L:
        plot_df = drv.head(top_n).copy()
        plot_df["Highlight"] = np.where(plot_df["Driver"] == story_driver, "Selected", "Other")
        fig = px.bar(
            plot_df.iloc[::-1],
            x="Strength",
            y="Driver",
            orientation="h",
            title="What drives consistent Myya selection",
            color="Highlight",
        )
        fig.update_layout(**PLOTLY_LAYOUT_BASE, legend_title_text="")
        plot(fig)

    with R:
        lvl = impact_level(float(drv[drv["Driver"] == story_driver]["Strength"].iloc[0]))
        meaning = DRIVER_EXPLANATIONS_BUSINESS.get(story_driver, "A measurable input that improves repeatability.")
        card(f"{lvl} impact", story_driver, meaning, pink=True)
        st.markdown(
            f"""
<div class="card" style="margin-top:12px;">
  <div class="cardTitle">Business takeaway</div>
  <div class="cardNote">
    Standardize <b>{story_driver}</b> → fewer exceptions → higher repeatability → faster scaling.
  </div>
</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("##### Visual evidence")
    top_products = df_f[TARGET_BRA].dropna().astype(str).value_counts().head(5).index.tolist()
    sub = df_f[df_f[TARGET_BRA].astype(str).isin(top_products)].copy()

    if pd.api.types.is_numeric_dtype(sub[story_driver]) if story_driver in sub.columns else False:
        fig = px.box(
            sub.dropna(subset=[story_driver, TARGET_BRA]),
            x=TARGET_BRA,
            y=story_driver,
            title=f"{story_driver} distribution across top Myya products",
        )
        fig.update_layout(**PLOTLY_LAYOUT_BASE)
        plot(fig)
    else:
        tmp2 = sub.dropna(subset=[story_driver, TARGET_BRA]).copy()
        tmp2[story_driver] = tmp2[story_driver].astype(str)
        share = tmp2.groupby([story_driver, TARGET_BRA]).size().reset_index(name="count")
        tot = share.groupby(story_driver)["count"].sum().reset_index(name="total")
        share = share.merge(tot, on=story_driver, how="left")
        share["share"] = share["count"] / share["total"]
        top_cats = tmp2[story_driver].value_counts().head(8).index.tolist()
        share = share[share[story_driver].isin(top_cats)]
        fig = px.bar(
            share,
            x="share",
            y=story_driver,
            color=TARGET_BRA,
            orientation="h",
            title=f"Product share by {story_driver}",
            hover_data=["count"],
        )
        fig.update_layout(**PLOTLY_LAYOUT_BASE)
        plot(fig)

    if show_details:
        table(drv, height=240)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# STORY BLOCK 3 — What gets chosen (Mix)
# =========================
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.markdown('<div class="storyBlock">', unsafe_allow_html=True)
st.markdown('<div class="blockKicker">What gets chosen</div>', unsafe_allow_html=True)

mix = df_f[TARGET_BRA].dropna().astype(str).value_counts()
mix_df = mix.reset_index()
mix_df.columns = ["Myya Bra", "Count"]
mix_df["Share"] = mix_df["Count"] / max(mix_df["Count"].sum(), 1)
mix_df["CumShare"] = mix_df["Share"].cumsum()

top1 = float(mix_df["Share"].iloc[0]) if len(mix_df) else np.nan
top3 = float(mix_df["Share"].head(3).sum()) if len(mix_df) else np.nan
std_index = hhi(mix_df["Share"].values)

st.markdown('<div class="blockHeadline">Selection behavior is measurable — and can be standardized for scale.</div>', unsafe_allow_html=True)
st.markdown('<div class="blockBody">This supports the win story: consistent selection patterns enable repeatable outcomes.</div>', unsafe_allow_html=True)

a, b, c = st.columns([0.44, 0.28, 0.28], gap="large")
with a:
    show = mix_df.head(top_n).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=show["Myya Bra"], y=show["Count"], name="Count"))
    fig.add_trace(go.Scatter(x=show["Myya Bra"], y=show["CumShare"], name="Cumulative share", yaxis="y2", mode="lines+markers"))
    fig.update_layout(
        **PLOTLY_LAYOUT_BASE,
        title="Mix concentration (Pareto)",
        xaxis_title="Myya product",
        yaxis_title="Count",
        yaxis2=dict(title="Cumulative share", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plot(fig)

with b:
    card("Top-1 share", pct(top1, 0) if pd.notna(top1) else "—", "Dominant product concentration")
    card("Top-3 share", pct(top3, 0) if pd.notna(top3) else "—", "Concentration across a small set")

with c:
    card("Standardization index", f"{std_index:.2f}" if pd.notna(std_index) else "—", "Higher = more concentrated behavior", pink=True)
    fig2 = px.treemap(mix_df.head(20), path=["Myya Bra"], values="Count", title="Mix map")
    fig2.update_layout(**PLOTLY_LAYOUT_BASE)
    plot(fig2)

if show_details:
    table(mix_df.head(25), height=260)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# STORY BLOCK 4 — Where to scale next (Segments)
# =========================
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.markdown('<div class="storyBlock">', unsafe_allow_html=True)
st.markdown('<div class="blockKicker">Where to scale next</div>', unsafe_allow_html=True)

seg_options = [c for c in SEGMENT_FIELDS if c in df_f.columns]
if not seg_options:
    st.info("No segment fields found in this dataset for the scaling view.")
else:
    seg_field = st.selectbox("Segment lens (for scaling decisions)", seg_options, index=0)

    work = df_f.copy()
    if pd.api.types.is_numeric_dtype(work[seg_field]):
        try:
            work["_seg"] = pd.qcut(work[seg_field], q=4, duplicates="drop")
        except Exception:
            work["_seg"] = pd.cut(work[seg_field], bins=4)
        work["_seg"] = work["_seg"].map(interval_to_label)
    else:
        work["_seg"] = work[seg_field].astype(str)

    sub = work[["_seg", TARGET_BRA]].dropna()
    if sub.empty:
        st.info("Not enough data for segment opportunity charts.")
    else:
        st.markdown('<div class="blockHeadline">Prioritize rollout where volume is high — and wins are repeatable.</div>', unsafe_allow_html=True)
        st.markdown('<div class="blockBody">This helps decide rollout, inventory alignment, and messaging by segment.</div>', unsafe_allow_html=True)

        seg_mix = sub.groupby(["_seg", TARGET_BRA]).size().reset_index(name="count")
        seg_tot = seg_mix.groupby("_seg")["count"].sum().reset_index(name="total")
        seg_mix = seg_mix.merge(seg_tot, on="_seg", how="left")
        seg_mix["share"] = seg_mix["count"] / seg_mix["total"]

        top_products = sub[TARGET_BRA].value_counts().head(7).index.astype(str).tolist()
        seg_mix = seg_mix[seg_mix[TARGET_BRA].astype(str).isin(top_products)].copy()

        L, R = st.columns([0.62, 0.38], gap="large")
        with L:
            fig = px.bar(
                seg_mix,
                x="share",
                y="_seg",
                color=TARGET_BRA,
                orientation="h",
                title=f"Product mix by segment ({seg_field})",
                hover_data=["count"],
            )
            fig.update_layout(**PLOTLY_LAYOUT_BASE)
            plot(fig)

        with R:
            seg_size = sub["_seg"].value_counts().reset_index()
            seg_size.columns = ["Segment", "Cases"]
            fig2 = px.bar(seg_size.iloc[::-1], x="Cases", y="Segment", orientation="h", title="Segment volume")
            fig2.update_layout(**PLOTLY_LAYOUT_BASE)
            plot(fig2)

            if show_details:
                table(seg_size, height=220)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# CLOSING
# =========================
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.markdown('<div class="storyBlock">', unsafe_allow_html=True)
st.markdown('<div class="blockKicker">Closing</div>', unsafe_allow_html=True)
st.markdown('<div class="blockHeadline">This is measurable replacement — not a marketing claim.</div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="blockBody">
For CEOs and customers: <span class="pill">alternatives are considered</span>, and Myya is still selected at a high rate.
That’s the simplest, most credible story to put on the website — backed by interactive evidence.
</div>
    """,
    unsafe_allow_html=True
)
st.caption("Note: Method labels are intentionally hidden to avoid buyer confusion; internal scoring ensures correctness.")
st.markdown('</div>', unsafe_allow_html=True)
