# myya.py — Interactive BI-style Dashboard (CEO/Website-ready, minimal jargon)
# Run: streamlit run myya.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from scipy.stats import spearmanr, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# =========================
# CONFIG
# =========================
# ---- DATA LOADING (URL ONLY) ----
data_url = data_url_input.strip() or get_data_url()
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

TARGET_BRA = "Myya Bra"
TARGET_SIZE = "Myya Bra Size"
OTHER_BRA_COLS = ["Other Bra 1", "Other Bra 2"]

# Candidate drivers (keep close to your original BI concept)
DRIVER_CANDIDATES = [
    "Underbust Measurement (in)",
    "Bra Size (Band Size)",
    "Bra Size (Cup Size)",
    "Breast Projection (Profile)",
    "Patient Surgery Type",
    "Side",
    "Patient Age",
    "Sternum→Back (in)",
]

SEGMENT_CANDIDATES = [
    "Underbust Measurement (in)",
    "Bra Size (Band Size)",
    "Patient Age",
    "Patient Surgery Type",
    "Side",
    "Breast Projection (Profile)",
]

DRIVER_EXPLANATIONS = {
    "Underbust Measurement (in)": "Anchors sizing decisions and reduces guesswork.",
    "Bra Size (Band Size)": "Clinician-familiar input that stabilizes recommendations.",
    "Bra Size (Cup Size)": "Adds volume context alongside band measurement.",
    "Breast Projection (Profile)": "Key differentiator for post-op fit and comfort.",
    "Patient Surgery Type": "Reconstruction context changes fit constraints.",
    "Side": "Laterality can reflect asymmetry and fit needs.",
    "Patient Age": "May correlate with comfort and preference differences.",
    "Sternum→Back (in)": "Adds torso geometry context for stability and comfort.",
}


# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="Myya Dashboard", layout="wide")


# =========================
# HELPERS
# =========================
def exists(df, c): 
    return c in df.columns

def is_num(df, c):
    return exists(df, c) and pd.api.types.is_numeric_dtype(df[c])

def safe_nunique(s: pd.Series) -> int:
    if s is None or s.empty:
        return 0
    return int(s.dropna().nunique())

def pct(x, digits=0):
    if pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"

def series_clean_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "—": np.nan, "–": np.nan})
    return s

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

def spearman_abs(df, x, y):
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return np.nan
    rho, _ = spearmanr(sub[x], sub[y])
    return float(abs(rho))

def eta_squared_by_groups(num: pd.Series, grp: pd.Series) -> float:
    sub = pd.DataFrame({"num": num, "grp": grp}).dropna()
    if sub.empty:
        return np.nan
    if sub["grp"].nunique() < 2:
        return np.nan
    grand_mean = sub["num"].mean()
    ss_total = ((sub["num"] - grand_mean) ** 2).sum()
    if ss_total <= 0:
        return np.nan
    ss_between = sub.groupby("grp")["num"].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
    return float(ss_between / ss_total)

def driver_strength(df, feature, target):
    """
    We compute a robust impact strength score internally.
    We do NOT expose the method name to the UI.
    """
    if feature not in df.columns or target not in df.columns:
        return np.nan

    x = df[feature]
    y = df[target]
    x_is_num = pd.api.types.is_numeric_dtype(x)
    y_is_num = pd.api.types.is_numeric_dtype(y)

    if x_is_num and y_is_num:
        return spearman_abs(df, feature, target)

    if (not x_is_num) and (not y_is_num):
        sub = df[[feature, target]].dropna()
        if sub.empty:
            return np.nan
        ct = pd.crosstab(sub[feature], sub[target]).values
        return cramers_v(ct)

    if (not x_is_num) and y_is_num:
        return eta_squared_by_groups(df[target], df[feature])

    if x_is_num and (not y_is_num):
        return eta_squared_by_groups(df[feature], df[target])

    return np.nan

def strength_level(v: float) -> str:
    if pd.isna(v):
        return "—"
    if v < 0.10:
        return "Low"
    if v < 0.30:
        return "Medium"
    return "High"

def interval_to_segment_label(x):
    # Converts pandas.Interval or other types into a JSON-safe, readable label
    if isinstance(x, pd.Interval):
        return f"{x.left:g}–{x.right:g}"
    return str(x)

@st.cache_data(show_spinner=False)
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    key_col = "Patient ID (Actual - Primary Key)"
    if key_col in df.columns:
        df[key_col] = df[key_col].astype(str).str.strip()
        df = df[df[key_col] != "EHR Data"].copy()

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = series_clean_str(df[c])

    # normalize a few common categoricals
    for c in ["Patient Surgery Type", "Side", "Breast Projection (Profile)"]:
        if exists(df, c):
            df[c] = series_clean_str(df[c]).str.replace(r"\s+", " ", regex=True).str.title()

    # numeric coercion
    for c in ["Patient Age", "Underbust Measurement (in)", "Bra Size (Band Size)", "Sternum→Back (in)"]:
        if exists(df, c):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def train_reco_model(df, features, target):
    """
    Recommendation engine for interactive Figure 2.
    Robust split: avoid stratify crash on tiny classes.
    """
    X = df[features].copy()
    y = df[target].copy()

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 20 or y.nunique(dropna=True) < 2:
        return None, "Not enough data in this view to generate recommendations."

    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample"
    )

    model = Pipeline([("prep", pre), ("rf", clf)])

    y_counts = y.value_counts(dropna=False)
    min_count = int(y_counts.min()) if not y_counts.empty else 0
    do_stratify = (y.nunique(dropna=True) > 1) and (min_count >= 2)

    test_size = 0.25
    if do_stratify:
        n_classes = int(y.nunique(dropna=True))
        min_test = n_classes / max(len(y), 1)
        test_size = max(test_size, min(0.4, min_test))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if do_stratify else None
    )

    model.fit(X_tr, y_tr)
    return model, None


# =========================
# LOAD DATA
# =========================
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at:\n{DATA_PATH}\n\nUpdate DATA_PATH in myya.py.")
    st.stop()

df = load_and_clean(DATA_PATH)
for req in [TARGET_BRA, TARGET_SIZE]:
    if req not in df.columns:
        st.error(f"Missing required column: {req}")
        st.stop()


# =========================
# SIDEBAR FILTERS (keep close to your original)
# =========================
with st.sidebar:
    st.markdown("## Filters")
    st.caption("These apply to all visuals and recommendations.")

    def multiselect_filter(col):
        if not exists(df, col):
            return None
        opts = sorted(df[col].dropna().astype(str).unique().tolist())
        if not opts:
            return None
        return st.multiselect(col, opts, default=opts)

    f_side = multiselect_filter("Side")
    f_surg = multiselect_filter("Patient Surgery Type")
    f_proj = multiselect_filter("Breast Projection (Profile)")

    age_range = None
    if is_num(df, "Patient Age") and df["Patient Age"].notna().any():
        mn = float(df["Patient Age"].min())
        mx = float(df["Patient Age"].max())
        age_range = st.slider("Patient Age Range", mn, mx, (mn, mx))

    st.divider()
    st.markdown("## Display")
    top_n = st.slider("Top N", 4, 15, 8)
    show_what_if = st.toggle("Enable interactive recommendation (Figure 2)", value=True)


# Apply filters
df_f = df.copy()
if f_side is not None:
    df_f = df_f[df_f["Side"].astype(str).isin(f_side)]
if f_surg is not None:
    df_f = df_f[df_f["Patient Surgery Type"].astype(str).isin(f_surg)]
if f_proj is not None:
    df_f = df_f[df_f["Breast Projection (Profile)"].astype(str).isin(f_proj)]
if age_range is not None and "Patient Age" in df_f.columns:
    df_f = df_f[df_f["Patient Age"].between(age_range[0], age_range[1], inclusive="both")]

if len(df_f) < 5:
    st.warning("Not enough rows after filters. Please broaden filters.")
    st.stop()


# =========================
# HEADER
# =========================
st.markdown(
    """
    <div style="padding:6px 0 10px 0;">
      <div style="font-size:14px; letter-spacing:0.08em; text-transform:uppercase; color:#666;">
        Myya Dashboard
      </div>
      <div style="font-size:36px; font-weight:850; line-height:1.05; margin-top:8px;">
        Key Influencers • Segmentation • Recommendation • Competitive Alternatives
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows (filtered)", f"{len(df_f):,}")
k2.metric("Myya bra types", f"{safe_nunique(df_f[TARGET_BRA]):,}")
k3.metric("Myya sizes", f"{safe_nunique(df_f[TARGET_SIZE]):,}")
if exists(df_f, "Underbust Measurement (in)"):
    k4.metric("Missing underbust", pct(df_f["Underbust Measurement (in)"].isna().mean()))
else:
    k4.metric("Missing underbust", "—")

st.caption(f"Data source (fixed): {DATA_PATH}")


# =========================
# TABS (keep your original structure)
# =========================
tabs = st.tabs([
    "Overview",
    "Drivers",
    "Segmentation",
    "Recommendation",
    "Alternatives"
])


# =========================
# TAB: OVERVIEW
# =========================
with tabs[0]:
    st.markdown("## Overview")
    st.caption("A quick snapshot for CEOs: what’s in view and what Myya is most often selected.")

    c1, c2 = st.columns([0.55, 0.45], gap="large")
    with c1:
        if TARGET_BRA in df_f.columns:
            dist = df_f[TARGET_BRA].dropna().astype(str).value_counts().head(top_n).reset_index()
            dist.columns = ["Myya Bra", "Count"]
            fig = px.bar(dist.iloc[::-1], x="Count", y="Myya Bra", orientation="h", title="Top Myya bra types (filtered)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Missing target column for distribution.")

    with c2:
        st.markdown("### Why sizing fails (context)")
        st.markdown(
            """
- **Brand inconsistency:** same label ≠ same fit  
- **Anchoring:** band-first or cup-first guessing  
- **Post-op shape shift:** projection/asymmetry changes fit  
- **Information friction:** measurements exist, but context isn’t systematically used  

**BI opportunity:** show drivers + explain recommendations so decisions feel obvious.
            """
        )


# =========================
# TAB: FIGURE 1 — DRIVERS (hide methods)
# =========================
with tabs[1]:
    st.markdown("## Key Drivers")
    st.caption("Drivers ranked by influence. We account for numeric and categorical factors internally, but we show only the insights.")

    target_choice = st.selectbox("Analyze drivers for", [TARGET_BRA, TARGET_SIZE], index=0)

    drivers = [c for c in DRIVER_CANDIDATES if c in df_f.columns]
    rows = []
    for d in drivers:
        s = driver_strength(df_f, d, target_choice)
        rows.append({
            "Driver": d,
            "Impact strength": float(s) if pd.notna(s) else np.nan,
        })

    rank = pd.DataFrame(rows).dropna(subset=["Impact strength"]).sort_values("Impact strength", ascending=False)
    rank["Impact level"] = rank["Impact strength"].map(strength_level)
    rank["Explanation"] = rank["Driver"].map(lambda x: DRIVER_EXPLANATIONS.get(x, "Clinically relevant factor influencing fit decisions."))

    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        fig = px.bar(
            rank.head(top_n).iloc[::-1],
            x="Impact strength",
            y="Driver",
            orientation="h",
            title="Key drivers (ranked)",
            hover_data=["Impact level"]
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        focus = st.selectbox("Explain a driver", rank["Driver"].head(top_n).tolist())
        focus_row = rank[rank["Driver"] == focus].iloc[0]
        st.markdown("### Plain-language explanation")
        st.markdown(f"**{focus}** — {focus_row['Explanation']}")
        st.markdown("### Strength (internal score)")
        st.progress(min(max(float(focus_row["Impact strength"]), 0.0), 1.0))
        st.caption("Displayed as an intuitive strength bar (no statistical jargon).")

    st.markdown("### Driver table")
    st.dataframe(
        rank.head(top_n)[["Driver", "Impact level", "Impact strength", "Explanation"]],
        width="stretch",
        height=320
    )


# =========================
# TAB: SEGMENTATION (BI-style) — fix Interval JSON issue
# =========================
with tabs[2]:
    st.markdown("## Segmentation")
    st.caption("Pick a driver and see how Myya selection shifts by segment. Designed to be understandable without a data background.")

    seg_driver = st.selectbox("Segment by driver", [c for c in SEGMENT_CANDIDATES if c in df_f.columns])

    # Prepare segments
    work = df_f.copy()
    if is_num(work, seg_driver):
        # bins chosen to keep readable; you can tweak
        # qcut can produce Interval -> convert to str later
        try:
            work["_segment"] = pd.qcut(work[seg_driver], q=4, duplicates="drop")
        except Exception:
            work["_segment"] = pd.cut(work[seg_driver], bins=4)
    else:
        work["_segment"] = work[seg_driver].astype(str)

    # Make segment JSON-safe
    work["_segment"] = work["_segment"].map(interval_to_segment_label)

    # Choose outcome to view
    outcome = st.selectbox("Outcome to compare", [TARGET_BRA, TARGET_SIZE], index=0)

    sub = work[["_segment", outcome]].dropna()
    if sub.empty:
        st.info("No data after segmentation for the chosen outcome.")
    else:
        seg = (
            sub.groupby(["_segment", outcome]).size().reset_index(name="count")
        )
        seg_tot = seg.groupby("_segment")["count"].sum().reset_index(name="segment_total")
        seg = seg.merge(seg_tot, on="_segment", how="left")
        seg["pct"] = seg["count"] / seg["segment_total"]

        left, right = st.columns([0.58, 0.42], gap="large")
        with left:
            # stacked bar: segment distribution
            # limit categories for readability
            top_cats = sub[outcome].value_counts().head(6).index.astype(str).tolist()
            seg2 = seg[seg[outcome].astype(str).isin(top_cats)].copy()
            fig = px.bar(
                seg2,
                x="pct",
                y="_segment",
                color=outcome,
                orientation="h",
                title=f"Outcome mix by segment — {seg_driver}",
                hover_data=["count"]
            )
            st.plotly_chart(fig, width="stretch")

        with right:
            st.markdown("### Segment table (share within segment)")
            # show top category per segment
            top_by_seg = (
                seg.sort_values(["_segment", "pct"], ascending=[True, False])
                   .groupby("_segment").head(3)
            )
            show_df = top_by_seg.rename(columns={"_segment": "segment"})[["segment", outcome, "count", "pct"]]
            st.dataframe(show_df, width="stretch", height=360)


# =========================
# TAB: FIGURE 2 — RECOMMENDATION (interactive)
# =========================
with tabs[3]:
    st.markdown("## Recommendation")
    st.caption("A what-if panel: adjust inputs and see suggested Myya bra and size. This is designed for demos.")

    if not show_what_if:
        st.info("Interactive recommendation is turned off in the sidebar.")
    else:
        # Use top drivers as features (plus a couple of core contexts if present)
        drivers = [c for c in DRIVER_CANDIDATES if c in df_f.columns]
        # rank quickly against bra selection to pick stable features
        tmp_rank = []
        for d in drivers:
            tmp_rank.append((d, driver_strength(df_f, d, TARGET_BRA)))
        tmp_rank = [x for x in tmp_rank if pd.notna(x[1])]
        tmp_rank.sort(key=lambda x: x[1], reverse=True)

        base_features = [x[0] for x in tmp_rank[:6]]
        for extra in ["Patient Surgery Type", "Breast Projection (Profile)", "Side"]:
            if extra in df_f.columns and extra not in base_features:
                base_features.append(extra)

        features = base_features

        model_bra, err_bra = train_reco_model(df_f, features, TARGET_BRA)
        model_size, err_size = train_reco_model(df_f, features, TARGET_SIZE)

        if model_bra is None or model_size is None:
            st.info(err_bra or err_size or "Recommendation unavailable for this view.")
        else:
            left, right = st.columns([0.55, 0.45], gap="large")

            with left:
                st.markdown("### Inputs")
                patient = {}
                for f in features:
                    if is_num(df_f, f):
                        default = float(df_f[f].dropna().median()) if df_f[f].dropna().size else 0.0
                        patient[f] = st.number_input(f, value=default)
                    else:
                        opts = sorted(df_f[f].dropna().astype(str).unique().tolist())
                        patient[f] = st.selectbox(f, options=opts if opts else ["(missing)"])

                p_df = pd.DataFrame([patient])

            with right:
                st.markdown("### Suggested outcome")
                bra_pred = model_bra.predict(p_df)[0]
                size_pred = model_size.predict(p_df)[0]

                c1, c2 = st.columns(2)
                c1.metric("Suggested Myya bra", str(bra_pred))
                c2.metric("Suggested size", str(size_pred))

                # Show confidence bars (more interactive & demo-friendly)
                try:
                    proba = model_bra.predict_proba(p_df)[0]
                    classes = model_bra.named_steps["rf"].classes_
                    prob_df = pd.DataFrame({"Myya Bra": classes.astype(str), "Probability": proba})
                    prob_df = prob_df.sort_values("Probability", ascending=False).head(5)
                    fig = px.bar(prob_df.iloc[::-1], x="Probability", y="Myya Bra", orientation="h",
                                 title="Top suggestions (confidence)")
                    st.plotly_chart(fig, width="stretch")
                except Exception:
                    st.caption("Confidence view unavailable for this model.")


# =========================
# TAB: FIGURE 3 — ALTERNATIVES (more interactive, no Sankey)
# =========================
with tabs[4]:
    st.markdown("## Competitive Alternatives")
    st.caption("Switch between views to understand which alternatives Myya replaces and where those replacements go.")

    available_other = [c for c in OTHER_BRA_COLS if c in df_f.columns]
    if not available_other:
        st.info("This dataset view does not include 'Other Bra 1/2' competitor fields.")
    else:
        tmp = df_f[[TARGET_BRA] + available_other].copy()
        tmp["competitor_mentioned"] = tmp[available_other].notna().any(axis=1)

        total_cases = len(tmp)
        considered = int(tmp["competitor_mentioned"].sum())
        selected = int(tmp.loc[tmp["competitor_mentioned"], TARGET_BRA].notna().sum())
        win_rate = selected / max(considered, 1)

        # Prevent competitor values that equal Myya bra names
        myya_set = set(df_f[TARGET_BRA].dropna().astype(str).str.strip().unique().tolist())

        comp_series = (
            tmp.loc[tmp["competitor_mentioned"], available_other]
              .stack().dropna().astype(str).str.strip()
        )
        comp_series = comp_series[~comp_series.isin(myya_set)]
        most_comp = comp_series.value_counts().index[0] if not comp_series.empty else "—"

        c1, c2, c3 = st.columns(3)
        c1.metric("Alternatives considered", f"{considered:,}", f"{considered/max(total_cases,1):.0%} of cases")
        c2.metric("Myya selected anyway", f"{win_rate:.0%}", "Evidence of replacement")
        c3.metric("Most replaced alternative", most_comp)

        # Build competitor->Myya pairs
        melted = tmp.melt(
            id_vars=[TARGET_BRA],
            value_vars=available_other,
            value_name="competitor"
        ).dropna(subset=["competitor", TARGET_BRA])

        melted["competitor"] = melted["competitor"].astype(str).str.strip()
        melted[TARGET_BRA] = melted[TARGET_BRA].astype(str).str.strip()
        melted = melted[~melted["competitor"].isin(myya_set)]

        if len(melted) < 2:
            st.info("Not enough competitor mentions to render charts (after cleaning).")
        else:
            view = st.radio(
                "Choose a view",
                ["Leaderboard", "Where replacements go (Heatmap)", "Segment drilldown"],
                horizontal=True
            )

            # Common controls for interactivity
            all_comp = melted["competitor"].value_counts().index.tolist()
            comp_focus = st.selectbox("Focus on a competitor (optional)", ["(All)"] + all_comp)

            if comp_focus != "(All)":
                melted_v = melted[melted["competitor"] == comp_focus].copy()
            else:
                melted_v = melted.copy()

            if view == "Leaderboard":
                metric = st.selectbox("Metric", ["Replaced cases (count)", "Share of replaced (percent)"], index=0)

                comp_counts = (
                    melted_v.groupby("competitor").size().reset_index(name="replaced_cases")
                    .sort_values("replaced_cases", ascending=False)
                )
                comp_counts["share"] = comp_counts["replaced_cases"] / max(comp_counts["replaced_cases"].sum(), 1)

                show = comp_counts.head(top_n).copy()

                if metric.startswith("Share"):
                    fig = px.bar(
                        show.iloc[::-1],
                        x="share",
                        y="competitor",
                        orientation="h",
                        title="Replacement leaderboard (share)",
                        hover_data=["replaced_cases"]
                    )
                else:
                    fig = px.bar(
                        show.iloc[::-1],
                        x="replaced_cases",
                        y="competitor",
                        orientation="h",
                        title="Replacement leaderboard (count)",
                        hover_data=["share"]
                    )
                st.plotly_chart(fig, width="stretch")

                st.markdown("### Table")
                st.dataframe(show, width="stretch", height=280)

            elif view == "Where replacements go (Heatmap)":
                mat = (
                    melted_v.groupby(["competitor", TARGET_BRA]).size().reset_index(name="count")
                )
                # limit axes for readability
                top_comp = (
                    mat.groupby("competitor")["count"].sum().sort_values(ascending=False).head(8).index.tolist()
                )
                top_myya = (
                    mat.groupby(TARGET_BRA)["count"].sum().sort_values(ascending=False).head(6).index.tolist()
                )
                mat2 = mat[mat["competitor"].isin(top_comp) & mat[TARGET_BRA].isin(top_myya)].copy()
                if mat2.empty:
                    mat2 = mat.copy()

                pivot = mat2.pivot_table(index="competitor", columns=TARGET_BRA, values="count",
                                         aggfunc="sum", fill_value=0)

                fig = px.imshow(pivot, aspect="auto", title="Replacement matrix (competitor × Myya bra)")
                st.plotly_chart(fig, width="stretch")

            else:
                st.markdown("### Segment drilldown")
                seg_dim = st.selectbox("Break down by", [c for c in ["Patient Surgery Type", "Side", "Breast Projection (Profile)"] if c in df_f.columns])
                if seg_dim:
                    # Add segment to melted rows by joining index alignment from df_f
                    # Safer: rebuild from df_f with id column
                    df_key = df_f.reset_index().rename(columns={"index": "_rid"})
                    tmp2 = df_key[["_rid", TARGET_BRA, seg_dim] + available_other].copy()
                    tmp2["competitor_mentioned"] = tmp2[available_other].notna().any(axis=1)

                    melt2 = tmp2.melt(
                        id_vars=["_rid", TARGET_BRA, seg_dim],
                        value_vars=available_other,
                        value_name="competitor"
                    ).dropna(subset=["competitor", TARGET_BRA])

                    melt2["competitor"] = melt2["competitor"].astype(str).str.strip()
                    melt2[TARGET_BRA] = melt2[TARGET_BRA].astype(str).str.strip()
                    melt2 = melt2[~melt2["competitor"].isin(myya_set)]

                    if comp_focus != "(All)":
                        melt2 = melt2[melt2["competitor"] == comp_focus]

                    grp = (
                        melt2.groupby([seg_dim, "competitor"]).size().reset_index(name="count")
                    )
                    # take top competitors overall
                    top_comp = melt2["competitor"].value_counts().head(6).index.tolist()
                    grp = grp[grp["competitor"].isin(top_comp)].copy()

                    fig = px.bar(
                        grp,
                        x="count",
                        y=seg_dim,
                        color="competitor",
                        orientation="h",
                        title=f"Competitor mentions by {seg_dim}",
                    )
                    st.plotly_chart(fig, width="stretch")

        st.markdown("### Why Myya wins (website-friendly copy)")
        st.markdown(
            """
- **Consistent sizing logic** across clinicians and brands  
- **Projection-aware fit** that adapts to post-op anatomy  
- **Lower decision friction** with clear, measurement-based guidance  
- **Workflow-friendly** inputs clinicians already collect
            """
        )
