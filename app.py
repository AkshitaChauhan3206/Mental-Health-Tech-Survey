"""
Mental Health in Tech Survey — Interactive EDA Dashboard
Built on the OSMI 2014 Mental Health in Tech Survey.

Run locally:
    streamlit run app.py

Deploy:
    Push this file, requirements.txt, and survey.csv to a GitHub repo,
    then deploy on https://share.streamlit.io (Streamlit Community Cloud).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------
# Page config & theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health in Tech Survey — EDA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1D3557"
TEAL = "#6FB3B8"
CORAL = "#E76F51"
PALETTE = ["#1D3557", "#E76F51", "#6FB3B8", "#8AB17D", "#F4A261", "#457B9D"]
PLOTLY_TEMPLATE = "plotly_white"


def style_fig(fig, height=430, title=None):
    """Apply consistent styling + hover formatting to every chart."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Arial", size=13, color="#2B2B2B"),
        title=dict(text=title, font=dict(size=16, color=NAVY)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial"),
    )
    return fig


# --------------------------------------------------------------------------
# Data loading & cleaning (cached)
# --------------------------------------------------------------------------
MALE_LABELS = [
    "male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ",
    "man", "msle", "mail", "malr", "cis man", "cis male", "guy (-ish) ^_^",
    "male leaning androgynous",
    "ostensibly male, unsure what that really means",
]
FEMALE_LABELS = [
    "female", "f", "woman", "femake", "female ", "cis-female/femme",
    "female (cis)", "femail", "cis female", "trans-female", "trans woman",
    "female (trans)",
]


def clean_gender(g):
    g = str(g).strip().lower()
    if g in MALE_LABELS:
        return "Male"
    elif g in FEMALE_LABELS:
        return "Female"
    else:
        return "Other"


@st.cache_data
def load_and_clean(path_or_buffer):
    raw = pd.read_csv(path_or_buffer)
    df = raw.copy()

    drop_cols = [c for c in ["comments", "Timestamp"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Clean Age: keep a realistic working-age range
    df = df[(df["Age"] >= 18) & (df["Age"] <= 75)]

    # Standardize Gender
    df["Gender"] = df["Gender"].apply(clean_gender)

    # Contextual missing-value handling
    if "state" in df.columns:
        df["state"] = df["state"].fillna("Not Applicable")
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].fillna(df["self_employed"].mode()[0])
    if "work_interfere" in df.columns:
        df["work_interfere"] = df["work_interfere"].fillna("Not Applicable")

    df.reset_index(drop=True, inplace=True)
    return raw, df


# Column descriptions used throughout the app (tooltips, data dictionary, chart captions)
COLUMN_DESCRIPTIONS = {
    "Age": "Respondent's age.",
    "Gender": "Respondent's gender, standardized into Male / Female / Other.",
    "Country": "Respondent's country of residence.",
    "state": "US state, if applicable (only meaningful for US respondents).",
    "self_employed": "Whether the respondent is self-employed.",
    "family_history": "Whether the respondent has a family history of mental illness.",
    "treatment": "Whether the respondent has sought treatment for a mental health condition (primary variable of interest).",
    "work_interfere": "How often mental health interferes with work, if the respondent has a condition.",
    "no_employees": "Number of employees at the respondent's company.",
    "remote_work": "Whether the respondent works remotely at least 50% of the time.",
    "tech_company": "Whether the employer is primarily a tech company.",
    "benefits": "Whether the employer provides mental health benefits.",
    "care_options": "Whether the respondent knows the mental health care options provided by their employer.",
    "wellness_program": "Whether the employer has discussed mental health as part of an employee wellness program.",
    "seek_help": "Whether the employer provides resources to learn about mental health and seeking help.",
    "anonymity": "Whether anonymity is protected if the employee chooses to take advantage of mental health resources.",
    "leave": "How easy it is perceived to be to take medical leave for a mental health condition.",
    "mental_health_consequence": "Whether the respondent thinks discussing mental health with their employer would have negative consequences.",
    "phys_health_consequence": "Whether the respondent thinks discussing a physical health issue with their employer would have negative consequences.",
    "coworkers": "Willingness to discuss a mental health issue with coworkers.",
    "supervisor": "Willingness to discuss a mental health issue with a direct supervisor.",
    "mental_health_interview": "Whether the respondent would bring up a mental health issue in a job interview.",
    "phys_health_interview": "Whether the respondent would bring up a physical health issue in a job interview.",
    "mental_vs_physical": "Whether the respondent feels the employer takes mental health as seriously as physical health.",
    "obs_consequence": "Whether the respondent has observed negative consequences for coworkers with mental health conditions.",
}

# Human-readable insight text shown under key charts
DRIVER_INSIGHTS = {
    "family_history": "Family history is the single strongest predictor in this dataset — respondents with a family history of mental illness seek treatment at roughly **double** the rate of those without one.",
    "work_interfere": "Treatment-seeking rises almost linearly with how often mental health interferes with work — from ~14% at *Never* up to ~85% at *Often*, suggesting interference is an early-warning signal.",
    "benefits": "Employees who know their employer offers mental health benefits seek treatment noticeably more than those who are unsure — **awareness** matters as much as the benefit itself.",
    "care_options": "Similar to benefits: simply being aware of available care options is associated with a higher treatment-seeking rate than not knowing.",
    "remote_work": "Remote work shows only a small relationship with treatment-seeking — a much weaker signal than family history or work interference.",
    "anonymity": "A majority of respondents don't know whether anonymity is protected when using mental health resources — a clear communication gap for employers to close.",
    "leave": "Close to half of respondents don't know how easy it would be to take mental-health-related leave, mirroring the anonymity awareness gap.",
    "mental_health_consequence": "Respondents who fear negative consequences from disclosure actually seek treatment at a *higher* rate — likely because those with a real diagnosis have more at stake, not because fear encourages treatment.",
    "no_employees": "Company size doesn't show a strong, consistent relationship with treatment-seeking on its own.",
    "wellness_program": "Wellness programs that explicitly mention mental health are associated with somewhat higher treatment-seeking rates.",
    "seek_help": "Employers who provide clear resources on how to seek help show a modest positive association with treatment-seeking.",
}

# --------------------------------------------------------------------------
# Sidebar — data source, about, & filters
# --------------------------------------------------------------------------
st.sidebar.title("🧠 Mental Health in Tech")
st.sidebar.caption("OSMI 2014 Survey — EDA Dashboard")

with st.sidebar.expander("ℹ️ About this project", expanded=False):
    st.markdown(
        """
This dashboard explores the **OSMI 2014 Mental Health in Tech Survey**
(1,259 responses, 48 countries) to understand what personal and
workplace factors relate to an employee **seeking treatment** for a
mental health condition.

**Explore:**
- Demographics of respondents
- What drives treatment-seeking (family history, work interference, benefits...)
- How every variable correlates with every other
- Any single column on its own

Use the filters below to slice the data — every chart updates live.
        """
    )

uploaded = st.sidebar.file_uploader("Upload survey.csv (optional)", type=["csv"])
data_source = uploaded if uploaded is not None else "survey.csv"

try:
    raw_df, df = load_and_clean(data_source)
except FileNotFoundError:
    st.error(
        "Couldn't find `survey.csv` next to this app. Upload it using the "
        "sidebar uploader, or place `survey.csv` in the same folder as `app.py`."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filters")
st.sidebar.caption(
    "Filters are cascading — each dropdown only shows options that actually "
    "exist given your other selections, so you can never pick a combination "
    "with zero matching rows."
)

FILTER_KEYS = [
    "f_country", "f_gender", "f_age", "f_treatment",
    "f_family", "f_remote", "f_size", "f_self_emp",
]


def reset_filters():
    """Clear every filter widget's stored state so it falls back to its default."""
    for k in FILTER_KEYS:
        if k in st.session_state:
            del st.session_state[k]


def safe_selectbox(label, options, key):
    """A selectbox whose stored value is auto-corrected if it's no longer valid
    for the current (cascaded) option list — prevents Streamlit's
    'default value is not part of the options' crash."""
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = options[0]
    return st.sidebar.selectbox(label, options, key=key)


# Apply filters one at a time, in order — each subsequent filter's options are
# computed from the data *already* narrowed by the filters above it. This is
# what guarantees every reachable combination has at least one matching row.
filtered = df.copy()

countries = ["All"] + sorted(filtered["Country"].unique().tolist())
sel_country = safe_selectbox("Country", countries, "f_country")
if sel_country != "All":
    filtered = filtered[filtered["Country"] == sel_country]

genders = ["All"] + sorted(filtered["Gender"].unique().tolist())
sel_gender = safe_selectbox("Gender", genders, "f_gender")
if sel_gender != "All":
    filtered = filtered[filtered["Gender"] == sel_gender]

age_min, age_max = int(filtered["Age"].min()), int(filtered["Age"].max())
if "f_age" not in st.session_state:
    st.session_state["f_age"] = (age_min, age_max)
else:
    lo, hi = st.session_state["f_age"]
    st.session_state["f_age"] = (max(lo, age_min), min(hi, age_max)) if lo <= hi else (age_min, age_max)
    if st.session_state["f_age"][0] > st.session_state["f_age"][1]:
        st.session_state["f_age"] = (age_min, age_max)
sel_age = st.sidebar.slider("Age range", age_min, age_max, key="f_age")
filtered = filtered[(filtered["Age"] >= sel_age[0]) & (filtered["Age"] <= sel_age[1])]

treatment_opt = ["All"] + sorted(filtered["treatment"].unique().tolist())
sel_treatment = safe_selectbox("Sought treatment?", treatment_opt, "f_treatment")
if sel_treatment != "All":
    filtered = filtered[filtered["treatment"] == sel_treatment]

family_opt = ["All"] + sorted(filtered["family_history"].unique().tolist())
sel_family = safe_selectbox("Family history of mental illness", family_opt, "f_family")
if sel_family != "All":
    filtered = filtered[filtered["family_history"] == sel_family]

remote_opt = ["All"] + sorted(filtered["remote_work"].unique().tolist())
sel_remote = safe_selectbox("Works remotely", remote_opt, "f_remote")
if sel_remote != "All":
    filtered = filtered[filtered["remote_work"] == sel_remote]

size_order = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
size_opt = ["All"] + [s for s in size_order if s in filtered["no_employees"].unique()]
sel_size = safe_selectbox("Company size", size_opt, "f_size")
if sel_size != "All":
    filtered = filtered[filtered["no_employees"] == sel_size]

self_emp_opt = ["All"] + sorted(filtered["self_employed"].unique().tolist())
sel_self_emp = safe_selectbox("Self-employed", self_emp_opt, "f_self_emp")
if sel_self_emp != "All":
    filtered = filtered[filtered["self_employed"] == sel_self_emp]

st.sidebar.button("🔄 Reset all filters", on_click=reset_filters)

st.sidebar.markdown("---")
st.sidebar.metric("Rows after filters", f"{len(filtered):,}", f"of {len(df):,} total")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🧠 Mental Health in Tech Survey — EDA Dashboard")
st.markdown(
    "Interactive exploration of the **OSMI 2014 Mental Health in Tech Survey** "
    "(1,259 responses across 48 countries). Use the sidebar to filter the data — "
    "every chart below updates live, and you can **hover over any chart** to see "
    "exact values."
)

if filtered.empty:
    st.warning("No rows match the current filter combination. Try widening your filters in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_demo, tab_drivers, tab_corr, tab_explore, tab_data = st.tabs(
    ["📊 Overview", "👥 Demographics", "🔍 What Drives Treatment",
     "🧮 Correlations", "🎛️ Explore Any Variable", "📄 Raw Data & Dictionary"]
)

# ================================= OVERVIEW =================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents (filtered)", f"{len(filtered):,}")
    c2.metric("Countries", filtered["Country"].nunique())
    treat_rate = (filtered["treatment"] == "Yes").mean() * 100
    c3.metric("Sought Treatment", f"{treat_rate:.1f}%")
    c4.metric("Median Age", f"{filtered['Age'].median():.0f}")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("#### Missing Values in the Original (Raw) Data")
        miss = raw_df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        miss_pct = (miss / len(raw_df) * 100).round(1)
        fig = px.bar(
            x=miss_pct.index, y=miss_pct.values,
            labels={"x": "Column", "y": "Missing %"},
            color=miss_pct.values, color_continuous_scale="Teal",
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Missing: %{y}%<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
        st.caption(
            "`state` and `work_interfere` are missing **by design** (not applicable to "
            "every respondent), not by data-collection error — see the Raw Data tab for "
            "how each column was cleaned."
        )

    with col2:
        st.markdown("#### Who Sought Treatment?")
        counts = filtered["treatment"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index, color_discrete_map={"Yes": CORAL, "No": NAVY},
            hole=0.45,
        )
        fig.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value} respondents (%{percent})<extra></extra>",
        )
        st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
        st.caption(
            "A near-even split makes `treatment` a well-balanced outcome — any factor "
            "that correlates strongly with it (explored in the next tabs) is a meaningful "
            "signal rather than a statistical artifact."
        )

    st.markdown("#### Age Distribution, Split by Treatment")
    fig = px.histogram(
        filtered, x="Age", color="treatment", nbins=25, barmode="overlay",
        color_discrete_map={"Yes": CORAL, "No": NAVY}, opacity=0.7,
    )
    fig.update_traces(hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>")
    st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
    st.caption(
        "Age shows almost no separation between the two treatment groups — unlike the "
        "behavioral/workplace factors explored in the **What Drives Treatment** tab, age "
        "alone isn't a strong predictor here."
    )

# ================================ DEMOGRAPHICS ================================
with tab_demo:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Age Distribution")
        fig = px.histogram(filtered, x="Age", nbins=25, color_discrete_sequence=[NAVY])
        fig.update_traces(hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>")
        st.plotly_chart(style_fig(fig, height=360), use_container_width=True)
        st.caption("Respondents skew young — mostly 23–40, peaking around 28–32, matching the typical tech workforce.")

    with col2:
        st.markdown("#### Gender Distribution (Cleaned)")
        order = filtered["Gender"].value_counts()
        fig = px.bar(
            x=order.index, y=order.values, color=order.index,
            color_discrete_sequence=PALETTE, labels={"x": "Gender", "y": "Count"},
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, height=360), use_container_width=True)
        st.caption("The sample is ~79% Male, reflecting tech's well-documented gender gap — interpret Female/Other findings with that smaller sample size in mind.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Top Countries by Respondent Count")
        top_c = filtered["Country"].value_counts().head(10).sort_values()
        fig = px.bar(
            x=top_c.values, y=top_c.index, orientation="h",
            color=top_c.values, color_continuous_scale="Teal",
            labels={"x": "Respondents", "y": ""},
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Respondents: %{x}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
        st.caption("The US and UK dominate the sample — findings generalize most reliably to those markets.")

    with col4:
        st.markdown("#### Company Size")
        order = [s for s in size_order if s in filtered["no_employees"].unique()]
        counts = filtered["no_employees"].value_counts().reindex(order)
        fig = px.bar(
            x=counts.index, y=counts.values, color=counts.index,
            color_discrete_sequence=PALETTE, labels={"x": "Company Size", "y": "Count"},
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
        st.caption("Respondents are fairly evenly spread across company sizes, so findings aren't dominated by any single bracket.")

# ============================= WHAT DRIVES TREATMENT =============================
with tab_drivers:
    st.markdown(
        "Compare **treatment-seeking** against any workplace or personal factor. "
        "Bars are colored by whether the respondent sought treatment — hover for exact counts."
    )
    driver_options = [
        "family_history", "work_interfere", "benefits", "care_options",
        "remote_work", "anonymity", "leave", "mental_health_consequence",
        "no_employees", "wellness_program", "seek_help",
    ]
    driver_choice = st.selectbox(
        "Choose a factor:", driver_options, index=0,
        format_func=lambda c: c.replace("_", " ").title(),
    )
    st.caption(f"**{driver_choice}** — {COLUMN_DESCRIPTIONS.get(driver_choice, '')}")

    order = None
    if driver_choice == "work_interfere":
        order = ["Not Applicable", "Never", "Rarely", "Sometimes", "Often"]
        order = [o for o in order if o in filtered[driver_choice].unique()]
    elif driver_choice == "leave":
        order = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]
        order = [o for o in order if o in filtered[driver_choice].unique()]
    elif driver_choice == "no_employees":
        order = [s for s in size_order if s in filtered[driver_choice].unique()]

    grouped = filtered.groupby([driver_choice, "treatment"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        grouped, x=driver_choice, y="count", color="treatment", barmode="group",
        category_orders={driver_choice: order} if order else None,
        color_discrete_map={"Yes": CORAL, "No": NAVY},
        labels={driver_choice: driver_choice.replace("_", " ").title(), "count": "Count"},
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>")
    st.plotly_chart(style_fig(fig, height=440, title=f"{driver_choice.replace('_', ' ').title()} vs Treatment Seeking"), use_container_width=True)

    if driver_choice in DRIVER_INSIGHTS:
        st.info(f"💡 **Insight:** {DRIVER_INSIGHTS[driver_choice]}")

    ctab = pd.crosstab(filtered[driver_choice], filtered["treatment"], normalize="index") * 100
    st.markdown("**Treatment rate (%) by category:**")
    st.dataframe(ctab.round(1).style.format("{:.1f}%"), use_container_width=True)

# ================================ CORRELATIONS ================================
with tab_corr:
    st.markdown(
        "This heatmap label-encodes every categorical column and computes pairwise "
        "correlation, so you can scan the **entire dataset** for relationships at once. "
        "Hover over any cell to see the exact correlation value."
    )
    df_encoded = filtered.copy()
    le = LabelEncoder()
    text_cols = df_encoded.select_dtypes(include=["object", "string", "category"]).columns
    for col in text_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    corr = df_encoded.corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z}<extra></extra>",
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=650, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "`treatment` correlates most strongly with `family_history` and `work_interfere`, "
        "confirming the patterns seen in the **What Drives Treatment** tab. Most other "
        "pairs show weak correlation, so the dataset doesn't suffer from severe "
        "multicollinearity."
    )

# ============================= EXPLORE ANY VARIABLE =============================
with tab_explore:
    st.markdown("Pick any categorical column to see its distribution in the filtered data — optionally split by treatment status.")

    cat_cols = [
        c for c in filtered.select_dtypes(include=["object", "string", "category"]).columns
        if c != "state"
    ]
    default_idx = cat_cols.index("Gender") if "Gender" in cat_cols else 0
    col_choice = st.selectbox("Column", cat_cols, index=default_idx, format_func=lambda c: c.replace("_", " ").title())
    split_by_treatment = st.checkbox("Split by treatment status", value=False)

    st.caption(f"**{col_choice}** — {COLUMN_DESCRIPTIONS.get(col_choice, 'No description available.')}")

    order = filtered[col_choice].value_counts().index.tolist()
    if split_by_treatment and col_choice != "treatment":
        grouped = filtered.groupby([col_choice, "treatment"], observed=True).size().reset_index(name="count")
        fig = px.bar(
            grouped, y=col_choice, x="count", color="treatment", orientation="h",
            category_orders={col_choice: order},
            color_discrete_map={"Yes": CORAL, "No": NAVY},
            labels={"count": "Count", col_choice: col_choice.replace("_", " ").title()},
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x}<extra></extra>")
    else:
        counts = filtered[col_choice].value_counts()
        fig = px.bar(
            y=counts.index, x=counts.values, orientation="h",
            color=counts.values, color_continuous_scale="Sunsetdark",
            labels={"x": "Count", "y": ""},
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, height=450), use_container_width=True)

# ============================== RAW DATA & DICTIONARY ==============================
with tab_data:
    st.markdown("#### Cleaned & Filtered Dataset")
    st.dataframe(filtered, use_container_width=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv, "filtered_survey.csv", "text/csv")

    st.markdown("#### 📖 Data Dictionary")
    dict_df = pd.DataFrame(
        [{"Column": k, "Description": v} for k, v in COLUMN_DESCRIPTIONS.items() if k in filtered.columns]
    )
    st.dataframe(dict_df, use_container_width=True, hide_index=True)

    with st.expander("🧹 Data cleaning notes"):
        st.markdown(
            """
- **Age** filtered to a realistic **18–75** range (raw data had negative ages and one entry of `99999999999`).
- **Gender** standardized from 49 free-text spellings into **Male / Female / Other**.
- **state** missing values filled with `"Not Applicable"` (only meaningful for US respondents).
- **self_employed** missing values filled with the column mode.
- **work_interfere** missing values filled with `"Not Applicable"` (only asked of respondents with a condition — NaN is meaningful, not random).
- **comments** and **Timestamp** dropped (free text / metadata, not analytically useful).
            """
        )

st.markdown("---")
st.caption(
    "Built on the OSMI 2014 Mental Health in Tech Survey. "
    "This dashboard is for exploratory analysis and is not a diagnostic or clinical tool."
)
