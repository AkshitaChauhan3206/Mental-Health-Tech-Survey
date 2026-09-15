"""
Mental Health in Tech Survey — Interactive EDA Dashboard
Streamlit app built on the OSMI 2014 Mental Health in Tech Survey.

Run locally:
    streamlit run app.py

Deploy:
    Push this file, requirements.txt, and survey.csv to a GitHub repo,
    then deploy on https://share.streamlit.io (Streamlit Community Cloud).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health in Tech Survey — EDA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

PRIMARY = "#1D3557"
ACCENT = "#E76F51"
TEAL = "#6FB3B8"

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

    # Drop low-value columns
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


# --------------------------------------------------------------------------
# Sidebar — data source & filters
# --------------------------------------------------------------------------
st.sidebar.title("🧠 Mental Health in Tech")
st.sidebar.caption("OSMI 2014 Survey — EDA Dashboard")

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
st.sidebar.subheader("Filters")

countries = ["All"] + sorted(df["Country"].unique().tolist())
sel_country = st.sidebar.selectbox("Country", countries, index=0)

genders = ["All"] + sorted(df["Gender"].unique().tolist())
sel_gender = st.sidebar.selectbox("Gender", genders, index=0)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
sel_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

treatment_opt = ["All", "Yes", "No"]
sel_treatment = st.sidebar.selectbox("Sought treatment?", treatment_opt, index=0)

filtered = df.copy()
if sel_country != "All":
    filtered = filtered[filtered["Country"] == sel_country]
if sel_gender != "All":
    filtered = filtered[filtered["Gender"] == sel_gender]
if sel_treatment != "All":
    filtered = filtered[filtered["treatment"] == sel_treatment]
filtered = filtered[(filtered["Age"] >= sel_age[0]) & (filtered["Age"] <= sel_age[1])]

st.sidebar.markdown("---")
st.sidebar.metric("Rows after filters", f"{len(filtered):,}", f"of {len(df):,} total")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🧠 Mental Health in Tech Survey — EDA Dashboard")
st.markdown(
    "Interactive exploration of the **OSMI 2014 Mental Health in Tech Survey** "
    "(1,259 responses across 48 countries). Use the sidebar to filter the data; "
    "every chart below updates live."
)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_demo, tab_drivers, tab_corr, tab_explore, tab_data = st.tabs(
    ["📊 Overview", "👥 Demographics", "🔍 What Drives Treatment",
     "🧮 Correlations", "🎛️ Explore Any Variable", "📄 Raw Data"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents (filtered)", f"{len(filtered):,}")
    c2.metric("Countries", filtered["Country"].nunique())
    treat_rate = (filtered["treatment"] == "Yes").mean() * 100 if len(filtered) else 0
    c3.metric("Sought Treatment", f"{treat_rate:.1f}%")
    c4.metric("Median Age", f"{filtered['Age'].median():.0f}" if len(filtered) else "—")

    st.markdown("### Missing Values in Raw Data")
    miss = raw_df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_pct = (miss / len(raw_df) * 100).round(1)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.barplot(x=miss.index, y=miss_pct.values, palette="viridis", ax=ax)
    ax.set_ylabel("Missing %")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
    st.caption(
        "`state` and `work_interfere` are missing *by design* (not applicable to "
        "every respondent) rather than by data-collection error — see the cleaning "
        "notes in the notebook for how each was handled."
    )

    st.markdown("### Proportion Who Sought Treatment")
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = filtered["treatment"].value_counts()
    if len(counts):
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=[ACCENT, PRIMARY], startangle=90)
    st.pyplot(fig)

# ---------------- DEMOGRAPHICS ----------------
with tab_demo:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Age Distribution")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        if len(filtered):
            sns.histplot(filtered["Age"], bins=25, kde=True, color=PRIMARY, ax=ax)
        st.pyplot(fig)
    with col2:
        st.markdown("#### Gender Distribution")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        order = filtered["Gender"].value_counts().index
        sns.countplot(x="Gender", data=filtered, order=order, palette="pastel", ax=ax)
        st.pyplot(fig)

    st.markdown("#### Top Countries by Respondent Count")
    fig, ax = plt.subplots(figsize=(10, 4))
    top_c = filtered["Country"].value_counts().head(10)
    sns.barplot(x=top_c.values, y=top_c.index, palette="mako", ax=ax)
    ax.set_xlabel("Number of Respondents")
    st.pyplot(fig)

    st.markdown("#### Company Size")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    order = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
    order = [o for o in order if o in filtered["no_employees"].unique()]
    sns.countplot(x="no_employees", data=filtered, order=order, palette="crest", ax=ax)
    st.pyplot(fig)

# ---------------- WHAT DRIVES TREATMENT ----------------
with tab_drivers:
    st.markdown(
        "Each chart below compares **treatment-seeking** against a workplace or "
        "personal factor. Bars are colored by whether the respondent sought treatment."
    )
    driver_choice = st.selectbox(
        "Choose a factor to compare against treatment-seeking:",
        [
            "family_history", "work_interfere", "benefits", "care_options",
            "remote_work", "anonymity", "leave", "mental_health_consequence",
            "no_employees", "wellness_program", "seek_help",
        ],
        index=0,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    order = None
    if driver_choice == "work_interfere":
        order = ["Not Applicable", "Never", "Rarely", "Sometimes", "Often"]
        order = [o for o in order if o in filtered[driver_choice].unique()]
    elif driver_choice == "leave":
        order = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]
        order = [o for o in order if o in filtered[driver_choice].unique()]
    sns.countplot(x=driver_choice, hue="treatment", data=filtered, order=order,
                  palette="Set2", ax=ax)
    ax.set_title(f"{driver_choice} vs Treatment Seeking")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

    if len(filtered) and filtered[driver_choice].nunique() > 0:
        ctab = pd.crosstab(filtered[driver_choice], filtered["treatment"], normalize="index") * 100
        st.markdown("**Treatment rate (%) by category:**")
        st.dataframe(ctab.round(1).style.format("{:.1f}%"), use_container_width=True)

# ---------------- CORRELATIONS ----------------
with tab_corr:
    st.markdown("#### Correlation Heatmap (Label-Encoded Variables)")
    from sklearn.preprocessing import LabelEncoder

    df_encoded = filtered.copy()
    if len(df_encoded):
        le = LabelEncoder()
        # Use astype(str) + explicit non-numeric detection so this works whether
        # pandas represents text columns as 'object' (older pandas) or the newer
        # dedicated 'string' dtype (pandas 2.x/3.x defaults on some platforms,
        # e.g. Streamlit Community Cloud) — relying on `dtype == object` alone
        # silently misses 'string' columns on those platforms.
        text_cols = df_encoded.select_dtypes(include=["object", "string", "category"]).columns
        for col in text_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        fig, ax = plt.subplots(figsize=(12, 9))
        corr = df_encoded.corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3, ax=ax)
        st.pyplot(fig)
    else:
        st.info("No rows match the current filters.")

# ---------------- EXPLORE ANY VARIABLE ----------------
with tab_explore:
    st.markdown("Pick any categorical column to see its distribution in the filtered data.")
    # Same 'object' vs 'string' dtype issue as above — check both.
    cat_cols = [
        c for c in filtered.select_dtypes(include=["object", "string", "category"]).columns
        if c != "state"
    ]
    if not cat_cols:
        st.info("No categorical columns available to explore.")
    else:
        default_idx = cat_cols.index("Gender") if "Gender" in cat_cols else 0
        col_choice = st.selectbox("Column", cat_cols, index=default_idx)
        if col_choice not in filtered.columns or filtered.empty:
            st.info("No data matches the current filters.")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            order = filtered[col_choice].value_counts().index
            sns.countplot(y=col_choice, data=filtered, order=order, palette="flare", ax=ax)
            ax.set_xlabel("Count")
            st.pyplot(fig)

# ---------------- RAW / CLEANED DATA ----------------
with tab_data:
    st.markdown("#### Cleaned & Filtered Dataset")
    st.dataframe(filtered, use_container_width=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", csv, "filtered_survey.csv", "text/csv")

    with st.expander("Data cleaning notes"):
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
