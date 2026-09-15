# 🧠 Mental Health in Tech Survey — EDA & Dashboard

An end-to-end data analysis project exploring the **OSMI 2014 Mental Health in Tech Survey** — 1,259 responses from tech employees across 48 countries — to understand what personal and workplace factors are associated with seeking mental health treatment.

🔗 **Live Demo:** [mental-health-tech-survey-emhxxtn7udfnqqeugle5nz.streamlit.app](https://mental-health-tech-survey-emhxxtn7udfnqqeugle5nz.streamlit.app/)
💻 **GitHub Repo:** [github.com/AkshitaChauhan3206/Mental-Health-Tech-Survey](https://github.com/AkshitaChauhan3206/Mental-Health-Tech-Survey)

---

## 📌 Project Description

Mental health remains an under-discussed issue in the tech industry despite its demanding, high-pressure nature. This project digs into the OSMI survey data to answer:

- Who is represented in the survey (age, gender, country, company size)?
- Which personal factors (e.g., family history) relate to treatment-seeking?
- Which workplace factors (benefits, care options, anonymity, leave policy) relate to treatment-seeking?
- What can tech employers do, based on the data, to build more supportive workplaces?

**Key findings:**
- **Family history** is the strongest single predictor — 74% of respondents with a family history sought treatment vs. 35% without.
- **Work interference** shows a near-linear relationship with treatment-seeking (14% → 85% as interference goes from "Never" to "Often").
- **Awareness**, not just availability, drives outcomes — knowing about benefits/care options lifts treatment rates more than the benefit's mere existence.
- **65%** of respondents don't know if anonymity is protected, and **45%** don't know how easy mental health leave would be to take — a clear communication gap employers can close.

---

## 📂 What's in This Repo

| File | Description |
|---|---|
| `Mental_Health_EDA.ipynb` | Full EDA notebook — data cleaning, 15 visualizations with insights, correlation heatmap, pair plot, and recommendations |
| `app.py` | Interactive Streamlit dashboard |
| `requirements.txt` | Python dependencies for the app |

---

## 🚀 Running the Streamlit App Locally

**1. Clone the repo**
```bash
git clone https://github.com/AkshitaChauhan3206/Mental-Health-Tech-Survey.git
cd Mental-Health-Tech-Survey
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`. Make sure `survey.csv` is in the same folder as `app.py` — or upload it via the sidebar uploader once the app is running.

### What the app does
- Filter the data live by **country, gender, age range, and treatment status**
- **Overview** tab — headline metrics, missing-value breakdown, treatment split
- **Demographics** tab — age, gender, country, and company-size distributions
- **What Drives Treatment** tab — compare treatment-seeking against any workplace/personal factor
- **Correlations** tab — full correlation heatmap across all encoded variables
- **Explore Any Variable** tab — pick any column and see its distribution
- **Raw Data** tab — view and download the filtered dataset

---

## 🌐 Deploying Your Own Copy

This app is deployed on **Streamlit Community Cloud**:
1. Push `app.py`, `requirements.txt`, and `survey.csv` to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **"New app"**.
3. Select the repo/branch and set the main file path to `app.py`.
4. Click **Deploy** — you'll get a public URL within a couple of minutes.

---

## 🛠️ Built With
Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Streamlit

---

## 👤 Author

**Akshita Chauhan**
