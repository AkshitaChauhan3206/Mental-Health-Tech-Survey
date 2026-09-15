# Mental Health in Tech Survey — Project Deliverables

This folder contains all three deliverables for the Mental Health in Tech Survey project.

## 1. `Mental_Health_EDA_Completed.ipynb`
Full EDA capstone notebook (matches the original template structure) with:
- Data cleaning (Age outliers, Gender standardization, contextual missing-value handling)
- 15 required charts, each with "why this chart / insight / business impact" markdown
- Correlation heatmap & pair plot
- Business recommendations and conclusion

Open in Jupyter, Google Colab, or VS Code. All charts are pre-rendered as embedded images, so you can view it without re-running anything — but you can also re-run all cells end-to-end (just make sure `survey.csv` is in the same folder).

## 2. `Mental_Health_in_Tech_Survey.pptx`
A 15-slide presentation summarizing the EDA findings and recommendations, built directly from the notebook's charts and conclusions. Ready to present as-is, or edit in PowerPoint / Google Slides.

## 3. Streamlit App (`app.py` + `requirements.txt`)
An interactive dashboard version of the EDA — filter by country, gender, age, and treatment status, and explore demographics, treatment drivers, correlations, and any variable on the fly.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Make sure `survey.csv` is in the same folder as `app.py` (or upload it via the sidebar uploader once the app is running).

### Deploy for free (Streamlit Community Cloud)
1. Create a new GitHub repo and push `app.py`, `requirements.txt`, and `survey.csv` to it.
2. Go to https://share.streamlit.io, sign in with GitHub, and click **"New app"**.
3. Select your repo/branch and set the main file path to `app.py`.
4. Click **Deploy** — you'll get a public URL in a couple of minutes.

### Deploy elsewhere
The app has no external dependencies beyond `requirements.txt`, so it also runs as-is on Render, Railway, Hugging Face Spaces (Streamlit SDK), or any Docker host running `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Data
`survey.csv` — the original OSMI 2014 Mental Health in Tech Survey (1,259 responses, 27 columns).
`cleaned_survey.csv` — the cleaned dataset (1,251 rows, 0 missing values) produced by the notebook's data-wrangling step, used for reference.
