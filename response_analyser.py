# response_analyser.py
import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from dotenv import load_dotenv
from openai import OpenAI

# --- CONFIG ---
st.set_page_config(page_title="Lexi Response Analyser", page_icon="🪞", layout="wide")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["google"], scope)
gs_client = gspread.authorize(creds)

# --- DATA LOAD ---
st.title("🪞 Lexi Response Similarity Explorer")

try:
    sheet = gs_client.open("lexi_live").worksheet("form responses")
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    st.success(f"Loaded {len(df)} responses from form responses sheet.")
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# --- BASIC CLEANING ---
df.columns = [c.strip().lower() for c in df.columns]
if "agent" not in df.columns or "suggestion" not in df.columns:
    st.error("Missing 'Agent' or 'Suggestion' columns.")
    st.stop()

# --- USER FILTERS ---
agents = df["agent"].dropna().unique().tolist()
selected_agents = st.multiselect("Select Agents to Compare:", agents, default=agents)
subset = df[df["agent"].isin(selected_agents)]

# --- TEXT VECTORIZATION ---
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(subset["suggestion"].astype(str))
similarity = cosine_similarity(tfidf_matrix)

# --- AGGREGATE BY AGENT ---
agent_avg = subset.groupby("agent")["suggestion"].apply(lambda x: " ".join(x))
agent_tfidf = vectorizer.fit_transform(agent_avg)
agent_sim = cosine_similarity(agent_tfidf)

# --- DISPLAY RESULTS ---
st.subheader("Similarity Matrix (by Agent)")
sim_df = pd.DataFrame(agent_sim, index=agent_avg.index, columns=agent_avg.index)
st.dataframe(sim_df.style.background_gradient(cmap="YlGnBu"))

# --- OPTIONAL: VISUAL HEATMAP ---
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(sim_df, annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# --- OPTIONAL: NATURAL LANGUAGE SUMMARY ---
if st.button("Summarise with GPT"):
    summary_prompt = (
        "Analyse the following similarity matrix between wardrobe agents' suggestions. "
        "Explain which agents most often agree in tone or content, and what that implies:\n\n"
        f"{sim_df.to_string()}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data analyst for Lexi's wardrobe AI."},
            {"role": "user", "content": summary_prompt},
        ],
    )
    st.markdown("### AI Summary")
    st.write(response.choices[0].message.content)
