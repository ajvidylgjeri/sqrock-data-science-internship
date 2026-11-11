from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import screener

HERE = Path(__file__).parent
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screening System")
st.caption("Rank resumes against a job description with NLP (TF-IDF + cosine similarity + skill matching). "
           "Sqrock IT Solutions Data Science Internship, Task 3.")


@st.cache_data
def sample_resumes():
    return screener.load_folder(HERE / "data" / "resumes")


@st.cache_data
def sample_jobs():
    return {p.stem.replace("_", " ").title(): p.read_text() for p in sorted((HERE / "data" / "job_descriptions").glob("*.txt"))}


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    method = st.radio("Text features", ["TF-IDF (recommended)", "Bag of Words"])
    w = st.slider("Weight of text similarity vs skill match", 0.0, 1.0, 0.6, 0.05,
                  help="Match % = w × text similarity + (1 − w) × skill coverage")
    threshold = st.slider("Shortlist threshold (match %)", 0, 100, 50, 5)
    top_n = st.number_input("Show top N candidates", 1, 100, 10)

# ---------------- Inputs ----------------
left, right = st.columns([1, 1])
with left:
    st.subheader("1. Job description")
    jobs = sample_jobs()
    options = list(jobs) + ["Write my own"]
    preset = st.selectbox("Start from a sample job", options, index=options.index("Data Scientist"))
    jd = st.text_area("Job description", value="" if preset == "Write my own" else jobs[preset], height=280)

with right:
    st.subheader("2. Resumes")
    use_sample = st.toggle("Use the 14 sample resumes", value=True)
    uploads = st.file_uploader("…and/or upload resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    resumes = list(sample_resumes()) if use_sample else []
    for f in uploads or []:
        try:
            resumes.append({"file": f.name, "text": screener.extract_text(f.name, f.getvalue())})
        except Exception as e:  # unreadable or scanned PDF
            st.warning(f"Could not read {f.name}: {e}")
    st.write(f"**{len(resumes)}** resumes loaded.")
    if jd.strip():
        st.write("**Skills detected in the job description:**")
        st.write(" ".join(f"`{s}`" for s in sorted(screener.extract_skills(jd))) or "_none from the skill vocabulary_")

if not jd.strip() or not resumes:
    st.info("Add a job description and at least one resume to start screening.")
    st.stop()

# ---------------- Results ----------------
ranked = screener.rank_resumes(jd, resumes, method="tfidf" if method.startswith("TF") else "bow",
                               w_similarity=w, threshold=threshold)
shortlist = ranked[ranked["Match %"] >= threshold]

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Resumes screened", len(ranked))
m2.metric("Shortlisted", len(shortlist))
m3.metric("Top match", ranked.iloc[0]["Candidate"], f"{ranked.iloc[0]['Match %']:.1f}%", delta_color="off")
m4.metric("Average match", f"{ranked['Match %'].mean():.1f}%")

tab_rank, tab_short, tab_detail = st.tabs(["🏆 Ranking", "✅ Shortlist", "🔍 Candidate detail"])

with tab_rank:
    top = ranked.head(int(top_n))
    fig = px.bar(top[::-1], x="Match %", y="Candidate", orientation="h", color="Status",
                 color_discrete_map={"✅ Shortlisted": "#37b24d", "❌ Rejected": "#ced4da"},
                 hover_data=["Cosine similarity", "Skill coverage %", "Years exp."], range_x=[0, 100])
    fig.add_vline(x=threshold, line_dash="dash", line_color="#e8590c")
    fig.update_layout(height=max(300, 38 * len(top)), margin=dict(t=10, b=10), legend_title=None)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(top[["Rank", "Candidate", "Match %", "Cosine similarity", "Skill coverage %", "Years exp.", "Status"]],
                 hide_index=True, width="stretch",
                 column_config={"Match %": st.column_config.ProgressColumn("Match %", min_value=0, max_value=100, format="%.1f%%")})

with tab_short:
    if shortlist.empty:
        st.warning("No candidate reaches the threshold. Try lowering it in the sidebar.")
    else:
        st.dataframe(shortlist[["Rank", "Candidate", "Email", "Match %", "Years exp.", "Matched skills", "Missing skills"]],
                     hide_index=True, width="stretch")
        st.download_button("Download shortlist CSV", shortlist.to_csv(index=False).encode(), "shortlist.csv", "text/csv")

with tab_detail:
    who = st.selectbox("Candidate", ranked["Candidate"])
    row = ranked[ranked["Candidate"] == who].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Match", f"{row['Match %']:.1f}%")
    c2.metric("Skill coverage", f"{row['Skill coverage %']:.0f}%")
    c3.metric("Years of experience", f"{row['Years exp.']:.0f}")
    st.markdown("**✅ Matched skills:** " + (" ".join(f"`{s}`" for s in row["Matched skills"].split(", ")) if row["Matched skills"] else "_none_"))
    st.markdown("**⚠️ Missing skills:** " + (" ".join(f"`{s}`" for s in row["Missing skills"].split(", ")) if row["Missing skills"] else "_none_"))
    with st.expander("Extracted resume text"):
        st.text(next(r["text"] for r in resumes if r["file"] == row["File"]))
