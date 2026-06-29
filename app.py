import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Student Result Analysis System",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; }
    h1 { font-size: 2rem !important; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Data generation ──────────────────────────────────────────────────────────

@st.cache_data
def generate_data(n=214, seed=42):
    rng = np.random.default_rng(seed)
    subjects = ["Mathematics", "Physics", "Chemistry", "English", "Computer Science"]

    # Per-subject mean scores (Chemistry hardest)
    subject_means = {"Mathematics": 58, "Physics": 54, "Chemistry": 49,
                     "English": 72, "Computer Science": 76}

    attendance = rng.integers(40, 100, n)
    assignments_submitted = rng.integers(2, 11, n)

    scores = {}
    for subj, mean in subject_means.items():
        base = rng.normal(mean, 12, n)
        # Students with good attendance/assignments tend to score higher
        boost = (attendance - 70) * 0.25 + (assignments_submitted - 6) * 1.5
        scores[subj] = np.clip(base + boost, 0, 100).round(1)

    df = pd.DataFrame(scores)
    df["avg_score"] = df[subjects].mean(axis=1).round(1)
    df["attendance"] = attendance
    df["assignments_submitted"] = assignments_submitted
    df["student_id"] = [f"S-{str(i+1).zfill(3)}" for i in range(n)]

    # Grade label
    def grade(avg):
        if avg >= 90: return "A+"
        elif avg >= 80: return "A"
        elif avg >= 70: return "B+"
        elif avg >= 60: return "B"
        elif avg >= 50: return "C"
        elif avg >= 40: return "D"
        else: return "F"

    df["grade"] = df["avg_score"].apply(grade)
    df["at_risk"] = (df["avg_score"] < 50).astype(int)

    return df, subjects


# ── Model training ────────────────────────────────────────────────────────────

@st.cache_resource
def train_model(df):
    features = ["avg_score", "attendance", "assignments_submitted"]
    X = df[features]
    y = df["at_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Safe", "At-Risk"], output_dict=True)
    importances = dict(zip(features, clf.feature_importances_))
    return clf, acc, report, importances


# ── Load data & model ─────────────────────────────────────────────────────────

df, subjects = generate_data()
clf, accuracy, report, importances = train_model(df)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("## 🎓 Student Result Analysis System")
st.markdown(
    "Analysing academic performance of **214 students** across 5 subjects · "
    "ML-powered at-risk detection · Personal Project 2025"
)
st.divider()

# ── Top metrics ───────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total students", "214")
c2.metric("Class average", f"{df['avg_score'].mean():.1f}")
c3.metric("Pass rate", f"{(df['avg_score'] >= 50).mean()*100:.1f}%")
c4.metric("At-risk students", int(df["at_risk"].sum()))
c5.metric("Model accuracy", f"{accuracy*100:.0f}%", delta="Random Forest")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Grade Overview",
    "📚 Subject Analysis",
    "⚠️ At-Risk Students",
    "🤖 Predict a Student",
])


# ── Tab 1 · Grade Overview ────────────────────────────────────────────────────

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Grade distribution")
        grade_order = ["F", "D", "C", "B", "B+", "A", "A+"]
        grade_counts = df["grade"].value_counts().reindex(grade_order, fill_value=0).reset_index()
        grade_counts.columns = ["Grade", "Count"]
        color_map = {"F": "#e34948", "D": "#eb6834", "C": "#eda100",
                     "B": "#2a78d6", "B+": "#2a78d6", "A": "#1baf7a", "A+": "#1baf7a"}
        fig = px.bar(grade_counts, x="Grade", y="Count",
                     color="Grade", color_discrete_map=color_map,
                     text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", height=320,
                          margin=dict(t=10, b=10))
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Students", gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Score distribution")
        fig2 = px.histogram(df, x="avg_score", nbins=20,
                            color_discrete_sequence=["#2a78d6"])
        fig2.add_vline(x=50, line_dash="dash", line_color="#e34948",
                       annotation_text="Pass line", annotation_position="top right")
        fig2.add_vline(x=df["avg_score"].mean(), line_dash="dot", line_color="#1baf7a",
                       annotation_text=f"Mean {df['avg_score'].mean():.1f}",
                       annotation_position="top left")
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=320, margin=dict(t=10, b=10), showlegend=False)
        fig2.update_xaxes(title="Average score")
        fig2.update_yaxes(title="Students", gridcolor="#f0f0f0")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Attendance vs average score")
    fig3 = px.scatter(df, x="attendance", y="avg_score",
                      color="grade", color_discrete_map=color_map,
                      hover_data=["student_id", "assignments_submitted"],
                      opacity=0.7, size_max=8)
    fig3.add_hline(y=50, line_dash="dash", line_color="#e34948", opacity=0.5)
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=350, margin=dict(t=10, b=10))
    fig3.update_xaxes(title="Attendance (%)", gridcolor="#f0f0f0")
    fig3.update_yaxes(title="Average score", gridcolor="#f0f0f0")
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 2 · Subject Analysis ──────────────────────────────────────────────────

with tab2:
    st.subheader("Subject-wise performance")

    subject_stats = pd.DataFrame({
        "Subject": subjects,
        "Mean": [df[s].mean() for s in subjects],
        "Median": [df[s].median() for s in subjects],
        "Pass rate (%)": [(df[s] >= 50).mean() * 100 for s in subjects],
        "Std dev": [df[s].std() for s in subjects],
    }).round(1)

    col1, col2 = st.columns(2)

    with col1:
        fig4 = go.Figure()
        colors = ["#e34948" if m < 50 else "#2a78d6" for m in subject_stats["Mean"]]
        fig4.add_trace(go.Bar(
            x=subject_stats["Mean"],
            y=subject_stats["Subject"],
            orientation="h",
            marker_color=colors,
            text=subject_stats["Mean"].apply(lambda x: f"{x:.1f}"),
            textposition="outside",
        ))
        fig4.add_vline(x=50, line_dash="dash", line_color="#e34948",
                       annotation_text="Pass (50)", annotation_position="top right")
        fig4.update_layout(title="Average score by subject",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=300, margin=dict(t=40, b=10))
        fig4.update_xaxes(range=[0, 100], gridcolor="#f0f0f0")
        fig4.update_yaxes(title=None)
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        fig5 = px.bar(subject_stats, x="Subject", y="Pass rate (%)",
                      color="Pass rate (%)",
                      color_continuous_scale=["#e34948", "#eda100", "#1baf7a"],
                      text="Pass rate (%)")
        fig5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig5.update_layout(title="Pass rate by subject",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=300, margin=dict(t=40, b=10), showlegend=False,
                           coloraxis_showscale=False)
        fig5.update_yaxes(range=[0, 110], gridcolor="#f0f0f0")
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Score spread per subject (box plot)")
    df_melted = df[subjects].melt(var_name="Subject", value_name="Score")
    fig6 = px.box(df_melted, x="Subject", y="Score", color="Subject",
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  points="outliers")
    fig6.add_hline(y=50, line_dash="dash", line_color="#e34948", opacity=0.6)
    fig6.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=350, showlegend=False, margin=dict(t=10, b=10))
    fig6.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Summary table")
    st.dataframe(subject_stats.set_index("Subject"), use_container_width=True)


# ── Tab 3 · At-Risk Students ──────────────────────────────────────────────────

with tab3:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("At-risk student roster")
        at_risk_df = df[df["at_risk"] == 1][
            ["student_id", "avg_score", "attendance", "assignments_submitted", "grade"] + subjects
        ].sort_values("avg_score").reset_index(drop=True)
        at_risk_df.columns = (["Student ID", "Avg Score", "Attendance %",
                                "Assignments"] + ["Grade"] + subjects)
        st.dataframe(at_risk_df.style.background_gradient(
            subset=["Avg Score"], cmap="RdYlGn", vmin=0, vmax=100
        ), use_container_width=True, height=380)

    with col_r:
        st.subheader("Model performance")
        st.metric("Accuracy", f"{accuracy*100:.0f}%")
        st.metric("At-risk precision", f"{report['At-Risk']['precision']*100:.0f}%")
        st.metric("At-risk recall", f"{report['At-Risk']['recall']*100:.0f}%")

        st.subheader("Feature importance")
        imp_df = pd.DataFrame({
            "Feature": ["Avg score", "Attendance", "Assignments"],
            "Importance": [importances["avg_score"],
                           importances["attendance"],
                           importances["assignments_submitted"]]
        }).sort_values("Importance", ascending=True)
        fig7 = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                      color_discrete_sequence=["#2a78d6"])
        fig7.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=220, margin=dict(t=10, b=10))
        fig7.update_xaxes(title=None, gridcolor="#f0f0f0")
        fig7.update_yaxes(title=None)
        st.plotly_chart(fig7, use_container_width=True)


# ── Tab 4 · Predict ───────────────────────────────────────────────────────────

with tab4:
    st.subheader("Predict at-risk status for a student")
    st.caption("Enter a student's details below to get a real-time prediction from the trained Random Forest model.")

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_score_input = st.slider("Average score", 0, 100, 65)
    with col2:
        attendance_input = st.slider("Attendance (%)", 0, 100, 75)
    with col3:
        assignments_input = st.slider("Assignments submitted (out of 10)", 0, 10, 7)

    if st.button("Run prediction", type="primary"):
        input_data = np.array([[avg_score_input, attendance_input, assignments_input]])
        prediction = clf.predict(input_data)[0]
        proba = clf.predict_proba(input_data)[0]

        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)

        if prediction == 1:
            res_col1.error("⚠️ **At-Risk**")
        else:
            res_col1.success("✅ **Not At-Risk**")

        res_col2.metric("At-risk probability", f"{proba[1]*100:.1f}%")
        res_col3.metric("Safe probability", f"{proba[0]*100:.1f}%")

        fig8 = go.Figure(go.Bar(
            x=["Safe", "At-Risk"],
            y=[proba[0]*100, proba[1]*100],
            marker_color=["#1baf7a", "#e34948"],
            text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
            textposition="outside",
        ))
        fig8.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=280, showlegend=False, yaxis_range=[0, 110],
                           margin=dict(t=10, b=10))
        fig8.update_yaxes(title="Probability (%)", gridcolor="#f0f0f0")
        st.plotly_chart(fig8, use_container_width=True)

        # Recommendation
        st.subheader("Recommendation")
        if prediction == 1:
            st.warning(
                "This student shows at-risk indicators. "
                "Suggested actions: schedule a counselling session, "
                "monitor Chemistry and Physics scores closely, "
                "and follow up on missing assignments."
            )
        else:
            st.info(
                "This student is performing within a safe range. "
                "Continue monitoring attendance — dips below 60% are an early warning signal."
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("Student Result Analysis System · Shailesh Pawar · Personal Project 2025 · Built with Python, Scikit-learn, and Streamlit")
