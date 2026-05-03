import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import io

from modelcode import run_topsis, scenario0, scenario1, scenario2, scenario3, strategies

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Supply Chain Decision Tool", layout="wide")

st.markdown("""<style>
[data-testid="stAppViewContainer"] {background-color: #f5ede6;}
</style>""", unsafe_allow_html=True)

# ================= START SCREEN =================
if "start" not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:
    st.title("🚨 Supply Chain Disruption?")
    st.write("A disruption is impacting your supply chain? No worries, this tool will help you find a solution.")
    if st.button("Let's find a Solution!"):
        st.session_state.start = True
    st.stop()

# ================= MAIN =================
st.title("📊 Supply Chain Decision Tool")

col1, col2 = st.columns([1, 2])

# ================= LEFT PANEL =================
with col1:
    scenario_choice = st.selectbox(
        "Scenario",
        [
            "Stable Environment",
            "Supplier Disruption",
            "Demand Uncertainty",
            "Multi-Risk Disruption"
        ]
    )

    scenario_map = {
        "Stable Environment": scenario0,
        "Supplier Disruption": scenario1,
        "Demand Uncertainty": scenario2,
        "Multi-Risk Disruption": scenario3
    }

    scenario_descriptions = {
        "Stable Environment": "Normal operations with low disruption risk.",
        "Supplier Disruption": "Supplier failure reduces capacity and increases lead time.",
        "Demand Uncertainty": "Demand surge with forecasting errors.",
        "Multi-Risk Disruption": "Simultaneous supply and logistics disruptions."
    }

    matrix = scenario_map[scenario_choice]

    st.info(scenario_descriptions[scenario_choice])
    st.caption("Scale: 1–5 (higher = better performance). Cost & complexity already transformed.")

# ================= RIGHT PANEL =================
with col2:
    default_weights = np.array([0.04, 0.38, 0.20, 0.10, 0.21, 0.02, 0.06])

    st.write("### 🎯 Define priorities")
    st.caption("0 = not important | 1 = very important")

    cost = st.slider("Cost Efficiency", 0.0, 1.0, float(default_weights[0]))
    risk = st.slider("Risk Reduction", 0.0, 1.0, float(default_weights[1]))
    flex = st.slider("Flexibility", 0.0, 1.0, float(default_weights[2]))
    resp = st.slider("Responsiveness", 0.0, 1.0, float(default_weights[3]))
    align = st.slider("Strategic Alignment", 0.0, 1.0, float(default_weights[4]))
    complexity = st.slider("Implementation Complexity", 0.0, 1.0, float(default_weights[5]))
    esg = st.slider("ESG", 0.0, 1.0, float(default_weights[6]))

weights = np.array([cost, risk, flex, resp, align, complexity, esg])
weights = weights / weights.sum()

criteria_names = [
    "Cost","Risk Reduction","Flexibility",
    "Responsiveness","Strategic Alignment",
    "Implementation Complexity","ESG"
]

# ================= TOPSIS =================
ci, ranking = run_topsis(matrix, weights)
ci_percent = ci * 100
best_idx = ranking[0]
best_strategy = strategies[best_idx]

gap = ci[ranking[0]] - ci[ranking[1]]
confidence_percent = round((gap / ci[ranking[0]]) * 100, 1)

# ================= ROBUSTNESS =================
np.random.seed(42)

def perturb(w):
    noise = np.random.uniform(-0.05, 0.05, len(w))
    nw = np.clip(w + noise, 0.001, None)
    return nw / nw.sum()

runs = 100
stable = sum(
    run_topsis(matrix, perturb(weights))[1][0] == best_idx
    for _ in range(runs)
)

stability = stable / runs

# ================= RESULTS =================
st.markdown("---")
st.success(f"🏆 Recommended Strategy: {best_strategy}")

colA, colB, colC = st.columns(3)
colA.metric("Decision Strength", f"{confidence_percent}%")
colB.metric("Robustness", f"{int(stability*100)}%")
colC.metric("Score", f"{round(ci_percent[best_idx],1)}%")

# ================= RANKING =================
st.subheader("🏆 Strategy Ranking")

ranking_df = pd.DataFrame({
    "Strategy": [strategies[i] for i in ranking],
    "Score (%)": [round(ci_percent[i],1) for i in ranking]
})

fig_rank = px.bar(
    ranking_df,
    x="Score (%)",
    y="Strategy",
    orientation="h",
    text="Score (%)"
)
fig_rank.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_rank, use_container_width=True)

# ================= SENSITIVITY =================
st.subheader("🎛️ Sensitivity Analysis")

selected_criterion = st.selectbox("Criterion to vary", criteria_names)
crit_idx = criteria_names.index(selected_criterion)

variation_range = np.linspace(0.01, 1, 30)

sens_results = []
for val in variation_range:
    temp_weights = weights.copy()
    temp_weights[crit_idx] = val
    temp_weights = temp_weights / temp_weights.sum()

    ci_temp, _ = run_topsis(matrix, temp_weights)

    for i, s in enumerate(strategies):
        sens_results.append({
            "Weight": val,
            "Strategy": s,
            "Score": ci_temp[i]
        })

sens_df = pd.DataFrame(sens_results)

fig_sens = px.line(sens_df, x="Weight", y="Score", color="Strategy")
st.plotly_chart(fig_sens, use_container_width=True)

# ================= SCENARIO COMPARISON =================
st.subheader("🌍 Scenario Comparison")

scenario_names = ["Stable", "Supplier", "Demand", "Multi-Risk"]
scenario_matrices = [scenario0, scenario1, scenario2, scenario3]

comparison_results = []
for name, mat in zip(scenario_names, scenario_matrices):
    ci_temp, rank_temp = run_topsis(mat, weights)
    comparison_results.append({
        "Scenario": name,
        "Best Strategy": strategies[rank_temp[0]],
        "Score": round(ci_temp[rank_temp[0]] * 100, 1)
    })

comparison_df = pd.DataFrame(comparison_results)

fig_compare = px.bar(comparison_df, x="Scenario", y="Score", color="Best Strategy")
st.plotly_chart(fig_compare, use_container_width=True)

# ================= PDF =================
def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Supply Chain Decision Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Scenario", styles["Heading2"]))
    elements.append(Paragraph(scenario_descriptions[scenario_choice], styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Best Strategy: {best_strategy}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Ranking Table
    elements.append(Paragraph("Strategy Ranking", styles["Heading2"]))
    ranking_table_data = [["Strategy", "Score (%)"]] + [
        [strategies[i], f"{round(ci_percent[i],1)}"]
        for i in ranking
    ]

    table = Table(ranking_table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("GRID", (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    # Scenario Comparison Table
    elements.append(Paragraph("Scenario Comparison", styles["Heading2"]))
    comp_data = [["Scenario", "Best Strategy", "Score (%)"]]

    for name, mat in zip(scenario_names, scenario_matrices):
        ci_temp, rank_temp = run_topsis(mat, weights)
        comp_data.append([
            name,
            strategies[rank_temp[0]],
            f"{round(ci_temp[rank_temp[0]] * 100,1)}"
        ])

    comp_table = Table(comp_data)
    comp_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("GRID", (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(comp_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "Interactive charts are available in the Streamlit dashboard.",
        styles["Italic"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ================= DOWNLOAD =================
st.markdown("---")

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

if st.button("📄 Generate Report"):
    st.session_state.pdf_data = generate_pdf()

if st.session_state.pdf_data is not None:
    st.download_button(
        "📥 Download PDF",
        data=st.session_state.pdf_data,
        file_name="decision_report.pdf",
        mime="application/pdf"
    )
