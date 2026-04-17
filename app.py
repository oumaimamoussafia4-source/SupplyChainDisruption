import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io
from modelcode import run_topsis, scenario1, scenario2, scenario3, strategies

st.set_page_config(page_title="Supply Chain Decision Tool", layout="wide")

st.markdown("""<style> [data-testid="stAppViewContainer"]  {background-color: #f5ede6;} </style>""", unsafe_allow_html=True)

if "start" not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:
    st.title("🚨 Supply Chain Disruption?")
    st.write("A disruption is impacting your supply chain? No worries, this tool will help you find a solution.")
    if st.button("Let's find a Solution!"):
        st.session_state.start = True
    st.stop()

st.title("📊 Supply Chain Decision Tool")

col1, col2 = st.columns([1,2])
with col1:
    scenario_choice = st.selectbox(
        "Scenario",
        ["Supplier Disruption", "Demand Uncertainty", "Multi-Risk Disruption"]
    )

    scenario_map = {
        "Supplier Disruption": scenario1,
        "Demand Uncertainty": scenario2,
        "Multi-Risk Disruption": scenario3
    }

    matrix = scenario_map[scenario_choice]

    scenario_descriptions = {
        "Supplier Disruption": "Upstream supplier failure reduces capacity and increases lead time, threatening production continuity.",
        "Demand Uncertainty": "Unpredictable demand surge with poor forecasts creates risks of stockouts or excess inventory.",
        "Multi-Risk Disruption": "Simultaneous supplier and logistics disruptions create compounded pressure on the supply chain."
    }

    st.info(scenario_descriptions[scenario_choice])

with col2:
    st.write("### 🎯 Define priorities")
    st.caption("0 = not important | 1 = very important")

    cost = st.slider("Cost Efficiency", 0.0, 1.0, 0.3)
    risk = st.slider("Risk Reduction", 0.0, 1.0, 0.6)
    flex = st.slider("Flexibility", 0.0, 1.0, 0.5)
    resp = st.slider("Responsiveness", 0.0, 1.0, 0.4)
    align = st.slider("Strategic Alignment", 0.0, 1.0, 0.4)
    complexity = st.slider("Implementation Complexity", 0.0, 1.0, 0.2)
    esg = st.slider("ESG", 0.0, 1.0, 0.3)

weights = np.array([cost, risk, flex, resp, align, complexity, esg])
weights = weights / weights.sum()

criteria_names = [
    "Cost Efficiency","Risk Reduction","Flexibility",
    "Responsiveness","Alignment","Implementation Complexity","ESG"
]

ci, ranking = run_topsis(matrix, weights)
ci_percent = ci * 100
best_idx = ranking[0]
best_strategy = strategies[best_idx]

norm = matrix / np.sqrt((matrix**2).sum(axis=0))
weighted = norm * weights
contrib = weighted[best_idx]

gap = ci[ranking[0]] - ci[ranking[1]]
confidence_percent = int((gap / ci[ranking[0]]) * 100)

np.random.seed(42)

def perturb(w):
    noise = np.random.uniform(-0.05, 0.05, len(w))
    nw = np.clip(w + noise, 0.001, None)
    return nw / nw.sum()

runs = 100
stable = 0

for _ in range(runs):
    new_ci, new_rank = run_topsis(matrix, perturb(weights))
    if new_rank[0] == best_idx:
        stable += 1

stability = stable / runs

win_count = np.zeros(len(strategies))

for _ in range(runs):
    new_ci, new_rank = run_topsis(matrix, perturb(weights))
    win_count[new_rank[0]] += 1

prob_df = pd.DataFrame({
    "Strategy": strategies,
    "Win Probability": win_count / runs
})

st.markdown("---")
st.success(f"🏆 Recommended Strategy: {best_strategy}")

colA, colB, colC = st.columns(3)
colA.metric("Decision Strength", f"{confidence_percent}%")
colB.metric("Robustness", f"{int(stability*100)}%")
colC.metric("Top Rank Score", f"{round(ci_percent[best_idx],1)}%")

st.markdown("---")
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
    text="Score (%)",
    color=ranking_df["Strategy"] == best_strategy,
    color_discrete_map={True: "#2ecc71", False: "#bdc3c7"}
)

fig_rank.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_rank, use_container_width=True)

st.markdown("---")
st.subheader("🧠 Decision Drivers")

top = np.argsort(contrib)[::-1][:3]
important = np.argsort(weights)[::-1][:3]
worst = np.argsort(contrib)[:2]

st.write("### 🔑 Key Insights")

st.write(f"""
**1. Your Priorities**
- {criteria_names[important[0]]}
- {criteria_names[important[1]]}

**2. Why this strategy performs best**
- Strong in: {criteria_names[top[0]]}, {criteria_names[top[1]]}

**3. Trade-offs to consider**
- Weaker in: {criteria_names[worst[0]]}, {criteria_names[worst[1]]}
""")

st.write("### 📌 Interpretation")

st.write(f"""
👉 Because you prioritized **{criteria_names[important[0]]}**,  
the model favors strategies strong in **{criteria_names[top[0]]}**,  
which explains why **{best_strategy}** is selected.
""")

if important[0] == top[0] or important[1] == top[0]:
    st.success("Strong alignment with your priorities.")
elif important[0] in worst:
    st.warning("Top priority not fully satisfied.")
else:
    st.info("Balanced compromise solution.")

st.markdown("---")
st.subheader("📈 Strategy Behavior")

risk_range = np.linspace(0.01, 1, 30)
results = []

for r in risk_range:
    temp_weights = weights.copy()
    temp_weights[1] = r
    temp_weights = temp_weights / temp_weights.sum()

    ci_temp, _ = run_topsis(matrix, temp_weights)

    for i, s in enumerate(strategies):
        results.append({
            "Risk Weight": r,
            "Strategy": s,
            "Score": ci_temp[i]
        })

df_line = pd.DataFrame(results)

fig_line = px.line(df_line, x="Risk Weight", y="Score", color="Strategy")
st.plotly_chart(fig_line, use_container_width=True)

st.subheader("🔄 Switching Points")

switch_points = []
previous_best = None

for r in risk_range:
    temp_weights = weights.copy()
    temp_weights[1] = r
    temp_weights = temp_weights / temp_weights.sum()

    ci_temp, rank_temp = run_topsis(matrix, temp_weights)
    current_best = strategies[rank_temp[0]]

    if previous_best and current_best != previous_best:
        switch_points.append({
            "Risk Weight": round(r, 2),
            "From": previous_best,
            "To": current_best
        })

    previous_best = current_best

if switch_points:
    st.dataframe(pd.DataFrame(switch_points))
else:
    st.success("Stable decision across all risk levels")

st.markdown("---")
st.subheader("⚖️ Trade-offs")

x_axis = st.selectbox("X-axis", criteria_names)
y_axis = st.selectbox("Y-axis", criteria_names, index=2)

scatter_df = pd.DataFrame({
    "Strategy": strategies,
    x_axis: norm[:, criteria_names.index(x_axis)],
    y_axis: norm[:, criteria_names.index(y_axis)]
})

fig_scatter = px.scatter(
    scatter_df,
    x=x_axis,
    y=y_axis,
    text="Strategy",
    size=[15 if s == best_strategy else 8 for s in strategies],
    color=[s == best_strategy for s in strategies],
    color_discrete_map={True: "#2ecc71", False: "#95a5a6"}
)

fig_scatter.update_traces(textposition='top center')
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.subheader("📊 Strategy Robustness")

if prob_df["Win Probability"].max() > 0.9:
    st.warning("One strategy dominates → very robust but low sensitivity.")

fig = px.bar(prob_df, x="Strategy", y="Win Probability", text="Win Probability")
fig.update_traces(textposition="outside")
fig.update_layout(yaxis=dict(range=[0,1]))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📌 Final Recommendation")

st.success(f"""
Based on your priorities, **{best_strategy}** is the most suitable strategy.

It performs strongly in your key focus areas and remains stable under uncertainty,
making it a reliable decision in this scenario.
""")

st.markdown("---")
st.subheader("📄 Export Report")

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Supply Chain Decision Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("1. Scenario Context", styles["Heading2"]))
    elements.append(Paragraph(scenario_descriptions[scenario_choice], styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("2. Decision Summary", styles["Heading2"]))
    elements.append(Paragraph(f"""
    Recommended Strategy: <b>{best_strategy}</b><br/>
    Decision Strength: {confidence_percent}%<br/>
    Robustness: {int(stability*100)}%
    """, styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("3. Strategy Ranking", styles["Heading2"]))

    table_data = [["Strategy", "Score (%)"]]
    for i in ranking:
        table_data.append([strategies[i], round(ci_percent[i],1)])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),1,colors.black)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("4. Decision Drivers", styles["Heading2"]))

    elements.append(Paragraph(f"""
    The decision is primarily driven by the importance assigned to 
    <b>{criteria_names[important[0]]}</b> and <b>{criteria_names[important[1]]}</b>.

    The selected strategy performs strongly in 
    <b>{criteria_names[top[0]]}</b> and <b>{criteria_names[top[1]]}</b>, 
    which explains its high ranking.

    However, it shows weaker performance in 
    <b>{criteria_names[worst[0]]}</b> and <b>{criteria_names[worst[1]]}</b>, 
    representing trade-offs that should be considered.
    """, styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("5. Robustness Analysis", styles["Heading2"]))

    max_prob = prob_df["Win Probability"].max()

    if max_prob > 0.9:
        robustness_text = "The analysis shows a dominant strategy that remains optimal across most simulations, indicating a highly robust but low-sensitivity decision."
    else:
        robustness_text = "The analysis indicates variability in optimal strategies under changing conditions, suggesting a more sensitive decision environment."

    elements.append(Paragraph(robustness_text, styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("6. Final Recommendation", styles["Heading2"]))

    elements.append(Paragraph(f"""
    Based on the defined priorities and multi-criteria evaluation, 
    <b>{best_strategy}</b> is identified as the most suitable strategy.

    It provides strong performance in key decision areas and maintains stability 
    under uncertainty, making it a reliable and well-balanced choice 
    for the given disruption scenario.
    """, styles["Normal"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.download_button(
    label="Download Full Report",
    data=generate_pdf(),
    file_name="decision_report.pdf",
    mime="application/pdf"
)