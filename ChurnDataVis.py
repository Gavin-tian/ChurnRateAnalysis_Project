import pandas as pd
import numpy as np
import plotly.express as px

# Load dataset
df = pd.read_csv("Employee_HR.csv")

# ==========================================================
# 1️⃣ Churn Rate by Department (Horizontal Bar Chart)
# ==========================================================
churn_by_dept = df.groupby('Department')['Churn'].mean().sort_values(ascending=True).reset_index()
churn_by_dept['Churn'] = churn_by_dept['Churn'] * 100  # convert to %

fig1 = px.bar(
    churn_by_dept,
    x='Churn',
    y='Department',
    orientation='h',
    text='Churn',
    color='Churn',
    color_continuous_scale='Blues',
    title='Churn Rate by Department (%)',
)
fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig1.update_layout(
    xaxis_title='Churn Rate (%)',
    yaxis_title='Department',
    title_font_size=18,
    plot_bgcolor='white',
    paper_bgcolor='white',
)
fig1.show()

# ==========================================================
# 2️⃣ Satisfaction vs. Evaluation by Churn (Scatter Plot)
# ==========================================================
fig2 = px.scatter(
    df,
    x='Satisfaction',
    y='Evaluation',
    color=df['Churn'].map({0: 'Stayed', 1: 'Left'}),
    color_discrete_map={'Stayed': '#1f77b4', 'Left': '#d62728'},
    title='Satisfaction vs Evaluation by Churn Status',
    hover_data=['Department', 'time_spent_company', 'Promotion'],
    opacity=0.7,
)
fig2.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
fig2.update_layout(
    xaxis_title='Satisfaction Score (0–10)',
    yaxis_title='Last Evaluation Score (0–10)',
    legend_title_text='Churn Status',
    title_font_size=18,
    plot_bgcolor='white',
    paper_bgcolor='white',
)
fig2.show()

# ==========================================================
# 3️⃣ Churn Rate by Satisfaction Level (Binned)
# ==========================================================
# Create satisfaction buckets
bins = [0, 3, 5, 7, 10]
labels = ['Low (0–3)', 'Medium (4–5)', 'High (6–7)', 'Very High (8–10)']
df['Satisfaction_Bin'] = pd.cut(df['Satisfaction'], bins=bins, labels=labels, include_lowest=True)

churn_by_sat = df.groupby('Satisfaction_Bin')['Churn'].mean().reset_index()
churn_by_sat['Churn'] = churn_by_sat['Churn'] * 100  # convert to %

fig3 = px.bar(
    churn_by_sat,
    x='Satisfaction_Bin',
    y='Churn',
    color='Churn',
    text='Churn',
    color_continuous_scale='Reds',
    title='Churn Rate by Satisfaction Level (%)',
)
fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig3.update_layout(
    xaxis_title='Satisfaction Level',
    yaxis_title='Churn Rate (%)',
    title_font_size=18,
    plot_bgcolor='white',
    paper_bgcolor='white',
)
fig3.show()
