# 📈 Churn Rate Analysis Project

This repository contains an end-to-end data analytics pipeline and visualization project exploring **customer churn behavior** and **business retention patterns**.  
It combines data preprocessing, feature engineering, model building, and interactive dashboarding to uncover key drivers of customer churn and support data-driven decision-making.

---

## 🧠 Project Overview

The project aims to answer:
- What customer segments are most likely to churn?
- Which features (usage frequency, demographics, etc.) most strongly predict churn?
- How can retention strategies be optimized to reduce churn rates?

By combining Python-based analysis with visual storytelling and interactive dashboards, the project translates raw data into actionable business insights.

---

## 🧩 Components

| Module | Description |
|---------|-------------|
| `data_preprocessing/` | Cleans raw datasets, handles missing values, standardizes numeric features |
| `feature_engineering/` | Generates key behavioral metrics and churn-risk indicators |
| `model_training/` | Trains and evaluates models (Logistic Regression, Random Forest, XGBoost) |
| `evaluation/` | Produces confusion matrices, ROC curves, precision-recall trade-offs |
| `dashboard/` | Tableau or Plotly dashboards to visualize churn by customer segments |
| `reports/` | Business summaries and presentation files |

---

## 🧰 Tech Stack

- **Python** – pandas, numpy, matplotlib, seaborn, scikit-learn  
- **SQL** – customer segmentation and aggregation queries  
- **Tableau / Plotly** – interactive dashboards  
- **Jupyter Notebook** – reproducible analysis workflow  

---

## 📊 Example Output: Laneige E-Commerce Dashboard (June 2025)

> A real-world visualization example from a related dataset analyzing product and video performance.

**Highlights:**
- 4,309 total orders generating **$152K revenue**
- **Average Order Value (AOV):** $39.31  
- **Sampling ratio:** 31.6%  
- Over **10K videos** analyzed (≈353/day)  
- Product concentration: top 3 SKUs contributed **80.9% of total orders**

**Actionable Insights:**
- Focus on top-performing SKUs (`Glaze Craze Serum`, `Mini Lip Balm Set`, `Bubble Tea Balm`)  
- Optimize ROI by reducing heavy sample reliance  
- Build look-alike audiences for high-conversion creative groups  
- Regional focus: Western & Northeastern U.S. show strongest profit margins:contentReference[oaicite:0]{index=0}

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Gavin-tian/ChurnRateAnalysis_Project.git

# Navigate to the directory
cd ChurnRateAnalysis_Project

# (Optional) create virtual environment
python3 -m venv venv
source venv/bin/activate  # on macOS/Linux
venv\Scripts\activate     # on Windows

# Install dependencies
pip install -r requirements.txt
