# Statistical Analysis & Interactive Dashboard

This repository contains two main components:

## 📊 Task 1: Time on Page vs Revenue Analysis

**Files:**
- `analysis.py` - Complete statistical analysis script
- `Time_Revenue_Analysis.pdf` - 3-page analysis report
- `code_appendix.html` - Code documentation
- `testdata (1).csv` - Dataset with 4,000 user sessions

**Key Findings:**
- Complex relationship between time on page and revenue
- Mixed correlations: Pearson (-0.555) vs Spearman (+0.608)
- Short sessions dominate (92% of data)
- Outliers heavily influence patterns
- Only 31% of revenue variance explained by time on page

**Methods Used:**
- Exploratory Data Analysis (EDA)
- Correlation analysis (Pearson & Spearman)
- Linear regression modeling
- Outlier detection and analysis
- Data visualization (scatter plots, box plots, bar charts)

## 🎯 Task 2: Interactive Statistical Dashboard

**Files:**
- `streamlit_dashboard.py` - Interactive web application
- `requirements.txt` - Python dependencies

**Features:**
- Demonstrates Central Limit Theorem (CLT)
- Interactive distribution selection (Normal, Exponential, Uniform, Poisson, Gamma, Beta, Chi-Square)
- Real-time parameter adjustment
- Sample size and count controls
- Statistical metrics and visualizations
- Q-Q plots for normality testing

**Live Demo:** [Deployed on Streamlit Cloud](https://share.streamlit.io/)

## 🚀 Quick Start

### For Analysis:
```bash
pip install pandas numpy matplotlib scipy
python analysis.py
```

### For Dashboard:
```bash
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

## 📋 Requirements

See `requirements.txt` for complete list:
- streamlit>=1.28.0
- numpy>=1.24.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- scipy>=1.10.0
- plotly>=5.15.0

## 📈 Statistical Methods Demonstrated

1. **Central Limit Theorem** - Sample means approach normal distribution
2. **Distribution Properties** - Mean, variance, skewness, kurtosis
3. **Normality Testing** - Q-Q plots and statistical tests
4. **Interactive Visualization** - Real-time parameter exploration

## 🎓 Educational Value

Perfect for:
- Statistics students learning CLT
- Data science practitioners
- Anyone interested in statistical distributions
- Interactive learning and experimentation
