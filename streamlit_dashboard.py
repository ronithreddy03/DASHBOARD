import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Statistical Distributions Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Statistical Distributions Explorer</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎛️ Controls")

# Distribution selection
distribution = st.sidebar.selectbox(
    "Choose Distribution:",
    ["Normal", "Exponential", "Uniform", "Poisson", "Gamma", "Beta", "Chi-Square"]
)

# Parameters based on distribution
if distribution == "Normal":
    mu = st.sidebar.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
    sigma = st.sidebar.slider("Standard Deviation (σ)", 0.1, 3.0, 1.0, 0.1)
    params = {"loc": mu, "scale": sigma}
elif distribution == "Exponential":
    lambda_param = st.sidebar.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)
    params = {"scale": 1/lambda_param}
elif distribution == "Uniform":
    a = st.sidebar.slider("Lower bound (a)", -5.0, 0.0, 0.0, 0.1)
    b = st.sidebar.slider("Upper bound (b)", 0.0, 10.0, 1.0, 0.1)
    params = {"low": a, "high": b}
elif distribution == "Poisson":
    lambda_param = st.sidebar.slider("Rate (λ)", 0.1, 20.0, 5.0, 0.1)
    params = {"mu": lambda_param}
elif distribution == "Gamma":
    alpha = st.sidebar.slider("Shape (α)", 0.1, 10.0, 2.0, 0.1)
    beta = st.sidebar.slider("Rate (β)", 0.1, 5.0, 1.0, 0.1)
    params = {"a": alpha, "scale": 1/beta}
elif distribution == "Beta":
    alpha = st.sidebar.slider("α", 0.1, 10.0, 2.0, 0.1)
    beta = st.sidebar.slider("β", 0.1, 10.0, 2.0, 0.1)
    params = {"a": alpha, "b": beta}
elif distribution == "Chi-Square":
    df = st.sidebar.slider("Degrees of Freedom", 1, 20, 5, 1)
    params = {"df": df}

# Sample size
sample_size = st.sidebar.slider("Sample Size", 10, 10000, 1000, 10)

# Number of samples for CLT
n_samples = st.sidebar.slider("Number of Samples (for CLT)", 10, 1000, 100, 10)

# Generate data
@st.cache_data
def generate_data(dist_name, params, n_samples, sample_size):
    """Generate data for the selected distribution"""
    if dist_name == "Normal":
        data = np.random.normal(**params, size=(n_samples, sample_size))
    elif dist_name == "Exponential":
        data = np.random.exponential(**params, size=(n_samples, sample_size))
    elif dist_name == "Uniform":
        data = np.random.uniform(**params, size=(n_samples, sample_size))
    elif dist_name == "Poisson":
        data = np.random.poisson(**params, size=(n_samples, sample_size))
    elif dist_name == "Gamma":
        data = np.random.gamma(**params, size=(n_samples, sample_size))
    elif dist_name == "Beta":
        data = np.random.beta(**params, size=(n_samples, sample_size))
    elif dist_name == "Chi-Square":
        data = np.random.chisquare(**params, size=(n_samples, sample_size))
    
    return data

# Generate the data
data = generate_data(distribution, params, n_samples, sample_size)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Distribution Visualization")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Original Distribution', 'Sample Means Distribution', 
                       'Q-Q Plot', 'Histogram with Theoretical PDF'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Original distribution (first sample)
    x_original = data[0]
    
    # Sample means
    sample_means = np.mean(data, axis=1)
    
    # Theoretical PDF
    x_range = np.linspace(min(x_original), max(x_original), 1000)
    if distribution == "Normal":
        theoretical_pdf = stats.norm.pdf(x_range, **params)
    elif distribution == "Exponential":
        theoretical_pdf = stats.expon.pdf(x_range, **params)
    elif distribution == "Uniform":
        theoretical_pdf = stats.uniform.pdf(x_range, **params)
    elif distribution == "Poisson":
        # For discrete distributions, we'll use a histogram approach
        theoretical_pdf = None
    elif distribution == "Gamma":
        theoretical_pdf = stats.gamma.pdf(x_range, **params)
    elif distribution == "Beta":
        theoretical_pdf = stats.beta.pdf(x_range, **params)
    elif distribution == "Chi-Square":
        theoretical_pdf = stats.chi2.pdf(x_range, **params)
    
    # Plot 1: Original distribution
    fig.add_trace(
        go.Histogram(x=x_original, nbinsx=30, name="Sample Data", 
                    marker_color='lightblue', opacity=0.7),
        row=1, col=1
    )
    
    # Plot 2: Sample means distribution
    fig.add_trace(
        go.Histogram(x=sample_means, nbinsx=30, name="Sample Means", 
                    marker_color='lightgreen', opacity=0.7),
        row=1, col=2
    )
    
    # Plot 3: Q-Q plot
    theoretical_quantiles = stats.probplot(x_original, dist="norm", plot=None)
    if len(theoretical_quantiles) >= 2 and len(theoretical_quantiles[0]) > 0 and len(theoretical_quantiles[1]) > 0:
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles[0], y=theoretical_quantiles[1], 
                      mode='markers', name="Q-Q Plot", marker_color='red'),
            row=2, col=1
        )
    
    # Add diagonal line for Q-Q plot
    if len(theoretical_quantiles) >= 2 and len(theoretical_quantiles[0]) > 0 and len(theoretical_quantiles[1]) > 0:
        min_val = min(theoretical_quantiles[0].min(), theoretical_quantiles[1].min())
        max_val = max(theoretical_quantiles[0].max(), theoretical_quantiles[1].max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name="Perfect Fit", line=dict(color='black', dash='dash')),
            row=2, col=1
        )
    
    # Plot 4: Histogram with theoretical PDF
    fig.add_trace(
        go.Histogram(x=x_original, nbinsx=30, name="Sample Data", 
                    marker_color='lightcoral', opacity=0.7),
        row=2, col=2
    )
    
    if theoretical_pdf is not None and len(theoretical_pdf) > 0:
        try:
            fig.add_trace(
                go.Scatter(x=x_range, y=theoretical_pdf * len(x_original) * (max(x_original) - min(x_original)) / 30,
                          mode='lines', name="Theoretical PDF", line=dict(color='red', width=2)),
                row=2, col=2
            )
        except:
            pass  # Skip if there's an error with theoretical PDF
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text=f"{distribution} Distribution Analysis"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Statistical Summary")
    
    # Calculate statistics
    mean_original = np.mean(x_original)
    std_original = np.std(x_original)
    mean_of_means = np.mean(sample_means)
    std_of_means = np.std(sample_means)
    
    # Display metrics
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Original Mean", f"{mean_original:.4f}")
    st.metric("Original Std Dev", f"{std_original:.4f}")
    st.metric("Mean of Sample Means", f"{mean_of_means:.4f}")
    st.metric("Std Dev of Sample Means", f"{std_of_means:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Central Limit Theorem info
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**Central Limit Theorem:**")
    st.write(f"Expected Std Dev of Means: {std_original/np.sqrt(sample_size):.4f}")
    st.write(f"Actual Std Dev of Means: {std_of_means:.4f}")
    st.write(f"Ratio: {std_of_means/(std_original/np.sqrt(sample_size)):.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Additional analysis
st.subheader("🔍 Detailed Analysis")

col3, col4 = st.columns(2)

with col3:
    st.write("**Distribution Properties:**")
    
    # Skewness and Kurtosis
    skewness = stats.skew(x_original)
    kurtosis = stats.kurtosis(x_original)
    
    st.write(f"Skewness: {skewness:.4f}")
    st.write(f"Kurtosis: {kurtosis:.4f}")
    
    # Normality test
    _, p_value = stats.normaltest(x_original)
    st.write(f"Normality Test p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write("❌ Data is NOT normally distributed")
    else:
        st.write("✅ Data appears to be normally distributed")

with col4:
    st.write("**Sample Statistics:**")
    
    # Percentiles
    percentiles = [25, 50, 75, 95, 99]
    for p in percentiles:
        value = np.percentile(x_original, p)
        st.write(f"{p}th percentile: {value:.4f}")
    
    # Range
    data_range = np.max(x_original) - np.min(x_original)
    st.write(f"Range: {data_range:.4f}")

# Interactive simulation
st.subheader("🎲 Interactive Simulation")

# Add some interactivity
if st.button("Generate New Sample"):
    st.cache_data.clear()
    st.rerun()

# Add distribution information
st.subheader("📚 Distribution Information")

dist_info = {
    "Normal": "The normal distribution is symmetric and bell-shaped. It's characterized by its mean and standard deviation. Many natural phenomena follow this distribution.",
    "Exponential": "The exponential distribution describes the time between events in a Poisson process. It's memoryless and always positive.",
    "Uniform": "The uniform distribution has equal probability for all values in its range. It's the simplest continuous distribution.",
    "Poisson": "The Poisson distribution describes the number of events occurring in a fixed interval. It's discrete and always non-negative.",
    "Gamma": "The gamma distribution is a two-parameter family of continuous probability distributions. It's always positive and can be skewed.",
    "Beta": "The beta distribution is defined on the interval [0,1]. It's very flexible and can take many shapes.",
    "Chi-Square": "The chi-square distribution is used in hypothesis testing and confidence intervals. It's always positive and right-skewed."
}

st.info(dist_info.get(distribution, "No information available for this distribution."))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Demonstrates Central Limit Theorem and Statistical Properties</p>
</div>
""", unsafe_allow_html=True)
