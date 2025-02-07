import streamlit as st
import numpy as np
import scipy.stats as stats
from scipy.stats import gamma
import plotly.graph_objects as go
from typing import Tuple, Callable

BRAND_COLORS = {
    'primary': '#1b3a6f',    # Main color for titles and primary elements
    'secondary': '#c5a46d',  # Secondary elements
    'background': '#f5f1e3', # Background color
    'accent1': '#5ab9ea',    # Accent for highlights
    'accent2': '#b7bbc2'     # Secondary accent
}

DISTRIBUTIONS = {
    'Normal (e.g., Blood Pressure Readings)': stats.norm,
    'Binomial (e.g., Treatment Success Count)': stats.binom,
    'Gamma (e.g., Patient Recovery Time)': stats.gamma,
    'Exponential (e.g., Time Between Emergency Admissions)': stats.expon
}

def get_distribution_params(dist_name: str) -> Tuple[dict, Callable]:
    """Get distribution parameters based on user input and return parameter dict and PDF/PMF function."""
    if 'Normal' in dist_name:
        mu = st.slider('Mean (μ)', -10.0, 10.0, 0.0, 
                      help="For blood pressure, this might be 120 mmHg")
        sigma = st.slider('Standard Deviation (σ)', 0.1, 5.0, 1.0,
                         help="Spread of measurements around the mean")
        return {'loc': mu, 'scale': sigma}, lambda x: DISTRIBUTIONS[dist_name].pdf(x, **{'loc': mu, 'scale': sigma})
    
    elif 'Binomial' in dist_name:
        n = st.slider('Number of Trials (n)', 1, 100, 20,
                     help="E.g., number of patients in the study")
        p = st.slider('Probability of Success (p)', 0.0, 1.0, 0.5,
                     help="E.g., probability of treatment success for each patient")
        return {'n': n, 'p': p}, lambda x: DISTRIBUTIONS[dist_name].pmf(x, n, p)
    
    elif 'Gamma' in dist_name:
        shape = st.slider('Shape (k)', 0.1, 10.0, 2.0,
                         help="Controls the basic shape of recovery time distribution")
        scale = st.slider('Scale (θ)', 0.1, 5.0, 1.0,
                         help="Stretches or compresses the distribution")
        return {'a': shape, 'scale': scale}, lambda x: DISTRIBUTIONS[dist_name].pdf(x, shape, scale=scale)
    
    else:  # Exponential
        scale = st.slider('Scale (β)', 0.1, 5.0, 1.0,
                         help="Average time between events (e.g., admissions)")
        return {'scale': scale}, lambda x: DISTRIBUTIONS[dist_name].pdf(x, scale=scale)

def plot_distribution(dist_name: str, params: dict, pdf_func: Callable, show_samples: bool, n_samples: int = 100):
    """Create and return a plot for the selected distribution."""
    fig = go.Figure()
    
    # Handle Binomial distribution separately as it's discrete
    if 'Binomial' in dist_name:
        n = params['n']
        x = np.arange(0, n + 1)  # All possible values from 0 to n
        y = [pdf_func(i) for i in x]
        
        # Create bar plot for PMF
        fig.add_trace(go.Bar(x=x, y=y, 
                           name='Theoretical PMF',
                           marker_color=BRAND_COLORS['primary']))
        
        if show_samples:
            samples = DISTRIBUTIONS[dist_name].rvs(**params, size=n_samples)
            # Calculate proportion for each possible value
            values, counts = np.unique(samples, return_counts=True)
            proportions = counts / n_samples
            
            # Create bar plot for samples
            fig.add_trace(go.Bar(x=values, 
                               y=proportions,
                               name='Sampled Proportion',
                               opacity=0.7,
                               marker_color=BRAND_COLORS['secondary']))
            
        fig.update_layout(
            xaxis_title="Number of Successes",
            yaxis_title="Probability",
            bargap=0.2
        )
    
    else:
        # For continuous distributions
        if 'Normal' in dist_name:
            range_val = np.abs(params.get('loc', 0)) + (4 * params.get('scale', 1))
            x = np.linspace(-range_val, range_val, 1000)
        else:
            x = np.linspace(0, 10, 1000)
        
        y = pdf_func(x)
        fig.add_trace(go.Scatter(x=x, y=y, 
                               mode='lines', 
                               name='Theoretical',
                               line_color=BRAND_COLORS['primary']))
        
        if show_samples:
            samples = DISTRIBUTIONS[dist_name].rvs(**params, size=n_samples)
            fig.add_trace(go.Histogram(x=samples, 
                                     histnorm='probability density',
                                     name='Sampled',
                                     opacity=0.7,
                                     marker_color=BRAND_COLORS['secondary'],
                                     nbinsx=30))
    
    fig.update_layout(
        title=dict(
            text=f"{dist_name.split('(')[0].strip()} Distribution",
            font=dict(family="Merriweather", size=24)
        ),
        font=dict(family="Lato"),
        showlegend=show_samples
    )
    
    if show_samples:
        return fig, samples
    else:
        return fig

def plot_clt_demonstration(dist_name: str, params: dict, sample_size: int, num_samples: int):
    """
    Create a visualization demonstrating the Central Limit Theorem by showing
    the distribution of sample means compared to a normal distribution.
    """
    # Generate multiple samples and calculate their means
    dist = DISTRIBUTIONS[dist_name]
    sample_means = []
    
    for _ in range(num_samples):
        sample = dist.rvs(**params, size=sample_size)
        sample_means.append(np.mean(sample))
    
    # Create figure for CLT demonstration
    fig = go.Figure()
    
    # Plot histogram of sample means
    fig.add_trace(go.Histogram(
        x=sample_means,
        histnorm='probability density',
        name='Distribution of Sample Means',
        opacity=0.7,
        marker_color=BRAND_COLORS['secondary'],
        nbinsx=30
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Central Limit Theorem Demonstration",
            font=dict(family="Merriweather", size=24)
        ),
        xaxis_title=f"Sample Means (n={sample_size})",
        yaxis_title="Density",
        font=dict(family="Lato"),
        showlegend=True,
    )
    
    return fig, sample_means

def main():
    st.set_page_config(page_title="Medical Statistics Explorer")
    
    st.title("Medical Distribution Explorer")
    st.markdown("""
    Explore common probability distributions in medical data analysis. 
    Select a distribution to visualize its properties and understand its medical applications.
    """)
    
    # Distribution selection
    dist_name = st.selectbox("Select Distribution", list(DISTRIBUTIONS.keys()))
    
    # Get distribution parameters and PDF/PMF function
    params, pdf_func = get_distribution_params(dist_name)
    
    # Sampling controls
    show_samples = st.checkbox("Show Sample Distribution", value=False,
                             help="Compare theoretical distribution with random samples")
    if show_samples:
        n_samples = st.slider("Number of Samples", 1, 50, 10)
    else:
        n_samples = 10
    
    # Create and display plot
    if show_samples:
        fig, samples = plot_distribution(dist_name, params, pdf_func, show_samples, n_samples)
    else:
        fig = plot_distribution(dist_name, params, pdf_func, show_samples, n_samples)
    st.plotly_chart(fig)
    
    # Calculate and display statistics
    dist = DISTRIBUTIONS[dist_name](**params)
    
    mean = dist.mean()
    std = dist.std()
    skew = dist.stats(moments='s')
    kurt = dist.stats(moments='k')
    
    st.subheader("Theoretical Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{mean:.2f}")
    with col2:
        st.metric("Std Dev", f"{std:.2f}")
    with col3:
        st.metric("Skewness", f"{float(skew):.2f}")
    with col4:
        st.metric("Kurtosis", f"{float(kurt):.2f}")

    if show_samples:
        st.subheader("Sample Statistics")
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        sample_skew = stats.skew(samples)
        sample_kurt = stats.kurtosis(samples)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sample Mean", f"{sample_mean:.2f}")
        with col2:
            st.metric("Sample Std", f"{sample_std:.2f}")
        with col3:
            st.metric("Sample Skewness", f"{sample_skew:.2f}")
        with col4:
            st.metric("Sample Kurtosis", f"{sample_kurt:.2f}")

    # Add CLT demonstration section
    st.markdown("---")
    show_clt = st.checkbox("Show Central Limit Theorem Demonstration", 
                          help="Visualize how sample means approach a normal distribution")
    
    if show_clt:
        st.markdown("""
        ### Central Limit Theorem Demonstration
        
        The Central Limit Theorem states that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the underlying distribution's shape. This is crucial in medical statistics because:
        
        - It allows us to make inferences about population parameters even when the original data isn't normally distributed
        - It helps us understand why many medical measurements tend toward normal distributions
        - It provides the foundation for many statistical tests used in medical research
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Sample Size (n)", 
                                  min_value=2, 
                                  max_value=50, 
                                  value=30,
                                  help="Number of observations in each sample")
        with col2:
            num_samples = st.slider("Number of Samples", 
                                  min_value=10, 
                                  max_value=500, 
                                  value=100,
                                  help="Number of samples to draw")
        
        # Create and display CLT demonstration
        clt_fig, _ = plot_clt_demonstration(dist_name, params, sample_size, num_samples)
        st.plotly_chart(clt_fig)


if __name__ == "__main__":
    main()