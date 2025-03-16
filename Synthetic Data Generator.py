#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from ctgan import CTGAN, TVAE
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import numpy as np
import os
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

torch.classes.__path__ = [] 

# Max file size (1 GB in bytes)
MAX_SIZE = 1_073_741_824  # 1 GB

# Cache the data loading and preprocessing
@st.cache_data
def load_and_preprocess(file, file_size):
    if file_size > MAX_SIZE:
        st.error(f"File too large! Max size is {MAX_SIZE / (1024 * 1024):.0f} MB.")
        return None
    if file_size == 0:
        st.error("Uploaded file is empty!")
        return None
    
    try:
        data = pd.read_csv(file)
        if data.empty or data.shape[1] == 0:
            st.error("File has no data or columns!")
            return None
        
        # Store original statistics for later comparison
        original_stats = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            original_stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'skew': data[col].skew()
            }
        
        # Handle missing values - use mean for numeric, mode for categorical
        # but preserve distribution characteristics
        for col in numeric_cols:
            if data[col].isna().sum() > 0:
                mean_val = data[col].mean()
                std_val = data[col].std()
                # Generate random values from same distribution for missing values
                missing_mask = data[col].isna()
                missing_count = missing_mask.sum()
                if missing_count > 0 and not np.isnan(std_val) and std_val > 0:
                    random_values = np.random.normal(mean_val, std_val, missing_count)
                    data.loc[missing_mask, col] = random_values
                else:
                    data[col].fillna(mean_val, inplace=True)
        
        # Fill categorical missing values with mode
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown", inplace=True)
            
        return data, original_stats
    except Exception as e:
        st.error(f"Invalid CSV file: {str(e)}")
        return None, None

# Check if model fits the data
def check_model_suitability(data, model_type, discrete_cols):
    num_cols = data.select_dtypes(include=[np.number]).columns
    cat_cols = discrete_cols
    total_cols = len(data.columns)
    cat_ratio = len(cat_cols) / total_cols if total_cols > 0 else 0
    data_size = data.shape[0]
    
    if model_type == "CTGAN":
        if data_size < 1000:
            return "Note: CTGAN works better with larger datasets (1000+ rows)."
        if cat_ratio == 0 or cat_ratio == 1:
            return "Note: CTGAN is best for mixed data (numeric + categorical), not purely one type."
    elif model_type == "TVAE":
        if cat_ratio > 0.7:
            return "Note: TVAE is better with numeric-heavy data, not mostly categorical."
        if data_size < 500:
            return "Note: TVAE needs decent-sized data (500+ rows) for good results."
    elif model_type == "GaussianCopula":
        if cat_ratio > 0.5:
            return "Note: GaussianCopula might struggle with mostly categorical data."
        for col in num_cols:
            if abs(data[col].skew()) > 1:
                return "Note: GaussianCopula works best with Gaussian-like numeric data."
    return None

# Function to analyze and compare distributions
def compare_distributions(original, synthetic, column, stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original data histogram
    sns.histplot(original[column], kde=True, ax=ax1)
    ax1.set_title(f"Original {column} Distribution")
    ax1.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
    
    # Synthetic data histogram
    sns.histplot(synthetic[column], kde=True, ax=ax2)
    ax2.set_title(f"Synthetic {column} Distribution")
    ax2.axvline(synthetic[column].mean(), color='r', linestyle='--', 
                label=f"Mean: {synthetic[column].mean():.2f}")
    
    # Stats comparison
    fig.suptitle(f"Distribution Comparison for {column}")
    plt.tight_layout()
    return fig

# Basic setup
st.title("Synthetic Data Generator Pro")

# File upload with validation
data_file = st.file_uploader("Upload your data file (CSV, max 1 GB)", type="csv")
if data_file is not None:
    file_size = data_file.size
    result = load_and_preprocess(data_file, file_size)
    
    if result is None:
        st.stop()
    
    data, original_stats = result
    
    # Show quick preview
    st.write("Data Preview:", data.head())
    st.write(f"File Size: {file_size / (1024 * 1024):.2f} MB")
    
    # Show distribution info
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        col_for_dist = st.selectbox("Select column to view distribution:", options=numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data[col_for_dist], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col_for_dist}")
        ax.axvline(data[col_for_dist].mean(), color='r', linestyle='--', 
                   label=f"Mean: {data[col_for_dist].mean():.2f}")
        ax.axvline(data[col_for_dist].mean() + data[col_for_dist].std(), color='g', linestyle='--', 
                   label=f"Mean + StdDev: {data[col_for_dist].mean() + data[col_for_dist].std():.2f}")
        ax.axvline(data[col_for_dist].mean() - data[col_for_dist].std(), color='g', linestyle='--', 
                   label=f"Mean - StdDev: {data[col_for_dist].mean() - data[col_for_dist].std():.2f}")
        ax.legend()
        st.pyplot(fig)
        
        # Show distribution statistics
        st.write(f"**Distribution Statistics for {col_for_dist}:**")
        st.write(f"- Mean: {data[col_for_dist].mean():.4f}")
        st.write(f"- Median: {data[col_for_dist].median():.4f}")
        st.write(f"- Std Dev: {data[col_for_dist].std():.4f}")
        st.write(f"- Skewness: {data[col_for_dist].skew():.4f} (0 = perfectly normal)")
        st.write(f"- Kurtosis: {data[col_for_dist].kurtosis():.4f} (0 = perfectly normal)")
        
        # Normality test
        stat, p_value = stats.normaltest(data[col_for_dist].dropna())
        st.write(f"- Normality Test p-value: {p_value:.6f} ({'Normal' if p_value > 0.05 else 'Non-normal'} distribution)")
    
    # Pick discrete columns
    discrete_options = data.select_dtypes(include=['object']).columns.tolist()
    selected_discrete = st.multiselect("Select discrete columns:", options=discrete_options, default=discrete_options)
    
    # Model choice and settings
    model_type = st.selectbox("Pick a model:", ["GaussianCopula", "CTGAN", "TVAE"])
    st.write("Note: GaussianCopula is best for preserving normal distributions!")
    
    num_samples = st.number_input("How many synthetic rows?", min_value=1, value=100)
    
    # Advanced settings expander
    with st.expander("Advanced Settings"):
        epochs = st.slider("Max epochs:", min_value=10, max_value=500, value=200)
        batch_size = st.slider("Batch size:", min_value=100, max_value=1000, value=500)
        
        # GaussianCopula specific settings
        if model_type == "GaussianCopula":
            enforce_bounds = st.checkbox("Enforce min/max values", value=True)
            enforce_rounding = st.checkbox("Round discrete values", value=True)
            
        # CTGAN/TVAE specific settings
        if model_type in ["CTGAN", "TVAE"]:
            st.warning("Neural models may not perfectly preserve normal distributions without careful tuning!")
    
    # Show model suitability warning
    warning = check_model_suitability(data, model_type, selected_discrete)
    if warning:
        st.warning(warning)
    
    # Generate button
    if st.button("Generate Synthetic Data"):
        try:
            with st.spinner(f"Training {model_type}..."):
                if model_type == "CTGAN":
                    # Adjust pac to fit dataset size
                    pac = 10 if data.shape[0] % 10 == 0 else 1  
                    model = CTGAN(
                        epochs=epochs,
                        batch_size=batch_size,
                        generator_dim=(256, 256, 256),
                        discriminator_dim=(256, 256, 256),
                        generator_lr=2e-4,
                        discriminator_lr=2e-4,
                        pac=pac,  
                        verbose=True
                    )
                    model.fit(data, selected_discrete)
                    synthetic_data = model.sample(num_samples)
                    
                elif model_type == "TVAE":
                    model = TVAE(
                        epochs=epochs,
                        batch_size=batch_size,
                        embedding_dim=128,
                        compress_dims=(128, 128),
                        decompress_dims=(128, 128),
                        l2scale=1e-5,
                        verbose=True
                    )
                    model.fit(data, selected_discrete)
                    synthetic_data = model.sample(num_samples)
                    
                else:  # GaussianCopula
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data)
                    model = GaussianCopulaSynthesizer(
                        metadata,
                        enforce_min_max_values=enforce_bounds if 'enforce_bounds' in locals() else True,
                        enforce_rounding=enforce_rounding if 'enforce_rounding' in locals() else True
                    )
                    model.fit(data)
                    synthetic_data = model.sample(num_samples)
            
            # Show results
            st.write("Synthetic Data Preview:", synthetic_data.head())
            
            # Compare distributions for one selected column
            if not numeric_cols.empty:
                st.subheader("Distribution Comparison")
                col_to_compare = st.selectbox("Select column to compare distributions:", options=numeric_cols)
                
                fig = compare_distributions(data, synthetic_data, col_to_compare, original_stats[col_to_compare])
                st.pyplot(fig)
                
                # Statistical comparison
                original_mean = data[col_to_compare].mean()
                synth_mean = synthetic_data[col_to_compare].mean()
                mean_diff = abs(original_mean - synth_mean)
                mean_diff_pct = (mean_diff / original_mean) * 100 if original_mean != 0 else 0
                
                original_std = data[col_to_compare].std()
                synth_std = synthetic_data[col_to_compare].std()
                std_diff_pct = (abs(original_std - synth_std) / original_std) * 100 if original_std != 0 else 0
                
                # KS test for distribution similarity
                ks_stat, ks_pvalue = stats.ks_2samp(data[col_to_compare].dropna(), 
                                                   synthetic_data[col_to_compare].dropna())
                
                st.write("**Statistical Comparison:**")
                st.write(f"- Mean difference: {mean_diff:.4f} ({mean_diff_pct:.2f}%)")
                st.write(f"- Std Dev difference: {abs(original_std - synth_std):.4f} ({std_diff_pct:.2f}%)")
                st.write(f"- KS test p-value: {ks_pvalue:.6f} ({'Similar' if ks_pvalue > 0.05 else 'Different'} distributions)")
                
                # Guidance based on results
                if ks_pvalue < 0.05 or mean_diff_pct > 5 or std_diff_pct > 10:
                    st.warning("⚠️ The synthetic data distribution differs significantly from the original. Consider:")
                    if model_type != "GaussianCopula":
                        st.write("- Switching to GaussianCopula which better preserves statistical distributions")
                    if epochs < 300 and model_type != "GaussianCopula":
                        st.write("- Increasing the number of training epochs")
                    st.write("- Increasing your sample size")
                else:
                    st.success("✅ The synthetic data preserves the original distribution well!")
            
            # Download button
            st.download_button(
                label="Download Synthetic Data",
                data=synthetic_data.to_csv(index=False),
                file_name=f"synthetic_{model_type}.csv"
            )
        except Exception as e:
            st.error(f"Error generating synthetic data: {str(e)}")

# Footer note
st.info("**This synthetic data generator focuses on preserving statistical distributions of the original data, particularly normal distributions. The GaussianCopula model is recommended for data that follows a normal distribution.**")
