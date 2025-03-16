# Synthetic Data Generator Pro

A powerful Streamlit application for generating high-quality synthetic data while preserving the statistical characteristics of your original datasets.

[LIVE DEMO](https://generatesyntheticdata.streamlit.app/)

## Features

- Generate synthetic data with three different models: GaussianCopula, CTGAN, and TVAE
- Preserve statistical distributions and relationships between variables
- Interactive visualization of distributions for comparison
- Automatic model recommendation based on your data characteristics
- Statistical validation of synthetic data quality
- Simple and intuitive user interface

## Installation

```bash
# Clone the repository
git clone https://github.com/username/synthetic-data-generator-pro.git
cd synthetic-data-generator-pro

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

This application requires the following packages:
- streamlit
- pandas
- numpy
- ctgan
- sdv
- torch
- scipy
- matplotlib
- seaborn

You can install all requirements using the provided `requirements.txt` file.

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your CSV data file (max 1GB)
3. Review data statistics and distributions
4. Select discrete columns in your dataset
5. Choose the synthetic data model (GaussianCopula, CTGAN, or TVAE)
6. Configure number of samples to generate
7. Adjust advanced settings if needed
8. Generate synthetic data
9. Download the resulting synthetic dataset

## Models

### GaussianCopula
- Best for preserving normal distributions
- Works well with numeric data following Gaussian patterns
- Maintains statistical relationships while protecting privacy
- Fast and efficient even with large datasets

### CTGAN (Conditional Tabular GAN)
- Excels with mixed data types (categorical + numeric)
- Best for larger datasets (1000+ rows)
- Can capture complex non-linear relationships
- May require careful tuning for distribution preservation

### TVAE (Tabular Variational Autoencoder)
- Good with numeric-heavy data
- Works best with medium to large datasets (500+ rows)
- Robust to missing values and outliers
- Balances privacy and utility well

## Statistical Validation

The app provides comprehensive validation of your synthetic data:
- Visual comparison of distributions
- Mean and standard deviation comparisons
- Kolmogorov-Smirnov test for distribution similarity
- Automated recommendations based on results

## Model Selection Guidelines

- Use **GaussianCopula** when:
  - Your data has mostly normal distributions
  - Statistical fidelity is the top priority
  - You need fast generation with limited computational resources

- Use **CTGAN** when:
  - You have complex relationships between variables
  - Your dataset has many categorical variables
  - You have a large dataset (1000+ rows)

- Use **TVAE** when:
  - You need a balance between privacy and utility
  - Your data is primarily numeric
  - You have at least 500 rows of data

## Advanced Settings

For expert users, additional settings are available:
- Control training epochs and batch size
- Enforce value bounds and rounding for GaussianCopula
- Adjust model architecture parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [SDV](https://github.com/sdv-dev/SDV) - Synthetic Data Vault
- [CTGAN](https://github.com/sdv-dev/CTGAN) - Conditional Tabular GAN
- [Streamlit](https://streamlit.io/) - Interactive Web Framework
