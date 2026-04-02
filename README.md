# Used Car Price Prediction — A Machine Learning Walkthrough

An educational notebook that walks through a complete machine learning pipeline to predict the price of second-hand cars based on their characteristics.

---

## What's inside

The notebook covers the full process from raw data to a working predictive model:

1. **Exploratory Data Analysis (EDA)** — duplicate removal, missing value handling, outlier detection, feature engineering and visualizations
2. **Simple Linear Regression** — individual analysis of each feature's contribution to price
3. **Multiple Linear Regression** — combining all features into a single model
4. **LASSO Regression** — regularized model with automatic feature selection
5. **Ridge Regression** — regularized model optimized for stability and generalization
6. **Polynomial Regression** — capturing non-linear relationships between numerical features and price

Each section explains not just *what* was done, but *why* — including alternative approaches and observations from experimenting with different configurations.

---

## Dataset

The dataset contains listings of second-hand cars with features such as make, model, mileage, engine power, fuel type, transmission, and more.

> The CSV file (`cars.csv`) is required to run the notebook.

---

## Results summary

| Model | R² (test) | R² (train) |
|---|---|---|
| Multiple Linear Regression | 0.862 | 0.902 |
| LASSO | 0.858 | 0.894 |
| **Ridge** | **0.864** | **0.895** |
| Polynomial (degree=2) | 0.840 | 0.850 |

**Best model: Ridge Regression** — highest test R² and smallest train-test gap.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Contact

Any questions or suggestions — feel free to reach out at **21alexrodri@gmail.com**
