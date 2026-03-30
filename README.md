# Predictive Modeling and Fairness in Health Insurance Pricing

This repository presents a machine learning project conducted as part of a Master's-level course in Machine Learning and Data Mining at the **University of Neuchâtel**, within the [Swiss Joint Master in Computer Science](https://mcs.unibnf.ch/).

The project was carried out under the supervision of **Prof. Christos Dimitrakakis**.

---

## Overview

Machine learning models are increasingly used in high-stakes domains such as healthcare and insurance pricing. In such settings, **accuracy alone is not sufficient** — models must also be evaluated for **fairness**.

This project investigates the relationship between predictive performance and fairness in the context of health insurance pricing.

We address two central questions:

- **Q1 — Cost Differences:**  
  Do smokers incur higher medical costs in the dataset?

- **Q2 — Algorithmic Fairness:**  
  Does a predictive model reproduce these differences without introducing additional bias?

The goal is to distinguish between:
- real-world risk differences  
- unfair behavior introduced by machine learning models  

---

## Dataset

- Synthetic but realistic health insurance dataset from [Kaggle](https://www.kaggle.com/datasets/sridharstreaks/insurance-data-for-machine-learning/data)
- ~1,000,000 observations
- Features include:
  - Demographics: age, gender, region
  - Lifestyle: BMI, exercise frequency, smoking status
  - Medical history (individual and family)
  - Insurance plan (coverage level)

- Target variable:
  - **Annual insurance charges (regression task)**

---

## Methodology

### Data Processing Pipeline

- Data cleaning and type enforcement
- Handling missing values
- One-hot encoding of categorical variables
- Train–test split (75% / 25%)

---

### Models Evaluated

- Linear Regression
- Elastic Net (cross-validated)
- Gradient Boosting (HistGradientBoosting)
- Neural Network (MLP Regressor)
- Generalized Linear Model (Gamma with log link)

### Evaluation Metrics

- **R²** — explained variance  
- **MAE** — average prediction error  
- **RMSE** — sensitivity to large errors  

**Selected model:** Linear Regression  
(best trade-off between performance, stability, and interpretability)

---

## Fairness Analysis

Fairness is evaluated with respect to **smoking status** (smokers vs non-smokers).

### 1. Regression-Based Fairness
- Residual analysis (prediction errors)
- Mean error comparison
- Mispricing rates (underpricing / overpricing)

### 2. Statistical Inference
- Bootstrap confidence intervals (2000 resamples)
- Test whether group differences are statistically significant

### 3. Classification-Style Fairness Metrics
(After converting charges into a high-cost classification task)

- Demographic Parity (DP)
- Equal Opportunity (EO)
- Equalized Odds (EOdds)

### 4. Stress Test (Bias Injection)

- Artificial bias introduced: **+5000 added to smokers**
- Validates that fairness metrics detect discrimination

---

## Key Results

- Smokers exhibit **higher true medical costs** in the dataset
- The selected model achieves:
  - **High predictive accuracy**
  - **Stable generalization performance**
- Error metrics (MAE, residuals) are:
  - **Nearly identical across groups**
- Bootstrap confidence intervals:
  - Include zero → **no statistically significant differences**
- Fairness metrics (EO, EOdds):
  - **Low values → balanced model behavior**
- Stress test confirms:
  - The evaluation pipeline **correctly detects unfair bias**

**Conclusion:**  
The model reflects real-world cost differences **without introducing additional unfairness**

---

## Repository Contents

- `ml-fairness-insurance-pricing` — implementation and experimental analysis
- `report.pdf` — full report with methodology, results, and discussion
- `presentation.pdf` — presentation slides

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Statsmodels
- PyTorch

---

## Key Takeaways

- Distinguishing **true data differences** from **algorithmic bias** is essential  
- Fairness should be evaluated through **error patterns**, not only outcomes  
- Standard ML models can be **accurate and fair simultaneously** when properly evaluated  
- Stress testing is critical to validate fairness metrics  

---

## Authors

- Allizha Theiventhiram — University of Neuchâtel
- Flaminia Trinca — University of Bern

---

## Notes

This project evaluates fairness from a **technical and statistical perspective**.

It does not make normative or legal claims about the use of sensitive attributes in real-world insurance systems.
