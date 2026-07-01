<h1 align="center">
  <img src="https://img.shields.io/badge/ISRO-Exoplanet%20Hackathon-orange?style=for-the-badge" alt="ISRO Hackathon"/>
  <br>AI-Enabled Exoplanet Detection Pipeline
</h1>

<p align="center">
  <strong>End-to-end automated pipeline for identifying exoplanet transits from noisy NASA TESS light curves.</strong><br>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/XGBoost-%23F37626.svg?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

---

## 🌌 Overview

Modern telescopes like NASA's TESS generate massive amounts of stellar brightness data. Finding exoplanets requires identifying tiny, periodic dips in stellar brightness (transits) among significant noise, stellar blending, and false positives like eclipsing binaries. 

This repository contains our comprehensive, fully automated pipeline that cleans telescope data, models physical transits using Box Least Squares (BLS) and Transit Least Squares (TLS), and classifies the candidates using a highly accurate AI model.

### 🏆 Model Performance
- **ROC-AUC Score**: `87.44%`
- **Recall**: `83.22%`
- **Top Candidate**: `TIC 455036659` (97.71% Planetary Probability)

---

## 🖥️ Mission Control Dashboard

We provide a beautiful, dark-mode Streamlit dashboard to visually explore the targets, view phase-folded light curves, and analyze the AI's classification verdict in real-time.

![Mission Control Dashboard UI](dashboard.png)

---

## ⚙️ Architecture & Pipeline Phases

1. **FITS Ingestion & Preprocessing**: Raw TESS light curve data is parsed using `Astropy` and `Lightkurve`.
2. **Noise Cleaning**: Advanced biweight detrending via `Wotan` removes stellar variability and systemic instrument drift.
3. **Period Discovery (BLS + TLS)**: 
   - Box Least Squares efficiently finds periodic signals.
   - Transit Least Squares models the physical shape (limb-darkening, U-shapes vs. V-shapes).
4. **Feature Engineering**: 29 specific physical features are extracted (SNR, depth, stability, etc.).
5. **AI Classification Engine**: An `XGBoost` & `PyTorch` model (trained on confirmed TOI datasets) evaluates the features to produce a planetary probability score. `SHAP` values guarantee explainability.

---

## 🚀 Quick Start Guide

### 1. Installation

Clone this repository and set up the environment:

```bash
git clone https://github.com/Therocky001/isro-exoplanet-pipeline-isolated.git
cd isro-exoplanet-pipeline-isolated

# It's recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation & Indexing

Rebuild the filtered DuckDB catalog index for the raw TESS TIC crossmatch table:
```bash
python run_data_prep.py
```

Build the fast labeled training manifest using NASA Exoplanet Archive TOI labels:
```bash
python -c "from bootstrap_toi_labels import build_fast_bootstrap_manifest; build_fast_bootstrap_manifest()"
```

### 3. Training the AI Model

Train the core classification model using the generated manifest:
```bash
python train_model.py --epochs 1 --batch-size 3 --manifest output/toi_label
```
*(Note: You can tweak the epochs and batch size depending on your hardware.)*

### 4. Launch the Mission Control Dashboard

To interactively explore the dataset and run the pipeline on specific targets (like `TIC 80423805`), launch the Streamlit app:
```bash
streamlit run app.py
```
This will open the web dashboard in your default browser at `http://localhost:8501`.

---

## 🛠️ Technology Stack

- **Science & Astronomy**: `Astropy`, `Lightkurve`, `Wotan`, `TransitLeastSquares (TLS)`
- **AI & Machine Learning**: `PyTorch`, `XGBoost`, `Scikit-Learn`, `SHAP`
- **Data & Processing**: `Python 3.9`, `Numba`, `NumPy`, `Pandas`, `DuckDB`
- **UI/Visualization**: `Streamlit`, `Matplotlib`

---

## 🔬 Future Work & Ablation Study

Our ablation studies revealed that only a small portion of the current dataset (8.8%) contains full TLS morphological enrichment. Expanding TLS coverage natively across the dataset is the most direct path to breaking the `>90%` ROC-AUC threshold in future iterations.
