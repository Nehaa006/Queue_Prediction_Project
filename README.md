# 🚦 Deep Learning Queue Length & Wait-Time Prediction System

An end-to-end computer vision and temporal sequence modeling pipeline designed to track multi-person queue dynamics and accurately predict waiting times from CCTV camera feeds using the **MOT17** benchmark dataset.

---

## 📌 Project Overview

Predicting queue wait times in public environments (e.g., healthcare, retail, airports) is challenging due to heavy occlusion, dynamic crowd movements, and non-linear line formations. 

This repository implements a **two-stage hybrid framework**:
1. **Spatial Tracking Stage:** Uses object tracking mechanisms (MOT17 ground truth / YOLO + DeepSORT) to count individuals and capture trajectory tracks.
2. **Temporal Prediction Stage:** Evaluates sequence architectures (1D-TCN, LSTM, CNN-1D, CSRNet) to convert trajectory features into continuous wait-time predictions.

### 🚀 Key Performance Highlights
* **Wait-Time Prediction Error (MAE):** **1.08 minutes** (~85% relative predictive precision).
* **Baseline Improvement:** **~8.5x error reduction** over traditional naive statistical baselines (10.24 min MAE).
* **Inference Speed:** **2.1 ms (`0.0021s`)** per sequence using 1D-TCN, making it ideal for real-time edge deployment.

---
### Architecture Diagram

<img width="1024" height="713" alt="image" src="https://github.com/user-attachments/assets/77bab87e-a987-4843-98f2-ffbfb130fc28" />

## 🛠️ Models Evaluated

The pipeline benchmarks six distinct statistical and deep learning architectures for wait-time prediction performance:

| Model | Model Type | MAE (min) | RMSE (min) | MSE | Latency (s) | Key Feature / Architecture Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D-TCN (Proposed)** | Temporal CNN | **1.08** | **1.42** | **2.02** | `0.0021` | Dilated causal convolutions capturing long-term temporal trends. |
| **LSTM-only** | Recurrent Neural Net | 1.10 | 1.43 | 2.04 | `0.0017` | Standard sequential memory baseline. |
| **CNN-only** | 1D Convolutional | 1.11 | 1.43 | 2.04 | `0.0010` | Local feature extraction without temporal dilation gates. |
| **CSRNet (adapted)** | Density Estimator | 1.23 | 1.54 | 2.37 | `0.0014` | Adapted crowd-density regression model. |
| **Linear Regression**| ML Baseline | 2.15 | 2.94 | 8.64 | `0.0001` | Non-deep-learning statistical baseline. |
| **Naive Baseline** | Heuristic Control | 10.24 | 11.98 | 143.52 | `0.0000` | Historical mean estimation baseline. |

---

### 1. Prerequisites
Ensure you have Python 3.10 or 3.11 installed. A GPU environment (CUDA supported) is recommended for training.

### 2. Environment Setup
```bash
# Clone the project repository
git clone https://github.com/Nehaa006/queue-prediction-project.git
cd queue-prediction-project

# Create a virtual environment
python -m venv env

# Activate environment (Windows PowerShell)
.\env\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
# source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Setup (MOT17)
This project utilizes the MOT17 (Multiple Object Tracking Benchmark) dataset.

Download the MOT17 dataset from Kaggle or the MOT Challenge Official Site.

Place and extract the dataset into the ./data/mot-17/ directory containing the train/ and test/ sequence directories.

## 🏃 Execution & Output Generation
To run the complete data extraction, feature engineering, model training, baseline evaluation, and visualization pipeline:

```bash
python main_pipeline.py
```

Execution Steps & Automated Deliverables:

* Data Parsing: Parses gt.txt ground-truth tracking files from ./data/mot-17/train.

* Feature Extraction: Computes queue density, velocity vectors, entry/exit frame indices, and individual dwell times.

* Model Benchmarking: Trains 1D-TCN alongside baseline models (LSTM, CNN, CSRNet, Linear Regression, Naive) across sequence inputs.

* Figure & CSV Generation: Automatically saves all publication-grade outputs into the ./paper_outputs/ directory:

* dataset_description.csv: Summary dataset statistics.

* table3_prediction.csv: Complete model comparative metrics.

* fig1_wait_time_density.png: Probability density distribution of wait times.

* fig2_queue_size_distribution.png: Patient and crowd queue volume distribution.

* fig3_training_loss.png: Neural network convergence curves.

* fig4_detection_performance.png: Object detection metric breakdown.

* fig5_tracking_performance.png: Multi-object tracking evaluation metrics.

* fig6_prediction_comparison.png: Multi-model MAE and RMSE performance comparison chart.

* fig7_predicted_vs_actual.png: Scatter plot correlation of predicted vs actual wait times.
