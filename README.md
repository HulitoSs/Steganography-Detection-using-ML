# 🧠 ML Stenography Detector

A machine learning pipeline for **binary classification** of statistical signal features using multiple classifiers and an ensemble voting model. The dataset contains ~70,000 samples of time-domain signal statistics extracted from EEG or similar time-series data.

---

## 📁 Project Structure

```
├── classifiers.ipynb          # Main notebook: EDA, training, evaluation
├── features_train_70000.csv   # Training set (55,999 samples)
├── features_test_70000.csv    # Test set
└── sweetviz_report.html       # Auto-generated EDA report
```

## 📊 Dataset

- **Samples:** ~70,000 (train + test), balanced binary classes (0 / 1)
- **Features (8):** `Kurtosis`, `Skewness`, `Std`, `Range`, `Median`, `Geometric_Mean`, `Mobility`, `Complexity`
- **Target:** `Tag` — binary label (0 or 1)
- All features are statistical descriptors of raw signal windows (time-domain)

## ⚙️ Pipeline

1. **EDA** — null checks, duplicate detection, correlation heatmap, Sweetviz HTML report
2. **Preprocessing** — MinMaxScaler normalization (0–1)
3. **Model Training & Evaluation** — accuracy + precision per model
4. **Ensemble** — Soft Voting Classifier combining best models
5. **Evaluation** — Confusion matrix visualization

## 🤖 Models Compared

| Model | Notes |
|-------|-------|
| Support Vector Classifier (SVC) | Sigmoid kernel, γ=1.0 |
| K-Nearest Neighbors (KNN) | Default params |
| Logistic Regression | L1 penalty, liblinear solver |
| Random Forest | 50 estimators, random_state=2 |
| **Voting Classifier** ⭐ | Soft vote: RF + KNN + LR |

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas matplotlib seaborn scikit-learn sweetviz
```

### Run

```bash
jupyter notebook "classifiers (1).ipynb"
```

Make sure `features_train_70000.csv` and `features_test_70000.csv` are in the same directory as the notebook.

## 📈 Results

Models are evaluated on **accuracy** and **precision**. Results are visualized as a grouped bar chart. The final **Soft Voting Classifier** (RF + KNN + LR) combines the strengths of individual models for improved generalization. A confusion matrix is generated for the ensemble's predictions.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `scikit-learn` | ML models, preprocessing, metrics |
| `pandas` | Data loading and manipulation |
| `seaborn` / `matplotlib` | Visualizations |
| `sweetviz` | Automated EDA report |

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
