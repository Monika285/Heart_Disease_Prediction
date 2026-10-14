# Heart Disease Prediction

A beginner-friendly machine-learning classification project for predicting heart-disease likelihood from structured patient data.

## Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

The training script compares Accuracy, Precision, Recall, F1-score and ROC-AUC and saves the model with the best F1-score.

## Setup

1. Put your dataset at `data/heart.csv`.
2. The CSV must have a target column named `target` or `num`.
3. Open a terminal in this project folder.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Important
This is an educational machine-learning project. Its output is a model prediction, not a medical diagnosis.
