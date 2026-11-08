from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST_FILE = ROOT / "test_data.csv"
TARGET_COLUMN = "is_spam"
CLASS_NAMES = ["Not spam", "Spam"]
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
DATASET_URL = "https://archive.ics.uci.edu/dataset/94/spambase"

st.set_page_config(page_title="Spam Email Classifier", layout="wide")


@st.cache_resource
def load_pipeline(file_name):
    # Load a saved pipeline once and reuse it across reruns
    return joblib.load(MODEL_DIR / file_name)


def required_feature_names():
    # Feature columns every saved pipeline was trained on
    first_file = next(iter(MODEL_FILES.values()))
    return list(load_pipeline(first_file).feature_names_in_)


def compute_metrics(y_true, y_pred, y_score):
    # Six evaluation metrics with spam as the positive class
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def find_problems(frame):
    # Collect reasons the data cannot be scored; an empty list means it is usable
    problems = []
    if TARGET_COLUMN not in frame.columns:
        problems.append(f"The target column '{TARGET_COLUMN}' is missing.")
    needed = required_feature_names()
    absent = [c for c in needed if c not in frame.columns]
    if absent:
        problems.append(f"{len(absent)} required feature column(s) are missing, for example: {absent[:5]}")
    if problems:
        return problems
    numeric_check = frame[needed].apply(pd.to_numeric, errors="coerce")
    if numeric_check.isna().any().any():
        problems.append("Feature columns must be numeric with no missing values.")
    target_values = set(frame[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        problems.append(f"The target column must contain only 0 and 1, but found: {sorted(target_values)[:5]}")
    elif len(target_values) < 2:
        problems.append("The target column must contain both classes (0 and 1) so AUC can be computed.")
    return problems


def get_test_frame(uploaded):
    # Use the uploaded CSV if present, otherwise fall back to the bundled test set
    if uploaded is not None:
        return pd.read_csv(uploaded), f"Uploaded file: {uploaded.name}"
    return pd.read_csv(DEFAULT_TEST_FILE), "Bundled test_data.csv (upload a file in the sidebar to replace it)"


def split_inputs(frame, pipeline):
    features = frame[list(pipeline.feature_names_in_)].apply(pd.to_numeric)
    labels = frame[TARGET_COLUMN].astype(int)
    return features, labels


@st.cache_data(show_spinner=False)
def score_all_models(frame):
    # Evaluate every saved pipeline on the same data for the comparison page
    rows = []
    for model_name, file_name in MODEL_FILES.items():
        pipe = load_pipeline(file_name)
        features, labels = split_inputs(frame, pipe)
        row = {"ML Model Name": model_name}
        row.update(compute_metrics(labels, pipe.predict(features), pipe.predict_proba(features)[:, 1]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("ML Model Name")


def show_evaluate_page(frame):
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    pipe = load_pipeline(MODEL_FILES[model_name])
    features, labels = split_inputs(frame, pipe)
    predicted = pipe.predict(features)
    spam_score = pipe.predict_proba(features)[:, 1]

    st.subheader(f"{model_name}: evaluation summary")
    values = compute_metrics(labels, predicted, spam_score)
    for col, metric in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
        col.metric(metric, f"{values[metric]:.4f}")

    tab_cm, tab_report, tab_preview = st.tabs(["Confusion matrix", "Classification report", "Test data preview"])

    with tab_cm:
        cm = confusion_matrix(labels, predicted, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(model_name)
        st.pyplot(fig)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel()
        for col, (label, count) in zip(
            st.columns(4),
            [("True negatives", tn), ("False positives", fp), ("False negatives", fn), ("True positives", tp)],
        ):
            col.metric(label, f"{count:,}")
        st.caption("A false positive is a legitimate email flagged as spam; a false negative is spam that slips through.")

    with tab_report:
        report = classification_report(
            labels, predicted, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4))

    with tab_preview:
        st.dataframe(frame.head(25))


def show_compare_page(frame):
    st.subheader("Compare all five models on the same data")
    table = score_all_models(frame)
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, props="font-weight: 700; text-decoration: underline;"))
    st.caption("The best value in each column is bold and underlined.")

    left, right = st.columns([1, 1])
    with left:
        chosen_metric = st.selectbox("Metric to plot", METRIC_NAMES)
        st.bar_chart(table[chosen_metric])
    with right:
        st.markdown("**Best model per metric**")
        best = pd.DataFrame({"Best model": table.idxmax(), "Score": table.max().round(4)})
        st.dataframe(best)


def show_about_page(frame):
    st.subheader("About this project")
    for col, (label, value) in zip(
        st.columns(3),
        [("Algorithms", "5"), ("Evaluation metrics", "6"), ("Retraining in app", "No")],
    ):
        col.metric(label, value)

    st.markdown(
        f"""
**Objective.** Classify an email as spam (1) or not spam (0) from the frequency of certain words and characters.

**Dataset.** [UCI Spambase]({DATASET_URL}): 4,601 emails and 57 numeric features. After removing 391 duplicate rows, 4,210 emails remain (about 40% spam).

**Method.**
- Stratified 80/20 train-test split with a fixed random seed; the held-out 20% is the bundled `test_data.csv`.
- Logistic Regression, KNN and SVM include standard scaling inside their saved pipelines; Decision Tree and Naive Bayes do not need it.
- This app loads the saved pipelines and only evaluates them. Nothing is retrained on uploaded data.

**Reading the metrics.** Spam is the positive class. Precision answers "of the emails flagged as spam, how many really were"; recall answers "of the real spam, how much was caught". Flagging a legitimate email is usually costlier than missing one spam message, so precision matters alongside accuracy, F1 and MCC.
"""
    )
    st.markdown("**Current test data**")
    st.write(f"{frame.shape[0]:,} rows and {frame.shape[1]} columns")
    st.dataframe(frame.head(10))


st.title("Spam Email Classifier")

st.sidebar.header("Test dataset")
uploaded_file = st.sidebar.file_uploader("Upload a labelled test CSV", type=["csv"])
st.sidebar.caption(f"Expected: the 57 feature columns plus the target column '{TARGET_COLUMN}' (1 = spam, 0 = not spam).")
page = st.sidebar.radio("Navigation", ["Evaluate model", "Compare models", "About"])

try:
    test_frame, source_note = get_test_frame(uploaded_file)
except Exception as err:
    st.error(f"Could not read the CSV file: {err}")
    st.stop()

issues = find_problems(test_frame)
if issues:
    for issue in issues:
        st.error(issue)
    st.stop()

st.caption(f"Data source: {source_note} | {len(test_frame):,} labelled rows")

if page == "Evaluate model":
    show_evaluate_page(test_frame)
elif page == "Compare models":
    show_compare_page(test_frame)
else:
    show_about_page(test_frame)