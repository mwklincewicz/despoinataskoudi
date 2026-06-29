import warnings
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings(
    action="ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.metrics import (
    make_scorer,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier


THRESHOLD = 0.45
RANDOM_STATE = 50
DATA_PATH = "enriched_employee_dataset.csv"


# =========================================================
# Feature engineering and data preparation
# =========================================================

def build_engineered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final engineered dataset for early burnout detection.
    Removes Mental Fatigue Score and constructs theory-driven features.
    """
    df_fe = df.copy()

    df_fe = df_fe.drop(columns=["Mental Fatigue Score"])

    df_fe["Work_Pressure"] = (
        df_fe["Work Hours per Week"] +
        df_fe["Deadline Pressure Score"]
    )

    df_fe["Organizational_Support"] = (
        df_fe["Manager Support Score"] +
        df_fe["Recognition Frequency"]
    )

    df_fe["Pressure_to_Support"] = (
        df_fe["Work_Pressure"] /
        (df_fe["Organizational_Support"] + 1)
    )

    return df_fe


def prepare_model_data(df: pd.DataFrame):
    """
    Prepare final X and y for binary burnout classification.
    """
    df_fe = build_engineered_dataset(df)

    df_model = df_fe.drop(columns=["Employee ID", "Date of Joining"]).copy()
    df_model = df_model.dropna().copy()

    df_model["Burn_Rate_Binary"] = (
        df_model["Burn Rate"] >= THRESHOLD
    ).astype(int)

    X = df_model.drop(columns=["Burn Rate", "Burn_Rate_Binary"]).copy()
    y = df_model["Burn_Rate_Binary"].copy()

    X = X.drop(columns=[
        "Work Hours per Week",
        "Deadline Pressure Score",
        "Manager Support Score",
        "Recognition Frequency",
    ])

    return df_fe, df_model, X, y


def build_preprocessors(X: pd.DataFrame):
    """
    Build separate preprocessors for linear and tree-based models.
    Linear models use scaling; tree models do not need scaling.
    """
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "string", "bool"]).columns

    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor_non_tree = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    return preprocessor_non_tree, preprocessor_tree, numerical_cols, categorical_cols


def get_models():
    """
    Return candidate models and hyperparameter grids.
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5],
            }
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            }
        }
    }


def get_scoring():
    return {
        "f1": "f1",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "mcc": make_scorer(matthews_corrcoef),
    }


# =========================================================
# Utilities
# =========================================================

def print_metric_summary(model_name: str, cv_results: dict):
    print(f"\n{model_name}")
    print(f"  F1-Score:  {np.mean(cv_results['test_f1']):.3f} ± {np.std(cv_results['test_f1'], ddof=1):.3f}")
    print(f"  ROC-AUC:   {np.mean(cv_results['test_roc_auc']):.3f} ± {np.std(cv_results['test_roc_auc'], ddof=1):.3f}")
    print(f"  Precision: {np.mean(cv_results['test_precision']):.3f} ± {np.std(cv_results['test_precision'], ddof=1):.3f}")
    print(f"  Recall:    {np.mean(cv_results['test_recall']):.3f} ± {np.std(cv_results['test_recall'], ddof=1):.3f}")
    print(f"  MCC:       {np.mean(cv_results['test_mcc']):.3f} ± {np.std(cv_results['test_mcc'], ddof=1):.3f}")

    if "train_f1" in cv_results:
        print(f"  Train F1:       {np.mean(cv_results['train_f1']):.3f} ± {np.std(cv_results['train_f1'], ddof=1):.3f}")
        print(f"  Train AUC:      {np.mean(cv_results['train_roc_auc']):.3f} ± {np.std(cv_results['train_roc_auc'], ddof=1):.3f}")
        print(f"  Train Precision:{np.mean(cv_results['train_precision']):.3f} ± {np.std(cv_results['train_precision'], ddof=1):.3f}")
        print(f"  Train Recall:   {np.mean(cv_results['train_recall']):.3f} ± {np.std(cv_results['train_recall'], ddof=1):.3f}")
        print(f"  Train MCC:      {np.mean(cv_results['train_mcc']):.3f} ± {np.std(cv_results['train_mcc'], ddof=1):.3f}")


def make_pipeline_for_model(name: str, spec: dict, preprocessor_non_tree, preprocessor_tree):
    if name in ["Random Forest", "XGBoost"]:
        return Pipeline(steps=[
            ("preprocessor", preprocessor_tree),
            ("model", spec["model"]),
        ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor_non_tree),
        ("model", spec["model"]),
    ])


def as_dense_matrix(X_transformed):
    """
    SHAP plots work more reliably with dense DataFrames.
    """
    if hasattr(X_transformed, "toarray"):
        return X_transformed.toarray()
    return X_transformed


# =========================================================
# EDA plots
# =========================================================

def run_eda(df: pd.DataFrame):
    print("\nThe shape of dataset is:", df.shape)

    print("\nThe names of the columns are:")
    print(df.columns)

    if df["Employee ID"].nunique() != len(df):
        print("Duplicate employee records detected.")
    else:
        print("Each employee appears only once. No duplicates.")

    print("\nMissing values in each column (full dataset):")
    print(df.isnull().sum())

    total_records = len(df)
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / total_records) * 100

    missing_summary = pd.DataFrame({
        "Missing Values": missing_count,
        "Missing Percentage (%)": missing_percentage,
    })
    missing_summary = missing_summary[missing_summary["Missing Values"] > 0]

    print("\nMissing summary (only columns with missing values):")
    print(missing_summary.to_string())

    x = df.drop(columns=["Burn Rate"])

    print(
        "The number of employees that have at least 1 missing value is",
        x.isnull().any(axis=1).sum()
    )
    print(
        "The percentage of rows contain missing values is",
        round((x.isnull().any(axis=1).sum() / len(x)) * 100, 2),
        "%"
    )
    print(
        "The number of employees that have more than one missing value is",
        (x.isnull().sum(axis=1) > 1).sum()
    )

    cols_with_missing = x.columns[x.isnull().sum() > 0]
    if len(cols_with_missing) > 1:
        print("\nCorrelation matrix of missing values:")
        print(x[cols_with_missing].isnull().corr())
    else:
        print("\nNot enough columns with missing values to compute correlation.")

    df.info()
    print("\nData types of each column of the dataset are:")
    print(df.dtypes)

    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = x.select_dtypes(include=["object", "string", "bool"]).columns

    print("\nContinuous (numerical) features:", numerical_cols.values)
    print("\nCategorical features:", categorical_cols.values)

    print("\nSummary statistics for numerical columns:\n")
    print(x[numerical_cols].describe().to_string())

    continuous_cols = [
        "Sleep Hours",
        "Work Hours per Week",
        "Years in Company",
        "Team Size",
    ]

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax = ax.flatten()

    for i, col in enumerate(continuous_cols):
        s = x[col].dropna()
        mu, sigma = s.mean(), s.std()

        ax[i].hist(
            s,
            bins=25,
            density=True,
            alpha=0.6,
            color="steelblue",
            edgecolor="white",
            linewidth=0.8,
        )

        x_vals = np.linspace(s.min(), s.max(), 200)
        y_vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((x_vals - mu) ** 2) / (2 * sigma ** 2)
        )

        ax[i].plot(x_vals, y_vals, linewidth=2, color="darkred")
        ax[i].set_title(col, fontsize=10)
        ax[i].set_xlabel(col, fontsize=9)
        ax[i].set_ylabel("Density", fontsize=9)

    plt.tight_layout(pad=0.8)
    plt.savefig("Feature_hists_norm.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    ordinal_cols = [
        "Designation",
        "Resource Allocation",
        "Work-Life Balance Score",
        "Manager Support Score",
        "Deadline Pressure Score",
        "Recognition Frequency",
    ]

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax = ax.flatten()

    for i, col in enumerate(ordinal_cols):
        counts = (
            x[col]
            .dropna()
            .astype(float)
            .round()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        ax[i].bar(counts.index.astype(str), counts.values, alpha=0.7)
        ax[i].set_title(col, fontsize=11)
        ax[i].set_xlabel("Level", fontsize=10)
        ax[i].set_ylabel("Count", fontsize=10)

    plt.tight_layout(pad=1.0)
    plt.savefig("Ordinal_barplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    fig = px.pie(
        df,
        names="WFH Setup Available",
        title="WFH Setup Availability"
    )
    fig.write_html("WFH_Setup_Availability.html")

    y = df["Burn Rate"]
    y_clean = y.dropna()

    print("\nThe top of the Target:")
    print(y_clean.head())
    print()
    print(y_clean.describe())

    plt.figure(figsize=(6, 4))
    plt.hist(
        y_clean,
        bins=25,
        density=True,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.8,
    )

    mu = y_clean.mean()
    sigma = y_clean.std()
    x_vals = np.linspace(y_clean.min(), y_clean.max(), 300)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((x_vals - mu) ** 2) / (2 * sigma ** 2)
    )
    plt.plot(x_vals, y_norm, linewidth=2, color="darkred")
    plt.title("Distribution of Burn Rate", fontweight="bold")
    plt.xlabel("Burn Rate")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("Burn_Rate_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    y_binary = (y_clean >= THRESHOLD).astype(int)

    print(f"\nFixed threshold used: {THRESHOLD:.2f}")
    print(y_binary.head(10))

    print(
        "After the threshold application, the number of Low to Moderate burnout (0) "
        "and High burnout (1) is:\n",
        y_binary.value_counts()
    )
    print("Missing values in y after thresholding:", y.isna().sum())
    print("Portion of how many are 0 and 1:\n", y_binary.value_counts(normalize=True) * 100)

    class_counts = [np.sum(y_binary == 0), np.sum(y_binary == 1)]
    class_labels = ["Low Risk", "High Risk"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(class_labels, class_counts, color=["purple", "pink"], width=0.75)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, axis="y")
    plt.title("Distribution of Target Classes", fontsize=16, fontweight="bold")
    plt.xlabel("Classes", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            f"{int(yval)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("Class_dist_barplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    boxplot_cols = [
        "Sleep Hours",
        "Work Hours per Week",
        "Resource Allocation",
        "Work-Life Balance Score",
        "Manager Support Score",
        "Deadline Pressure Score",
        "Recognition Frequency",
    ]

    plt.figure(figsize=(15, 7))
    plt.boxplot(
        [x[col].dropna() for col in boxplot_cols],
        tick_labels=boxplot_cols,
        notch=True,
        patch_artist=True,
        boxprops=dict(linestyle="-", linewidth=2, color="navy"),
        medianprops=dict(linestyle="-", linewidth=2, color="firebrick"),
        whiskerprops=dict(linestyle="--", linewidth=2, color="black"),
        capprops=dict(linestyle="-", linewidth=2, color="grey"),
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, axis="y")
    plt.title("Distribution of Selected Features (Boxplots)", fontsize=18, fontweight="bold")
    plt.xlabel("Features", fontsize=16)
    plt.ylabel("Values", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("Employee_Feature_Boxplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    correlation_matrix = x[numerical_cols].corr()

    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmax=1, vmin=-1)
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Feature Correlation Matrix")

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(
                j,
                i,
                np.around(correlation_matrix.iloc[i, j], decimals=2),
                ha="center",
                va="center",
                color="w",
            )

    plt.tight_layout()
    plt.savefig("Correlation_heatmap_before.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    corr_pairs = upper_triangle.stack().reset_index()
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]

    print("\nStrong positive correlations (> 0.7):")
    print(corr_pairs[corr_pairs["Correlation"] > 0.7], "\n")

    print("Strong negative correlations (< -0.7):")
    print(corr_pairs[corr_pairs["Correlation"] < -0.7])


# =========================================================
# Model comparison plot
# =========================================================

def plot_model_comparison(results: dict):
    models_list = ["Logistic Regression", "Random Forest", "XGBoost"]
    metrics = {
        "F1-score": "test_f1",
        "ROC-AUC": "test_roc_auc",
        "Precision": "test_precision",
        "Recall": "test_recall",
        "MCC": "test_mcc",
    }

    means = {
        metric: [np.mean(results[m][key]) for m in models_list]
        for metric, key in metrics.items()
    }
    stds = {
        metric: [np.std(results[m][key], ddof=1) for m in models_list]
        for metric, key in metrics.items()
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()

    for ax, (metric_name, _) in zip(axes, metrics.items()):
        bars = ax.bar(
            models_list,
            means[metric_name],
            yerr=stds[metric_name],
            capsize=6,
            alpha=0.85,
        )

        ax.set_title(metric_name, fontsize=14, fontweight="bold")
        ax.set_ylim(0.0, max(means[metric_name]) + 0.1)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", labelrotation=25)
        ax.tick_params(axis="both", labelsize=11)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for bar, mean_val in zip(bars, means[metric_name]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_val - 0.05,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Comparison of Classifier Performance", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Classifier_comparison_multimetric.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Confusion matrix analysis
# =========================================================

def plot_nested_cv_confusion_matrices(
    X,
    y,
    best_model_name,
    models,
    preprocessor_non_tree,
    preprocessor_tree,
    outer_cv,
    inner_cv
):
    """
    Create confusion matrices using out-of-fold predictions from nested CV.
    This avoids evaluating the final model on the same data used for training.
    """

    spec = models[best_model_name]

    y_true_all = []
    y_pred_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = make_pipeline_for_model(
            best_model_name,
            spec,
            preprocessor_non_tree,
            preprocessor_tree
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            cv=inner_cv,
            scoring="recall",
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Normalize by true labels (recall view)
    ConfusionMatrixDisplay.from_predictions(
        y_true_all,
        y_pred_all,
        normalize="true",
        cmap="Blues",
        ax=axes[0],
        display_labels=["Low Risk", "High Risk"]
    )
    axes[0].set_title("Normalized by True (Recall)")

    # Normalize by predicted labels (precision view)
    ConfusionMatrixDisplay.from_predictions(
        y_true_all,
        y_pred_all,
        normalize="pred",
        cmap="Blues",
        ax=axes[1],
        display_labels=["Low Risk", "High Risk"]
    )
    axes[1].set_title("Normalized by Predicted (Precision)")

    # Normalize by all samples (overall proportions)
    ConfusionMatrixDisplay.from_predictions(
        y_true_all,
        y_pred_all,
        normalize="all",
        cmap="Blues",
        ax=axes[2],
        display_labels=["Low Risk", "High Risk"]
    )
    axes[2].set_title("Normalized by All (Proportion)")

    plt.tight_layout()
    plt.savefig("Confusion_Matrices_Nested_CV.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    return y_true_all, y_pred_all


def build_nested_cv_error_results(
    X,
    y,
    best_model_name,
    models,
    preprocessor_non_tree,
    preprocessor_tree,
    outer_cv,
    inner_cv
):
    """
    Build an out-of-fold results table for error analysis.
    The table keeps the original feature scale and adds:
    y_true, y_pred, y_proba, and error_type.
    """

    spec = models[best_model_name]
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = make_pipeline_for_model(
            best_model_name,
            spec,
            preprocessor_non_tree,
            preprocessor_tree
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            cv=inner_cv,
            scoring="recall",
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)
        y_proba = grid_search.predict_proba(X_test)[:, 1]

        results_clf = X_test.copy()
        results_clf["outer_fold"] = fold_idx
        results_clf["y_true"] = y_test.values
        results_clf["y_pred"] = y_pred
        results_clf["y_proba"] = y_proba

        results_clf["error_type"] = "unknown"

        results_clf.loc[
            (results_clf["y_true"] == 1) & (results_clf["y_pred"] == 1),
            "error_type"
        ] = "true_positive"

        results_clf.loc[
            (results_clf["y_true"] == 0) & (results_clf["y_pred"] == 0),
            "error_type"
        ] = "true_negative"

        results_clf.loc[
            (results_clf["y_true"] == 0) & (results_clf["y_pred"] == 1),
            "error_type"
        ] = "false_positive"

        results_clf.loc[
            (results_clf["y_true"] == 1) & (results_clf["y_pred"] == 0),
            "error_type"
        ] = "false_negative"

        fold_results.append(results_clf)

    results_clf_all = pd.concat(fold_results, axis=0)
    results_clf_all.to_csv("Nested_CV_Error_Analysis_Results.csv", index=True)

    print("\nError type counts from nested CV out-of-fold predictions:")
    print(results_clf_all["error_type"].value_counts())

    print("\nMean predicted probability by error type:")
    print(results_clf_all.groupby("error_type")["y_proba"].mean().sort_values(ascending=False))

    return results_clf_all


def plot_prediction_confidence_by_error_type(results_clf):
    """
    Plot predicted-probability distributions by confusion matrix outcome type.
    """

    plt.figure(figsize=(9, 5))

    bins = 20

    colors = {
        "true_positive": "#4CAF50",
        "true_negative": "#1f77b4",
        "false_positive": "#ff7f0e",
        "false_negative": "#d62728"
    }

    for error_type, color in colors.items():
        subset = results_clf.loc[
            results_clf["error_type"] == error_type,
            "y_proba"
        ]

        if len(subset) > 0:
            plt.hist(
                subset,
                bins=bins,
                alpha=0.6,
                label=error_type.replace("_", " ").title(),
                color=color
            )

    plt.xlabel("Predicted probability (class 1)")
    plt.ylabel("Frequency")
    plt.title("Prediction Confidence by Outcome Type")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("Prediction_Confidence_by_Error_Type.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_level_error_analysis(
    results_clf,
    feature="Work_Pressure",
    n_bins=10
):
    """
    Analyze false positive and false negative rates across quantiles
    of a selected original-scale feature.
    """

    if feature not in results_clf.columns:
        print(f"\nFeature-level error analysis skipped: '{feature}' not found in results table.")
        return None

    results_feature = results_clf.copy()

    results_feature["error_type_feature"] = "correct"
    results_feature.loc[
        (results_feature["y_true"] == 0) & (results_feature["y_pred"] == 1),
        "error_type_feature"
    ] = "false_positive"
    results_feature.loc[
        (results_feature["y_true"] == 1) & (results_feature["y_pred"] == 0),
        "error_type_feature"
    ] = "false_negative"

    group_col = f"{feature}_group"

    results_feature[group_col] = pd.qcut(
        results_feature[feature],
        q=n_bins,
        duplicates="drop"
    )

    summary = results_feature.groupby(group_col, observed=False).apply(
        lambda df: pd.Series({
            "n_samples": len(df),
            "fp_rate": ((df["y_true"] == 0) & (df["y_pred"] == 1)).mean(),
            "fn_rate": ((df["y_true"] == 1) & (df["y_pred"] == 0)).mean(),
            "error_rate": (df["y_true"] != df["y_pred"]).mean()
        })
    ).reset_index()

    summary["group_label"] = summary[group_col].astype(str)

    safe_feature_name = feature.replace(" ", "_").replace("/", "_")
    summary.to_csv(
        f"Feature_Level_Error_Analysis_{safe_feature_name}.csv",
        index=False
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax2.bar(
        summary["group_label"],
        summary["n_samples"],
        alpha=0.25,
        label="Sample count"
    )

    ax1.plot(
        summary["group_label"],
        summary["fp_rate"],
        marker="o",
        linewidth=2,
        label="False Positive Rate"
    )
    ax1.plot(
        summary["group_label"],
        summary["fn_rate"],
        marker="o",
        linewidth=2,
        label="False Negative Rate"
    )

    ax1.set_xlabel(f"{feature} quantile")
    ax1.set_ylabel("Rate")
    ax2.set_ylabel("Number of samples")
    ax1.set_title(
        f"False Positive and False Negative Rates Across {feature} Quantiles"
    )

    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1.set_xticklabels(
        ax1.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        f"Feature_Level_Error_Analysis_{safe_feature_name}.pdf",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"\nFeature-level error analysis for {feature}:")
    print(summary.to_string(index=False))

    return summary


def plot_calibration_curve_from_error_results(
    results_clf,
    n_bins=10
):
    """
    Plot a calibration curve using nested-CV out-of-fold predicted probabilities.
    """

    y_true = results_clf["y_true"].values
    y_proba = results_clf["y_proba"].values

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins
    )

    calibration_df = pd.DataFrame({
        "mean_predicted_probability": prob_pred,
        "observed_frequency": prob_true
    })
    calibration_df.to_csv("Calibration_Curve_Data.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("Calibration_Curve.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nCalibration curve data:")
    print(calibration_df.to_string(index=False))

    return calibration_df


# =========================================================
# SHAP analysis
# =========================================================

def fit_final_model(best_model_name, best_params, X, y, models, preprocessor_non_tree, preprocessor_tree):
    """
    Retrain the final model on the full modeling dataset using the selected hyperparameters.
    This follows the standard workflow: CV for evaluation, then final model for interpretation.
    """
    spec = models[best_model_name]
    final_pipe = make_pipeline_for_model(
        best_model_name,
        spec,
        preprocessor_non_tree,
        preprocessor_tree,
    )

    clean_params = {
        key: value
        for key, value in best_params.items()
        if key in spec["params"]
    }

    final_pipe.set_params(**clean_params)
    final_pipe.fit(X, y)
    return final_pipe


def run_shap_analysis(final_pipe, X, best_model_name: str, max_display: int = 15):
    """
    SHAP-based replacement for model-specific feature importance.
    Produces global and local explanations for the final predictive model.
    """
    print(f"\nRunning SHAP analysis for final model: {best_model_name}")

    preprocessor = final_pipe.named_steps["preprocessor"]
    model = final_pipe.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    X_transformed = as_dense_matrix(X_transformed)
    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )

    # To keep runtime manageable, explain a representative sample if the dataset is large.
    if len(X_transformed_df) > 3000:
        X_shap = X_transformed_df.sample(
            n=3000,
            random_state=RANDOM_STATE,
        )
    else:
        X_shap = X_transformed_df.copy()

    if best_model_name in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer(X_shap)
    elif best_model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_shap)
        shap_values_raw = explainer(X_shap)
    else:
        explainer = shap.Explainer(model, X_shap)
        shap_values_raw = explainer(X_shap)

    # For binary classifiers, some SHAP versions return shape (n, p, 2).
    if len(shap_values_raw.values.shape) == 3:
        shap_values = shap.Explanation(
            values=shap_values_raw.values[:, :, 1],
            base_values=shap_values_raw.base_values[:, 1],
            data=shap_values_raw.data,
            feature_names=shap_values_raw.feature_names,
        )
    else:
        shap_values = shap_values_raw

    # -------------------------
    # Global SHAP bar plot
    # -------------------------
    plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_global_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # SHAP beeswarm plot
    # -------------------------
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_beeswarm.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Local waterfall explanation
    # -------------------------
    sample_id = 0
    plt.figure()
    shap.plots.waterfall(shap_values[sample_id], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_waterfall_sample0.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Export SHAP importance table
    # -------------------------
    shap_importance = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        "mean_shap": shap_values.values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv("SHAP_global_importance.csv", index=False)

    print("\nTop SHAP features:")
    print(shap_importance.head(max_display).to_string(index=False))

    # -------------------------
    # Theory-driven SHAP interaction-style scatter
    # -------------------------
    if (
        "num__Work_Pressure" in shap_values.feature_names and
        "num__Organizational_Support" in shap_values.feature_names
    ):
        plt.figure()
        shap.plots.scatter(
            shap_values[:, "num__Work_Pressure"],
            color=shap_values[:, "num__Organizational_Support"],
            show=False,
        )
        plt.tight_layout()
        plt.savefig("SHAP_WorkPressure_by_OrganizationalSupport.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    return shap_values, shap_importance


def run_shap_stability_across_outer_folds(results, model_name: str, X: pd.DataFrame, top_k: int = 15):
    """
    Optional stability analysis: computes SHAP global importance for each outer-fold model.
    This is closer to an 'explanation stability across folds' design.
    """
    print(f"\nRunning SHAP stability analysis across outer folds for: {model_name}")

    fold_tables = []

    for fold_idx, grid in enumerate(results[model_name]["estimator"], start=1):
        pipe = grid.best_estimator_
        preprocessor = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["model"]

        X_transformed = as_dense_matrix(preprocessor.transform(X))
        feature_names = preprocessor.get_feature_names_out()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        if len(X_transformed_df) > 1500:
            X_shap = X_transformed_df.sample(n=1500, random_state=RANDOM_STATE + fold_idx)
        else:
            X_shap = X_transformed_df.copy()

        if model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer(X_shap)
        elif model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_shap)
            shap_values_raw = explainer(X_shap)
        else:
            explainer = shap.Explainer(model, X_shap)
            shap_values_raw = explainer(X_shap)

        if len(shap_values_raw.values.shape) == 3:
            values = shap_values_raw.values[:, :, 1]
        else:
            values = shap_values_raw.values

        fold_imp = pd.DataFrame({
            "feature": feature_names,
            f"fold_{fold_idx}": np.abs(values).mean(axis=0),
        })
        fold_tables.append(fold_imp.set_index("feature"))

    stability = pd.concat(fold_tables, axis=1)
    stability_summary = pd.DataFrame({
        "mean_abs_shap": stability.mean(axis=1),
        "std_abs_shap": stability.std(axis=1, ddof=1),
    }).sort_values("mean_abs_shap", ascending=False)

    stability_summary.to_csv("SHAP_stability_across_outer_folds.csv")

    plot_df = stability_summary.head(top_k).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(
        plot_df.index,
        plot_df["mean_abs_shap"],
        xerr=plot_df["std_abs_shap"],
        capsize=5,
        alpha=0.85,
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"SHAP Stability Across Outer CV Folds – {model_name}")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("SHAP_stability_across_outer_folds.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nTop stable SHAP features across folds:")
    print(stability_summary.head(top_k).to_string())

    return stability_summary


# =========================================================
# Main workflow
# =========================================================

def main():
    df = pd.read_csv(DATA_PATH, sep=";")

    run_eda(df)

    df_fe, df_model, X, y = prepare_model_data(df)

    engineered_corr = df_fe[
        ["Work_Pressure", "Organizational_Support", "Pressure_to_Support", "Burn Rate"]
    ].corr()

    print("\nCorrelation matrix of engineered features and target:")
    print(engineered_corr)

    preprocessor_non_tree, preprocessor_tree, numerical_cols, categorical_cols = build_preprocessors(X)
    models = get_models()
    scoring = get_scoring()

    print("\nFinal modeling features:")
    print(X.columns.tolist())

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}

    for name, spec in models.items():
        print(f"\nRunning model: {name}")

        pipeline = make_pipeline_for_model(
            name,
            spec,
            preprocessor_non_tree,
            preprocessor_tree,
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            cv=inner_cv,
            scoring="recall",
            n_jobs=1,
        )

        cv_results = cross_validate(
            estimator=grid_search,
            X=X,
            y=y,
            cv=outer_cv,
            scoring=scoring,
            return_estimator=True,
            return_train_score=True,
            n_jobs=1,
        )

        results[name] = cv_results
        print_metric_summary(name, cv_results)

    # Select best model by recall, matching the current project logic.
    best_model = max(results, key=lambda m: np.mean(results[m]["test_recall"]))
    best_params_per_fold = [est.best_params_ for est in results[best_model]["estimator"]]

    print(f"\n{best_model} – best hyperparameters per outer fold:")
    for i, params in enumerate(best_params_per_fold, 1):
        print(f"  Fold {i}: {params}")

    plot_model_comparison(results)

    y_true_cv, y_pred_cv = plot_nested_cv_confusion_matrices(
        X=X,
        y=y,
        best_model_name=best_model,
        models=models,
        preprocessor_non_tree=preprocessor_non_tree,
        preprocessor_tree=preprocessor_tree,
        outer_cv=outer_cv,
        inner_cv=inner_cv
    )

    results_clf = build_nested_cv_error_results(
        X=X,
        y=y,
        best_model_name=best_model,
        models=models,
        preprocessor_non_tree=preprocessor_non_tree,
        preprocessor_tree=preprocessor_tree,
        outer_cv=outer_cv,
        inner_cv=inner_cv
    )

    plot_prediction_confidence_by_error_type(results_clf)

    feature_error_summary = plot_feature_level_error_analysis(
        results_clf=results_clf,
        feature="Work_Pressure",
        n_bins=10
    )

    calibration_df = plot_calibration_curve_from_error_results(
        results_clf=results_clf,
        n_bins=10
    )

    # Use the most frequent best-parameter configuration across folds.
    best_params = pd.Series([str(p) for p in best_params_per_fold]).mode().iloc[0]
    best_params = eval(best_params)

    final_pipe = fit_final_model(
        best_model_name=best_model,
        best_params=best_params,
        X=X,
        y=y,
        models=models,
        preprocessor_non_tree=preprocessor_non_tree,
        preprocessor_tree=preprocessor_tree,
    )

    run_shap_analysis(
        final_pipe=final_pipe,
        X=X,
        best_model_name=best_model,
        max_display=15,
    )

    # Optional: uncomment if you want thesis-level explanation stability across folds.
    # run_shap_stability_across_outer_folds(results, best_model, X, top_k=15)


def repeated_nested_cv():
    df = pd.read_csv(DATA_PATH, sep=";")
    _, _, X, y = prepare_model_data(df)

    preprocessor_non_tree, preprocessor_tree, _, _ = build_preprocessors(X)
    models = get_models()
    scoring = get_scoring()

    outer_cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=RANDOM_STATE,
    )

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}

    for name, spec in models.items():
        print(f"\nRepeated nested CV – running model: {name}")

        base_pipe = make_pipeline_for_model(
            name,
            spec,
            preprocessor_non_tree,
            preprocessor_tree,
        )

        grid_search = GridSearchCV(
            estimator=base_pipe,
            param_grid=spec["params"],
            cv=inner_cv,
            scoring="recall",
            n_jobs=1,
        )

        cv_results = cross_validate(
            estimator=grid_search,
            X=X,
            y=y,
            cv=outer_cv,
            scoring=scoring,
            return_estimator=False,
            n_jobs=1,
        )

        results[name] = cv_results
        n_outer = len(cv_results["test_roc_auc"])

        print(f"\n{name}")
        print(f"  F1-Score:  {np.mean(cv_results['test_f1']):.3f} ± {np.std(cv_results['test_f1'], ddof=1):.3f} (n={n_outer})")
        print(f"  ROC-AUC:   {np.mean(cv_results['test_roc_auc']):.3f} ± {np.std(cv_results['test_roc_auc'], ddof=1):.3f} (n={n_outer})")
        print(f"  Precision: {np.mean(cv_results['test_precision']):.3f} ± {np.std(cv_results['test_precision'], ddof=1):.3f} (n={n_outer})")
        print(f"  Recall:    {np.mean(cv_results['test_recall']):.3f} ± {np.std(cv_results['test_recall'], ddof=1):.3f} (n={n_outer})")
        print(f"  MCC:       {np.mean(cv_results['test_mcc']):.3f} ± {np.std(cv_results['test_mcc'], ddof=1):.3f} (n={n_outer})")


if __name__ == "__main__":
    main()

    # Keep this commented while debugging SHAP, because it is computationally expensive.
    # repeated_nested_cv()
