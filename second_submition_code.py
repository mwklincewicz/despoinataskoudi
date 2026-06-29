import warnings
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings(
    action="ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

from collections import Counter

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
from xgboost import XGBClassifier


THRESHOLD = 0.45
RANDOM_STATE = 50
N_REPEATS = 10   # repeats for the repeated nested CV stability check


# Feature groups

CONTEXTUAL = [
    "Gender", "Company Type", "WFH Setup Available",
    "Designation", "Years in Company", "Team Size",
]

# Raw well-being / resource variables NOT used to build engineered features.
WELLBEING_RAW = [
    "Resource Allocation", "Sleep Hours", "Work-Life Balance Score",
]

# Raw building blocks of the engineered features (NOT used as predictors here;
# listed only so they can be included in the shared listwise-deletion mask).
DEMAND_RESOURCE_RAW = [
    "Work Hours per Week", "Deadline Pressure Score",
    "Manager Support Score", "Recognition Frequency",
]

ENGINEERED = [
    "Work_Pressure", "Organizational_Support", "Pressure_to_Support",
]

# ----------------------------------------------------------------------
#   A = conservative contextual-only baseline (NO Manager Support)
#   B = contextual + raw well-being + engineered  (the final/engineered model)
# ----------------------------------------------------------------------
CONFIGS = {
    "A_baseline":   CONTEXTUAL,
    "B_engineered": CONTEXTUAL + WELLBEING_RAW + ENGINEERED,
}

ENGINEERED_CONFIG = "B_engineered"   # config used for the split table and SHAP


#Eda
def describe_dataset(df: pd.DataFrame) -> None:
    """Shape, column names, duplicate check, dtypes and structural info."""
    print("\nThe shape of dataset is:", df.shape)

    print("\nThe names of the columns are:")
    print(df.columns)

    if df["Employee ID"].nunique() != len(df):
        print("Duplicate employee records detected.")
    else:
        print("Each employee appears only once. No duplicates.")

    df.info()
    print("\nData types of each column of the dataset are:")
    print(df.dtypes)


def missing_value_analysis(df: pd.DataFrame, x: pd.DataFrame) -> None:
    """Per-column missingness, per-row missingness and missingness correlation."""
    print("\nMissing values in each column (full dataset):")
    print(df.isnull().sum())

    print("\nPercentage of missing values in each column:")
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

    print(
        "The number of employees that have at least 1 missing value is",
        x.isnull().any(axis=1).sum(),
    )
    print(
        "The percentage of rows contain missing values is",
        round((x.isnull().any(axis=1).sum() / len(x)) * 100, 2),
        "%",
    )
    print(
        "The number of employees that have more than one missing value is",
        (x.isnull().sum(axis=1) > 1).sum(),
    )

    cols_with_missing = x.columns[x.isnull().sum() > 0]
    if len(cols_with_missing) > 1:
        missing_corr = x[cols_with_missing].isnull().corr()
        print("\nCorrelation matrix of missing values:")
        print(missing_corr)
    else:
        print("\nNot enough columns with missing values to compute correlation.")


def summary_statistics(x: pd.DataFrame):
    """Split numerical/categorical features and print describe() for numerics."""
    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = x.select_dtypes(include=["object", "string", "bool"]).columns

    print("\nContinuous (numerical) features:", numerical_cols.values)
    print()
    print("Categorical features:", categorical_cols.values)

    print("\nSummary statistics for numerical columns:\n")
    print(x[numerical_cols].describe().to_string())

    return numerical_cols, categorical_cols


def plot_feature_distributions(df: pd.DataFrame, x: pd.DataFrame) -> None:
    """Histograms (with normal overlay), ordinal barplots and the WFH pie."""
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
            s, bins=25, density=True, alpha=0.6,
            color="steelblue", edgecolor="white", linewidth=0.8,
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
            x[col].dropna().astype(float).round().astype(int)
            .value_counts().sort_index()
        )
        ax[i].bar(counts.index.astype(str), counts.values, alpha=0.7)
        ax[i].set_title(col, fontsize=11)
        ax[i].set_xlabel("Level", fontsize=10)
        ax[i].set_ylabel("Count", fontsize=10)
        ax[i].tick_params(axis="x", rotation=0)

    plt.tight_layout(pad=1.0)
    plt.savefig("Ordinal_barplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    fig = px.pie(df, names="WFH Setup Available", title="WFH Setup Availability")
    fig.write_html("WFH_Setup_Availability.html")


def plot_target_analysis(df: pd.DataFrame, x: pd.DataFrame) -> None:
    """Burn Rate distribution, binarisation, class balance and feature boxplots."""
    y = df["Burn Rate"]
    y_clean = y.dropna()

    print("\nThe top of the Target:")
    print(y_clean.head())
    print()
    print(y_clean.describe())

    plt.figure(figsize=(6, 4))
    plt.hist(y_clean, bins=25, density=True, alpha=0.6,
             edgecolor="black", linewidth=0.8)

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
        y_binary.value_counts(),
    )
    print("Missing values in y after thresholding:", y.isna().sum())

    class_counts = [np.sum(y_binary == 0), np.sum(y_binary == 1)]
    class_labels = ["Low Risk", "High Risk"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(class_labels, class_counts, color=["purple", "pink"], width=0.75)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, axis="y")
    plt.title("Distribution of Target Classes", fontsize=16, fontweight="bold")
    plt.xlabel("Classes", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 20,
                 f"{int(yval)}", ha="center", va="bottom",
                 fontweight="bold", fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("Class_dist_barplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("Portion of how many are 0 and 1:\n",
          y_binary.value_counts(normalize=True) * 100)

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
    boxprops = dict(linestyle="-", linewidth=2, color="navy")
    medianprops = dict(linestyle="-", linewidth=2, color="firebrick")
    whiskerprops = dict(linestyle="--", linewidth=2, color="black")
    capprops = dict(linestyle="-", linewidth=2, color="grey")

    plt.boxplot(
        [x[col].dropna() for col in boxplot_cols],
        tick_labels=boxplot_cols,
        notch=True, patch_artist=True,
        boxprops=boxprops, medianprops=medianprops,
        whiskerprops=whiskerprops, capprops=capprops,
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, axis="y")
    plt.title("Distribution of Selected Features (Boxplots)",
              fontsize=18, fontweight="bold")
    plt.xlabel("Features", fontsize=16)
    plt.ylabel("Values", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("Employee_Feature_Boxplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlations(x: pd.DataFrame) -> None:
    """Pearson correlation heatmap and strong positive/negative pairs."""
    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    correlation_matrix = x[numerical_cols].corr(method="pearson")

    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmax=1, vmin=-1)
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)),
               correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)),
               correlation_matrix.columns)
    plt.title("Pearson Feature Correlation Matrix")

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, np.around(correlation_matrix.iloc[i, j], decimals=2),
                     ha="center", va="center", color="w")

    plt.tight_layout()
    plt.savefig("Correlation_heatmap_before.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    corr = x[numerical_cols].corr(method="pearson")
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_pairs = upper_triangle.stack().reset_index()
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]

    strong_positive = corr_pairs[corr_pairs["Correlation"] > 0.7]
    print("\nStrong positive Pearson correlations (> 0.7):")
    print(strong_positive, "\n")

    strong_negative = corr_pairs[corr_pairs["Correlation"] < -0.7]
    print("Strong negative Pearson correlations (< -0.7):")
    print(strong_negative)


def plot_engineered_correlation(df_fe: pd.DataFrame) -> None:
    """Pearson correlation between the engineered JD-R features and Burn Rate."""
    engineered_corr = df_fe[
        ["Work_Pressure", "Organizational_Support", "Pressure_to_Support", "Burn Rate"]
    ].corr(method="pearson")

    print("\nPearson correlation matrix of engineered features and target:")
    print(engineered_corr)

    plt.figure(figsize=(7, 5))
    plt.imshow(engineered_corr, cmap="coolwarm", vmax=1, vmin=-1)
    plt.colorbar()
    plt.xticks(range(len(engineered_corr.columns)),
               engineered_corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(engineered_corr.columns)), engineered_corr.columns)
    plt.title("Pearson Correlation: Engineered Features and Burn Rate")

    for i in range(len(engineered_corr.columns)):
        for j in range(len(engineered_corr.columns)):
            plt.text(j, i, np.around(engineered_corr.iloc[i, j], decimals=2),
                     ha="center", va="center", color="w")

    plt.tight_layout()
    plt.savefig("Correlation_engineered_features_target.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()


def run_eda(df: pd.DataFrame) -> None:
    """Run the full EDA pipeline (overview -> plots -> correlations)."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    describe_dataset(df)

    x = df.drop(columns=["Burn Rate"])   # X used for EDA only (target excluded)

    missing_value_analysis(df, x)
    summary_statistics(x)
    plot_feature_distributions(df, x)
    plot_target_analysis(df, x)
    plot_correlations(x)

    df_fe = build_engineered_dataset(df, verbose=False)
    plot_engineered_correlation(df_fe)

    print("\nEDA complete. Figures saved as PDF / HTML in the working directory.")


# --------------------------------
def build_engineered_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove the burnout proxy and construct the three JD-R features."""
    df_fe = df.copy()
    df_fe = df_fe.drop(columns=["Mental Fatigue Score"])

    df_fe["Work_Pressure"] = (
        df_fe["Work Hours per Week"] + df_fe["Deadline Pressure Score"]
    )
    df_fe["Organizational_Support"] = (
        df_fe["Manager Support Score"] + df_fe["Recognition Frequency"]
    )
    df_fe["Pressure_to_Support"] = (
        df_fe["Work_Pressure"] / (df_fe["Organizational_Support"] + 1)
    )

    if verbose:
        example_cols = [
            "Work Hours per Week",
            "Deadline Pressure Score",
            "Manager Support Score",
            "Recognition Frequency",
            "Work_Pressure",
            "Organizational_Support",
            "Pressure_to_Support",
        ]
        print("\nEXAMPLE FEATURE ENGINEERING ROW")
        print(df_fe[example_cols].head(1).to_string(index=False))

    return df_fe


def build_shared_dataset(df: pd.DataFrame):

    df_fe = build_engineered_dataset(df)

    # Union of all predictors + raw building blocks + the CONTINUOUS target.
    # The continuous "Burn Rate" MUST be in the deletion mask so that rows with a
    # missing burnout outcome are DROPPED, not silently labelled 0 (NaN >= 0.45
    # evaluates to False). The binary target is built AFTER deletion.
    used_cols = sorted(set(
        CONTEXTUAL + WELLBEING_RAW + DEMAND_RESOURCE_RAW + ENGINEERED
    ))
    needed = used_cols + ["Burn Rate"]

    df_shared = df_fe[needed].dropna().copy()

    # Binarise on complete-case rows only, then drop the continuous target.
    df_shared["Burn_Rate_Binary"] = (df_shared["Burn Rate"] >= THRESHOLD).astype(int)
    df_shared = df_shared.drop(columns=["Burn Rate"])

    prevalence = df_shared["Burn_Rate_Binary"].mean() * 100
    print(f"\nShared complete-case dataset: n = {len(df_shared)}")
    print(f"High-risk burnout prevalence: {prevalence:.1f}%")
    return df_shared


def get_Xy(df_shared: pd.DataFrame, feature_list):
    """Select a configuration's predictors from the shared dataframe."""
    X = df_shared[feature_list].copy()
    y = df_shared["Burn_Rate_Binary"].copy()
    return X, y


def build_preprocessors(X: pd.DataFrame):
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "string", "bool"]).columns

    num_t = Pipeline(steps=[("scaler", StandardScaler())])
    cat_t = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre_non_tree = ColumnTransformer(
        transformers=[("num", num_t, numerical_cols),
                      ("cat", cat_t, categorical_cols)],
        remainder="drop",
    )
    pre_tree = ColumnTransformer(
        transformers=[("num", "passthrough", numerical_cols),
                      ("cat", cat_t, categorical_cols)],
        remainder="drop",
    )
    return pre_non_tree, pre_tree


def get_models():
    return {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            "params": {"model__C": [0.1, 1.0, 10.0]},
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {"model__n_estimators": [50, 100, 200],
                       "model__max_depth": [None, 5]},
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            "params": {"model__n_estimators": [100, 200],
                       "model__learning_rate": [0.05, 0.1],
                       "model__max_depth": [3, 5]},
        },
    }


def get_scoring():
    return {
        "f1": "f1", "roc_auc": "roc_auc",
        "precision": "precision", "recall": "recall",
    }



# NESTED CROSS-VALIDATION

def run_nested_cv(df_shared, config_name, feature_list, outer_cv, inner_cv):
    """
    Nested CV for all models under one feature configuration.
    GridSearchCV uses multi-metric inner scoring with refit on recall, so the
    inner validation scores for every metric are available afterwards.
    Returns (results, X, y) where results[model_name] is the cross_validate dict
    (with fitted estimators and train scores).
    """
    X, y = get_Xy(df_shared, feature_list)
    pre_non_tree, pre_tree = build_preprocessors(X)
    models = get_models()
    scoring = get_scoring()

    print(f"\n{'='*70}\nCONFIG: {config_name}  (n={len(X)}, features={len(feature_list)})")
    print(f"Features: {list(X.columns)}\n{'='*70}")

    results = {}
    for name, spec in models.items():
        pre = pre_tree if name in ("Random Forest", "XGBoost") else pre_non_tree
        pipe = Pipeline(steps=[("preprocessor", pre), ("model", spec["model"])])

        grid = GridSearchCV(
            pipe, spec["params"], cv=inner_cv,
            scoring=scoring, refit="recall", n_jobs=1,
        )
        cv = cross_validate(
            grid, X, y, cv=outer_cv, scoring=scoring,
            return_estimator=True, return_train_score=True, n_jobs=1,
        )
        results[name] = cv

        rec = np.mean(cv["test_recall"]); recs = np.std(cv["test_recall"], ddof=1)
        f1 = np.mean(cv["test_f1"]); f1s = np.std(cv["test_f1"], ddof=1)
        auc = np.mean(cv["test_roc_auc"]); aucs = np.std(cv["test_roc_auc"], ddof=1)
        prec = np.mean(cv["test_precision"]); precs = np.std(cv["test_precision"], ddof=1)

        print(f"\n{name}")
        print(f"  Recall    {rec:.3f} +/- {recs:.3f}")
        print(f"  F1        {f1:.3f} +/- {f1s:.3f}")
        print(f"  ROC-AUC   {auc:.3f} +/- {aucs:.3f}")
        print(f"  Precision {prec:.3f} +/- {precs:.3f}")

        # Best hyperparameters selected by the inner grid search, per outer fold.
        fold_params = [est.best_params_ for est in cv["estimator"]]
        param_counts = Counter(tuple(sorted(p.items())) for p in fold_params)
        modal = dict(max(param_counts, key=param_counts.get))
        all_same = len(param_counts) == 1
        print(f"  Best params per outer fold:")
        for i, p in enumerate(fold_params, 1):
            print(f"    Fold {i}: {p}")
        if all_same:
            print(f"  -> Identical across all outer folds: {modal}")
        else:
            print(f"  -> Modal (most frequent) configuration: {modal}")

    return results, X, y


def test_summary_rows(results, config_name):
    """Outer-test metrics (mean +/- sd) per model -> A vs B comparison rows."""
    rows = []
    for name, cv in results.items():
        def stat(k):
            return np.mean(cv[k]), np.std(cv[k], ddof=1)
        rec, recs = stat("test_recall")
        f1, f1s = stat("test_f1")
        auc, aucs = stat("test_roc_auc")
        prec, precs = stat("test_precision")
        rows.append({
            "Config": config_name, "Model": name,
            "Recall": f"{rec:.3f} ± {recs:.3f}",
            "F1": f"{f1:.3f} ± {f1s:.3f}",
            "ROC-AUC": f"{auc:.3f} ± {aucs:.3f}",
            "Precision": f"{prec:.3f} ± {precs:.3f}",
        })
    return rows


def _validation_scores(cv, metric_keys):
    """Mean inner-CV validation score (per metric) at the selected hyperparameters,
    averaged across the outer folds."""
    out = {k: [] for k in metric_keys}
    for est in cv["estimator"]:
        bi = est.best_index_
        for k in metric_keys:
            out[k].append(est.cv_results_[f"mean_test_{k}"][bi])
    return {k: float(np.mean(v)) for k, v in out.items()}


def train_val_test_table(results, config_name):
    """Train / Validation / Test performance per model (F1, ROC-AUC, Precision, Recall)."""
    metric_keys = ["f1", "roc_auc", "precision", "recall"]
    rows = []
    for name, cv in results.items():
        train = {k: float(np.mean(cv[f"train_{k}"])) for k in metric_keys}
        test = {k: float(np.mean(cv[f"test_{k}"])) for k in metric_keys}
        val = _validation_scores(cv, metric_keys)
        for set_name, d in (("Train", train), ("Validation", val), ("Test", test)):
            rows.append({
                "Config": config_name, "Model": name, "Set": set_name,
                "F1": round(d["f1"], 3),
                "ROC-AUC": round(d["roc_auc"], 3),
                "Precision": round(d["precision"], 3),
                "Recall": round(d["recall"], 3),
            })
    return pd.DataFrame(rows)


def split_size_table(X, y, outer_cv, inner_splits=5):
    """
    Per-fold sample sizes under the nested CV procedure. Within each outer
    development set, one inner fold (1/inner_splits) is taken as the validation
    portion, the rest as training, giving the 64% / 16% / 20% framing.
    """
    rows = []
    for i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        dev = len(tr_idx)
        test = len(te_idx)
        val = int(round(dev / inner_splits))
        train = dev - val
        test_high = float(y.iloc[te_idx].mean()) * 100
        rows.append({
            "Fold": i, "Train": train, "Validation": val,
            "Test": test, "Test high-risk %": round(test_high, 1),
        })

    df = pd.DataFrame(rows)
    n = len(X)
    avg = {
        "Fold": "Average (%)",
        "Train": round(df["Train"].mean() / n * 100, 1),
        "Validation": round(df["Validation"].mean() / n * 100, 1),
        "Test": round(df["Test"].mean() / n * 100, 1),
        "Test high-risk %": round(df["Test high-risk %"].mean(), 1),
    }
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df



# REPEATED NESTED CROSS-VALIDATION (stability check)

def run_repeated_nested_cv(df_shared, config_name, feature_list,
                           n_splits=5, n_repeats=N_REPEATS, inner_splits=5):
    """
    Repeated nested CV for one feature configuration. The outer loop is a
    RepeatedStratifiedKFold (n_splits x n_repeats), giving n_splits*n_repeats
    outer-test estimates per model; the inner loop is a stratified K-fold grid
    search refit on recall. Reports mean +/- sd over all outer folds and repeats,
    used to assess the stability of the single-run nested CV estimates.
    Returns a list of summary rows (one per model).
    """
    X, y = get_Xy(df_shared, feature_list)
    pre_non_tree, pre_tree = build_preprocessors(X)
    models = get_models()
    scoring = get_scoring()

    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE,
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE,
    )

    print(f"\n{'='*70}")
    print(f"REPEATED NESTED CV: {config_name}  "
          f"({n_splits} splits x {n_repeats} repeats = "
          f"{n_splits * n_repeats} outer fits per model)")
    print(f"{'='*70}")

    rows = []
    for name, spec in models.items():
        pre = pre_tree if name in ("Random Forest", "XGBoost") else pre_non_tree
        pipe = Pipeline(steps=[("preprocessor", pre), ("model", spec["model"])])

        grid = GridSearchCV(
            pipe, spec["params"], cv=inner_cv,
            scoring=scoring, refit="recall", n_jobs=1,
        )
        cv = cross_validate(
            grid, X, y, cv=outer_cv, scoring=scoring,
            return_estimator=False, n_jobs=1,
        )

        n_outer = len(cv["test_recall"])
        rec = np.mean(cv["test_recall"]); recs = np.std(cv["test_recall"], ddof=1)
        f1 = np.mean(cv["test_f1"]); f1s = np.std(cv["test_f1"], ddof=1)
        auc = np.mean(cv["test_roc_auc"]); aucs = np.std(cv["test_roc_auc"], ddof=1)
        prec = np.mean(cv["test_precision"]); precs = np.std(cv["test_precision"], ddof=1)

        print(f"\n{name}  (n={n_outer})")
        print(f"  Recall    {rec:.3f} +/- {recs:.3f}")
        print(f"  F1        {f1:.3f} +/- {f1s:.3f}")
        print(f"  ROC-AUC   {auc:.3f} +/- {aucs:.3f}")
        print(f"  Precision {prec:.3f} +/- {precs:.3f}")

        rows.append({
            "Config": config_name, "Model": name, "Outer fits": n_outer,
            "Recall": f"{rec:.3f} ± {recs:.3f}",
            "F1": f"{f1:.3f} ± {f1s:.3f}",
            "ROC-AUC": f"{auc:.3f} ± {aucs:.3f}",
            "Precision": f"{prec:.3f} ± {precs:.3f}",
        })

    return rows


# BEST-MODEL SELECTION (on the engineered config) + per-fold hyperparameters

def select_best_model(results, config_name=ENGINEERED_CONFIG):
    """
    Select the best model within a configuration by mean outer-test recall,
    breaking ties on F1 and then ROC-AUC. Recall is the lead criterion because it
    is the study's primary objective (correctly identifying high-risk employees);
    F1 and ROC-AUC act as deterministic tie-breakers when recall is identical
    (e.g. Random Forest and XGBoost both reaching recall = 1.000 under config B).
    Prints the best model, its selection scores, and its tuned hyperparameters per
    outer fold, then returns the best model name.
    """
    def key(name):
        cv = results[name]
        return (
            np.mean(cv["test_recall"]),
            np.mean(cv["test_f1"]),
            np.mean(cv["test_roc_auc"]),
        )

    best_model = max(results, key=key)
    rec, f1, auc = key(best_model)

    print(f"\n{'='*70}")
    print(f"BEST MODEL ({config_name}): {best_model}")
    print(f"{'='*70}")
    print(f"Selection scores -> Recall {rec:.3f} | F1 {f1:.3f} | ROC-AUC {auc:.3f}")

    best_params = [est.best_params_ for est in results[best_model]["estimator"]]
    print(f"\n{best_model} - best hyperparameters per outer fold:")
    for i, params in enumerate(best_params, 1):
        print(f"  Fold {i}: {params}")

    # Modal hyperparameters: the configuration most frequently selected across the
    # outer folds (ties resolved by first occurrence). Used to fit a single final
    # model on the full shared dataset for interpretation.
    param_counts = Counter(tuple(sorted(p.items())) for p in best_params)
    modal_params = dict(max(param_counts, key=param_counts.get))
    print(f"\n{best_model} - modal hyperparameters (used for final SHAP model): "
          f"{modal_params}")

    return best_model, modal_params


def fit_final_model(df_shared, feature_list, model_name, modal_params):
    """
    Fit ONE final pipeline (preprocessing + model) on the FULL shared dataset using
    the modal best hyperparameters from nested CV. This is the model explained by
    SHAP. Fitting a single final model for interpretation avoids the bias of
    explaining a fold-specific estimator (trained on only 4/5 of the data) while
    computing SHAP over the whole sample, which would let the model "see" its own
    training rows. Evaluation remains the nested-CV estimates; this refit is for
    interpretation only.
    """
    X, y = get_Xy(df_shared, feature_list)
    pre_non_tree, pre_tree = build_preprocessors(X)
    spec = get_models()[model_name]

    pre = pre_tree if model_name in ("Random Forest", "XGBoost") else pre_non_tree
    pipe = Pipeline(steps=[("preprocessor", pre), ("model", spec["model"])])
    pipe.set_params(**modal_params)

    print(f"\nFitting final {model_name} on full shared dataset "
          f"(n={len(X)}) for SHAP interpretation...")
    pipe.fit(X, y)
    return pipe, X



# SHAP INTERPRETABILITY (final best model, fit on the full shared dataset)

def run_shap_analysis(final_pipe, X, model_name):
    """
    SHAP replaces model-specific feature importance. Explains the FINAL pipeline
    (best model fit on the full shared dataset) using the transformed feature
    space. TreeExplainer is appropriate for the tree-based models (Random Forest,
    XGBoost) this study selects from.
    """
    print(f"\nRunning SHAP analysis on final model: {model_name} ...")

    preprocessor = final_pipe.named_steps["preprocessor"]
    model = final_pipe.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    if len(X_transformed_df) > 3000:
        X_shap = X_transformed_df.sample(n=3000, random_state=RANDOM_STATE)
    else:
        X_shap = X_transformed_df.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)

    # If SHAP returns values for both classes, keep class 1 = high burnout risk
    if len(shap_values.values.shape) == 3:
        shap_values = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )

    plt.figure()
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_global_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_beeswarm.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    sample_id = 0
    plt.figure()
    shap.plots.waterfall(shap_values[sample_id], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_waterfall_sample0.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    shap_importance = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        "mean_shap": shap_values.values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv("SHAP_global_importance.csv", index=False)

    print("\nTop SHAP features:")
    print(shap_importance.head(15).to_string(index=False))

    if (
        "num__Work_Pressure" in shap_values.feature_names
        and "num__Organizational_Support" in shap_values.feature_names
    ):
        plt.figure()
        shap.plots.scatter(
            shap_values[:, "num__Work_Pressure"],
            color=shap_values[:, "num__Organizational_Support"],
            show=False,
        )
        plt.tight_layout()
        plt.savefig("SHAP_WorkPressure_by_OrganizationalSupport.pdf",
                    dpi=300, bbox_inches="tight")
        plt.close()



# MAIN

def main():
    df = pd.read_csv("enriched_employee_dataset.csv", sep=";")

    # 1. EDA
    run_eda(df)

    #  2. Shared complete-case dataset -> identical N for A and B ----
    df_shared = build_shared_dataset(df)
    y_shared = df_shared["Burn_Rate_Binary"]

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    #  3. Nested CV for both configurations ----
    all_test_rows = []
    perf_tables = []
    results_by_config = {}
    X_by_config = {}

    for config_name, feature_list in CONFIGS.items():
        results, X, y = run_nested_cv(
            df_shared, config_name, feature_list, outer_cv, inner_cv
        )
        results_by_config[config_name] = results
        X_by_config[config_name] = X
        all_test_rows.extend(test_summary_rows(results, config_name))
        perf_tables.append(train_val_test_table(results, config_name))

    # 4a. A vs B comparison (outer-test metrics) ----
    summary = pd.DataFrame(all_test_rows)
    print("\n\n===== A vs B COMPARISON (shared N) =====")
    print(summary.to_string(index=False))
    summary.to_csv("config_comparison.csv", index=False)

    # 4b. Train / Validation / Test performance (both configs)
    perf = pd.concat(perf_tables, ignore_index=True)
    print("\n\n===== TRAIN / VALIDATION / TEST PERFORMANCE =====")
    print(perf.to_string(index=False))
    perf.to_csv("model_performance_train_val_test.csv", index=False)

    print(f"\n--- Engineered configuration only ({ENGINEERED_CONFIG}) ---")
    print(perf[perf["Config"] == ENGINEERED_CONFIG].to_string(index=False))

    # 4c. Per-fold split sizes (data-level; reported on the engineered config)
    X_eng = X_by_config[ENGINEERED_CONFIG]
    splits = split_size_table(X_eng, y_shared, outer_cv, inner_splits=5)
    print("\n\n===== PER-FOLD CV SPLITS =====")
    print(splits.to_string(index=False))
    splits.to_csv("cv_splits.csv", index=False)

    # 4d. Repeated nested CV (stability check, both configs)
    repeated_rows = []
    for config_name, feature_list in CONFIGS.items():
        repeated_rows.extend(
            run_repeated_nested_cv(df_shared, config_name, feature_list)
        )
    repeated = pd.DataFrame(repeated_rows)
    print("\n\n===== REPEATED NESTED CV (shared N) =====")
    print(repeated.to_string(index=False))
    repeated.to_csv("repeated_nested_cv.csv", index=False)

    # 5a. Best model on the engineered config (B): select by recall,
    #          ties broken on F1 then ROC-AUC; report tuned hyperparameters
    #          per outer fold and the modal hyperparameter set. ----
    print(f"\n\n===== BEST MODEL ON {ENGINEERED_CONFIG} =====")
    best_model_name, modal_params = select_best_model(
        results_by_config[ENGINEERED_CONFIG], ENGINEERED_CONFIG
    )

    # 5b. Fit ONE final model on the full shared dataset (interpretation
    #          only) and run SHAP on it, avoiding the bias of explaining a
    #          fold-specific estimator over the whole sample. ----
    final_pipe, X_final = fit_final_model(
        df_shared, CONFIGS[ENGINEERED_CONFIG], best_model_name, modal_params
    )
    run_shap_analysis(final_pipe, X_final, best_model_name)

    print("\nDone. Saved: config_comparison.csv, model_performance_train_val_test.csv, "
          "cv_splits.csv, repeated_nested_cv.csv, SHAP_global_importance.csv + figures.")


if __name__ == "__main__":
    main()