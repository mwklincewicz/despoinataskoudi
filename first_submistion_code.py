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
    cross_validate
)
from sklearn.metrics import make_scorer, matthews_corrcoef
from xgboost import XGBClassifier


THRESHOLD = 0.45
RANDOM_STATE = 50


def build_engineered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final engineered dataset for early burnout detection.
    Removes Mental Fatigue Score and constructs theory-driven features.
    """
    df_fe = df.copy()

    # Remove direct burnout-like proxy
    df_fe = df_fe.drop(columns=["Mental Fatigue Score"])

    # Early-demand feature
    df_fe["Work_Pressure"] = (
        df_fe["Work Hours per Week"] +
        df_fe["Deadline Pressure Score"]
    )

    # Early-resource feature
    df_fe["Organizational_Support"] = (
        df_fe["Manager Support Score"] +
        df_fe["Recognition Frequency"]
    )

    # Pressure-support trade-off feature
    df_fe["Pressure_to_Support"] = (
        df_fe["Work_Pressure"] /
        (df_fe["Organizational_Support"] + 1)
    )

    return df_fe


def prepare_model_data(df: pd.DataFrame):
    """
    Prepare final X and y for modeling.
    """
    df_fe = build_engineered_dataset(df)

    df_model = df_fe.drop(columns=["Employee ID", "Date of Joining"]).copy()
    df_model = df_model.dropna().copy()

    df_model["Burn_Rate_Binary"] = (df_model["Burn Rate"] >= THRESHOLD).astype(int)

    X = df_model.drop(columns=["Burn Rate", "Burn_Rate_Binary"]).copy()
    y = df_model["Burn_Rate_Binary"].copy()

    # Remove original variables used in engineered features
    X = X.drop(columns=[
        "Work Hours per Week",
        "Deadline Pressure Score",
        "Manager Support Score",
        "Recognition Frequency"
    ])

    return df_fe, df_model, X, y


def build_preprocessors(X: pd.DataFrame):
    """
    Build preprocessors for linear and tree-based models.
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
    Return the modeling dictionary.
    """
    models = {
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
    return models


def get_scoring():
    """
    Return evaluation metrics including MCC.
    """
    scoring = {
        "f1": "f1",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "mcc": make_scorer(matthews_corrcoef)
    }
    return scoring


def run_shap_analysis(results, X):
    """
    SHAP replaces model-specific feature importance.

    This function explains the trained XGBoost pipeline from the nested CV
    results using the transformed feature space produced by the preprocessing
    pipeline.
    """
    print("\nRunning SHAP analysis...")

    # Use XGBoost as the main explainable tree-based model
    shap_model_name = "XGBoost"

    # Take the best trained XGBoost pipeline from the first outer fold
    best_pipe = results[shap_model_name]["estimator"][0].best_estimator_

    preprocessor = best_pipe.named_steps["preprocessor"]
    model = best_pipe.named_steps["model"]

    # Transform data exactly as the model sees it
    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index
    )

    # Use a sample for speed and readable SHAP plots
    if len(X_transformed_df) > 3000:
        X_shap = X_transformed_df.sample(
            n=3000,
            random_state=RANDOM_STATE
        )
    else:
        X_shap = X_transformed_df.copy()

    # TreeSHAP for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)

    # If SHAP returns values for both classes, keep class 1 = high burnout risk
    if len(shap_values.values.shape) == 3:
        shap_values = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=shap_values.data,
            feature_names=shap_values.feature_names
        )

    # SHAP global bar plot
    plt.figure()
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_global_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP beeswarm plot
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_beeswarm.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP local waterfall plot for one employee
    sample_id = 0
    plt.figure()
    shap.plots.waterfall(shap_values[sample_id], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_waterfall_sample0.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP importance table
    shap_importance = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        "mean_shap": shap_values.values.mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv("SHAP_global_importance.csv", index=False)

    print("\nTop SHAP features:")
    print(shap_importance.head(15).to_string(index=False))

    # SHAP dependence / interaction-style plot:
    # Work Pressure colored by Organizational Support
    if (
        "num__Work_Pressure" in shap_values.feature_names and
        "num__Organizational_Support" in shap_values.feature_names
    ):
        plt.figure()
        shap.plots.scatter(
            shap_values[:, "num__Work_Pressure"],
            color=shap_values[:, "num__Organizational_Support"],
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            "SHAP_WorkPressure_by_OrganizationalSupport.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def main():
    # =========================
    # Load dataset
    # =========================
    df = pd.read_csv("enriched_employee_dataset.csv", sep=";")

    print("\nThe shape of dataset is:", df.shape)

    print("\nThe names of the columns are:")
    print(df.columns)

    # Check duplicates
    if df["Employee ID"].nunique() != len(df):
        print("Duplicate employee records detected.")
    else:
        print("Each employee appears only once. No duplicates.")

    print("\nMissing values in each column (full dataset):")
    print(df.isnull().sum())

    # Missing values percentage
    print("\nPercentage of missing values in each column:")

    total_records = len(df)
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / total_records) * 100

    missing_summary = pd.DataFrame({
        "Missing Values": missing_count,
        "Missing Percentage (%)": missing_percentage
    })

    missing_summary = missing_summary[missing_summary["Missing Values"] > 0]

    print("\nMissing summary (only columns with missing values):")
    print(missing_summary.to_string())

    # X for EDA only
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

    # Correlation of missingness
    cols_with_missing = x.columns[x.isnull().sum() > 0]

    if len(cols_with_missing) > 1:
        missing_corr = x[cols_with_missing].isnull().corr()
        print("\nCorrelation matrix of missing values:")
        print(missing_corr)
    else:
        print("\nNot enough columns with missing values to compute correlation.")

    df.info()
    print("\nData types of each column of the dataset are:")
    print(df.dtypes)

    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = x.select_dtypes(include=["object", "string", "bool"]).columns

    print("\nContinuous (numerical) features:", numerical_cols.values)
    print()
    print("Categorical features:", categorical_cols.values)

    print("\nSummary statistics for numerical columns:\n")
    print(x[numerical_cols].describe().to_string())

    # =========================
    # EDA plots
    # =========================
    continuous_cols = [
        "Sleep Hours",
        "Work Hours per Week",
        "Years in Company",
        "Team Size"
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
            linewidth=0.8
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
        ax[i].tick_params(axis="x", rotation=0)

    plt.tight_layout(pad=1.0)
    plt.savefig("Ordinal_barplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Boolean/categorical visualization
    fig = px.pie(
        df,
        names="WFH Setup Available",
        title="WFH Setup Availability"
    )
    fig.write_html("WFH_Setup_Availability.html")

    # =========================
    # Target analysis
    # =========================
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
        linewidth=0.8
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
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            f"{int(yval)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12
        )

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("Class_dist_barplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("Portion of how many are 0 and 1:\n", y_binary.value_counts(normalize=True) * 100)

    boxplot_cols = [
        "Sleep Hours",
        "Work Hours per Week",
        "Resource Allocation",
        "Work-Life Balance Score",
        "Manager Support Score",
        "Deadline Pressure Score",
        "Recognition Frequency"
    ]

    plt.figure(figsize=(15, 7))

    boxprops = dict(linestyle="-", linewidth=2, color="navy")
    medianprops = dict(linestyle="-", linewidth=2, color="firebrick")
    whiskerprops = dict(linestyle="--", linewidth=2, color="black")
    capprops = dict(linestyle="-", linewidth=2, color="grey")

    plt.boxplot(
        [x[col].dropna() for col in boxplot_cols],
        tick_labels=boxplot_cols,
        notch=True,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops
    )

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, axis="y")
    plt.title("Distribution of Selected Features (Boxplots)", fontsize=18, fontweight="bold")
    plt.xlabel("Features", fontsize=16)
    plt.ylabel("Values", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("Employee_Feature_Boxplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Pearson correlation heatmap before modeling
    # =========================
    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    correlation_matrix = x[numerical_cols].corr(method="pearson")

    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmax=1, vmin=-1)
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Pearson Feature Correlation Matrix")

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(
                j, i,
                np.around(correlation_matrix.iloc[i, j], decimals=2),
                ha="center",
                va="center",
                color="w"
            )

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

    # =========================
    # Feature engineering for EARLY DETECTION
    # =========================
    df_fe, df_model, X, y = prepare_model_data(df)

    engineered_corr = df_fe[
        ["Work_Pressure", "Organizational_Support", "Pressure_to_Support", "Burn Rate"]
    ].corr(method="pearson")

    print("\nPearson correlation matrix of engineered features and target:")
    print(engineered_corr)

    plt.figure(figsize=(7, 5))
    plt.imshow(engineered_corr, cmap="coolwarm", vmax=1, vmin=-1)
    plt.colorbar()
    plt.xticks(range(len(engineered_corr.columns)), engineered_corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(engineered_corr.columns)), engineered_corr.columns)
    plt.title("Pearson Correlation: Engineered Features and Burn Rate")

    for i in range(len(engineered_corr.columns)):
        for j in range(len(engineered_corr.columns)):
            plt.text(
                j, i,
                np.around(engineered_corr.iloc[i, j], decimals=2),
                ha="center",
                va="center",
                color="w"
            )

    plt.tight_layout()
    plt.savefig("Correlation_engineered_features_target.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    preprocessor_non_tree, preprocessor_tree, numerical_cols, categorical_cols = build_preprocessors(X)
    models = get_models()
    scoring = get_scoring()

    print("\nFinal modeling features:")
    print(X.columns.tolist())

    outer_split_num = 5
    inner_split_num = 5

    outer_cv = StratifiedKFold(
        n_splits=outer_split_num,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_split_num,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for name, spec in models.items():
        print(f"\nRunning model: {name}")

        if name in ["Random Forest", "XGBoost"]:
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor_tree),
                ("model", spec["model"])
            ])
        else:
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor_non_tree),
                ("model", spec["model"])
            ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            cv=inner_cv,
            scoring="recall",
            n_jobs=1
        )

        cv_results = cross_validate(
            estimator=grid_search,
            X=X,
            y=y,
            cv=outer_cv,
            scoring=scoring,
            return_estimator=True,
            return_train_score=True,
            n_jobs=1
        )

        results[name] = cv_results

        print(f"\n{name}")
        print(f"  F1-Score:  {np.mean(cv_results['test_f1']):.3f} ± {np.std(cv_results['test_f1'], ddof=1):.3f}")
        print(f"  ROC-AUC:   {np.mean(cv_results['test_roc_auc']):.3f} ± {np.std(cv_results['test_roc_auc'], ddof=1):.3f}")
        print(f"  Precision: {np.mean(cv_results['test_precision']):.3f} ± {np.std(cv_results['test_precision'], ddof=1):.3f}")
        print(f"  Recall:    {np.mean(cv_results['test_recall']):.3f} ± {np.std(cv_results['test_recall'], ddof=1):.3f}")
        print(f"  MCC:       {np.mean(cv_results['test_mcc']):.3f} ± {np.std(cv_results['test_mcc'], ddof=1):.3f}")

        print(f"  Train F1:       {np.mean(cv_results['train_f1']):.3f} ± {np.std(cv_results['train_f1'], ddof=1):.3f}")
        print(f"  Train AUC:      {np.mean(cv_results['train_roc_auc']):.3f} ± {np.std(cv_results['train_roc_auc'], ddof=1):.3f}")
        print(f"  Train Precision:{np.mean(cv_results['train_precision']):.3f} ± {np.std(cv_results['train_precision'], ddof=1):.3f}")
        print(f"  Train Recall:   {np.mean(cv_results['train_recall']):.3f} ± {np.std(cv_results['train_recall'], ddof=1):.3f}")
        print(f"  Train MCC:      {np.mean(cv_results['train_mcc']):.3f} ± {np.std(cv_results['train_mcc'], ddof=1):.3f}")

    # =========================
    # Best model
    # =========================
    best_model = max(results, key=lambda m: np.mean(results[m]["test_recall"]))
    best_params = [est.best_params_ for est in results[best_model]["estimator"]]

    print(f"\n{best_model} – best hyperparameters per outer fold:")
    for i, params in enumerate(best_params, 1):
        print(f"  Fold {i}: {params}")

    # =========================
    # Model comparison plot
    # =========================
    models_list = ["Logistic Regression", "Random Forest", "XGBoost"]
    metrics = {
        "F1-score": "test_f1",
        "ROC-AUC": "test_roc_auc",
        "Precision": "test_precision",
        "Recall": "test_recall",
        "MCC": "test_mcc"
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
            alpha=0.85
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
                fontweight="bold"
            )

    # Hide empty subplot
    if len(axes) > len(metrics):
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis("off")

    fig.suptitle(
        "Comparison of Classifier Performance",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Classifier_comparison_multimetric.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # SHAP INTERPRETABILITY
    # =========================
    run_shap_analysis(results, X)


def repeated_nested_cv():
    df = pd.read_csv("enriched_employee_dataset.csv", sep=";")

    _, _, X, y = prepare_model_data(df)

    preprocessor_non_tree, preprocessor_tree, _, _ = build_preprocessors(X)
    models = get_models()
    scoring = get_scoring()

    outer_split_num = 5
    inner_split_num = 5
    n_repeats = 10

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_split_num,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_split_num,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for name, spec in models.items():
        if name in ["Random Forest", "XGBoost"]:
            base_pipe = Pipeline(steps=[
                ("preprocessor", preprocessor_tree),
                ("model", spec["model"])
            ])
        else:
            base_pipe = Pipeline(steps=[
                ("preprocessor", preprocessor_non_tree),
                ("model", spec["model"])
            ])

        grid_search = GridSearchCV(
            estimator=base_pipe,
            param_grid=spec["params"],
            cv=inner_cv,
            scoring="recall",
            n_jobs=1
        )

        cv_results = cross_validate(
            estimator=grid_search,
            X=X,
            y=y,
            cv=outer_cv,
            scoring=scoring,
            return_estimator=False,
            n_jobs=1
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

    # Το αφήνουμε κλειστό για να μη βαραίνει το run.
    # Αν το χρειαστείς, ξεσχολίασε την επόμενη γραμμή.
    # repeated_nested_cv()
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
    cross_validate
)
from sklearn.metrics import make_scorer, matthews_corrcoef
from xgboost import XGBClassifier


THRESHOLD = 0.45
RANDOM_STATE = 50


def build_engineered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final engineered dataset for early burnout detection.
    Removes Mental Fatigue Score and constructs theory-driven features.
    """
    df_fe = df.copy()

    # Remove direct burnout-like proxy
    df_fe = df_fe.drop(columns=["Mental Fatigue Score"])

    # Early-demand feature
    df_fe["Work_Pressure"] = (
        df_fe["Work Hours per Week"] +
        df_fe["Deadline Pressure Score"]
    )

    # Early-resource feature
    df_fe["Organizational_Support"] = (
        df_fe["Manager Support Score"] +
        df_fe["Recognition Frequency"]
    )

    # Pressure-support trade-off feature
    df_fe["Pressure_to_Support"] = (
        df_fe["Work_Pressure"] /
        (df_fe["Organizational_Support"] + 1)
    )

    return df_fe


def prepare_model_data(df: pd.DataFrame):
    """
    Prepare final X and y for modeling.
    """
    df_fe = build_engineered_dataset(df)

    df_model = df_fe.drop(columns=["Employee ID", "Date of Joining"]).copy()
    df_model = df_model.dropna().copy()

    df_model["Burn_Rate_Binary"] = (df_model["Burn Rate"] >= THRESHOLD).astype(int)

    X = df_model.drop(columns=["Burn Rate", "Burn_Rate_Binary"]).copy()
    y = df_model["Burn_Rate_Binary"].copy()

    # Remove original variables used in engineered features
    X = X.drop(columns=[
        "Work Hours per Week",
        "Deadline Pressure Score",
        "Manager Support Score",
        "Recognition Frequency"
    ])

    return df_fe, df_model, X, y


def build_preprocessors(X: pd.DataFrame):
    """
    Build preprocessors for linear and tree-based models.
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
    Return the modeling dictionary.
    """
    models = {
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
    return models


def get_scoring():
    """
    Return evaluation metrics including MCC.
    """
    scoring = {
        "f1": "f1",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "mcc": make_scorer(matthews_corrcoef)
    }
    return scoring


def run_shap_analysis(results, X):
    """
    SHAP replaces model-specific feature importance.

    This function explains the trained XGBoost pipeline from the nested CV
    results using the transformed feature space produced by the preprocessing
    pipeline.
    """
    print("\nRunning SHAP analysis...")

    # Use XGBoost as the main explainable tree-based model
    shap_model_name = "XGBoost"

    # Take the best trained XGBoost pipeline from the first outer fold
    best_pipe = results[shap_model_name]["estimator"][0].best_estimator_

    preprocessor = best_pipe.named_steps["preprocessor"]
    model = best_pipe.named_steps["model"]

    # Transform data exactly as the model sees it
    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index
    )

    # Use a sample for speed and readable SHAP plots
    if len(X_transformed_df) > 3000:
        X_shap = X_transformed_df.sample(
            n=3000,
            random_state=RANDOM_STATE
        )
    else:
        X_shap = X_transformed_df.copy()

    # TreeSHAP for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)

    # If SHAP returns values for both classes, keep class 1 = high burnout risk
    if len(shap_values.values.shape) == 3:
        shap_values = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=shap_values.data,
            feature_names=shap_values.feature_names
        )

    # SHAP global bar plot
    plt.figure()
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_global_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP beeswarm plot
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_beeswarm.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP local waterfall plot for one employee
    sample_id = 0
    plt.figure()
    shap.plots.waterfall(shap_values[sample_id], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("SHAP_waterfall_sample0.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP importance table
    shap_importance = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        "mean_shap": shap_values.values.mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv("SHAP_global_importance.csv", index=False)

    print("\nTop SHAP features:")
    print(shap_importance.head(15).to_string(index=False))

    # SHAP dependence / interaction-style plot:
    # Work Pressure colored by Organizational Support
    if (
        "num__Work_Pressure" in shap_values.feature_names and
        "num__Organizational_Support" in shap_values.feature_names
    ):
        plt.figure()
        shap.plots.scatter(
            shap_values[:, "num__Work_Pressure"],
            color=shap_values[:, "num__Organizational_Support"],
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            "SHAP_WorkPressure_by_OrganizationalSupport.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def main():
    # =========================
    # Load dataset
    # =========================
    df = pd.read_csv("enriched_employee_dataset.csv", sep=";")

    print("\nThe shape of dataset is:", df.shape)

    print("\nThe names of the columns are:")
    print(df.columns)

    # Check duplicates
    if df["Employee ID"].nunique() != len(df):
        print("Duplicate employee records detected.")
    else:
        print("Each employee appears only once. No duplicates.")

    print("\nMissing values in each column (full dataset):")
    print(df.isnull().sum())

    # Missing values percentage
    print("\nPercentage of missing values in each column:")

    total_records = len(df)
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / total_records) * 100

    missing_summary = pd.DataFrame({
        "Missing Values": missing_count,
        "Missing Percentage (%)": missing_percentage
    })

    missing_summary = missing_summary[missing_summary["Missing Values"] > 0]

    print("\nMissing summary (only columns with missing values):")
    print(missing_summary.to_string())

    # X for EDA only
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

    # Correlation of missingness
    cols_with_missing = x.columns[x.isnull().sum() > 0]

    if len(cols_with_missing) > 1:
        missing_corr = x[cols_with_missing].isnull().corr()
        print("\nCorrelation matrix of missing values:")
        print(missing_corr)
    else:
        print("\nNot enough columns with missing values to compute correlation.")

    df.info()
    print("\nData types of each column of the dataset are:")
    print(df.dtypes)

    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = x.select_dtypes(include=["object", "string", "bool"]).columns

    print("\nContinuous (numerical) features:", numerical_cols.values)
    print()
    print("Categorical features:", categorical_cols.values)

    print("\nSummary statistics for numerical columns:\n")
    print(x[numerical_cols].describe().to_string())

    # =========================
    # EDA plots
    # =========================
    continuous_cols = [
        "Sleep Hours",
        "Work Hours per Week",
        "Years in Company",
        "Team Size"
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
            linewidth=0.8
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
        ax[i].tick_params(axis="x", rotation=0)

    plt.tight_layout(pad=1.0)
    plt.savefig("Ordinal_barplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Boolean/categorical visualization
    fig = px.pie(
        df,
        names="WFH Setup Available",
        title="WFH Setup Availability"
    )
    fig.write_html("WFH_Setup_Availability.html")

    # =========================
    # Target analysis
    # =========================
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
        linewidth=0.8
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
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            f"{int(yval)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12
        )

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("Class_dist_barplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("Portion of how many are 0 and 1:\n", y_binary.value_counts(normalize=True) * 100)

    boxplot_cols = [
        "Sleep Hours",
        "Work Hours per Week",
        "Resource Allocation",
        "Work-Life Balance Score",
        "Manager Support Score",
        "Deadline Pressure Score",
        "Recognition Frequency"
    ]

    plt.figure(figsize=(15, 7))

    boxprops = dict(linestyle="-", linewidth=2, color="navy")
    medianprops = dict(linestyle="-", linewidth=2, color="firebrick")
    whiskerprops = dict(linestyle="--", linewidth=2, color="black")
    capprops = dict(linestyle="-", linewidth=2, color="grey")

    plt.boxplot(
        [x[col].dropna() for col in boxplot_cols],
        tick_labels=boxplot_cols,
        notch=True,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops
    )

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, axis="y")
    plt.title("Distribution of Selected Features (Boxplots)", fontsize=18, fontweight="bold")
    plt.xlabel("Features", fontsize=16)
    plt.ylabel("Values", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("Employee_Feature_Boxplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Pearson correlation heatmap before modeling
    # =========================
    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    correlation_matrix = x[numerical_cols].corr(method="pearson")

    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmax=1, vmin=-1)
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Pearson Feature Correlation Matrix")

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(
                j, i,
                np.around(correlation_matrix.iloc[i, j], decimals=2),
                ha="center",
                va="center",
                color="w"
            )

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

    # =========================
    # Feature engineering for EARLY DETECTION
    # =========================
    df_fe, df_model, X, y = prepare_model_data(df)

    engineered_corr = df_fe[
        ["Work_Pressure", "Organizational_Support", "Pressure_to_Support", "Burn Rate"]
    ].corr(method="pearson")

    print("\nPearson correlation matrix of engineered features and target:")
    print(engineered_corr)

    plt.figure(figsize=(7, 5))
    plt.imshow(engineered_corr, cmap="coolwarm", vmax=1, vmin=-1)
    plt.colorbar()
    plt.xticks(range(len(engineered_corr.columns)), engineered_corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(engineered_corr.columns)), engineered_corr.columns)
    plt.title("Pearson Correlation: Engineered Features and Burn Rate")

    for i in range(len(engineered_corr.columns)):
        for j in range(len(engineered_corr.columns)):
            plt.text(
                j, i,
                np.around(engineered_corr.iloc[i, j], decimals=2),
                ha="center",
                va="center",
                color="w"
            )

    plt.tight_layout()
    plt.savefig("Correlation_engineered_features_target.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    preprocessor_non_tree, preprocessor_tree, numerical_cols, categorical_cols = build_preprocessors(X)
    models = get_models()
    scoring = get_scoring()

    print("\nFinal modeling features:")
    print(X.columns.tolist())

    outer_split_num = 5
    inner_split_num = 5

    outer_cv = StratifiedKFold(
        n_splits=outer_split_num,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_split_num,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for name, spec in models.items():
        print(f"\nRunning model: {name}")

        if name in ["Random Forest", "XGBoost"]:
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor_tree),
                ("model", spec["model"])
            ])
        else:
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor_non_tree),
                ("model", spec["model"])
            ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            cv=inner_cv,
            scoring="recall",
            n_jobs=1
        )

        cv_results = cross_validate(
            estimator=grid_search,
            X=X,
            y=y,
            cv=outer_cv,
            scoring=scoring,
            return_estimator=True,
            return_train_score=True,
            n_jobs=1
        )

        results[name] = cv_results

        print(f"\n{name}")
        print(f"  F1-Score:  {np.mean(cv_results['test_f1']):.3f} ± {np.std(cv_results['test_f1'], ddof=1):.3f}")
        print(f"  ROC-AUC:   {np.mean(cv_results['test_roc_auc']):.3f} ± {np.std(cv_results['test_roc_auc'], ddof=1):.3f}")
        print(f"  Precision: {np.mean(cv_results['test_precision']):.3f} ± {np.std(cv_results['test_precision'], ddof=1):.3f}")
        print(f"  Recall:    {np.mean(cv_results['test_recall']):.3f} ± {np.std(cv_results['test_recall'], ddof=1):.3f}")
        print(f"  MCC:       {np.mean(cv_results['test_mcc']):.3f} ± {np.std(cv_results['test_mcc'], ddof=1):.3f}")

        print(f"  Train F1:       {np.mean(cv_results['train_f1']):.3f} ± {np.std(cv_results['train_f1'], ddof=1):.3f}")
        print(f"  Train AUC:      {np.mean(cv_results['train_roc_auc']):.3f} ± {np.std(cv_results['train_roc_auc'], ddof=1):.3f}")
        print(f"  Train Precision:{np.mean(cv_results['train_precision']):.3f} ± {np.std(cv_results['train_precision'], ddof=1):.3f}")
        print(f"  Train Recall:   {np.mean(cv_results['train_recall']):.3f} ± {np.std(cv_results['train_recall'], ddof=1):.3f}")
        print(f"  Train MCC:      {np.mean(cv_results['train_mcc']):.3f} ± {np.std(cv_results['train_mcc'], ddof=1):.3f}")

    # =========================
    # Best model
    # =========================
    best_model = max(results, key=lambda m: np.mean(results[m]["test_recall"]))
    best_params = [est.best_params_ for est in results[best_model]["estimator"]]

    print(f"\n{best_model} – best hyperparameters per outer fold:")
    for i, params in enumerate(best_params, 1):
        print(f"  Fold {i}: {params}")

    # =========================
    # Model comparison plot
    # =========================
    models_list = ["Logistic Regression", "Random Forest", "XGBoost"]
    metrics = {
        "F1-score": "test_f1",
        "ROC-AUC": "test_roc_auc",
        "Precision": "test_precision",
        "Recall": "test_recall",
        "MCC": "test_mcc"
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
            alpha=0.85
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
                fontweight="bold"
            )

    # Hide empty subplot
    if len(axes) > len(metrics):
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis("off")

    fig.suptitle(
        "Comparison of Classifier Performance",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Classifier_comparison_multimetric.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # SHAP INTERPRETABILITY
    # =========================
    run_shap_analysis(results, X)


def repeated_nested_cv():
    df = pd.read_csv("enriched_employee_dataset.csv", sep=";")

    _, _, X, y = prepare_model_data(df)

    preprocessor_non_tree, preprocessor_tree, _, _ = build_preprocessors(X)
    models = get_models()
    scoring = get_scoring()

    outer_split_num = 5
    inner_split_num = 5
    n_repeats = 10

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_split_num,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_split_num,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for name, spec in models.items():
        if name in ["Random Forest", "XGBoost"]:
            base_pipe = Pipeline(steps=[
                ("preprocessor", preprocessor_tree),
                ("model", spec["model"])
            ])
        else:
            base_pipe = Pipeline(steps=[
                ("preprocessor", preprocessor_non_tree),
                ("model", spec["model"])
            ])

        grid_search = GridSearchCV(
            estimator=base_pipe,
            param_grid=spec["params"],
            cv=inner_cv,
            scoring="recall",
            n_jobs=1
        )

        cv_results = cross_validate(
            estimator=grid_search,
            X=X,
            y=y,
            cv=outer_cv,
            scoring=scoring,
            return_estimator=False,
            n_jobs=1
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


    # repeated_nested_cv()
