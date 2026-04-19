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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from xgboost import XGBClassifier


def main():
    # =========================
    # Load dataset
    # =========================
    df = pd.read_csv("enriched_employee_dataset.csv", sep=";")

    # Remove Mental Fatigue Score completely
    df = df.drop(columns=["Mental Fatigue Score"])

    print("\nThe shape of dataset is:", df.shape)

    print("\nThe names of the columns are:")
    print(df.columns)

    # Duplicate check
    if df["Employee ID"].nunique() != len(df):
        print("Duplicate employee records detected.")
    else:
        print("Each employee appears only once. No duplicates.")

    print("\nMissing values in each column (full dataset):")
    print(df.isnull().sum())

    # =========================
    # Conservative feature set
    # =========================
    safe_features = [
        "Gender",
        "Company Type",
        "WFH Setup Available",
        "Designation",
        "Years in Company",
        "Team Size",
        "Manager Support Score"
    ]

    df_model = df[safe_features + ["Burn Rate"]].copy()
    df_model = df_model.dropna().copy()

    threshold = 0.45
    df_model["Burn_Rate_Binary"] = (df_model["Burn Rate"] >= threshold).astype(int)

    X = df_model[safe_features].copy()
    y = df_model["Burn_Rate_Binary"].copy()

    print("\nFeatures used in the conservative model:")
    print(X.columns.tolist())

    print("\nClass distribution:")
    print(y.value_counts())
    print("\nClass proportions (%):")
    print(y.value_counts(normalize=True) * 100)

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "string", "bool"]).columns

    print("\nNumerical columns:", numerical_cols.tolist())
    print("Categorical columns:", categorical_cols.tolist())

    # =========================
    # Preprocessing
    # =========================
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

    # =========================
    # Modeling
    # =========================
    random_state = 50
    outer_split_num = 5
    inner_split_num = 5

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=random_state, max_iter=1000),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=random_state),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5],
            }
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=random_state,
                eval_metric="logloss"
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            }
        }
    }

    outer_cv = StratifiedKFold(
        n_splits=outer_split_num,
        shuffle=True,
        random_state=random_state
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_split_num,
        shuffle=True,
        random_state=random_state
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
            scoring=["f1", "roc_auc", "precision", "recall"],
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

        print(f"  Train F1:  {np.mean(cv_results['train_f1']):.3f} ± {np.std(cv_results['train_f1'], ddof=1):.3f}")
        print(f"  Train AUC: {np.mean(cv_results['train_roc_auc']):.3f} ± {np.std(cv_results['train_roc_auc'], ddof=1):.3f}")
        print(f"  Train Precision: {np.mean(cv_results['train_precision']):.3f} ± {np.std(cv_results['train_precision'], ddof=1):.3f}")
        print(f"  Train Recall:    {np.mean(cv_results['train_recall']):.3f} ± {np.std(cv_results['train_recall'], ddof=1):.3f}")

    # =========================
    # Best model
    # =========================
    best_model = max(results, key=lambda m: np.mean(results[m]["test_recall"]))
    best_params = [est.best_params_ for est in results[best_model]["estimator"]]

    print(f"\n{best_model} – best hyperparameters per outer fold:")
    for i, params in enumerate(best_params, 1):
        print(f"  Fold {i}: {params}")


if __name__ == "__main__":
    main()