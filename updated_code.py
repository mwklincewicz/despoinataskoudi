
import warnings
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings(
    action="ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning
)

def main():

    import pandas as pd
    df = pd.read_csv("enriched_employee_dataset.csv", sep=";")

    y = df["Burn Rate"]
    x = df.drop(columns=["Burn Rate"])

    print("\nThe shape of dataset is:", df.shape)

    print("\nThe names of the columns are:")
    print(df.columns)

    # A check for duplicate observations just to see if we have unique registrations
    if df["Employee ID"].nunique() != len(df):
        print("Duplicate employee records detected.")
    else:
        print("Each employee appears only once. No duplicates.")

    print("\nMissing values in each column (full dataset):")
    print(df.isnull().sum())
    # There are missing values in Resource Allocation, Mental Fatigue Score, and Burn Rate.

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

    # Burnout Υ has 4.49%, that is 1,124 rows out of 22,750
    # If we drop them, we still have ~21,600 rows

    # How many rows (observations) in X contain at least one missing value.
    print("The number of employees that have at least 1 missing value is",
          x.isnull().any(axis=1).sum())
    ##3296 this is the 3.51%

    # percentage of observations (rows) with at least one missing feature value
    print("The percentage of rows contain missing values is ", round(
        (x.isnull().any(axis=1).sum() / len(x)) * 100, 2), "%")

    # Τhe number of observations (rows) that contain more than one missing value across the feature set.
    print("The number of employees that have more than one missing value is",
          (x.isnull().sum(axis=1) > 1).sum())
    # 202 means that is less than 1% of dataset

    # Correlation Analysis of Missing Values
    cols_with_missing = x.columns[x.isnull().sum() > 0]

    if len(cols_with_missing) > 1:
        missing_corr = x[cols_with_missing].isnull().corr()

        print("\nCorrelation matrix of missing values:")
        print(missing_corr)
    else:
        print("\nNot enough columns with missing values to compute correlation.")

    # Comments
    # Shows no systematic co-occurrence of missing data across variables (0.046568)
    # There is a LOW Missing CORRELATION BETWEEN THE VARIABLES THAT ARE MISSING TOGETHER
    # That means that If an employee is missing Resource Allocation
    # that does NOT strongly increase the probability that Mental Fatigue Score is also missing.

    print(df.info())
    print("\nData types of each column of the dataset are:", df.dtypes)

    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = x.select_dtypes(include=["object", "bool"]).columns

    print("Continuous (numerical) features:", numerical_cols.values)
    print()
    print("Categorical features:", categorical_cols.values)

    # Statistics for numerical columns
    print("\nSummary statistics for numerical columns:")
    print()
    print(x[numerical_cols].describe().to_string())

# Distribution and sweekness line for continuous
    import numpy as np
    import matplotlib.pyplot as plt

    continuous_cols = [
        "Sleep Hours",
        "Work Hours per Week",
        "Mental Fatigue Score",
        "Years in Company",
        "Team Size"
    ]

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    ax = ax.flatten()

    for i, col in enumerate(continuous_cols):
        s = x[col].dropna()  # x = DataFrame
        mu, sigma = s.mean(), s.std()

        ax[i].hist(s, bins=25, density=True, alpha=0.6,
                   color="steelblue", edgecolor="white", linewidth=0.8)

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
    plt.subplots_adjust(wspace=0.25)
    plt.close()


    # Ordinal Distribution display
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

    # Boolean Variables Display
    import plotly.express as px

    fig = px.pie(
        df,
        names="WFH Setup Available",
        title="WFH Setup Availability"
    )
    fig.show()

    # Analysis of y Target
    y = df["Burn Rate"]
    print("The top of the Target: ", y.head())
    print()
    print(y.describe())

    # Distribution of Burn Rate display
    plt.figure(figsize=(6, 4))
    plt.hist(y, bins=25, density=True, alpha=0.6, edgecolor="black", linewidth=0.8)

    mu = y.mean()
    sigma = y.std()
    x_vals = np.linspace(y.min(), y.max(), 300)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x_vals - mu) ** 2) / (2 * sigma ** 2))
    plt.plot(x_vals, y_norm, linewidth=2, color="darkred")

    plt.title("Distribution of Burn Rate", fontweight="bold")
    plt.xlabel("Burn Rate")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

    # Remove missing values from the target variable before thresholding,
    # so that NaN observations are not incorrectly classified as 0 (Low Risk).
    # The binary target is created only from valid Burn Rate values.

    # Apply a Threshold
    print()
    y_clean = df["Burn Rate"].dropna()
    threshold = 0.45
    y_binary = (y_clean >= threshold).astype(int)
    x_clean = x.loc[y_clean.index]
    print("Threshold (mean Burn Rate):", threshold)
    print(y_binary.head(10))

    # How many are 0 and 1
    print("After the threshold application, "
          "the number of Low to Moderate burnout (zero) and"
          " high burnout (one) is""\n",
          y_binary.value_counts())

    # Μising values in y after preprocessing
    print("Μissing values in y after the threshold application:",
          np.isnan(y).sum())

    # Distribution of the target class
    class_counts = [np.sum(y_binary == 0), np.sum(y_binary == 1)]
    class_labels = ['Low  Risk', 'High Risk']

    plt.figure(figsize=(6, 4))
    bars = plt.bar(class_labels, class_counts, color=['purple', 'pink'], width=0.75)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, axis='y')

    plt.title('Distribution of Target Classes', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            f"{int(yval)}",
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12
        )
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig('Class_dist_barplot.pdf', dpi=300)
    plt.close()

    # Portion of how many are 0 and 1
    print("Portion of how many are 0 and 1:" "\n",
          y_binary.value_counts(normalize=True) * 100)

    # Box plots display
    boxplot_cols = [
        "Sleep Hours",
        "Work Hours per Week",
        "Mental Fatigue Score",
        "Resource Allocation",
        "Work-Life Balance Score",
        "Manager Support Score",
        "Deadline Pressure Score",
        "Recognition Frequency"
    ]

    plt.figure(figsize=(15, 7))

    boxprops = dict(linestyle='-', linewidth=2, color='navy')
    medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
    whiskerprops = dict(linestyle='--', linewidth=2, color='black')
    capprops = dict(linestyle='-', linewidth=2, color='grey')

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

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    plt.title('Distribution of Selected Features (Boxplots)', fontsize=18, fontweight='bold')
    plt.xlabel('Features', fontsize=16)
    plt.ylabel('Values', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig('Employee_Feature_Boxplots.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Only numerical FEATURES (exclude target)
    numerical_cols = x.select_dtypes(include=["int64", "float64"]).columns
    correlation_matrix = x[numerical_cols].corr()

    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmax=1, vmin=-1)
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Feature Correlation Matrix')

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(
                j, i,
                np.around(correlation_matrix.iloc[i, j], decimals=2),
                ha="center", va="center", color="w"
            )

    plt.tight_layout()
    plt.savefig('Correlation_heatmap_before.pdf', dpi=300)
    plt.close()

    # Correlation of Numericals
    corr = x[numerical_cols].corr()
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    corr_pairs = upper_triangle.stack().reset_index()
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]

    # Strong Correlattion
    strong_positive = corr_pairs[corr_pairs["Correlation"] > 0.7]
    print("Strong positive correlations (> 0.7):")
    print(strong_positive, "\n")

    # Strong Negative
    strong_negative = corr_pairs[corr_pairs["Correlation"] < -0.7]
    print("Strong negative correlations (< -0.7):")
    print(strong_negative)

    ###### Feature Engineering based on the Job Demands–Resources model

    df_fe = df.copy()

    df_fe["Work_Pressure"] = (
        df_fe["Work Hours per Week"] +
        df_fe["Deadline Pressure Score"] +
        df_fe["Mental Fatigue Score"]
    )

    df_fe["Well_Being"] = (
        df_fe["Recognition Frequency"] +
        df_fe["Work-Life Balance Score"] +
        df_fe["Sleep Hours"]
    )

    ########## Preprocessing steps###

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
    from xgboost import XGBClassifier

    # Use of feature-engineered dataframe df_fe
    threshold = 0.45

    df_model = df_fe.drop(columns=["Employee ID", "Date of Joining"]).copy()
    df_model = df_model.dropna().copy()

    df_model["Burn_Rate_Binary"] = (df_model["Burn Rate"] >= threshold).astype(int)

    X = df_model.drop(columns=["Burn Rate", "Burn_Rate_Binary"])
    y = df_model["Burn_Rate_Binary"]

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "string", "bool"]).columns

#############preprocessing###################
    # Numerical: scaling only
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Categorical: one-hot only
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Preprocessor for non-tree models
    preprocessor_non_tree = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    # Preprocessor for tree-based models
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

################################################################33
    # Model evaluation

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
            scoring="recall", ############################# # Recall is used as the optimization metric in GridSearchCV,
                                 # since the primary objective is to minimize false negatives
                                # and ensure high detection of high burnout risk employees.
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

############################# Best Model###############################333

    # Best Model (based on Recall)
    best_model = max(results, key=lambda m: np.mean(results[m]["test_recall"]))
    best_params = [est.best_params_ for est in results[best_model]["estimator"]]

    print(f"\n{best_model} – best hyperparameters per outer fold:")
    for i, params in enumerate(best_params, 1):
        print(f"  Fold {i}: {params}")

    # Comparison of Classifier Performance
    models_list = ["Logistic Regression", "Random Forest", "XGBoost"]
    metrics = {
        "F1-score": "test_f1",
        "ROC-AUC": "test_roc_auc",
        "Precision": "test_precision",
        "Recall": "test_recall"
    }

    means = {
        metric: [np.mean(results[m][key]) for m in models_list]
        for metric, key in metrics.items()
    }
    stds = {
        metric: [np.std(results[m][key], ddof=1) for m in models_list]
        for metric, key in metrics.items()
    }

    colors = plt.cm.viridis(np.linspace(start=0.2, stop=0.8, num=len(models_list)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric_name, _) in zip(axes, metrics.items()):

        bars = ax.bar(
            models_list,
            means[metric_name],
            yerr=stds[metric_name],
            capsize=6,
            color=colors,
            alpha=0.85
        )

        ax.set_title(metric_name, fontsize=14, fontweight="bold")
        ax.set_ylim(0.5, max(means[metric_name]) + 0.1)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        ax.tick_params(axis="x", labelrotation=25)
        ax.tick_params(axis="both", labelsize=11)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for bar, mean_val in zip(bars, means[metric_name]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_val - 0.1,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold"
            )

    fig.suptitle(
        "Comparison of Classifier Performance",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("Classifier_comparison_multimetric.pdf", dpi=300)
    plt.close()



    from sklearn.ensemble import BaggingClassifier

    ################# LOGISTIC REGRESSION ROBUST FEATURE IMPORTANCE ##################

    # Use the same preprocessor structure as Logistic Regression
    X_lr = X.copy()
    y_lr = y.copy()

    # Fit the non-tree preprocessor on the full final dataset
    X_lr_transformed = preprocessor_non_tree.fit_transform(X_lr, y_lr)

    # Get transformed feature names
    lr_feature_names = preprocessor_non_tree.get_feature_names_out()

    # Bagging Logistic Regression
    n_estimators = 100

    bagged_model = BaggingClassifier(
        estimator=LogisticRegression(random_state=random_state, max_iter=1000),
        n_estimators=n_estimators,
        random_state=random_state
    )

    bagged_model.fit(X_lr_transformed, y_lr)

    # Collect coefficients from all bagged estimators
    coefficients = []
    for est in bagged_model.estimators_:
        coefficients.append(est.coef_.ravel())

    coefficients = np.vstack(coefficients)

    # Robust feature importance = mean coefficient / standard error
    feature_importance = np.mean(coefficients, axis=0) / (
        np.std(coefficients, axis=0, ddof=1) / np.sqrt(n_estimators)
    )

    # Create DataFrame
    lr_fi_df = pd.DataFrame({
        "feature": lr_feature_names,
        "importance": feature_importance
    })

    # Sort by absolute importance
    lr_fi_df = lr_fi_df.reindex(
        lr_fi_df["importance"].abs().sort_values(ascending=False).index
    )

    # Plot top features
    top_k = 15
    lr_plot_df = lr_fi_df.head(top_k).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(
        lr_plot_df["feature"],
        lr_plot_df["importance"]
    )
    plt.xlabel("Robust coefficient importance")
    plt.ylabel("Feature")
    plt.title("Logistic Regression Feature Importance (Bagging-based)")
    plt.tight_layout()
    plt.savefig("Logistic_bagging_feature_importance.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Print top features
    print("\nTop Logistic Regression features (bagging-based importance):")
    print(lr_fi_df.head(15).to_string(index=False))

    ################# LOGISTIC REGRESSION ODDS RATIOS ##################

    mean_coef = np.mean(coefficients, axis=0)
    odds_ratios = np.exp(mean_coef)

    # Create dataframe
    odds_df = pd.DataFrame({
        "Feature": lr_feature_names,
        "odds_ratios": odds_ratios
    })

    # Sort by strength of effect (distance from OR=1)
    odds_df["effect_strength"] = np.abs(np.log(odds_df["odds_ratios"]))
    odds_df = odds_df.sort_values("effect_strength", ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(odds_df["Feature"], odds_df["odds_ratios"])
    plt.axhline(1, linestyle="--", color="red")  # No-effect line

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Feature")
    plt.ylabel("Odds Ratio")
    plt.title("Odds Ratios in Logistic Regression")

    # Log scale is more appropriate for odds ratios
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("Logistic_odds_ratios.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nTop Logistic Regression features by odds ratio effect strength:")
    print(odds_df.head(15).to_string(index=False))

    ################# GLOBAL FEATURE IMPORTANCE FOR TREE MODELS ##################

    def get_transformed_feature_names(preprocessor):
        return preprocessor.get_feature_names_out()

    def collapse_onehot_mean(df_imp, original_columns):
        df_imp = df_imp.copy()
        df_imp["feature_clean"] = df_imp["feature"].str.replace(r"^(num__|cat__)", "", regex=True)

        def get_base_feature(name):
            for col in original_columns:
                if name == col or name.startswith(col + "_"):
                    return col
            return name

        df_imp["base_feature"] = df_imp["feature_clean"].apply(get_base_feature)

        return df_imp.groupby("base_feature")["importance"].mean()

    def rf_importance(pipe):
        rf = pipe.named_steps["model"]
        pre = pipe.named_steps["preprocessor"]

        return pd.DataFrame({
            "feature": get_transformed_feature_names(pre),
            "importance": rf.feature_importances_
        })

    def xgb_importance(pipe):
        xgb = pipe.named_steps["model"]
        pre = pipe.named_steps["preprocessor"]

        return pd.DataFrame({
            "feature": get_transformed_feature_names(pre),
            "importance": xgb.feature_importances_
        })

    def importance_across_folds(results_dict, model_name, importance_fn, original_columns):
        fold_importances = []

        for grid in results_dict[model_name]["estimator"]:
            pipe = grid.best_estimator_
            df_imp = importance_fn(pipe)
            collapsed = collapse_onehot_mean(df_imp, original_columns)
            fold_importances.append(collapsed)

        return fold_importances

    def aggregate_importances(importances):
        combined = pd.concat(importances, axis=1)

        return pd.DataFrame({
            "importance_mean": combined.mean(axis=1),
            "importance_std": combined.std(axis=1, ddof=1)
        }).sort_values("importance_mean", ascending=False)

    def plot_feature_importances_subplots(importance_dict, top_k=10):
        n_models = len(importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(14, 8), sharex=False)

        if n_models == 1:
            axes = [axes]

        for ax, (model_name, df_imp) in zip(axes, importance_dict.items()):
            df_plot = df_imp.head(top_k).iloc[::-1]

            ax.barh(
                df_plot.index,
                df_plot["importance_mean"],
                xerr=df_plot["importance_std"],
                color="steelblue",
                alpha=0.85,
                capsize=5
            )

            ax.set_title(model_name, fontsize=14, fontweight="bold")
            ax.set_xlabel("Mean importance")
            ax.grid(axis="x", linestyle="--", alpha=0.6)

            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle(
            "Global Feature Importance Across Tree-Based Models\n"
            "(Mean ± variability across outer CV folds)",
            fontsize=16,
            fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig("Tree_model_feature_importances_subplots.pdf", dpi=300)
        plt.close()

    rf_fold_importances = importance_across_folds(
        results, "Random Forest", rf_importance, X.columns
    )
    xgb_fold_importances = importance_across_folds(
        results, "XGBoost", xgb_importance, X.columns
    )

    importance_dict = {
        "Random Forest": aggregate_importances(rf_fold_importances),
        "XGBoost": aggregate_importances(xgb_fold_importances)
    }

    plot_feature_importances_subplots(importance_dict, top_k=10)


############################################################################3
import numpy as np


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import (
        StratifiedKFold,
        RepeatedStratifiedKFold,
        GridSearchCV,
        cross_validate
    )


def repeated_nested_cv():
    import pandas as pd
    # Load data
    df = pd.read_csv("enriched_employee_dataset.csv", sep=";")

    # Preprocessing (dropna baseline)

    ###########Feature engineering based on the Job Demands and Resources literature
    df_fe = df.copy()

    df_fe["Work_Pressure"] = (
        df_fe["Work Hours per Week"] +
        df_fe["Deadline Pressure Score"] +
        df_fe["Mental Fatigue Score"]
    )

    df_fe["Well_Being"] = (
        df_fe["Recognition Frequency"] +
        df_fe["Work-Life Balance Score"] +
        df_fe["Sleep Hours"]
    )

    df_model = df_fe.drop(columns=["Employee ID", "Date of Joining"])
    df_model = df_model.dropna()

    X = df_model.drop(columns=["Burn Rate"])
    threshold = 0.45
    y = (df_model["Burn Rate"] >= threshold).astype(int)

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

    # Models + grids

    random_state = 50

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=random_state, max_iter=1000),
            "params": {"model__C": [0.1, 1.0, 10.0]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=random_state),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5]
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
                "model__max_depth": [3, 5]
            }
        }
    }

    # Repeated Nested CV hahahaha

    outer_split_num = 5
    inner_split_num = 5
    n_repeats = 50 ################3

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_split_num,
        n_repeats=n_repeats,
        random_state=random_state
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_split_num,
        shuffle=True,
        random_state=random_state
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
            scoring=["f1", "roc_auc", "precision", "recall"],
            return_estimator=True,
            n_jobs=1
        )

        results[name] = cv_results
        n_outer = len(cv_results["test_roc_auc"])

        print(f"\n{name}")
        print(f"  F1-Score:  {np.mean(cv_results['test_f1']):.3f} ± {np.std(cv_results['test_f1'], ddof=1):.3f} (n={n_outer})")
        print(f"  ROC-AUC:   {np.mean(cv_results['test_roc_auc']):.3f} ± {np.std(cv_results['test_roc_auc'], ddof=1):.3f} (n={n_outer})")
        print(f"  Precision: {np.mean(cv_results['test_precision']):.3f} ± {np.std(cv_results['test_precision'], ddof=1):.3f} (n={n_outer})")
        print(f"  Recall:    {np.mean(cv_results['test_recall']):.3f} ± {np.std(cv_results['test_recall'], ddof=1):.3f} (n={n_outer})")


print()

if __name__ == "__main__":
    main()
    repeated_nested_cv()