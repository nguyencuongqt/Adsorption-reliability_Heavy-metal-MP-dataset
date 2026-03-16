from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import STUDY_ROOT, load_config
from .data import (
    add_null_features,
    build_model_features,
    build_dataset_structure_table,
    build_feature_dictionary,
    dataset_hash,
    get_real_feature_columns,
    load_dataset,
)
from .importance import (
    aggregate_feature_importance,
    aggregate_mechanism_importance,
    elastic_net_coef_importance,
    lightgbm_builtin_importance,
    lightgbm_shap_importance,
    permutation_importance_frame,
    summarize_method_reliability,
)
from .models import fit_model
from .reporting import (
    plot_mechanism_heatmap,
    plot_null_benchmark,
    plot_stability_heatmap,
    plot_validation_hierarchy,
    save_table,
)
from .validation import build_generalization_gap_table, build_splits, regression_metrics, summarize_metrics


def run_rq1() -> dict[str, str]:
    config = load_config()
    results_dir = STUDY_ROOT / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    logs_dir = results_dir / "logs"
    for path in [results_dir, tables_dir, figures_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(config)
    real_features = get_real_feature_columns(df, config)
    feature_df = build_model_features(df)
    y = df[config.target_column].to_numpy(dtype=float)
    splits = build_splits(df, config)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for split_number, split in enumerate(splits, start=1):
        X_train = feature_df.iloc[split.train_index]
        X_test = feature_df.iloc[split.test_index]
        y_train = y[split.train_index]
        y_test = y[split.test_index]

        for model_offset, model_name in enumerate(config.rq1_models):
            fitted = fit_model(model_name, X_train, y_train, random_state=config.random_state + split_number + model_offset)
            y_pred = fitted.predict(X_test)
            metrics = regression_metrics(y_test, y_pred)
            metric_rows.append(
                {
                    "regime": split.regime,
                    "split_id": split.split_id,
                    "model": model_name,
                    **metrics,
                    "n_train": int(len(split.train_index)),
                    "n_test": int(len(split.test_index)),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "regime": split.regime,
                        "split_id": split.split_id,
                        "model": model_name,
                        "row_index": split.test_index,
                        "aut_id": df.iloc[split.test_index]["aut_id"].to_numpy(),
                        "exp_id": df.iloc[split.test_index]["exp_id"].to_numpy(),
                        "y_true": y_test,
                        "y_pred": y_pred,
                    }
                )
            )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    summary_df = summarize_metrics(metrics_df)
    gap_df = build_generalization_gap_table(summary_df)
    structure_df = build_dataset_structure_table(df, real_features)
    dictionary_df = build_feature_dictionary(real_features, config.feature_inventory_path)

    save_table(structure_df, tables_dir / "table_dataset_grouped_structure.csv")
    save_table(metrics_df, tables_dir / "rq1_fold_metrics.csv")
    save_table(summary_df, tables_dir / "table_performance_across_validation_regimes.csv")
    save_table(gap_df, tables_dir / "table_generalization_gaps.csv")
    save_table(dictionary_df, tables_dir / "feature_dictionary.csv")
    save_table(predictions_df, tables_dir / "rq1_out_of_fold_predictions.csv")
    plot_validation_hierarchy(summary_df, figures_dir / "figure_validation_hierarchy_performance_drop.png")

    manifest = {
        "study": "adsorption_reliability_study",
        "question": "RQ1",
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": dataset_hash(config.dataset_path),
        "n_rows": int(len(df)),
        "n_features": int(len(real_features)),
        "models": config.rq1_models,
        "regimes": sorted(pd.unique(metrics_df["regime"])),
    }
    (logs_dir / "rq1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in {"metrics": tables_dir / "table_performance_across_validation_regimes.csv", "gaps": tables_dir / "table_generalization_gaps.csv"}.items()}


def run_rq2() -> dict[str, str]:
    config = load_config()
    results_dir = STUDY_ROOT / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    logs_dir = results_dir / "logs"
    for path in [results_dir, tables_dir, figures_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(config)
    df_with_nulls, null_features = add_null_features(df, config)
    real_features = get_real_feature_columns(df, config)
    null_df = df_with_nulls[null_features].copy()
    X = build_model_features(df_with_nulls, null_features_df=null_df)
    y = df_with_nulls[config.target_column].to_numpy(dtype=float)
    splits = build_splits(df_with_nulls, config)

    importance_rows: list[pd.DataFrame] = []
    regime_counts: dict[str, int] = {}
    for split_number, split in enumerate(splits, start=1):
        if split.regime == "leave_one_study_out":
            continue
        if regime_counts.get(split.regime, 0) >= 5:
            continue
        regime_counts[split.regime] = regime_counts.get(split.regime, 0) + 1
        X_train = X.iloc[split.train_index]
        X_test = X.iloc[split.test_index]
        y_train = y[split.train_index]
        y_test = y[split.test_index]

        for model_offset, model_name in enumerate(config.rq2_models):
            fitted = fit_model(model_name, X_train, y_train, random_state=config.random_state + 1000 + split_number + model_offset)
            method_frames: list[tuple[str, pd.DataFrame]] = []
            if model_name == "elastic_net":
                method_frames.append(("elastic_net_coef", elastic_net_coef_importance(fitted)))
                method_frames.append(("elastic_net_permutation", permutation_importance_frame(fitted, X_test, y_test, random_state=config.random_state + split_number)))
            elif model_name == "lightgbm":
                method_frames.append(("lightgbm_permutation", permutation_importance_frame(fitted, X_test, y_test, random_state=config.random_state + split_number)))
                method_frames.append(("lightgbm_builtin_gain", lightgbm_builtin_importance(fitted)))
                method_frames.append(("lightgbm_shap", lightgbm_shap_importance(fitted, X_test)))
            for method_name, frame in method_frames:
                frame = frame.assign(regime=split.regime, split_id=split.split_id, model=model_name, method=method_name)
                importance_rows.append(frame)

    importances_df = pd.concat(importance_rows, ignore_index=True)
    feature_summary = aggregate_feature_importance(importances_df)
    mechanism_summary = aggregate_mechanism_importance(importances_df)
    reliability_summary, split_reliability, stability_df = summarize_method_reliability(importances_df)

    save_table(importances_df, tables_dir / "rq2_importance_by_split.csv")
    save_table(feature_summary, tables_dir / "rq2_feature_importance_summary.csv")
    save_table(mechanism_summary, tables_dir / "rq2_mechanism_group_importance.csv")
    save_table(reliability_summary, tables_dir / "table_feature_importance_method_reliability.csv")
    save_table(split_reliability, tables_dir / "rq2_split_level_reliability.csv")
    save_table(stability_df, tables_dir / "rq2_stability_by_method_regime.csv")

    dictionary_df = build_feature_dictionary(real_features, config.feature_inventory_path, null_features=null_features)
    save_table(dictionary_df, tables_dir / "rq2_feature_dictionary_with_nulls.csv")

    plot_mechanism_heatmap(mechanism_summary, figures_dir / "figure_mechanism_group_importance_across_methods.png")
    plot_null_benchmark(reliability_summary, figures_dir / "figure_null_feature_benchmarking.png")
    plot_stability_heatmap(stability_df, figures_dir / "figure_importance_stability_heatmap.png")

    manifest = {
        "study": "adsorption_reliability_study",
        "question": "RQ2",
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": dataset_hash(config.dataset_path),
        "n_rows": int(len(df_with_nulls)),
        "n_real_features": int(len(real_features)),
        "n_null_features": int(len(null_features)),
        "models": config.rq2_models,
        "methods": sorted(pd.unique(importances_df["method"])),
        "regimes": sorted(pd.unique(importances_df["regime"])),
    }
    (logs_dir / "rq2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in {"reliability": tables_dir / "table_feature_importance_method_reliability.csv", "mechanism": tables_dir / "rq2_mechanism_group_importance.csv"}.items()}
