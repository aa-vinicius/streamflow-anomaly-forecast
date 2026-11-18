from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor


warnings.filterwarnings("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--outlet", default="19094")
parser.add_argument("--h", type=int, required=True)
parser.add_argument(
    "--data-root",
    type=Path,
    default=Path("data"),
    help="Private local data directory (never committed).",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("outputs"),
    help="Directory for generated models, predictions and diagnostics.",
)
args = parser.parse_args()

OUTLET = str(args.outlet)
H = int(args.h)
if OUTLET not in {"19094", "19097"}:
    raise ValueError("--outlet must be 19094 or 19097")
if H not in {1, 2, 3, 4, 5}:
    raise ValueError("--h must be an integer from 1 to 5")

SOURCE = args.data_root / "processed" / OUTLET / f"matriz_completa_{OUTLET}_Dmais{H}.parquet"
OUTPUT = args.output_root / OUTLET / f"Dmais{H}"
MODELS = OUTPUT / "modelos"
if not SOURCE.exists():
    raise FileNotFoundError(
        f"Private modelling matrix not found: {SOURCE}. "
        "See DATA_ACCESS.md; data files must remain outside version control."
    )
OUTPUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 20261117
TARGET = "alvo_A5_m3s"
VALIDATION_YEARS = [2020, 2021, 2022]


def continuous_metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    corr = float(pearsonr(y, pred).statistic) if len(y) > 1 and np.std(y) > 0 and np.std(pred) > 0 else np.nan
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "bias": float(np.mean(pred - y)),
        "correlation": corr,
        "r2": float(r2_score(y, pred)),
    }


def binary_metrics(y, signal):
    y = np.asarray(y, bool)
    signal = np.asarray(signal, bool)
    precision, recall, f1, _ = precision_recall_fscore_support(y, signal, average="binary", zero_division=0)
    tp = int(np.sum(y & signal)); fp = int(np.sum(~y & signal))
    fn = int(np.sum(y & ~signal)); tn = int(np.sum(~y & ~signal))
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def select_threshold(y, prob, min_precision=0.70):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        m = binary_metrics(y, prob >= threshold)
        rows.append({"threshold": float(threshold), **m})
    eligible = [r for r in rows if r["precision"] >= min_precision and r["tp"] > 0]
    if eligible:
        best = max(eligible, key=lambda r: (r["recall"], r["f1"], r["precision"]))
        rule = f"maximizar recall com precisao >= {min_precision:.2f}"
    else:
        best = max(rows, key=lambda r: (r["f1"], r["recall"], r["precision"]))
        rule = "maximizar F1; restricao de precisao nao atingida"
    return best, rule


def temporal_mask(frame, start, end, window_start=None, window_end=None):
    mask = frame["emissao_D"].between(pd.Timestamp(start), pd.Timestamp(end))
    if window_start is not None:
        mask &= frame["inicio_janela"].ge(pd.Timestamp(window_start))
    if window_end is not None:
        mask &= frame["fim_janela"].le(pd.Timestamp(window_end))
    return mask & frame["status_linha"].eq("OK")


def add_derived(frame):
    frame = frame.copy()
    frame["A_persistencia_m3s"] = frame["Q_L_m3s"] - frame["C5_referencia_m3s"]
    frame["dQ_1d_m3s"] = frame["Q_L_m3s"] - frame["Q_Lm1_m3s"]
    frame["dQ_3d_m3s"] = frame["Q_L_m3s"] - frame["Q_Lm3_m3s"]
    frame["dQ_7d_m3s"] = frame["Q_L_m3s"] - frame["Q_Lm7_m3s"]
    frame["slope_Q_3d_m3s_d"] = frame["dQ_3d_m3s"] / 3.0
    frame["slope_Q_7d_m3s_d"] = frame["dQ_7d_m3s"] / 7.0
    qcols = ["Q_L_m3s", "Q_Lm1_m3s", "Q_Lm3_m3s", "Q_Lm7_m3s"]
    frame["Q_media_antecedente_m3s"] = frame[qcols].mean(axis=1)
    frame["Q_amplitude_antecedente_m3s"] = frame[qcols].max(axis=1) - frame[qcols].min(axis=1)
    frame["A_persistencia_menos_P80_m3s"] = frame["A_persistencia_m3s"] - frame["P80_A5_m3s"]
    doy = frame["centro_t"].dt.dayofyear.to_numpy()
    frame["sazonal_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["sazonal_cos"] = np.cos(2 * np.pi * doy / 365.25)
    frame["residuo_persistencia_m3s"] = frame[TARGET] - frame["A_persistencia_m3s"]
    frame["evento_P80"] = frame[TARGET] > frame["P80_A5_m3s"]
    return frame


REG_GRID = {
    "ELASTIC_NET": [
        {"alpha": a, "l1_ratio": l} for a, l in [(0.01, 0.5), (0.1, 0.5), (0.1, 0.9), (1.0, 0.5)]
    ],
    "SVR_RBF": [
        {"C": c, "epsilon": e, "gamma": g}
        for c, e, g in [(50.0, 5.0, "scale"), (100.0, 20.0, "scale"), (500.0, 5.0, 0.03), (500.0, 20.0, 0.03)]
    ],
    "EXTRA_TREES": [
        {"max_depth": d, "min_samples_leaf": l, "max_features": m}
        for d, l, m in [(None, 1, 1.0), (None, 3, 0.7), (12, 1, 1.0), (20, 3, 0.7)]
    ],
    "XGBOOST": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "min_child_weight": mc}
        for n, d, lr, mc in [(300, 2, 0.04, 1), (400, 3, 0.05, 1), (300, 3, 0.04, 5), (450, 2, 0.04, 5)]
    ],
}


def make_regressor(name, params):
    if name == "ELASTIC_NET":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=50000, random_state=RANDOM_STATE)),
        ])
    if name == "SVR_RBF":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(C=params["C"], epsilon=params["epsilon"], gamma=params["gamma"], cache_size=1000)),
        ])
    if name == "EXTRA_TREES":
        return ExtraTreesRegressor(
            n_estimators=350, max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"], random_state=RANDOM_STATE, n_jobs=-1,
        )
    if name == "XGBOOST":
        return XGBRegressor(
            objective="reg:squarederror", n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], min_child_weight=params["min_child_weight"],
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=4, tree_method="hist",
        )
    raise ValueError(name)


def fit_regressor(model, name, X, y, sample_weight):
    if isinstance(model, Pipeline):
        model.fit(X, y, model__sample_weight=sample_weight)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


CLF_GRID = {
    "LOGISTIC": [{"C": c} for c in [0.1, 1.0, 10.0]],
    "EXTRA_TREES_CLF": [
        {"max_depth": d, "min_samples_leaf": l, "max_features": m}
        for d, l, m in [(None, 1, "sqrt"), (None, 3, 0.7), (12, 1, 0.7), (20, 3, "sqrt")]
    ],
    "XGBOOST_CLF": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n, d, lr in [(250, 2, 0.04), (350, 3, 0.04), (400, 2, 0.06)]
    ],
}


def make_classifier(name, params):
    if name == "LOGISTIC":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=params["C"], class_weight="balanced", max_iter=10000, random_state=RANDOM_STATE)),
        ])
    if name == "EXTRA_TREES_CLF":
        return ExtraTreesClassifier(
            n_estimators=400, max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"], class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
    if name == "XGBOOST_CLF":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_estimators=params["n_estimators"],
            max_depth=params["max_depth"], learning_rate=params["learning_rate"],
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, scale_pos_weight=4.5,
            random_state=RANDOM_STATE, n_jobs=4, tree_method="hist",
        )
    raise ValueError(name)


df = pd.read_parquet(SOURCE)
for col in ["emissao_D", "centro_t", "inicio_janela", "fim_janela", "data_L_assumida"]:
    df[col] = pd.to_datetime(df[col])
df = add_derived(df)
wrf_cols = [c for c in df.columns if c.startswith("WRF__")]

hydro_features = [
    "Q_L_m3s", "Q_Lm1_m3s", "Q_Lm3_m3s", "Q_Lm7_m3s", "C5_referencia_m3s",
    "A_persistencia_m3s", "dQ_1d_m3s", "dQ_3d_m3s", "dQ_7d_m3s",
    "slope_Q_3d_m3s_d", "slope_Q_7d_m3s_d", "Q_media_antecedente_m3s",
    "Q_amplitude_antecedente_m3s", "A_persistencia_menos_P80_m3s", "sazonal_sin", "sazonal_cos",
]
feature_sets = {"HD": hydro_features, "HDW": hydro_features + wrf_cols}

folds = []
oof_parts = []
for year in VALIDATION_YEARS:
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    val_start = pd.Timestamp(f"{year}-01-01")
    val_end = pd.Timestamp(f"{year}-12-31")
    train_idx = df.index[temporal_mask(df, "2018-01-01", train_end, "2018-01-01", train_end)].to_numpy()
    val_idx = df.index[temporal_mask(df, val_start, val_end, val_start, val_end)].to_numpy()
    folds.append({"year": year, "train_idx": train_idx, "val_idx": val_idx})
    part = df.loc[val_idx, ["emissao_D", "centro_t", TARGET, "P20_A5_m3s", "P80_A5_m3s", "A_persistencia_m3s", "evento_P80"]].copy()
    part["validation_year"] = year
    oof_parts.append(part)
oof = pd.concat(oof_parts, ignore_index=True)
y_oof = oof[TARGET].to_numpy(float)
base_oof = oof["A_persistencia_m3s"].to_numpy(float)
p80_oof = oof["P80_A5_m3s"].to_numpy(float)
event_oof = oof["evento_P80"].to_numpy(bool)

reg_rows = []
reg_predictions = {}
for model_name, grid in REG_GRID.items():
    for feature_name, features in feature_sets.items():
        for weight_high in [1.0, 3.0]:
            for params in grid:
                parts = []
                for fold in folds:
                    train = df.loc[fold["train_idx"]]
                    valid = df.loc[fold["val_idx"]]
                    model = make_regressor(model_name, params)
                    sw = np.where(train["evento_P80"].to_numpy(bool), weight_high, 1.0)
                    fit_regressor(model, model_name, train[features], train["residuo_persistencia_m3s"], sw)
                    residual_pred = model.predict(valid[features])
                    parts.append(valid["A_persistencia_m3s"].to_numpy(float) + residual_pred)
                raw = np.concatenate(parts)
                bias = float(np.mean(raw - y_oof))
                corrected = raw - bias
                key = f"{model_name}__{feature_name}__W{weight_high}__{json.dumps(params, sort_keys=True)}"
                reg_predictions[key] = corrected
                cm = continuous_metrics(y_oof, corrected)
                bm = binary_metrics(event_oof, corrected > p80_oof)
                reg_rows.append({
                    "candidate": key, "model": model_name, "feature_set": feature_name,
                    "weight_high": weight_high, "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "bias_oof": bias, **cm,
                    "p80_precision": bm["precision"], "p80_recall": bm["recall"], "p80_f1": bm["f1"],
                })

reg_df = pd.DataFrame(reg_rows).sort_values(["mae", "p80_f1"], ascending=[True, False])
reg_df.to_csv(OUTPUT / "selecao_regressao_validacao.csv", index=False, encoding="utf-8-sig")
best_row = reg_df.iloc[0]
best_key = best_row["candidate"]
best_pred = reg_predictions[best_key]

blend_rows = []
blend_preds = {}
for w in np.linspace(0, 1, 21):
    pred = base_oof + w * (best_pred - base_oof)
    bias = float(np.mean(pred - y_oof))
    pred = pred - bias
    cm = continuous_metrics(y_oof, pred)
    bm = binary_metrics(event_oof, pred > p80_oof)
    blend_rows.append({"weight_model": float(w), "bias_oof": bias, **cm, **{f"p80_{k}": v for k, v in bm.items()}})
    blend_preds[float(w)] = pred
blend_df = pd.DataFrame(blend_rows).sort_values(["mae", "p80_f1"], ascending=[True, False])
blend_df.to_csv(OUTPUT / "selecao_peso_hibrido_validacao.csv", index=False, encoding="utf-8-sig")
best_blend = blend_df.iloc[0]
blend_weight = float(best_blend["weight_model"])
winner_oof = blend_preds[blend_weight]

abs_res = np.abs(y_oof - winner_oof)
def conformal_q(values):
    values = np.asarray(values, float)
    level = min(math.ceil((len(values) + 1) * 0.90) / len(values), 1.0)
    return float(np.quantile(values, level, method="higher"))

q_global = conformal_q(abs_res)
pred_high_oof = winner_oof > p80_oof
q_high = conformal_q(abs_res[pred_high_oof]) if pred_high_oof.sum() >= 30 else q_global
q_other = conformal_q(abs_res[~pred_high_oof]) if (~pred_high_oof).sum() >= 30 else q_global

# Classificador probabilístico direto para P80.
clf_rows = []
clf_probs = {}
for clf_name, grid in CLF_GRID.items():
    for feature_name, features in feature_sets.items():
        for params in grid:
            parts = []
            for fold in folds:
                train = df.loc[fold["train_idx"]]
                valid = df.loc[fold["val_idx"]]
                clf = make_classifier(clf_name, params)
                clf.fit(train[features], train["evento_P80"].astype(int))
                parts.append(clf.predict_proba(valid[features])[:, 1])
            raw_prob = np.concatenate(parts)
            key = f"{clf_name}__{feature_name}__{json.dumps(params, sort_keys=True)}"
            clf_probs[key] = raw_prob
            threshold_raw, _ = select_threshold(event_oof, raw_prob)
            clf_rows.append({
                "candidate": key, "model": clf_name, "feature_set": feature_name,
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "brier_raw": float(brier_score_loss(event_oof, raw_prob)),
                "threshold_raw": threshold_raw["threshold"],
                "precision_raw": threshold_raw["precision"], "recall_raw": threshold_raw["recall"], "f1_raw": threshold_raw["f1"],
            })
clf_df = pd.DataFrame(clf_rows).sort_values(["f1_raw", "brier_raw"], ascending=[False, True])
clf_df.to_csv(OUTPUT / "selecao_classificador_validacao.csv", index=False, encoding="utf-8-sig")
best_clf_row = clf_df.iloc[0]
best_clf_key = best_clf_row["candidate"]
best_prob_raw = clf_probs[best_clf_key]
isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
isotonic.fit(best_prob_raw, event_oof.astype(int))
prob_oof = isotonic.transform(best_prob_raw)
threshold_info, threshold_rule = select_threshold(event_oof, prob_oof, min_precision=0.70)

# Congelamento antes de abrir 2023-2024 para comparação de desenvolvimento.
best_model_name = str(best_row["model"])
best_feature_name = str(best_row["feature_set"])
best_features = feature_sets[best_feature_name]
best_params = json.loads(best_row["params"])
best_weight_high = float(best_row["weight_high"])
best_model_bias = float(best_row["bias_oof"])
blend_bias = float(best_blend["bias_oof"])

best_clf_name = str(best_clf_row["model"])
best_clf_feature_name = str(best_clf_row["feature_set"])
best_clf_features = feature_sets[best_clf_feature_name]
best_clf_params = json.loads(best_clf_row["params"])

frozen = {
    "version": "V2_HIBRIDO_TODOS_OS_DIAS",
    "created_before_development_comparison": True,
    "outlet": OUTLET, "h": H,
    "training_period": "2018-2022", "validation_years": VALIDATION_YEARS,
    "development_comparison_period": "2023-2024", "final_reserved_period": "2025",
    "regression": {
        "model": best_model_name, "feature_set": best_feature_name, "features": best_features,
        "params": best_params, "weight_high": best_weight_high,
        "model_bias_oof_m3s": best_model_bias, "blend_weight_model": blend_weight,
        "blend_bias_oof_m3s": blend_bias,
    },
    "classifier_P80": {
        "model": best_clf_name, "feature_set": best_clf_feature_name, "features": best_clf_features,
        "params": best_clf_params, "threshold": threshold_info["threshold"], "threshold_rule": threshold_rule,
        "validation_metrics": threshold_info,
    },
    "PI90": {"q_global": q_global, "q_predicted_high": q_high, "q_other": q_other},
}
(OUTPUT / "SELECAO_CONGELADA_ANTES_DESENVOLVIMENTO.json").write_text(
    json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
)

train_idx = df.index[temporal_mask(df, "2018-01-01", "2022-12-31", "2018-01-01", "2022-12-31")]
dev_idx = df.index[temporal_mask(df, "2023-01-01", "2024-12-31", "2023-01-01", "2025-12-31")]
train = df.loc[train_idx].copy()
dev = df.loc[dev_idx].copy()

reg_model = make_regressor(best_model_name, best_params)
sw = np.where(train["evento_P80"].to_numpy(bool), best_weight_high, 1.0)
fit_regressor(reg_model, best_model_name, train[best_features], train["residuo_persistencia_m3s"], sw)
residual_dev = reg_model.predict(dev[best_features])
hybrid_dev = dev["A_persistencia_m3s"].to_numpy(float) + residual_dev - best_model_bias
winner_dev_prebias = dev["A_persistencia_m3s"].to_numpy(float) + blend_weight * (hybrid_dev - dev["A_persistencia_m3s"].to_numpy(float))
winner_dev = winner_dev_prebias - blend_bias

clf_model = make_classifier(best_clf_name, best_clf_params)
clf_model.fit(train[best_clf_features], train["evento_P80"].astype(int))
prob_dev_raw = clf_model.predict_proba(dev[best_clf_features])[:, 1]
prob_dev = isotonic.transform(prob_dev_raw)

pred_high_dev = winner_dev > dev["P80_A5_m3s"].to_numpy(float)
halfwidth = np.where(pred_high_dev, q_high, q_other)
lower = winner_dev - halfwidth
upper = winner_dev + halfwidth

predictions = dev[[
    "emissao_D", "centro_t", TARGET, "classe_A5", "P20_A5_m3s", "P80_A5_m3s",
    "C5_referencia_m3s", "Q5_observada_m3s", "A_persistencia_m3s",
]].copy()
predictions["A5_hibrida_sem_correcao_final_m3s"] = winner_dev_prebias
predictions["A5_prevista_V2_m3s"] = winner_dev
predictions["PI90_inferior_A5_m3s"] = lower
predictions["PI90_superior_A5_m3s"] = upper
predictions["PI90_coberto"] = predictions[TARGET].between(lower, upper)
predictions["Q5_prevista_V2_m3s"] = winner_dev + predictions["C5_referencia_m3s"]
predictions["probabilidade_A5_acima_P80"] = prob_dev
predictions["sinal_P80_probabilistico"] = prob_dev >= threshold_info["threshold"]
predictions["sinal_P80_central"] = winner_dev > predictions["P80_A5_m3s"]
predictions.to_csv(OUTPUT / "previsoes_desenvolvimento_2023_2024.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

y_dev = dev[TARGET].to_numpy(float)
event_dev = dev["evento_P80"].to_numpy(bool)
base_dev = dev["A_persistencia_m3s"].to_numpy(float)
central_p80 = binary_metrics(event_dev, winner_dev > dev["P80_A5_m3s"].to_numpy(float))
prob_p80 = binary_metrics(event_dev, prob_dev >= threshold_info["threshold"])

class_metrics = {}
for class_name, group in predictions.groupby("classe_A5"):
    class_metrics[class_name] = continuous_metrics(group[TARGET], group["A5_prevista_V2_m3s"])
    class_metrics[class_name]["pi90_coverage"] = float(group["PI90_coberto"].mean())

summary = {
    "version": "V2_HIBRIDO_TODOS_OS_DIAS", "outlet": OUTLET, "h": H,
    "selection": frozen,
    "validation": {
        "persistence": continuous_metrics(y_oof, base_oof),
        "winner": continuous_metrics(y_oof, winner_oof),
        "winner_P80_central": binary_metrics(event_oof, winner_oof > p80_oof),
        "classifier_P80": {**threshold_info, "brier_calibrated": float(brier_score_loss(event_oof, prob_oof))},
    },
    "development_2023_2024": {
        "persistence": continuous_metrics(y_dev, base_dev),
        "winner": continuous_metrics(y_dev, winner_dev),
        "winner_P80_central": central_p80,
        "classifier_P80": {**prob_p80, "threshold": threshold_info["threshold"], "brier": float(brier_score_loss(event_dev, prob_dev))},
        "PI90_coverage": float(predictions["PI90_coberto"].mean()),
        "PI90_mean_width_m3s": float(np.mean(2 * halfwidth)),
        "class_metrics": class_metrics,
    },
    "note": "2023-2024 e comparacao de desenvolvimento; 2025 permanece reservado e nao foi aberto.",
}
(OUTPUT / "resumo_V2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

joblib.dump({"model": reg_model, "metadata": frozen["regression"]}, MODELS / "regressao_residual.joblib")
joblib.dump({"model": clf_model, "isotonic": isotonic, "metadata": frozen["classifier_P80"]}, MODELS / "classificador_P80.joblib")

print(json.dumps({
    "outlet": OUTLET, "h": H,
    "winner_regression": best_model_name, "features": best_feature_name,
    "weight_high": best_weight_high, "blend_weight_model": blend_weight,
    "validation_mae": summary["validation"]["winner"]["mae"],
    "validation_persistence_mae": summary["validation"]["persistence"]["mae"],
    "development_mae": summary["development_2023_2024"]["winner"]["mae"],
    "development_persistence_mae": summary["development_2023_2024"]["persistence"]["mae"],
    "development_classifier_P80": summary["development_2023_2024"]["classifier_P80"],
    "development_PI90_coverage": summary["development_2023_2024"]["PI90_coverage"],
}, ensure_ascii=False, indent=2))
