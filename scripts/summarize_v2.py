from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(sys.argv[1])
outlet = sys.argv[2] if len(sys.argv) > 2 else root.parent.name
outdir = root / "RESUMO_CONSOLIDADO"
outdir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
})


def episodes(mask):
    pos = np.flatnonzero(np.asarray(mask, bool))
    if len(pos) == 0:
        return []
    groups = []
    a = b = int(pos[0])
    for cur in pos[1:]:
        cur = int(cur)
        if cur != b + 1:
            groups.append((a, b))
            a = cur
        b = cur
    groups.append((a, b))
    return groups


rows = []
details = {}
v1_fallback = {1: 54.78, 2: 61.52, 3: 72.12, 4: 75.61, 5: 68.50}


def get_v1_mae(h):
    candidates = [
        root.parent / f"PILOTO_D{h}_P80_TREINO_2018_2022_TESTE_2023_2024" / "resumo_piloto.json",
        root.parent / f"PILOTO_D{h}_TREINO_2018_2022_TESTE_2023_2024" / "resumo_piloto.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return float(payload["test_metrics_winner"]["mae"])
    return v1_fallback[h]

for h in range(1, 6):
    base = root / f"Dmais{h}"
    summary = json.loads((base / "resumo_V2.json").read_text(encoding="utf-8"))
    df = pd.read_csv(base / "previsoes_desenvolvimento_2023_2024.csv", parse_dates=["emissao_D", "centro_t"])
    df = df.sort_values("centro_t").reset_index(drop=True)
    observed = df["alvo_A5_m3s"].to_numpy(float) > df["P80_A5_m3s"].to_numpy(float)
    signal = df["sinal_P80_probabilistico"].astype(str).str.lower().eq("true").to_numpy()
    obs_events = episodes(observed)
    sig_events = episodes(signal)
    detected = sum(bool(signal[a:b + 1].any()) for a, b in obs_events)
    false_sig_events = sum(not bool(observed[a:b + 1].any()) for a, b in sig_events)

    dev = summary["development_2023_2024"]
    high = dev["class_metrics"].get("ACIMA", {})
    normal = dev["class_metrics"].get("NORMAL", {})
    clf = dev["classifier_P80"]
    winner = summary["selection"]["regression"]
    gain = 1.0 - dev["winner"]["mae"] / dev["persistence"]["mae"]
    row = {
        "horizonte": f"D+{h}",
        "modelo_residual": winner["model"],
        "peso_modelo_hibrido": winner["blend_weight_model"],
        "peso_dias_P80": winner["weight_high"],
        "mae_V1_m3s": get_v1_mae(h),
        "mae_persistencia_m3s": dev["persistence"]["mae"],
        "mae_V2_m3s": dev["winner"]["mae"],
        "ganho_V2_vs_persistencia": gain,
        "rmse_V2_m3s": dev["winner"]["rmse"],
        "bias_V2_m3s": dev["winner"]["bias"],
        "r2_V2": dev["winner"]["r2"],
        "mae_normal_V2_m3s": normal.get("mae"),
        "mae_acima_P80_V2_m3s": high.get("mae"),
        "bias_acima_P80_V2_m3s": high.get("bias"),
        "precisao_classificador_P80": clf["precision"],
        "deteccao_diaria_classificador_P80": clf["recall"],
        "f1_classificador_P80": clf["f1"],
        "episodios_observados": len(obs_events),
        "episodios_detectados": detected,
        "taxa_deteccao_episodios": detected / len(obs_events) if obs_events else np.nan,
        "episodios_sinal_sem_sobreposicao": false_sig_events,
        "pi90_coverage": dev["PI90_coverage"],
        "pi90_mean_width_m3s": dev["PI90_mean_width_m3s"],
    }
    rows.append(row)
    details[f"D+{h}"] = row

    fig, (ax, axp) = plt.subplots(2, 1, figsize=(13, 6.7), sharex=True, gridspec_kw={"height_ratios": [4, 1.2]}, constrained_layout=True)
    ax.fill_between(df["centro_t"], df["PI90_inferior_A5_m3s"], df["PI90_superior_A5_m3s"], color="#F4A261", alpha=0.2, label="PI90 V2")
    ax.plot(df["centro_t"], df["alvo_A5_m3s"], color="#202020", lw=1.15, label="A₅ observada")
    ax.plot(df["centro_t"], df["A5_prevista_V2_m3s"], color="#E76F51", lw=1.15, label="A₅ prevista V2")
    ax.plot(df["centro_t"], df["P80_A5_m3s"], color="#B2182B", lw=1.1, ls="--", label="P80 sazonal")
    ax.axhline(0, color="#888888", lw=0.7)
    ax.set_ylabel("A₅ (m³/s)")
    ax.set_title(f"Exutório {outlet} — V2 híbrida D+{h} — desenvolvimento 2023–2024")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper right")

    threshold = summary["selection"]["classifier_P80"]["threshold"]
    axp.plot(df["centro_t"], df["probabilidade_A5_acima_P80"], color="#457B9D", lw=1.0, label="Probabilidade A₅>P80")
    axp.axhline(threshold, color="#E76F51", ls="--", lw=1.0, label=f"Limiar {threshold:.2f}")
    axp.fill_between(df["centro_t"], 0, 1, where=observed, color="#264653", alpha=0.12, label="Observado acima do P80")
    axp.set_ylim(0, 1.02)
    axp.set_ylabel("Probabilidade")
    axp.set_xlabel("Centro da janela pentadal")
    axp.grid(axis="y", alpha=0.25)
    axp.legend(frameon=False, ncol=3, loc="upper right")
    axp.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axp.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    axp.tick_params(axis="x", rotation=35)
    fig.savefig(outdir / f"serie_V2_Dmais{h}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / f"serie_V2_Dmais{h}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

summary_df = pd.DataFrame(rows)
summary_df.to_csv(outdir / "resumo_consolidado_V2.csv", index=False, encoding="utf-8-sig")
(outdir / "resumo_consolidado_V2.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

h = np.arange(1, 6)
fig, axes = plt.subplots(3, 1, figsize=(10.8, 10.8), constrained_layout=True)

ax = axes[0]
ax.plot(h, summary_df["mae_V1_m3s"], marker="o", lw=2, color="#9E9E9E", label="V1 congelada")
ax.plot(h, summary_df["mae_persistencia_m3s"], marker="s", lw=2, color="#2A9D8F", label="Persistência")
ax.plot(h, summary_df["mae_V2_m3s"], marker="D", lw=2.2, color="#E76F51", label="V2 híbrida")
ax.set_xticks(h, [f"D+{x}" for x in h])
ax.set_ylabel("MAE (m³/s)")
ax.set_title("(a) Desempenho contínuo em todos os dias")
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, ncol=3)

ax = axes[1]
ax.plot(h, 100 * summary_df["precisao_classificador_P80"], marker="o", lw=2, label="Precisão diária", color="#E76F51")
ax.plot(h, 100 * summary_df["deteccao_diaria_classificador_P80"], marker="s", lw=2, label="Detecção diária", color="#457B9D")
ax.plot(h, 100 * summary_df["taxa_deteccao_episodios"], marker="^", lw=2, label="Detecção de episódios", color="#2A9D8F")
ax.set_xticks(h, [f"D+{x}" for x in h])
ax.set_ylim(0, 105)
ax.set_ylabel("Percentual (%)")
ax.set_title("(b) Classificador probabilístico de A₅>P80")
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, ncol=3)

ax = axes[2]
ax.plot(h, 100 * summary_df["pi90_coverage"], marker="D", lw=2, color="#6A4C93", label="Cobertura PI90")
ax.axhline(90, color="#777777", ls="--", lw=1, label="Cobertura nominal")
ax2 = ax.twinx()
ax2.plot(h, summary_df["pi90_mean_width_m3s"], marker="o", lw=2, color="#F4A261", label="Largura média")
ax.set_xticks(h, [f"D+{x}" for x in h])
ax.set_ylim(80, 100)
ax.set_ylabel("Cobertura (%)")
ax2.set_ylabel("Largura média (m³/s)")
ax.set_title("(c) Incerteza da V2")
ax.grid(axis="y", alpha=0.3)
lines = ax.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, frameon=False, ncol=3, loc="upper center")

fig.suptitle(f"Exutório {outlet} — comparação da V2 híbrida para D+1 a D+5", fontsize=14, fontweight="bold")
fig.text(0.01, -0.018, "Nota: 2023–2024 é desenvolvimento comparativo; 2025 permanece reservado e não foi utilizado.", fontsize=9)
fig.savefig(outdir / "comparacao_consolidada_V2_D1_D5.png", dpi=260, bbox_inches="tight", facecolor="white")
fig.savefig(outdir / "comparacao_consolidada_V2_D1_D5.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(summary_df.to_json(orient="records", force_ascii=False, indent=2))
