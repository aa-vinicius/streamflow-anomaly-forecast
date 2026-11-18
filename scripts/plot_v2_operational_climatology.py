from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(sys.argv[1])
OUTLET = sys.argv[2] if len(sys.argv) > 2 else ROOT.parent.name
OUT = ROOT / "FIGURAS_OPERACIONAIS_V2_VS_CLIMATOLOGIA"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

COL = {
    "obs": "#202020",
    "v2": "#E76F51",
    "clim": "#457B9D",
    "p80": "#B2182B",
    "pi": "#F4A261",
    "event": "#264653",
    "central": "#F8961E",
    "prob": "#D62828",
}


def episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    pos = np.flatnonzero(np.asarray(mask, bool))
    if len(pos) == 0:
        return []
    groups: list[tuple[int, int]] = []
    a = b = int(pos[0])
    for cur in pos[1:]:
        cur = int(cur)
        if cur != b + 1:
            groups.append((a, b))
            a = cur
        b = cur
    groups.append((a, b))
    return groups


def binary_metrics(observed: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, bool)
    signal = np.asarray(signal, bool)
    tp = int(np.sum(observed & signal))
    fp = int(np.sum(~observed & signal))
    fn = int(np.sum(observed & ~signal))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    obs_events = episodes(observed)
    detected = sum(bool(signal[a:b + 1].any()) for a, b in obs_events)
    event_recall = detected / len(obs_events) if obs_events else 0.0
    return {"precision": precision, "recall": recall, "event_recall": event_recall}


def add_binary_spans(ax, dates: pd.Series, mask: np.ndarray, y: float, color: str, height: float = 0.68) -> None:
    nums = mdates.date2num(pd.to_datetime(dates).dt.to_pydatetime())
    for a, b in episodes(mask):
        left = nums[a] - 0.45
        width = nums[b] - nums[a] + 0.9
        ax.broken_barh([(left, width)], (y - height / 2, height), facecolors=color, edgecolors="none")


rows: list[dict[str, float | str]] = []

for h in range(1, 6):
    source = ROOT / f"Dmais{h}" / "previsoes_desenvolvimento_2023_2024.csv"
    df = pd.read_csv(source, parse_dates=["emissao_D", "centro_t"]).sort_values("centro_t").reset_index(drop=True)

    obs = df["alvo_A5_m3s"].to_numpy(float)
    pred = df["A5_prevista_V2_m3s"].to_numpy(float)
    p80 = df["P80_A5_m3s"].to_numpy(float)
    clim = np.zeros(len(df), dtype=float)
    observed_high = obs > p80
    signal_prob = df["sinal_P80_probabilistico"].astype(str).str.lower().eq("true").to_numpy()
    signal_central = pred > p80
    signal_clim = clim > p80

    met_v2 = binary_metrics(observed_high, signal_prob)
    met_clim = binary_metrics(observed_high, signal_clim)
    mae_v2 = float(np.mean(np.abs(obs - pred)))
    mae_clim = float(np.mean(np.abs(obs - clim)))
    rmse_v2 = float(np.sqrt(np.mean((obs - pred) ** 2)))
    rmse_clim = float(np.sqrt(np.mean((obs - clim) ** 2)))
    skill_mae = 1.0 - mae_v2 / mae_clim
    rows.append({
        "horizonte": f"D+{h}",
        "MAE_V2_m3s": mae_v2,
        "MAE_climatologia_m3s": mae_clim,
        "RMSE_V2_m3s": rmse_v2,
        "RMSE_climatologia_m3s": rmse_clim,
        "skill_MAE_V2_vs_climatologia": skill_mae,
        "deteccao_diaria_V2": met_v2["recall"],
        "precisao_diaria_V2": met_v2["precision"],
        "deteccao_episodios_V2": met_v2["event_recall"],
        "deteccao_diaria_climatologia": met_clim["recall"],
        "precisao_diaria_climatologia": met_clim["precision"],
        "deteccao_episodios_climatologia": met_clim["event_recall"],
    })

    # Figura operacional no padrão solicitado.
    fig = plt.figure(figsize=(16, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[4.5, 1.45, 2.0])
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1], sharex=ax)
    axb = fig.add_subplot(gs[2])

    ax.fill_between(df["centro_t"], df["PI90_inferior_A5_m3s"], df["PI90_superior_A5_m3s"],
                    color=COL["pi"], alpha=0.17, label="PI90 da V2")
    ax.plot(df["centro_t"], obs, color=COL["obs"], lw=1.45, label="A₅ observada")
    ax.plot(df["centro_t"], pred, color=COL["v2"], lw=1.35, label="A₅ prevista — V2 híbrida")
    ax.plot(df["centro_t"], p80, color=COL["p80"], lw=1.25, ls="--", label="P80 sazonal")
    ax.plot(df["centro_t"], clim, color=COL["clim"], lw=1.0, ls=":", label="Climatologia: A₅=0")
    ax.set_ylabel("Anomalia pentadal A₅ (m³/s)")
    ax.set_title("(a) Previsão da anomalia, PI90, P80 e referência climatológica")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(frameon=False, ncol=5, loc="upper right")
    ax.tick_params(labelbottom=False)

    signal_rows = [
        (observed_high, 3, COL["event"]),
        (signal_prob, 2, COL["prob"]),
        (signal_central, 1, COL["central"]),
        (signal_clim, 0, COL["clim"]),
    ]
    for mask, y, color in signal_rows:
        add_binary_spans(axs, df["centro_t"], mask, y, color)
    axs.set_yticks([3, 2, 1, 0], ["Observada > P80", "V2 probabilística", "V2 central > P80", "Climatologia > P80"])
    axs.set_ylim(-0.6, 3.6)
    axs.set_title("(b) Ocorrência diária dos sinais de anomalia alta")
    axs.grid(axis="x", alpha=0.25)
    axs.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    axs.tick_params(axis="x", rotation=35)

    categories = ["Detecção de\nepisódios", "Detecção\ndiária", "Precisão\ndiária"]
    v2_values = 100 * np.array([met_v2["event_recall"], met_v2["recall"], met_v2["precision"]])
    clim_values = 100 * np.array([met_clim["event_recall"], met_clim["recall"], met_clim["precision"]])
    x = np.arange(3)
    width = 0.34
    b1 = axb.bar(x - width / 2, v2_values, width, color=COL["v2"], label="V2 híbrida")
    b2 = axb.bar(x + width / 2, clim_values, width, color=COL["clim"], label="Climatologia (A₅=0)")
    axb.bar_label(b1, fmt="%.1f%%", padding=3)
    axb.bar_label(b2, fmt="%.1f%%", padding=3)
    axb.set_xticks(x, categories)
    axb.set_ylim(0, 112)
    axb.set_ylabel("Percentual (%)")
    axb.set_title("(c) Sensibilidade e confiabilidade operacional: V2 versus climatologia")
    axb.grid(axis="y", alpha=0.28)
    axb.legend(frameon=False, ncol=2, loc="upper center")

    fig.suptitle(f"Sinais de anomalia alta — Exutório {OUTLET}, V2 híbrida D+{h} (2023–2024)",
                 fontsize=15, fontweight="bold")
    fig.text(0.01, -0.012,
             f"Nota: climatologia = A₅ previsto igual a zero (Q₅ previsto = C₅). MAE: V2 {mae_v2:.1f} m³/s; "
             f"climatologia {mae_clim:.1f} m³/s; ganho relativo {100 * skill_mae:.1f}%. "
             "P80 é um percentil sazonal da anomalia, não um limiar de inundação.", fontsize=9)
    fig.savefig(OUT / f"operacional_V2_vs_climatologia_Dmais{h}.png", dpi=230, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"operacional_V2_vs_climatologia_Dmais{h}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Comparação em vazão reconstruída: Q5 = C5 + A5.
    q_obs = df["Q5_observada_m3s"].to_numpy(float)
    q_v2 = df["Q5_prevista_V2_m3s"].to_numpy(float)
    c5 = df["C5_referencia_m3s"].to_numpy(float)
    q_lo = c5 + df["PI90_inferior_A5_m3s"].to_numpy(float)
    q_hi = c5 + df["PI90_superior_A5_m3s"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(15, 5.8), constrained_layout=True)
    ax.fill_between(df["centro_t"], q_lo, q_hi, color=COL["pi"], alpha=0.17, label="PI90 da vazão V2")
    ax.plot(df["centro_t"], q_obs, color=COL["obs"], lw=1.35, label="Q₅ observada")
    ax.plot(df["centro_t"], q_v2, color=COL["v2"], lw=1.25, label="Q₅ prevista — V2")
    ax.plot(df["centro_t"], c5, color=COL["clim"], lw=1.2, ls="--", label="C₅ climatológica")
    ax.set_title(f"Vazão pentadal reconstruída — V2 versus climatologia — Exutório {OUTLET}, D+{h}", fontweight="bold")
    ax.set_ylabel("Vazão pentadal Q₅ (m³/s)")
    ax.set_xlabel("Centro da janela pentadal")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    ax.tick_params(axis="x", rotation=35)
    fig.savefig(OUT / f"vazao_reconstruida_V2_vs_C5_Dmais{h}.png", dpi=230, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"vazao_reconstruida_V2_vs_C5_Dmais{h}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "metricas_V2_vs_climatologia_D1_D5.csv", index=False, encoding="utf-8-sig")

# Síntese comparativa D+1 a D+5.
x = np.arange(5)
labels = metrics["horizonte"].tolist()
fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), constrained_layout=True)
width = 0.36

for ax, v2_col, clim_col, title, ylabel in [
    (axes[0], "MAE_V2_m3s", "MAE_climatologia_m3s", "(a) Erro absoluto médio", "MAE (m³/s)"),
    (axes[1], "RMSE_V2_m3s", "RMSE_climatologia_m3s", "(b) Raiz do erro quadrático médio", "RMSE (m³/s)"),
]:
    b1 = ax.bar(x - width / 2, metrics[v2_col], width, color=COL["v2"], label="V2 híbrida")
    b2 = ax.bar(x + width / 2, metrics[clim_col], width, color=COL["clim"], label="Climatologia")
    ax.bar_label(b1, fmt="%.1f", padding=2)
    ax.bar_label(b2, fmt="%.1f", padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

ax = axes[2]
skills = 100 * metrics["skill_MAE_V2_vs_climatologia"].to_numpy(float)
bars = ax.bar(x, skills, width=0.55, color=COL["v2"])
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, labels)
ax.set_ylabel("Redução do MAE (%)")
ax.set_title("(c) Ganho da V2 em relação à climatologia")
ax.grid(axis="y", alpha=0.25)

fig.suptitle(f"Exutório {OUTLET} — V2 híbrida versus climatologia para D+1 a D+5 (2023–2024)",
             fontsize=14, fontweight="bold")
fig.text(0.01, -0.012, "Nota: climatologia = A₅ previsto igual a zero, ou Q₅ previsto igual a C₅. 2025 não foi utilizado.", fontsize=9)
fig.savefig(OUT / "comparacao_consolidada_V2_vs_climatologia_D1_D5.png", dpi=240, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "comparacao_consolidada_V2_vs_climatologia_D1_D5.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(metrics.to_string(index=False))
