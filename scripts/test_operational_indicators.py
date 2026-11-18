from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def episodes(mask: pd.Series) -> list[tuple[int, int]]:
    idx = mask[mask].index.to_list()
    if not idx:
        return []
    out: list[tuple[int, int]] = []
    start = prev = idx[0]
    for cur in idx[1:]:
        if cur != prev + 1:
            out.append((start, prev))
            start = cur
        prev = cur
    out.append((start, prev))
    return out


def main() -> None:
    src = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outlet = sys.argv[3] if len(sys.argv) > 3 else "19094"
    horizon = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    df["emissao_D"] = pd.to_datetime(df["emissao_D"])
    df["centro_t"] = pd.to_datetime(df["centro_t"])
    df = df.sort_values("centro_t").reset_index(drop=True)

    df["obs_alta"] = df["alvo_A5_m3s"] > df["P80_A5_m3s"]
    point_col = "A5_prevista_corrigida_m3s" if "A5_prevista_corrigida_m3s" in df else "A5_prevista_m3s"
    lower_col = "PI90_inferior_A5_corrigido_m3s" if "PI90_inferior_A5_corrigido_m3s" in df else "PI90_inferior_A5_m3s"
    upper_col = "PI90_superior_A5_corrigido_m3s" if "PI90_superior_A5_corrigido_m3s" in df else "PI90_superior_A5_m3s"
    df["preventivo"] = df[upper_col] > df["P80_A5_m3s"]
    df["provavel"] = df[point_col] > df["P80_A5_m3s"]
    df["alta_confianca"] = df[lower_col] > df["P80_A5_m3s"]

    modes = {
        "PREVENTIVO_PI90_SUPERIOR": "preventivo",
        "PROVAVEL_PREVISAO_CENTRAL": "provavel",
        "ALTA_CONFIANCA_PI90_INFERIOR": "alta_confianca",
    }

    event_rows = []
    obs_events = episodes(df["obs_alta"])
    for event_id, (a, b) in enumerate(obs_events, 1):
        row = {
            "evento_observado_id": event_id,
            "centro_inicio": df.loc[a, "centro_t"].date().isoformat(),
            "centro_fim": df.loc[b, "centro_t"].date().isoformat(),
            "duracao_dias": int(b - a + 1),
            "A5_observada_max_m3s": float(df.loc[a:b, "alvo_A5_m3s"].max()),
            "excesso_observado_max_sobre_P80_m3s": float(
                (df.loc[a:b, "alvo_A5_m3s"] - df.loc[a:b, "P80_A5_m3s"]).max()
            ),
        }
        for label, col in modes.items():
            hits = df.loc[a:b][df.loc[a:b, col]]
            key = label.lower()
            row[f"{key}_detectado"] = bool(not hits.empty)
            if hits.empty:
                row[f"{key}_primeira_emissao"] = ""
                row[f"{key}_primeiro_centro"] = ""
                row[f"{key}_antecedencia_inicio_dias"] = ""
            else:
                first = hits.iloc[0]
                row[f"{key}_primeira_emissao"] = first["emissao_D"].date().isoformat()
                row[f"{key}_primeiro_centro"] = first["centro_t"].date().isoformat()
                row[f"{key}_antecedencia_inicio_dias"] = int(
                    (df.loc[a, "centro_t"] - first["emissao_D"]).days
                )
        event_rows.append(row)

    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(outdir / "teste_operacional_por_evento.csv", index=False)

    daily_rows = []
    summary_modes = {}
    for label, col in modes.items():
        tp = int((df[col] & df["obs_alta"]).sum())
        fp = int((df[col] & ~df["obs_alta"]).sum())
        fn = int((~df[col] & df["obs_alta"]).sum())
        tn = int((~df[col] & ~df["obs_alta"]).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        event_col = f"{label.lower()}_detectado"
        detected_events = int(event_df[event_col].sum()) if not event_df.empty else 0
        signal_events = episodes(df[col])
        false_signal_events = 0
        for a, b in signal_events:
            if not bool(df.loc[a:b, "obs_alta"].any()):
                false_signal_events += 1
        rec = {
            "modo": label,
            "TP_dias": tp,
            "FP_dias": fp,
            "FN_dias": fn,
            "TN_dias": tn,
            "precisao": precision,
            "revocacao": recall,
            "F1": f1,
            "eventos_observados": len(obs_events),
            "eventos_detectados": detected_events,
            "taxa_deteccao_eventos": detected_events / len(obs_events) if obs_events else None,
            "episodios_sinal_emitido": len(signal_events),
            "episodios_sinal_sem_sobreposicao_evento": false_signal_events,
        }
        daily_rows.append(rec)
        summary_modes[label] = rec

    pd.DataFrame(daily_rows).to_csv(outdir / "teste_operacional_resumo.csv", index=False)

    summary = {
        "escopo": f"Piloto {outlet} D+{horizon}; teste independente 2023-2024",
        "definicao_evento": "Sequencia continua de centros t com A5 observada acima do P80 sazonal",
        "antecedencia": "D+1: cada sinal e emitido um dia antes do centro t avaliado",
        "numero_eventos_observados": len(obs_events),
        "modos": summary_modes,
        "observacao": (
            "P80 e limiar de anomalia sazonal, nao limiar de cheia. "
            "O resultado avalia sinal de anomalia alta."
        ),
    }
    (outdir / "teste_operacional_resumo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# Teste operacional dos sinais de anomalia alta",
        "",
        f"Escopo: exutório {outlet}, D+{horizon}, teste independente 2023–2024.",
        "",
        "Evento observado: sequência contínua de centros t com A5 observada acima do P80 sazonal.",
        "P80 não é limiar de cheia; é limiar sazonal de anomalia alta.",
        "",
        "| Modo | Precisão diária | Detecção diária | Eventos detectados | Episódios de sinal sem sobreposição |",
        "|---|---:|---:|---:|---:|",
    ]
    for rec in daily_rows:
        precision_text = f"{rec['precisao']:.1%}" if rec["precisao"] is not None else "— (sem sinais)"
        recall_text = f"{rec['revocacao']:.1%}" if rec["revocacao"] is not None else "—"
        report.append(
            f"| {rec['modo']} | {precision_text} | {recall_text} | "
            f"{rec['eventos_detectados']}/{rec['eventos_observados']} | "
            f"{rec['episodios_sinal_sem_sobreposicao_evento']} |"
        )
    (outdir / "RELATORIO_TESTE_OPERACIONAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
