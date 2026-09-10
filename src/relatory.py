"""
Persistent reporting for inference runs.

Each run writes three files into its own directory, plus one line in a
shared index:

    output/relatorios/<run_id>/relatorio.txt   human-readable summary
    output/relatorios/<run_id>/predicoes.csv   raw per-municipality output
    output/relatorios/<run_id>/perdidos.csv    confirmed cases never flagged
    output/relatorios/index.csv                one line per run
"""

import subprocess
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "output" / "relatorios"


def get_git_commit() -> str | None:
    """
    Return the current short commit hash, or None outside a git repo.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_metrics(prediction, consolidated, arrival, season):
    """
    Compute every number the report shows, so the text file and the index
    can never disagree with each other.

    Returns the metrics dict plus the two frames the report writes out.
    """
    arrival_season = arrival[arrival["safra"] == season]
    confirmed = set(arrival_season["municipio_id"])

    # The alert date is the first day the latch closed, which is what the
    # map actually shows.
    alert_date = (
        prediction[prediction["trava_positiva"] == 1]
        .groupby("municipio_id")["data"]
        .min()
        .rename("data_alerta")
        .reset_index()
    )
    alert_date["teve_ocorrencia"] = alert_date["municipio_id"].isin(confirmed)

    total = len(consolidated)
    latched = len(alert_date)
    hits = int(alert_date["teve_ocorrencia"].sum())

    m = {
        "municipios": total,
        "travados": latched,
        "cobertura": latched / total if total else 0.0,
        "ocorrencias_na_safra": len(confirmed),
        "taxa_acerto": hits / latched if latched else 0.0,
        # What a random pick of the same size would achieve. Without it,
        # "14 hits out of 69" means nothing.
        "taxa_base": len(confirmed) / total if total else 0.0,
        "perdidos": len(confirmed - set(alert_date["municipio_id"])),
        "prob_media": float(prediction["predito_prob"].mean()),
        "prob_mediana": float(prediction["predito_prob"].median()),
    }
    m["ganho_sobre_acaso"] = m["taxa_acerto"] - m["taxa_base"]

    comparison = alert_date.merge(arrival_season, on="municipio_id", how="inner")

    if comparison.empty:
        m["confirmados_alertados"] = 0
        return m, alert_date, comparison

    lead = (
        pd.to_datetime(comparison["data_chegada_real"])
        - pd.to_datetime(comparison["data_alerta"])
    ).dt.days
    comparison = comparison.assign(antecedencia_dias=lead)

    n = len(lead)
    m["confirmados_alertados"] = n
    m["alertados_antes"] = int((lead > 0).sum())
    m["antecedencia_mediana"] = float(lead.median())
    m["antecedencia_media"] = float(lead.mean())
    m["antecedencia_std"] = float(lead.std())
    m["antecedencia_min"] = int(lead.min())
    m["antecedencia_max"] = int(lead.max())

    m["faixas"] = {
        "mais de 21 dias antes": int((lead > 21).sum()),
        "de 8 a 21 dias antes": int(((lead > 7) & (lead <= 21)).sum()),
        "de 1 a 7 dias antes": int(((lead > 0) & (lead <= 7)).sum()),
        "no dia ou depois": int((lead <= 0).sum()),
    }

    # Does the model tell municipalities apart, or does it flag everyone
    # at once? Close to zero means the lead time above comes from blanket
    # coverage, not from prediction.
    by_group = alert_date.groupby("teve_ocorrencia")["data_alerta"].median()
    m["discriminacao_dias"] = (
        (by_group[False] - by_group[True]).days if len(by_group) == 2 else None
    )

    # The same question as a single number: do municipalities that
    # latched earlier carry a higher chance of an actual occurrence?
    # 0.5 means no signal at all.
    if alert_date["teve_ocorrencia"].nunique() == 2:
        score = -pd.to_datetime(alert_date["data_alerta"]).astype("int64")
        m["auc_espacial"] = float(
            roc_auc_score(alert_date["teve_ocorrencia"].astype(int), score)
        )
    else:
        m["auc_espacial"] = None

    return m, alert_date, comparison


def write_report(
    prediction,
    consolidated,
    arrival,
    season,
    base_date,
    config: dict,
) -> str:
    """
    Write the run report, the raw predictions and the missed cases, then
    append one summary line to the shared index.
    """
    executed_at = config.get("executed_at") or pd.Timestamp.now()
    run_id = f"{base_date:%Y%m%d}_{executed_at:%H%M%S}"

    run_dir = REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    m, alert_date, comparison = build_metrics(prediction, consolidated, arrival, season)

    config = {**config, "git_commit": get_git_commit()}

    with open(run_dir / "relatorio.txt", "w", encoding="utf-8") as f:
        f.write("=" * 62 + "\n")
        f.write(f"AVALIAÇÃO — {base_date:%Y-%m-%d}  |  safra {season}\n")
        f.write("=" * 62 + "\n")

        f.write("\nConfiguração\n")
        for k, v in config.items():
            f.write(f"  {k:<30} {v}\n")

        f.write("\nCobertura\n")
        f.write(f"  {'municípios avaliados':<30} {m['municipios']:>8d}\n")
        f.write(
            f"  {'com alerta ativo':<30} {m['travados']:>8d}"
            f"  ({m['cobertura']:.1%})\n"
        )
        f.write(f"  {'ocorrências na safra':<30} {m['ocorrencias_na_safra']:>8d}\n")
        f.write(f"  {'probabilidade média':<30} {m['prob_media']:>8.3f}\n")
        f.write(f"  {'probabilidade mediana':<30} {m['prob_mediana']:>8.3f}\n")

        f.write("\nAcerto\n")
        f.write(f"  {'taxa de acerto do alerta':<30} {m['taxa_acerto']:>8.1%}\n")
        f.write(f"  {'taxa base (acaso)':<30} {m['taxa_base']:>8.1%}\n")
        f.write(f"  {'ganho sobre o acaso':<30} {m['ganho_sobre_acaso']:>+8.1%}\n")
        f.write(f"  {'ocorrências sem alerta algum':<30} {m['perdidos']:>8d}\n")

        if m["confirmados_alertados"]:
            n = m["confirmados_alertados"]

            f.write("\nAntecedência (dias antes da ocorrência)\n")
            f.write(f"  {'municípios confirmados':<30} {n:>8d}\n")
            f.write(
                f"  {'alertados ANTES':<30} {m['alertados_antes']:>8d}"
                f"  ({m['alertados_antes'] / n:.1%})\n"
            )
            f.write(f"  {'mediana':<30} {m['antecedencia_mediana']:>8.1f}\n")
            f.write(f"  {'média':<30} {m['antecedencia_media']:>8.1f}\n")
            f.write(f"  {'desvio padrão':<30} {m['antecedencia_std']:>8.1f}\n")
            f.write(f"  {'melhor caso':<30} {m['antecedencia_max']:>8d}\n")
            f.write(f"  {'pior caso':<30} {m['antecedencia_min']:>8d}\n")

            f.write("\n  Distribuição:\n")
            for label, qty in m["faixas"].items():
                f.write(f"    {label:<28} {qty:>6d}  ({qty / n:.1%})\n")

            f.write("\nDiscriminação\n")
            if m["discriminacao_dias"] is not None:
                f.write(
                    f"  {'confirmados travaram antes':<30} "
                    f"{m['discriminacao_dias']:>8d} dias\n"
                )
            if m["auc_espacial"] is not None:
                f.write(f"  {'AUC espacial':<30} {m['auc_espacial']:>8.3f}\n")
            f.write(
                "  Perto de zero dia e de AUC 0.5, o modelo não separa\n"
                "  os municípios: a antecedência acima vem da cobertura.\n"
            )

        f.write("\n" + "=" * 62 + "\n")

    # Raw output, so any later analysis can be redone without rerunning
    # the whole pipeline.
    cols = [
        c
        for c in (
            "municipio_id",
            "municipio",
            "trava_positiva",
            "predito_prob",
            "data_chegada_prevista",
            "data_chegada_real",
        )
        if c in consolidated.columns
    ]
    consolidated[cols].to_csv(run_dir / "predicoes.csv", index=False, sep=";")

    # The costly misses: they show up in no other metric, because the
    # lead time is computed with an inner join.
    latched = set(alert_date["municipio_id"])
    missed = arrival[arrival["safra"] == season]
    missed = missed[~missed["municipio_id"].isin(latched)]
    missed.to_csv(run_dir / "perdidos.csv", index=False, sep=";")

    append_to_index(run_id, base_date, season, config, m)

    return str(run_dir / "relatorio.txt")


def append_to_index(run_id, base_date, season, config, m) -> None:
    """
    One line per run, so configurations can be compared side by side
    without opening each report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = REPORTS_DIR / "index.csv"

    row = {
        "run_id": run_id,
        "data_base": f"{base_date:%Y-%m-%d}",
        "safra": season,
        "threshold": config.get("threshold"),
        "beta": config.get("beta"),
        "veto_days": config.get("veto_days"),
        "min_consecutive": config.get("min_consecutive"),
        "git_commit": config.get("git_commit"),
        "travados": m["travados"],
        "cobertura": round(m["cobertura"], 4),
        "ocorrencias": m["ocorrencias_na_safra"],
        "perdidos": m["perdidos"],
        "taxa_acerto": round(m["taxa_acerto"], 4),
        "taxa_base": round(m["taxa_base"], 4),
        "ganho_sobre_acaso": round(m["ganho_sobre_acaso"], 4),
        "antecedencia_mediana": m.get("antecedencia_mediana"),
        "discriminacao_dias": m.get("discriminacao_dias"),
        "auc_espacial": (
            round(m["auc_espacial"], 4) if m.get("auc_espacial") is not None else None
        ),
    }

    pd.DataFrame([row]).to_csv(
        index_path,
        mode="a",
        header=not index_path.exists(),
        index=False,
    )
