"""
Monta uma galeria HTML com os mapas de uma única configuração:
classificador com limiar 0.5 calibrado com beta=1, e veto do regressor
em 13 dias.

As 24 datas das duas safras aparecem numa sequência só, ordenadas, para
que a evolução da cobertura seja lida de ponta a ponta.

Uso:
    python galeria.py                    # tudo
    python galeria.py --season 2024/2025
"""

import argparse
import base64
import io
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "output" / "relatorios" / "semanal"
GALLERY_PATH = REPORTS_DIR / "galeria.html"

# Vermelho para alerta, verde para sem alerta: a mesma paleta do mapa
# interativo, para que a galeria não confunda quem já viu o produto.
COLOR_CONFIRMED = "#8B0000"
COLOR_ALERT = "#d73027"
COLOR_NONE = "#1a9850"

THUMB_SIZE = (3.2, 2.6)
THUMB_DPI = 70

# Configuração única que a galeria mostra. Comparar mapas de cortes
# diferentes lado a lado confunde mais do que informa.
TARGET_THRESHOLD = 0.5
TARGET_BETA = 1.0
TARGET_VETO_DAYS = 13


def load_index() -> pd.DataFrame:
    """
    Read the run index, keep only the target configuration and label each
    run with it.
    """
    index_path = REPORTS_DIR / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Índice não encontrado: {index_path}")

    df = pd.read_csv(index_path)

    # Comparar por diferença absoluta, e não por igualdade: um float que
    # veio de um cálculo falharia silenciosamente no ==.
    df = df[
        (df["threshold"].sub(TARGET_THRESHOLD).abs() < 1e-6)
        & (df["veto_days"] == TARGET_VETO_DAYS)
    ]

    if "beta" in df.columns:
        df = df[df["beta"].sub(TARGET_BETA).abs() < 1e-6]

    if df.empty:
        raise ValueError(
            f"Nenhuma execução com threshold={TARGET_THRESHOLD}, "
            f"beta={TARGET_BETA} e veto={TARGET_VETO_DAYS} no índice."
        )

    df["config"] = (
        "thr="
        + df["threshold"].astype(str)
        + "  veto="
        + df["veto_days"].astype(str)
        + "  minc="
        + df["min_consecutive"].astype(str)
    )

    return df


def render_thumb(geojson_path: Path, points_path: Path | None) -> str:
    """
    Draw one map as a base64 PNG, so the whole gallery is a single
    self-contained file that opens without a server.
    """
    gdf = gpd.read_file(geojson_path)

    fig, ax = plt.subplots(figsize=THUMB_SIZE)

    if "confirmado" in gdf.columns:
        colors = [
            COLOR_CONFIRMED
            if c
            else (COLOR_ALERT if t else COLOR_NONE)
            for c, t in zip(
                gdf["confirmado"].fillna(0).astype(int),
                gdf["trava_positiva"].fillna(0).astype(int),
            )
        ]
    else:
        colors = [
            COLOR_ALERT if t else COLOR_NONE
            for t in gdf["trava_positiva"].fillna(0).astype(int)
        ]

    gdf.plot(ax=ax, color=colors, edgecolor="white", linewidth=0.15)

    if points_path and points_path.exists():
        points = gpd.read_file(points_path)
        if not points.empty:
            points.plot(
                ax=ax, color="#000000", markersize=8, edgecolor="white", linewidth=0.3
            )

    ax.set_axis_off()
    fig.tight_layout(pad=0)

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=THUMB_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode()


def build_panel(row) -> dict | None:
    """
    One panel per run, or None when that run produced no map.
    """
    run_dir = REPORTS_DIR / row["run_id"]
    stamp = pd.to_datetime(row["data_base"]).strftime("%Y%m%d")

    polygons = run_dir / f"municipios_{stamp}.geojson"
    if not polygons.exists():
        return None

    points = run_dir / f"ocorrencias_{stamp}.geojson"

    return {
        "data_base": row["data_base"],
        "safra": row["safra"],
        "cobertura": row["cobertura"],
        "nao_alertados": row["perdidos"],
        "auc": row.get("auc_espacial"),
        "discriminacao": row.get("discriminacao_dias"),
        "png": render_thumb(polygons, points),
    }


def card_html(panel: dict) -> str:
    auc = f"{panel['auc']:.3f}" if pd.notna(panel.get("auc")) else "—"
    disc = (
        f"{int(panel['discriminacao'])}d"
        if pd.notna(panel.get("discriminacao"))
        else "—"
    )

    return f"""
      <figure class="card">
        <img src="data:image/png;base64,{panel['png']}" alt="{panel['data_base']}">
        <figcaption>
          <b>{panel['data_base']}</b>
          <span class="safra">{panel['safra']}</span>
          <span class="met">
            cobertura do alerta do modelo {panel['cobertura']:.0%} &middot; casos sem alerta {panel['nao_alertados']}
          </span>
          <span class="met">AUC {auc} &middot; disc {disc}</span>
        </figcaption>
      </figure>"""


def build_sections(df: pd.DataFrame) -> tuple[str, int]:
    """
    One section per configuration left after the filter, which in
    practice means one per min_consecutive value.
    """
    sections = []
    total = 0

    for config in sorted(df["config"].unique()):
        subset = df[df["config"] == config]

        # Todas as datas das duas safras numa sequência só, para que a
        # evolução da cobertura seja lida de ponta a ponta.
        subset = subset.sort_values("data_base")

        panels = [p for p in (build_panel(r) for _, r in subset.iterrows()) if p]

        if not panels:
            continue

        total += len(panels)
        print(f"  {config}  |  {len(panels)} mapas")

        cards = "".join(card_html(p) for p in panels)

        # Averages let the reader judge the configuration before looking
        # at any individual map.
        mean_auc = subset["auc_espacial"].mean()
        mean_cov = subset["cobertura"].mean()
        mean_lost = subset["perdidos"].mean()

        sections.append(
            f"""
  <section>
    <h2>{config}</h2>
    <p class="resumo">
      média — cobertura do alerta do modelo {mean_cov:.0%} &middot;
      pontos não alertados {mean_lost:.1f} &middot;
      AUC espacial {mean_auc:.3f}
    </p>
    <div class="grid">{cards}
    </div>
  </section>"""
        )

    return "".join(sections), total


def write_gallery(sections_html: str, total: int, output_path: Path) -> None:
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Ferrugem asiática — galeria de execuções</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fafafa;
          color: #222; }}
  h1 {{ font-size: 21px; margin-bottom: 4px; }}
  h2 {{ font-size: 15px; font-family: ui-monospace, monospace; margin: 0 0 2px;
        padding-top: 14px; }}
  .sub {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .legend {{ font-size: 13px; margin-bottom: 24px; }}
  .legend span {{ display: inline-block; width: 12px; height: 12px;
                  vertical-align: middle; margin-right: 4px; }}
  section {{ background: white; border: 1px solid #ddd; border-radius: 8px;
             padding: 14px 18px; margin-bottom: 22px; }}
  .resumo {{ font-size: 12px; color: #666; margin: 0 0 6px; }}
  .grid {{ display: grid; gap: 12px;
           grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }}
  .card {{ margin: 0; }}
  .card img {{ width: 100%; display: block; }}
  .safra {{ display: block; color: #999; font-size: 10px; }}
  figcaption {{ font-size: 11px; line-height: 1.45; margin-top: 4px; }}
  .met {{ display: block; color: #777; }}
</style>
</head>
<body>
<h1>Ferrugem asiática — galeria de execuções</h1>
<p class="sub">
  {total} mapas — classificador com limiar {TARGET_THRESHOLD}
  (beta {TARGET_BETA}) e veto do regressor em {TARGET_VETO_DAYS} dias.
</p>
<p class="legend">
  <span style="background:{COLOR_CONFIRMED}"></span> confirmado pelo consórcio
  &nbsp;&nbsp;
  <span style="background:{COLOR_ALERT}"></span> alerta do modelo
  &nbsp;&nbsp;
  <span style="background:{COLOR_NONE}"></span> sem alerta
  &nbsp;&nbsp;
  <span style="background:#000000;border-radius:50%"></span> ocorrência registrada
</p>
{sections_html}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Monta uma galeria HTML com os mapas da configuração escolhida.",
    )
    parser.add_argument("--season", default=None, help='ex: "2024/2025"')
    parser.add_argument(
        "--output",
        default=str(GALLERY_PATH),
        help="caminho do HTML de saída",
    )
    args = parser.parse_args()

    df = load_index()

    if args.season:
        df = df[df["safra"] == args.season]

    if df.empty:
        print("Nenhuma execução corresponde aos filtros.")
        return

    print(f"{len(df)} execuções no índice, desenhando os mapas...")

    sections_html, total = build_sections(df)

    if total == 0:
        print("Nenhum mapa encontrado. A varredura rodou com --no-geojson?")
        return

    output_path = Path(args.output)
    write_gallery(sections_html, total, output_path)

    size_mb = output_path.stat().st_size / 1e6
    print(f"\n{total} mapas em {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
