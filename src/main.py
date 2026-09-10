import argparse
import configparser
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from Helpers.mapping_real_occorrences import (
    map_occurrences_to_municipalities,
    normalize_municipio_id,
)
from Helpers.season import season_of
import features
import generate_geojson
import hybrid_model
from relatory import write_report

# Keep the token
_token = None

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.cfg"

OUTPUT_DIR = BASE_DIR / "output"

PREC_DIR = OUTPUT_DIR / "precipitacao"
FEATURES_DIR = OUTPUT_DIR / "features"
GEOJSON_DIR = OUTPUT_DIR / "geojson"
CONSORCIO_DIR = OUTPUT_DIR / "consorcio"


def read_cfg():
    """
    Read config.cfg from the repository
    """
    cfg = configparser.ConfigParser()
    if not cfg.read(CONFIG_PATH, encoding="utf-8"):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    return cfg


def base_url() -> str:
    """
    Return the Airflow base URL
    """
    return read_cfg().get("airflow", "base_url")


def get_token() -> str:
    """
    Authenticate with Airflow and return an access code

    This access code is cached
    """
    global _token
    if _token:
        return _token

    cfg = read_cfg()
    r = requests.post(
        f"{base_url()}/auth/token",
        json={
            "username": cfg.get("airflow", "username"),
            "password": cfg.get("airflow", "password"),
        },
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Failed to authenticate on airflow: {r.status_code}")

    _token = r.json()["access_token"]
    return _token


def headers() -> dict:
    """
    Return the authorization header Airflow expects every request
    """
    return {"Authorization": f"Bearer {get_token()}"}


def run_dag(dag_id: str, conf: dict) -> str:
    """
    Trigger a DAG run and returns immediately, without waiting for it
    """
    run_id = f"repo__{datetime.now():%Y%m%dT%H%M%S}__{uuid.uuid4().hex[:6]}"

    resp = requests.post(
        f"{base_url()}/api/v2/dags/{dag_id}/dagRuns",
        headers=headers(),
        json={
            "dag_run_id": run_id,
            "conf": conf,
            "logical_date": None,
        },
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to trigger DAG {dag_id}: {resp.status_code}")

    print(f"Triggered {dag_id}")
    return run_id


def wait_dag(dag_id: str, run_id: str) -> None:
    """
    Wait for the DAG run to succeed or fail
    """
    interval = 15
    deadline = time.time() + 60 * 60
    last_state = None

    while time.time() < deadline:
        r = requests.get(
            f"{base_url()}/api/v2/dags/{dag_id}/dagRuns/{run_id}",
            headers=headers(),
            timeout=30,
        )
        r.raise_for_status()
        state = r.json().get("state", "unknown")

        # prints only in state transitions
        if state != last_state:
            print(f"{dag_id}: {state}")
            last_state = state
        if state == "success":
            return
        if state in ("failed", "upstream_failed"):
            raise RuntimeError(f"DAG {dag_id} failed")

        time.sleep(interval)

    raise TimeoutError(f"DAG {dag_id} did not finish in time")


def exec_dag(dag_id: str, conf: dict) -> None:
    """
    Trigger a DAG and block until it finishes
    """
    wait_dag(dag_id, run_dag(dag_id, conf))


def collect(remote_path: str) -> str:
    """
    Bring a file produced by the DAG into the output directory.

    Running on the same machine as Airflow, the file is already on this
    filesystem, so a local copy replaces the scp round trip — and the
    password prompt that came with it.
    """
    work_dir = PREC_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    destination = work_dir / Path(remote_path).name

    if os.path.exists(remote_path):
        shutil.copy2(remote_path, destination)
        print(f"Collected {destination}")
        return str(destination)

    # Fallback for running from outside the server.
    cfg = read_cfg()
    ssh_user = cfg.get("server", "username")
    ssh_host = cfg.get("server", "host")
    source = f"{ssh_user}@{ssh_host}:{remote_path}"

    r = subprocess.run(
        ["scp", source, str(destination)], capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"scp failed on {source}:\n{r.stderr}\n")

    print(f"Collected {destination}")
    return str(destination)


def send_geojson(
    local_path: str, base_date: pd.Timestamp, prefix: str = "ferrugem"
) -> None:
    """
    Publish a GeoJSON as {prefix}_DD-MM-YYYY.geojson, copying locally
    when the destination is on this same machine.
    """
    cfg = read_cfg()
    remote_dir = cfg.get("server", "dir_rust")
    name = f"{prefix}_{base_date.strftime('%d-%m-%Y')}.geojson"

    if os.path.isdir(remote_dir) or os.path.isdir(os.path.dirname(remote_dir)):
        os.makedirs(remote_dir, exist_ok=True)
        shutil.copy2(local_path, os.path.join(remote_dir, name))
        print(f"enviado: {remote_dir}/{name}")
        return

    ssh_usuario = cfg.get("server", "username")
    ssh_host = cfg.get("server", "host")
    destination = f"{ssh_usuario}@{ssh_host}:{remote_dir}/{name}"

    subprocess.run(
        ["ssh", f"{ssh_usuario}@{ssh_host}", "mkdir", "-p", remote_dir],
        check=True,
    )

    r = subprocess.run(["scp", local_path, destination], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Falha ao enviar {local_path}:\n{r.stderr}\n")

    print(f"enviado: {remote_dir}/{name}")


def print_evaluation(prediction, consolidated, arrival, season, base_date, config):
    """
    Print every metric used to judge the run, so a single console output
    tells whether this configuration is better than the previous one.
    """
    total = len(consolidated)
    positivos = int(consolidated["trava_positiva"].sum())

    # The alert date is the first day the latch closed, which is what the
    # map actually shows.
    data_alerta = (
        prediction[prediction["trava_positiva"] == 1]
        .groupby("municipio_id")["data"]
        .min()
        .rename("data_alerta")
        .reset_index()
    )

    arrival_season = arrival[arrival["safra"] == season]
    confirmados = set(arrival_season["municipio_id"])

    comparacao = data_alerta.merge(arrival_season, on="municipio_id", how="inner")

    print("\n" + "=" * 62)
    print(f"AVALIAÇÃO — {base_date:%Y-%m-%d}  |  safra {season}")
    print("=" * 62)

    print("\nConfiguração")
    for k, v in config.items():
        print(f"  {k:<32} {v}")

    print("\nCobertura")
    print(f"  {'municípios avaliados':<32} {total:>8d}")
    print(f"  {'com alerta ativo':<32} {positivos:>8d}  ({positivos / total:.1%})")
    print(f"  {'instâncias processadas':<32} {len(prediction):>8d}")

    print("\nProbabilidade prevista (último dia da série)")
    prob = prediction["predito_prob"]
    print(f"  {'média':<32} {prob.mean():>8.3f}")
    print(f"  {'mediana':<32} {prob.median():>8.3f}")

    if comparacao.empty:
        print("\nSem ocorrências confirmadas para comparar nesta safra.")
        print("=" * 62 + "\n")
        return

    antecedencia = (
        pd.to_datetime(comparacao["data_chegada_real"])
        - pd.to_datetime(comparacao["data_alerta"])
    ).dt.days

    n = len(antecedencia)
    antes = int((antecedencia > 0).sum())

    print("\nAntecedência do alerta (dias antes da ocorrência confirmada)")
    print(f"  {'municípios confirmados':<32} {n:>8d}")
    print(f"  {'alertados ANTES da ocorrência':<32} {antes:>8d}  ({antes / n:.1%})")
    print(f"  {'mediana':<32} {antecedencia.median():>8.1f}")
    print(f"  {'média':<32} {antecedencia.mean():>8.1f}")
    print(f"  {'desvio padrão':<32} {antecedencia.std():>8.1f}")
    print(f"  {'melhor caso':<32} {antecedencia.max():>8.0f}")
    print(f"  {'pior caso':<32} {antecedencia.min():>8.0f}")

    print("\n  Distribuição:")
    faixas = [
        ("mais de 21 dias antes", antecedencia > 21),
        ("de 8 a 21 dias antes", (antecedencia > 7) & (antecedencia <= 21)),
        ("de 1 a 7 dias antes", (antecedencia > 0) & (antecedencia <= 7)),
        ("no dia ou depois", antecedencia <= 0),
    ]
    for label, mask in faixas:
        q = int(mask.sum())
        print(f"    {label:<30} {q:>6d}  ({q / n:.1%})")

    # Does the model actually tell municipalities apart, or does it just
    # flag everyone early? If both groups latched around the same date,
    # the lead time above is an artefact of blanket coverage, not of
    # prediction.
    data_alerta["teve_ocorrencia"] = data_alerta["municipio_id"].isin(confirmados)

    print("\nDiscriminação — data em que a trava fechou, por grupo")
    resumo = (
        data_alerta.groupby("teve_ocorrencia")["data_alerta"]
        .agg(["count", "min", "median", "max"])
        .rename(index={False: "sem ocorrência", True: "com ocorrência"})
    )
    for grupo, linha in resumo.iterrows():
        print(
            f"  {grupo:<20} n={linha['count']:>4}  "
            f"primeira={linha['min']:%Y-%m-%d}  "
            f"mediana={linha['median']:%Y-%m-%d}  "
            f"última={linha['max']:%Y-%m-%d}"
        )

    if len(resumo) == 2:
        delta = (
            resumo.loc["sem ocorrência", "median"]
            - resumo.loc["com ocorrência", "median"]
        ).days
        print(
            f"\n  Municípios com ocorrência travaram {delta} dias antes "
            "dos demais (mediana)."
        )
        print("  Valor próximo de zero indica que o modelo não distingue.")

    # Confirmed occurrences the model never flagged: the costly misses.
    alertados = set(data_alerta["municipio_id"])
    perdidos = confirmados - alertados

    print(f"\n  {'ocorrências sem alerta algum':<32} {len(perdidos):>8d}")

    print("=" * 62 + "\n")


def main(
    base_date: pd.Timestamp,
    classifier_threshold=None,
    regressor_threshold=None,
    beta=None,
    min_consecutive: int = 1,
) -> tuple[str, str]:
    cfg = read_cfg()
    stamp = base_date.strftime("%Y%m%d")
    season = season_of(base_date)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating features for {stamp}")

    # --- precipitation ---
    historic_days_prec = 300
    prec_begin = base_date - timedelta(days=historic_days_prec)
    exec_dag(
        "dag_gerar_csv_chuva_param",
        {
            "DATA_INICIO": prec_begin.strftime("%Y-%m-%d"),
            "DATA_FIM": base_date.strftime("%Y-%m-%d"),
        },
    )

    prec_dir = cfg.get("server", "dir_prec")
    prec_csv = collect(
        f"{prec_dir}/prep_{prec_begin:%Y-%m-%d}_{base_date:%Y-%m-%d}.csv"
    )

    # --- features ---
    municipalities, precipitation = features.load_data(prec_csv)

    light_cols = [
        "municipio_id",
        "municipio",
        "segment_id",
        "ocorrencia_latitude",
        "ocorrencia_longitude",
    ]
    points = pd.DataFrame(municipalities[light_cols])

    df = features.process_all_cores(points, precipitation, base_date)

    features_path = str(FEATURES_DIR / f"features_{stamp}.csv")
    features.save_results(df, features_path)

    # --- model ---
    prediction = hybrid_model.predict(df, classifier_threshold, regressor_threshold)
    prediction = hybrid_model.apply_latch(prediction, min_consecutive)
    consolidated = hybrid_model.consolidate_by_municipalitie(prediction)
    consolidated["municipio_id"] = normalize_municipio_id(consolidated["municipio_id"])

    # --- confirmed occurrences, collected beforehand by the consortium
    # extractor script ---
    occurrences_csv = BASE_DIR / cfg.get("paths", "occurrences_csv")
    if not os.path.exists(occurrences_csv):
        raise FileNotFoundError(
            f"Consortium file not found: {occurrences_csv}. "
            "Run the extractor before generating the map."
        )

    occurrences = pd.read_csv(occurrences_csv, encoding="utf-8-sig")

    arrival = map_occurrences_to_municipalities(occurrences_csv)
    consolidated = consolidated.merge(
        arrival[arrival["safra"] == season], on="municipio_id", how="left"
    )

    config = {
        "threshold": classifier_threshold,
        "beta": beta,
        "veto_days": regressor_threshold,
        "min_consecutive": min_consecutive,
    }

    write_report(prediction, consolidated, arrival, season, base_date, config)

    # --- geojson ---
    polygons_path, points_path = generate_geojson.generate_map_files(
        consolidated, occurrences, str(GEOJSON_DIR), base_date, season
    )

    # send_geojson(polygons_path, base_date, prefix="ferrugem")
    # send_geojson(points_path, base_date, prefix="ocorrencias")

    return polygons_path, points_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera o mapa de risco de ferrugem asiática para uma data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=None,
        metavar="YYYY-MM-DD",
        help="data base da simulação (padrão: hoje)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="limiar do classificador (padrão: o calibrado no treino)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="beta com que o limiar foi calibrado; só registra no relatório",
    )
    parser.add_argument(
        "--veto-days",
        type=int,
        default=None,
        help="dias acima dos quais o alerta é vetado pelo regressor",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=1,
        help="dias positivos seguidos para fechar a trava",
    )

    args = parser.parse_args()

    data = (
        pd.to_datetime(args.date, format="%Y-%m-%d")
        if args.date
        else pd.to_datetime(datetime.now().date())
    )

    main(data, args.threshold, args.veto_days, args.beta, args.min_consecutive)
