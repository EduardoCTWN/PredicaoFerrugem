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
from Helpers import adjacency
from Helpers import correction

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

def main(
    base_date: pd.Timestamp,
    classifier_threshold: float,
    regressor_threshold: int,
    min_consecutive: int,
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

    df["municipio_id"] = normalize_municipio_id(df["municipio_id"])

    # confirmed occurrences, collected beforehand by the consortium
    # extractor script
    occurrences_csv = BASE_DIR / cfg.get("paths", "occurrences_csv")
    if not os.path.exists(occurrences_csv):
        raise FileNotFoundError(
            f"Consortium file not found: {occurrences_csv}. "
            "Run the extractor before generating the map."
        )

    occurrences = pd.read_csv(occurrences_csv, encoding="utf-8-sig")

    arrival = map_occurrences_to_municipalities(occurrences_csv)

    # --- model ---
    prediction = hybrid_model.predict(df, classifier_threshold, regressor_threshold)
    prediction = hybrid_model.apply_latch(prediction, min_consecutive)
    prediction = adjacency.neighbours_alert(prediction)
    prediction = correction.correct_model(prediction, arrival, season)
    consolidated = hybrid_model.consolidate_by_municipalitie(prediction)
    consolidated["municipio_id"] = normalize_municipio_id(consolidated["municipio_id"])

    consolidated = consolidated.merge(
        arrival[arrival["safra"] == season], on="municipio_id", how="left"
    )

    config = {
        "threshold": classifier_threshold,
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
        required=True,
        help="corte do classificador",
    )
    parser.add_argument(
        "--veto-days",
        type=int,
        required=True,
        help="valor de veto_days",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        required=True,
        help="valor de min_consecutive",
    )

    args = parser.parse_args()

    data = (
        pd.to_datetime(args.date, format="%Y-%m-%d")
        if args.date
        else pd.to_datetime(datetime.now().date())
    )

    main(data, args.threshold, args.veto_days, args.min_consecutive)