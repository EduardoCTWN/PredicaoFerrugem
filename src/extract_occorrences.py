from pathlib import Path
import time
import random
import requests
import pandas as pd

# ============================================================
# sETUPCONFIGURAÇÃO
# ============================================================

BASE_URL = "http://www.consorcioantiferrugem.net/rest"

# Paraná == 41
UF_ID = 41
UF_SIGLA = "PR"

# ------------------------------------------------------------
# SAFRAS QUE DESEJA EXTRAIR
#
# Formato:
#     ID_DA_API: "nome da safra"
#
# Exemplo:
# SAFRAS = {
# 40: "2025/2026",
# 39: "2024/2025",
# 38: "2023/2024",
# }
#
# ------------------------------------------------------------

SAFRAS = {
    40: "2025/2026",
    38: "2024/2025",
    36: "2023/2024",
    35: "2022/2023",
}

# Apenas ocorrências reais em áreas comerciais que é ID == 1
TIPO_AREA_ID = 1

# Intervalo entre requisições
SLEEP_MIN = 0.4
SLEEP_MAX = 0.9

MAX_RETRIES = 3

session = requests.Session()

session.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/152.0.0.0 Safari/537.36"
        ),
    }
)


def get_json(url, max_retries=MAX_RETRIES):

    for tentativa in range(1, max_retries + 1):

        try:
            response = session.get(url, timeout=30)

            # Servidor pediu para diminuir requisições
            if response.status_code == 429:

                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    espera = int(retry_after)
                else:
                    espera = 10 * tentativa

                print(f"    HTTP 429. " f"Aguardando {espera}s...")

                time.sleep(espera)
                continue

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:

            print(f"    Erro HTTP " f"({tentativa}/{max_retries}): {e}")

        except ValueError as e:

            print(f"    JSON inválido " f"({tentativa}/{max_retries}): {e}")

        if tentativa < max_retries:

            espera = 3 * tentativa

            print(f"    Tentando novamente em {espera}s...")

            time.sleep(espera)

    return None


def extrair_safra_parana(safra_id, safra_nome):

    print("\n" + "=" * 70)
    print(f"SAFRA {safra_nome} - PARANÁ")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Buscar municípios do Paraná que possuem ocorrência
    # --------------------------------------------------------

    url_municipios = (
        f"{BASE_URL}/ocorrencias/all/safra/" f"{safra_id}/{UF_ID}/0/0/(5,1)/0"
    )

    print("\nBuscando municípios...")

    resultado = get_json(url_municipios)

    if resultado is None:
        print("Não foi possível consultar a safra.")
        return pd.DataFrame()

    estados = resultado.get("data", [])

    if not estados:
        print("Nenhum dado encontrado.")
        return pd.DataFrame()

    # Como UF_ID = 41, esperamos apenas o Paraná.
    # Mesmo assim percorremos todos os blocos retornados.
    municipios = []

    total_reportado = 0

    for estado in estados:

        total_reportado += estado.get("totalOcorrencias", 0) or 0

        for municipio in estado.get("ocorrencias", []):

            municipios.append(
                {
                    "codigo_ibge": municipio.get("id"),
                    "municipio": municipio.get("nome"),
                    "qtd_agregada": municipio.get("qtd"),
                    "qtdgeral_agregada": municipio.get("qtdgeral"),
                }
            )

    municipios_unicos = {}

    for municipio in municipios:

        codigo = municipio["codigo_ibge"]

        if codigo is not None:
            municipios_unicos[codigo] = municipio

    municipios = list(municipios_unicos.values())

    print(f"Municípios encontrados: {len(municipios)}")

    print(f"Total reportado pelo mapa: {total_reportado}")

    # --------------------------------------------------------
    # Buscar registros individuais de cada município
    # --------------------------------------------------------

    ocorrencias_comerciais = []

    for i, municipio in enumerate(municipios, start=1):

        codigo_ibge = municipio["codigo_ibge"]
        nome_municipio = municipio["municipio"]

        print(f"[{i}/{len(municipios)}] " f"{nome_municipio} " f"({codigo_ibge})")

        url_detalhes = f"{BASE_URL}/ocorrencias/cidade/" f"{codigo_ibge}/{safra_id}"

        resultado_cidade = get_json(url_detalhes)

        if resultado_cidade is None:

            print("    -> erro ao obter detalhes")

            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

            continue

        registros = resultado_cidade.get("data", [])

        # ----------------------------------------------------
        # IMPORTANTE:
        # Somente registros COMERCIAL
        # tipoDeAreaId == 1
        # ----------------------------------------------------

        comerciais = [
            registro
            for registro in registros
            if registro.get("tipoDeAreaId") == TIPO_AREA_ID
        ]

        print(
            f"    total retornado: {len(registros)} | " f"COMERCIAL: {len(comerciais)}"
        )

        for registro in comerciais:

            registro["municipio"] = nome_municipio
            registro["codigo_ibge"] = codigo_ibge

            registro["uf"] = UF_SIGLA

            registro["safra"] = safra_nome
            registro["safra_id"] = safra_id

            ocorrencias_comerciais.append(registro)

        # Pequena pausa para não acabar sendo banido
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    # --------------------------------------------------------
    # 3. DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(ocorrencias_comerciais)

    if df.empty:

        print(f"\nNenhuma ocorrência comercial " f"encontrada em {safra_nome}.")

        return df

    # --------------------------------------------------------
    # Remover duplicatas pelo ID
    # --------------------------------------------------------

    antes = len(df)

    df = df.drop_duplicates(subset=["id"], keep="first")

    duplicatas = antes - len(df)

    if duplicatas:
        print(f"Duplicatas removidas: {duplicatas}")

    if "data" in df.columns:

        df["data"] = pd.to_datetime(df["data"], errors="coerce")

    df = df[df["tipoDeAreaId"] == TIPO_AREA_ID].copy()

    df = df.sort_values(["data", "municipio", "id"]).reset_index(drop=True)

    print("\nResumo:")

    print(f"  Safra: {safra_nome}")

    print(f"  Ocorrências COMERCIAL: {len(df)}")

    print(f"  Municípios: " f"{df['codigo_ibge'].nunique()}")

    print(f"  Primeira data: " f"{df['data'].min()}")

    print(f"  Última data: " f"{df['data'].max()}")

    # Comparação com endpoint agregado
    print(f"  Total reportado pelo mapa: " f"{total_reportado}")

    if len(df) == total_reportado:

        print("  ✓ quantidade validada")

    else:

        print("  ⚠ quantidade diferente do " "total reportado pelo mapa")

    return df


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "dados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# EXECUTAR TODAS AS SAFRAS INFORMADAS
# ============================================================

todos_dfs = []


for safra_id, safra_nome in SAFRAS.items():

    df_safra = extrair_safra_parana(safra_id, safra_nome)

    if df_safra.empty:
        continue

    # --------------------------------------------------------
    # Salvar cada safra separadamente
    # --------------------------------------------------------

    nome_arquivo = safra_nome.replace("/", "_")

    arquivo_csv = OUTPUT_DIR / f"ocorrencias_PR_{nome_arquivo}.csv"

    # arquivo_parquet = (
    #    f"ocorrencias_PR_{nome_arquivo}.parquet"
    # )

    df_safra.to_csv(arquivo_csv, index=False, encoding="utf-8-sig")

    # df_safra.to_parquet(
    #    arquivo_parquet,
    #    index=False
    # )

    print(f"\nSalvo: {arquivo_csv}")

    todos_dfs.append(df_safra)


# ============================================================
# JUNTAR TODAS AS SAFRAS
# ============================================================

if todos_dfs:

    df_final = pd.concat(todos_dfs, ignore_index=True)

    # ID provavelmente é único globalmente,
    # mas não assumimos isso entre safras.
    df_final = df_final.drop_duplicates(subset=["safra_id", "id"])

    df_final = df_final.sort_values(["data", "municipio", "id"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Salvar arquivo único
    # --------------------------------------------------------

    df_final.to_csv(
        OUTPUT_DIR / "ocorrencias_PR_todas_safras.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # df_final.to_parquet(
    #    "ocorrencias_PR_todas_safras.parquet",
    #    index=False
    # )

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)

    print(f"Total de ocorrências: " f"{len(df_final)}")

    print("\nOcorrências por safra:")

    print(df_final["safra"].value_counts().sort_index())

    print("\nArquivos conjuntos:")

    print("ocorrencias_PR_todas_safras.csv")

    # print(
    #    "ocorrencias_PR_todas_safras.parquet"
    # )

else:

    print("\nNenhuma ocorrência foi coletada.")
