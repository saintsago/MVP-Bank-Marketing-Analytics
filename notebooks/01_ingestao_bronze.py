# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingestão — Camada Bronze
# MAGIC
# MAGIC **Objetivo:** carregar o arquivo bruto `bank-additional-full.csv` (UCI Bank Marketing) para uma tabela Delta no schema `bronze`,
# MAGIC **sem nenhuma transformação de conteúdo**, acrescentando apenas metadados de ingestão.
# MAGIC
# MAGIC | Item | Valor |
# MAGIC |---|---|
# MAGIC | Fonte | UCI Machine Learning Repository — [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
# MAGIC | Arquivo | `bank-additional-full.csv` (41.188 linhas × 21 colunas, separador `;`) |
# MAGIC | Licença | CC BY 4.0 — Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. https://doi.org/10.24432/C5K306 |
# MAGIC | Landing zone | Volume Unity Catalog `mvp_saas_analytics_pipeline.bronze.landing` |
# MAGIC | Destino | `mvp_saas_analytics_pipeline.bronze.raw_bank_marketing` (Delta) |
# MAGIC
# MAGIC **Decisões de design**
# MAGIC - Todas as colunas são lidas como `string` (`inferSchema = false`): o bronze preserva o dado exatamente como chegou; tipagem e validação ficam na silver.
# MAGIC - Carga **full e idempotente** (`overwrite`): o dataset é estático, então reprocessar sempre produz o mesmo resultado. O histórico de versões fica no Delta Lake (time travel).
# MAGIC - Metadados de ingestão: `_ingestion_timestamp`, `_source_file` e `_source_row_number` (posição da linha no arquivo, necessária porque o arquivo
# MAGIC   está ordenado cronologicamente e não tem coluna de ano nem identificador de registro).

# COMMAND ----------

# MAGIC %run ./00_utils

# COMMAND ----------

import io
import os
import urllib.request
import zipfile

from pyspark.sql import Window, functions as F

EXPECTED_ROWS = 41_188
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Coleta: garantir o arquivo bruto na landing zone
# MAGIC
# MAGIC O arquivo é enviado para o Volume `landing` (upload pela CLI do Databricks: `databricks fs cp bank-additional-full.csv dbfs:/Volumes/.../landing/`).
# MAGIC Se ele não estiver lá, o notebook tenta baixá-lo direto da UCI. O zip oficial é **aninhado** (`bank+marketing.zip` → `bank-additional.zip` → CSV).

# COMMAND ----------

spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_BRONZE}.{LANDING_VOLUME}
    COMMENT 'Landing zone: arquivos brutos recebidos da fonte (UCI Bank Marketing) antes da ingestão no bronze'
""")


def download_from_uci(target_path: str) -> None:
    """Baixa o zip oficial da UCI e extrai somente o bank-additional-full.csv para o volume."""
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    member = next(n for n in inner.namelist() if n.endswith(SOURCE_FILE_NAME) and "__MACOSX" not in n)
    with open(target_path, "wb") as f:
        f.write(inner.read(member))


if os.path.exists(SOURCE_PATH):
    print(f"Arquivo já presente na landing zone: {SOURCE_PATH}")
else:
    print(f"Arquivo ausente. Baixando de {UCI_ZIP_URL} ...")
    try:
        download_from_uci(SOURCE_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível baixar da UCI ({e}). Faça o upload manual do CSV para {LANDING_PATH} "
            "(Catalog Explorer → Volume → Upload, ou `databricks fs cp`)."
        ) from e

display(dbutils.fs.ls(LANDING_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Leitura do CSV bruto + metadados de ingestão
# MAGIC
# MAGIC - `_source_file` vem da coluna oculta `_metadata.file_path` (`input_file_name()` não é suportado no Unity Catalog/serverless).
# MAGIC - `_source_row_number` numera as linhas na ordem do arquivo (1 = primeira linha de dados, logo após o cabeçalho). O arquivo tem ~5,8 MB e é lido
# MAGIC   numa única partição, então `monotonically_increasing_id()` preserva a ordem original; a checagem da seção 3 confirma isso.

# COMMAND ----------

df_raw = (
    spark.read.format("csv")
    .option("header", True)
    .option("sep", ";")
    .option("quote", '"')
    .option("inferSchema", False)
    .load(SOURCE_PATH)
)

source_columns = df_raw.columns

df_bronze = (
    df_raw
    .select("*", F.col("_metadata.file_path").alias("_source_file"))
    .withColumn("_mono_id", F.monotonically_increasing_id())
    .withColumn("_source_row_number", F.row_number().over(Window.orderBy("_mono_id")))
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .select(*[quote_ident(c) for c in source_columns], "_source_row_number", "_ingestion_timestamp", "_source_file")
)

df_bronze.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validações de entrada
# MAGIC
# MAGIC Checagens mínimas para aceitar o arquivo: volume, layout de colunas e ordem das linhas (o arquivo começa em maio/2008 e termina em novembro/2010, segundo a UCI).

# COMMAND ----------

dq = DQReport("bronze")

row_count = df_bronze.count()
first_row = df_bronze.orderBy("_source_row_number").select("age", "job", "month").first()
last_row = df_bronze.orderBy(F.desc("_source_row_number")).select("month").first()

dq.check("Total de linhas do arquivo", row_count, EXPECTED_ROWS)
dq.check("Colunas fora do layout esperado", len(set(source_columns) ^ set(EXPECTED_COLUMNS)), 0)
dq.check("Números de linha repetidos", row_count - df_bronze.select("_source_row_number").distinct().count(), 0)
dq.check("Ordem preservada: 1ª linha = (56, housemaid, may)", 0 if tuple(first_row) == ("56", "housemaid", "may") else 1, 0)
dq.check("Ordem preservada: última linha = nov", 0 if last_row.month == "nov" else 1, 0)

display(dq.to_df())
dq.assert_ok()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Documentação (catálogo de dados) e escrita da tabela Delta

# COMMAND ----------

BRONZE_TABLE_COMMENT = (
    "Camada bronze: cópia fiel do arquivo bank-additional-full.csv (UCI Bank Marketing, campanhas de telemarketing de um banco português, "
    "mai/2008 a nov/2010). Todas as colunas de origem em string, sem tratamento. Carga full idempotente. Licença CC BY 4.0."
)

BRONZE_COLUMN_COMMENTS = {
    "age": "Idade do cliente em anos (texto bruto).",
    "job": "Profissão do cliente (texto bruto). Domínio: admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown.",
    "marital": "Estado civil (texto bruto). Domínio: divorced (inclui viúvos), married, single, unknown.",
    "education": "Escolaridade (texto bruto). Domínio: basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown.",
    "default": "Possui crédito em inadimplência? (texto bruto). Domínio: no, yes, unknown.",
    "housing": "Possui financiamento imobiliário? (texto bruto). Domínio: no, yes, unknown.",
    "loan": "Possui empréstimo pessoal? (texto bruto). Domínio: no, yes, unknown.",
    "contact": "Canal do último contato (texto bruto). Domínio: cellular, telephone.",
    "month": "Mês do último contato, abreviado (texto bruto). Domínio: jan a dec.",
    "day_of_week": "Dia da semana do último contato (texto bruto). Domínio: mon a fri.",
    "duration": "Duração do último contato em segundos (texto bruto). Atenção: só é conhecida após a ligação (data leakage).",
    "campaign": "Número de contatos realizados com o cliente nesta campanha, incluindo o último (texto bruto).",
    "pdays": "Dias desde o último contato de uma campanha anterior; 999 = sem registro (texto bruto).",
    "previous": "Número de contatos realizados com o cliente antes desta campanha (texto bruto).",
    "poutcome": "Resultado da campanha anterior (texto bruto). Domínio: failure, nonexistent, success.",
    "emp.var.rate": "Taxa de variação do emprego, indicador trimestral de Portugal (texto bruto).",
    "cons.price.idx": "Índice de preços ao consumidor, indicador mensal (texto bruto).",
    "cons.conf.idx": "Índice de confiança do consumidor, indicador mensal (texto bruto).",
    "euribor3m": "Taxa Euribor de 3 meses, indicador diário (texto bruto).",
    "nr.employed": "Número de empregados em milhares, indicador trimestral (texto bruto).",
    "y": "Variável-alvo: o cliente aderiu ao depósito a prazo? (texto bruto). Domínio: yes, no.",
    "_source_row_number": "Metadado de ingestão: posição da linha no arquivo de origem (1 = primeira linha de dados). Preserva a ordem cronológica e identifica o registro.",
    "_ingestion_timestamp": "Metadado de ingestão: data/hora em que a carga foi executada.",
    "_source_file": "Metadado de ingestão: caminho completo do arquivo de origem no volume landing.",
}

(
    with_comments(df_bronze, BRONZE_COLUMN_COMMENTS)
    .write.format("delta")
    .mode("overwrite")
    .saveAsTable(TBL_BRONZE)
)

apply_comments(TBL_BRONZE, BRONZE_TABLE_COMMENT, BRONZE_COLUMN_COMMENTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evidências da carga

# COMMAND ----------

display(spark.table(TBL_BRONZE).orderBy("_source_row_number").limit(10))

# COMMAND ----------

display(spark.sql(f"DESCRIBE TABLE EXTENDED {TBL_BRONZE}"))

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {TBL_BRONZE}").select("version", "timestamp", "operation", "operationMetrics"))

# COMMAND ----------

print(f"✔ Bronze carregado: {spark.table(TBL_BRONZE).count():,} linhas em {TBL_BRONZE}")
