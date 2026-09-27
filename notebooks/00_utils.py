# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Utilitários compartilhados
# MAGIC
# MAGIC Configuração e funções reutilizadas pelos notebooks `01` a `04`. Não é executado sozinho: cada notebook o carrega com `%run ./00_utils`.
# MAGIC
# MAGIC | Bloco | Conteúdo |
# MAGIC |---|---|
# MAGIC | Configuração | nomes de catálogo, schemas, volume e tabelas (fonte única da verdade) |
# MAGIC | `with_comments` / `apply_comments` | grava a documentação de cada tabela e coluna no **Unity Catalog** (catálogo de dados) |
# MAGIC | `DQReport` | registra checagens de qualidade de dados, exibe o resultado e interrompe o pipeline se uma checagem crítica falhar |

# COMMAND ----------

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CATALOG = "mvp-bank-marketing-analytics"  # o Unity Catalog guarda nomes em minúsculas
CATALOG_SQL = f"`{CATALOG}`"  # o hífen exige crases em comandos SQL; caminhos de volume usam o nome puro
SCHEMA_BRONZE, SCHEMA_SILVER, SCHEMA_GOLD = "bronze", "silver", "gold"

LANDING_VOLUME = "landing"
LANDING_PATH = f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/{LANDING_VOLUME}"
SOURCE_FILE_NAME = "bank-additional-full.csv"
SOURCE_PATH = f"{LANDING_PATH}/{SOURCE_FILE_NAME}"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

TBL_BRONZE = f"{CATALOG_SQL}.{SCHEMA_BRONZE}.raw_bank_marketing"
TBL_SILVER = f"{CATALOG_SQL}.{SCHEMA_SILVER}.slv_bank_marketing"
TBL_DIM_CLIENTE = f"{CATALOG_SQL}.{SCHEMA_GOLD}.dim_cliente"
TBL_DIM_CAMPANHA = f"{CATALOG_SQL}.{SCHEMA_GOLD}.dim_campanha"
TBL_DIM_CONTEXTO = f"{CATALOG_SQL}.{SCHEMA_GOLD}.dim_contexto_economico"
TBL_FATO = f"{CATALOG_SQL}.{SCHEMA_GOLD}.fato_contato"

# COMMAND ----------

# ---------------------------------------------------------------------------
# Catálogo de dados: comentários de tabela e coluna no Unity Catalog
# ---------------------------------------------------------------------------
from pyspark.sql import DataFrame, functions as F


def quote_ident(name: str) -> str:
    """Envolve um identificador em crases (necessário para colunas com ponto, ex.: `emp.var.rate`)."""
    return "`" + name.replace("`", "``") + "`"


def sql_str(text: str) -> str:
    """Literal de string SQL com escape de barra invertida e aspas simples."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def with_comments(df: DataFrame, column_comments: dict) -> DataFrame:
    """Embute o comentário de cada coluna nos metadados do schema, para que ele seja gravado junto com a tabela Delta."""
    return df.select([
        F.col(quote_ident(c)).alias(c, metadata={"comment": column_comments[c]}) if c in column_comments else F.col(quote_ident(c))
        for c in df.columns
    ])


def apply_comments(table_fqn: str, table_comment: str, column_comments: dict) -> None:
    """Garante que tabela e colunas estão documentadas no Unity Catalog.

    Compara com o `information_schema` e só executa `COMMENT`/`ALTER` no que estiver diferente,
    evitando poluir o histórico Delta a cada reprocessamento. Falha se alguma coluna ficar sem descrição.
    """
    catalog, schema, table = table_fqn.split(".")
    where = f"table_schema = '{schema}' AND table_name = '{table}'"

    current_table = spark.sql(f"SELECT comment FROM {catalog}.information_schema.tables WHERE {where}").first()
    if current_table is None or current_table.comment != table_comment:
        spark.sql(f"COMMENT ON TABLE {table_fqn} IS {sql_str(table_comment)}")

    current_cols = {
        r.column_name: r.comment
        for r in spark.sql(f"SELECT column_name, comment FROM {catalog}.information_schema.columns WHERE {where}").collect()
    }
    undocumented = set(current_cols) - set(column_comments)
    unknown = set(column_comments) - set(current_cols)
    assert not undocumented, f"Colunas sem descrição no catálogo: {sorted(undocumented)}"
    assert not unknown, f"Descrição para colunas inexistentes: {sorted(unknown)}"

    for col, comment in column_comments.items():
        if current_cols[col] != comment:
            spark.sql(f"ALTER TABLE {table_fqn} ALTER COLUMN {quote_ident(col)} COMMENT {sql_str(comment)}")


# COMMAND ----------

# ---------------------------------------------------------------------------
# Qualidade de dados: registro de checagens
# ---------------------------------------------------------------------------
class DQReport:
    """Acumula checagens de qualidade e produz um relatório tabular.

    Cada checagem compara um valor observado com o esperado:
      - `critical=True`  → divergência = FALHA (interrompe o pipeline em `assert_ok`);
      - `critical=False` → divergência = ALERTA (problema conhecido/documentado cujo volume mudou).
    Problemas conhecidos entram com o volume documentado como "esperado", o que permite detectar
    mudanças na fonte (drift) sem bloquear a carga.
    """

    SCHEMA = "camada string, checagem string, observado long, esperado long, status string, tratamento string"

    def __init__(self, layer: str):
        self.layer = layer
        self.rows = []

    def check(self, name: str, observed, expected=0, critical: bool = True, treatment: str = "") -> None:
        observed, expected = int(observed), int(expected)
        status = "OK" if observed == expected else ("FALHA" if critical else "ALERTA")
        self.rows.append((self.layer, name, observed, expected, status, treatment))

    def to_df(self) -> DataFrame:
        return spark.createDataFrame(self.rows, self.SCHEMA)

    def assert_ok(self) -> None:
        failures = [r for r in self.rows if r[4] == "FALHA"]
        assert not failures, "Checagens críticas falharam:\n" + "\n".join(f"  - {r[1]}: observado={r[2]}, esperado={r[3]}" for r in failures)
