# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Modelagem — Camada Gold (Star Schema)
# MAGIC
# MAGIC **Objetivo:** decompor a silver em um **modelo dimensional (esquema estrela)**, pronto para consumo analítico.
# MAGIC
# MAGIC ```
# MAGIC                    ┌──────────────────┐
# MAGIC                    │   dim_cliente    │
# MAGIC                    │  (perfil demogr.)│
# MAGIC                    └────────┬─────────┘
# MAGIC                             │ sk_cliente
# MAGIC ┌──────────────────┐  ┌─────┴────────────┐  ┌────────────────────────┐
# MAGIC │   dim_campanha   ├──┤   fato_contato   ├──┤ dim_contexto_economico │
# MAGIC │ (quando/como)    │  │ 1 linha/contato  │  │   (indicadores macro)  │
# MAGIC └──────────────────┘  └──────────────────┘  └────────────────────────┘
# MAGIC     sk_campanha                                 sk_contexto_economico
# MAGIC ```
# MAGIC
# MAGIC | Tabela | Tipo | Grão | Chave natural |
# MAGIC |---|---|---|---|
# MAGIC | `dim_cliente` | Dimensão | 1 perfil demográfico distinto | age, job, marital, education, credit_default, housing, loan |
# MAGIC | `dim_campanha` | Dimensão | 1 combinação distinta de ano, mês, dia, canal e resultado anterior | contact_year, month_num, day_of_week_num, contact, poutcome |
# MAGIC | `dim_contexto_economico` | Dimensão | 1 cenário macroeconômico distinto | emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed |
# MAGIC | `fato_contato` | Fato | 1 cliente contatado na campanha (último contato) | contact_id |
# MAGIC
# MAGIC **Decisões de modelagem**
# MAGIC - **Chaves substitutas (surrogate keys)** inteiras, geradas com `row_number()` sobre as combinações distintas da chave natural **ordenadas**. São determinísticas:
# MAGIC   reprocessar o mesmo dado gera as mesmas chaves.
# MAGIC - O dataset **não tem identificador de cliente**, então `dim_cliente` é uma dimensão de **perfil**: clientes com atributos idênticos compartilham a mesma linha.
# MAGIC - `dim_contexto_economico` isola os 5 indicadores macro, que se repetem em milhares de contatos do mesmo período (375 cenários para 41 mil contatos).
# MAGIC - PK e FK são declaradas no Unity Catalog como constraints **informativas**. Elas documentam o modelo e habilitam o diagrama de relacionamento no Catalog Explorer;
# MAGIC   a integridade referencial é garantida pelas checagens da seção 3.

# COMMAND ----------

# MAGIC %run ./00_utils

# COMMAND ----------

from pyspark.sql import DataFrame, Window, functions as F

CLIENTE_NK = ["age", "job", "marital", "education", "credit_default", "housing", "loan"]
CAMPANHA_NK = ["contact_year", "month_num", "day_of_week_num", "contact", "poutcome"]
CONTEXTO_NK = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]

FACT_MEASURES = ["duration", "duration_minutes", "campaign", "pdays", "previous", "is_previously_contacted", "is_subscribed"]

df_silver = spark.table(TBL_SILVER)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Construção das dimensões

# COMMAND ----------


def build_dimension(df: DataFrame, sk: str, natural_key: list, columns: list) -> DataFrame:
    """Gera uma dimensão com as combinações distintas de `columns` e uma surrogate key inteira, ordenada pela chave natural."""
    return (
        df.select(*columns).distinct()
        .withColumn(sk, F.row_number().over(Window.orderBy(*natural_key)))
        .select(sk, *columns)
    )


dim_cliente = build_dimension(
    df_silver, "sk_cliente", CLIENTE_NK,
    ["age", "age_group", "job", "marital", "education", "credit_default", "housing", "loan"],
)
dim_campanha = build_dimension(
    df_silver, "sk_campanha", CAMPANHA_NK,
    ["contact_year", "month", "month_num", "day_of_week", "day_of_week_num", "contact", "poutcome"],
)
dim_contexto = build_dimension(df_silver, "sk_contexto_economico", CONTEXTO_NK, CONTEXTO_NK)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Construção da fato
# MAGIC Cada contato da silver recebe as surrogate keys das três dimensões por *join* na chave natural.

# COMMAND ----------

fato_contato = (
    df_silver
    .join(dim_cliente.select("sk_cliente", *CLIENTE_NK), CLIENTE_NK, "left")
    .join(dim_campanha.select("sk_campanha", *CAMPANHA_NK), CAMPANHA_NK, "left")
    .join(dim_contexto.select("sk_contexto_economico", *CONTEXTO_NK), CONTEXTO_NK, "left")
    .select("contact_id", "sk_cliente", "sk_campanha", "sk_contexto_economico", *FACT_MEASURES)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validação de integridade (antes de gravar)
# MAGIC O *join* é `left`, então um contato sem correspondência na dimensão apareceria com FK nula em vez de sumir silenciosamente. As checagens abaixo garantem que isso não ocorre.

# COMMAND ----------

dq = DQReport("gold")

silver_stats = df_silver.agg(F.count("*").alias("rows"), F.sum(F.col("is_subscribed").cast("int")).alias("subs")).first()
fato_stats = fato_contato.agg(
    F.count("*").alias("rows"),
    F.countDistinct("contact_id").alias("ids"),
    F.sum(F.col("is_subscribed").cast("int")).alias("subs"),
    *[F.sum(F.col(fk).isNull().cast("int")).alias(f"null_{fk}") for fk in ["sk_cliente", "sk_campanha", "sk_contexto_economico"]],
).first()

dq.check("Linhas da fato = linhas da silver", abs(fato_stats.rows - silver_stats.rows), 0)
dq.check("contact_id repetido na fato", fato_stats.rows - fato_stats.ids, 0)
dq.check("Total de adesões fato = silver", abs(fato_stats.subs - silver_stats.subs), 0)
for fk in ["sk_cliente", "sk_campanha", "sk_contexto_economico"]:
    dq.check(f"FK nula (sem correspondência): {fk}", fato_stats[f"null_{fk}"], 0)

for name, dim, sk, nk in [
    ("dim_cliente", dim_cliente, "sk_cliente", CLIENTE_NK),
    ("dim_campanha", dim_campanha, "sk_campanha", CAMPANHA_NK),
    ("dim_contexto_economico", dim_contexto, "sk_contexto_economico", CONTEXTO_NK),
]:
    s = dim.agg(F.count("*").alias("rows"), F.countDistinct(sk).alias("sks"), F.countDistinct(*nk).alias("nks")).first()
    dq.check(f"{name}: surrogate key repetida", s.rows - s.sks, 0)
    dq.check(f"{name}: chave natural repetida", s.rows - s.nks, 0)

display(dq.to_df())
dq.assert_ok()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Catálogo de dados

# COMMAND ----------

TABLE_COMMENTS = {
    TBL_DIM_CLIENTE: "Gold | Dimensão de perfil demográfico do cliente. Não existe ID de cliente na fonte: cada linha é um perfil distinto "
                     "(idade, profissão, estado civil, escolaridade, crédito). Origem: silver.slv_bank_marketing.",
    TBL_DIM_CAMPANHA: "Gold | Dimensão do contato: quando (ano inferido, mês, dia da semana), por qual canal e qual foi o resultado da campanha anterior. "
                      "Origem: silver.slv_bank_marketing.",
    TBL_DIM_CONTEXTO: "Gold | Dimensão de contexto macroeconômico de Portugal no momento do contato (emprego, preços, confiança, Euribor). "
                      "Origem: silver.slv_bank_marketing.",
    TBL_FATO: "Gold | Fato de contatos de telemarketing. Grão: 1 cliente contatado na campanha (atributos do último contato). "
              "Métricas de esforço (campaign, duration) e resultado (is_subscribed). duration tem data leakage.",
}

COLUMN_COMMENTS = {
    TBL_DIM_CLIENTE: {
        "sk_cliente": "Surrogate key do perfil de cliente (PK). row_number() ordenado pela chave natural.",
        "age": "Idade do cliente em anos (17 a 98).",
        "age_group": "Faixa etária: 17-24, 25-34, 35-44, 45-54, 55-64, 65+.",
        "job": "Profissão (12 categorias, inclui unknown).",
        "marital": "Estado civil: divorced (inclui viúvos), married, single, unknown.",
        "education": "Escolaridade (8 categorias, inclui unknown).",
        "credit_default": "Possui crédito em inadimplência? no, yes, unknown.",
        "housing": "Possui financiamento imobiliário? no, yes, unknown.",
        "loan": "Possui empréstimo pessoal? no, yes, unknown.",
    },
    TBL_DIM_CAMPANHA: {
        "sk_campanha": "Surrogate key da combinação de contato (PK).",
        "contact_year": "Ano do contato (2008-2010), inferido pela ordem cronológica do arquivo de origem.",
        "month": "Mês do contato (abreviado). Não há contatos em jan e fev.",
        "month_num": "Número do mês (1-12).",
        "day_of_week": "Dia da semana do contato (mon a fri).",
        "day_of_week_num": "Número do dia da semana (1 = mon ... 5 = fri).",
        "contact": "Canal do contato: cellular ou telephone.",
        "poutcome": "Resultado da campanha anterior: failure, nonexistent, success.",
    },
    TBL_DIM_CONTEXTO: {
        "sk_contexto_economico": "Surrogate key do cenário macroeconômico (PK).",
        "emp_var_rate": "Taxa de variação do emprego (trimestral).",
        "cons_price_idx": "Índice de preços ao consumidor (mensal).",
        "cons_conf_idx": "Índice de confiança do consumidor (mensal; valores negativos = pessimismo).",
        "euribor3m": "Taxa Euribor de 3 meses em % (diária).",
        "nr_employed": "Número de empregados em milhares (trimestral).",
    },
    TBL_FATO: {
        "contact_id": "Identificador do contato (PK); linhagem até a linha do arquivo de origem (bronze._source_row_number).",
        "sk_cliente": "FK para dim_cliente.",
        "sk_campanha": "FK para dim_campanha.",
        "sk_contexto_economico": "FK para dim_contexto_economico.",
        "duration": "Duração do último contato em segundos. DATA LEAKAGE: usar só em análise descritiva.",
        "duration_minutes": "Duração do último contato em minutos. DATA LEAKAGE: usar só em análise descritiva.",
        "campaign": "Número de contatos com o cliente nesta campanha (inclui o último).",
        "pdays": "Dias desde o contato da campanha anterior; NULL = sem registro.",
        "previous": "Número de contatos antes desta campanha.",
        "is_previously_contacted": "True se o cliente foi contatado em campanha anterior (previous > 0).",
        "is_subscribed": "Variável-alvo: True se o cliente aderiu ao depósito a prazo.",
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Escrita das tabelas Delta e constraints PK/FK
# MAGIC Para permitir o reprocessamento idempotente, as FKs da fato são removidas antes de reescrever as dimensões e recriadas no final.

# COMMAND ----------

FOREIGN_KEYS = {  # nome -> (coluna na fato, dimensão referenciada)
    "fk_fato_contato_cliente": ("sk_cliente", TBL_DIM_CLIENTE),
    "fk_fato_contato_campanha": ("sk_campanha", TBL_DIM_CAMPANHA),
    "fk_fato_contato_contexto": ("sk_contexto_economico", TBL_DIM_CONTEXTO),
}
PRIMARY_KEYS = {
    TBL_DIM_CLIENTE: ("pk_dim_cliente", "sk_cliente"),
    TBL_DIM_CAMPANHA: ("pk_dim_campanha", "sk_campanha"),
    TBL_DIM_CONTEXTO: ("pk_dim_contexto_economico", "sk_contexto_economico"),
    TBL_FATO: ("pk_fato_contato", "contact_id"),
}


def existing_constraints() -> set:
    return {r.constraint_name for r in spark.sql(
        f"SELECT constraint_name FROM {CATALOG_SQL}.information_schema.table_constraints WHERE table_schema = '{SCHEMA_GOLD}'"
    ).collect()}


if spark.catalog.tableExists(TBL_FATO):
    for fk_name in FOREIGN_KEYS:
        spark.sql(f"ALTER TABLE {TBL_FATO} DROP CONSTRAINT IF EXISTS {fk_name}")

for table, df in [(TBL_DIM_CLIENTE, dim_cliente), (TBL_DIM_CAMPANHA, dim_campanha), (TBL_DIM_CONTEXTO, dim_contexto), (TBL_FATO, fato_contato)]:
    (
        with_comments(df, COLUMN_COMMENTS[table])
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )
    apply_comments(table, TABLE_COMMENTS[table], COLUMN_COMMENTS[table])

current = existing_constraints()
for table, (pk_name, pk_col) in PRIMARY_KEYS.items():
    if pk_name not in current:
        spark.sql(f"ALTER TABLE {table} ALTER COLUMN {pk_col} SET NOT NULL")
        spark.sql(f"ALTER TABLE {table} ADD CONSTRAINT {pk_name} PRIMARY KEY ({pk_col})")
for fk_name, (fk_col, dim_table) in FOREIGN_KEYS.items():
    if fk_name not in current:
        spark.sql(f"ALTER TABLE {TBL_FATO} ADD CONSTRAINT {fk_name} FOREIGN KEY ({fk_col}) REFERENCES {dim_table}")

display(spark.sql(f"""
    SELECT table_name, constraint_name, constraint_type
    FROM {CATALOG_SQL}.information_schema.table_constraints
    WHERE table_schema = '{SCHEMA_GOLD}'
    ORDER BY table_name, constraint_type DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Reconciliação pós-carga (tabelas persistidas)
# MAGIC Verifica, já nas tabelas gravadas, que não existem contatos órfãos (FK sem dimensão) e que a taxa de conversão da gold é idêntica à da silver.

# COMMAND ----------

dq_post = DQReport("gold (pós-carga)")
for fk_name, (fk_col, dim_table) in FOREIGN_KEYS.items():
    orphans = spark.table(TBL_FATO).join(spark.table(dim_table), fk_col, "left_anti").count()
    dq_post.check(f"Contatos órfãos em {fk_col}", orphans, 0)

rates = spark.sql(f"""
    SELECT (SELECT ROUND(100 * AVG(CAST(is_subscribed AS INT)), 4) FROM {TBL_SILVER}) AS taxa_silver,
           (SELECT ROUND(100 * AVG(CAST(is_subscribed AS INT)), 4) FROM {TBL_FATO})   AS taxa_gold
""").first()
dq_post.check("Taxa de conversão silver = gold (x10.000)", round(abs(rates.taxa_silver - rates.taxa_gold) * 10_000), 0)

display(dq_post.to_df())
dq_post.assert_ok()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evidências

# COMMAND ----------

display(spark.sql(f"""
    SELECT 'dim_cliente' AS tabela, COUNT(*) AS linhas FROM {TBL_DIM_CLIENTE}
    UNION ALL SELECT 'dim_campanha', COUNT(*) FROM {TBL_DIM_CAMPANHA}
    UNION ALL SELECT 'dim_contexto_economico', COUNT(*) FROM {TBL_DIM_CONTEXTO}
    UNION ALL SELECT 'fato_contato', COUNT(*) FROM {TBL_FATO}
"""))

# COMMAND ----------

# Exemplo de consulta no modelo estrela: conversão por canal e resultado da campanha anterior
display(spark.sql(f"""
    SELECT c.contact, c.poutcome,
           COUNT(*) AS contatos,
           ROUND(100 * AVG(CAST(f.is_subscribed AS INT)), 2) AS taxa_conversao_pct
    FROM {TBL_FATO} f
    JOIN {TBL_DIM_CAMPANHA} c ON f.sk_campanha = c.sk_campanha
    GROUP BY c.contact, c.poutcome
    ORDER BY c.contact, c.poutcome
"""))

# COMMAND ----------

display(spark.table(TBL_FATO).orderBy("contact_id").limit(10))

# COMMAND ----------

print("✔ Gold gravada:", ", ".join(t.split(".")[-1] for t in [TBL_DIM_CLIENTE, TBL_DIM_CAMPANHA, TBL_DIM_CONTEXTO, TBL_FATO]))
