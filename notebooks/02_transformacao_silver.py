# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Transformação — Camada Silver
# MAGIC
# MAGIC **Objetivo:** transformar o bronze em um dado **limpo, tipado, validado e enriquecido**, na relação 1:1 com a fonte (sem joins).
# MAGIC
# MAGIC | Item | Valor |
# MAGIC |---|---|
# MAGIC | Origem | `mvp_saas_analytics_pipeline.bronze.raw_bank_marketing` |
# MAGIC | Destino | `mvp_saas_analytics_pipeline.silver.slv_bank_marketing` (Delta) |
# MAGIC | Grão | 1 linha = 1 cliente contatado na campanha (atributos do último contato) |
# MAGIC
# MAGIC **Transformações** (a seção 6 documenta cada decisão)
# MAGIC 1. Tipagem (`try_cast`), com checagem de falhas de conversão.
# MAGIC 2. Enriquecimento temporal: `month_num`, `day_of_week_num` e `contact_year` (inferido pela ordem cronológica do arquivo).
# MAGIC 3. Remoção de **12 linhas duplicadas**.
# MAGIC 4. Renomeação para snake_case (`emp.var.rate` → `emp_var_rate` …) e `default` → `credit_default`.
# MAGIC 5. `pdays = 999` → `null`; flag `is_previously_contacted` baseada em `previous > 0`.
# MAGIC 6. Derivadas: `duration_minutes`, `age_group`, `is_subscribed` (a partir de `y`).
# MAGIC 7. Remoção dos metadados de ingestão (`_ingestion_timestamp`, `_source_file`); `_source_row_number` vira `contact_id`.

# COMMAND ----------

# MAGIC %run ./00_utils

# COMMAND ----------

from pyspark.sql import Window, functions as F

FIRST_YEAR = 2008  # UCI: "ordered by date (from May 2008 to November 2010)"

MONTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
DAY_NUM = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}

INT_COLUMNS = ["age", "duration", "campaign", "pdays", "previous"]
DOUBLE_COLUMNS = {  # nome na origem -> nome na silver
    "emp.var.rate": "emp_var_rate",
    "cons.price.idx": "cons_price_idx",
    "cons.conf.idx": "cons_conf_idx",
    "euribor3m": "euribor3m",
    "nr.employed": "nr_employed",
}
CATEGORICAL_COLUMNS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome", "y"]

# Domínios válidos, conforme o dicionário de dados da UCI (bank-additional-names.txt)
DOMAINS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
            "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"],
    "credit_default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": list(MONTH_NUM),
    "day_of_week": list(DAY_NUM),
    "poutcome": ["failure", "nonexistent", "success"],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Diagnóstico do dado bruto
# MAGIC
# MAGIC Antes de transformar, medimos o bronze: valores nulos/vazios, cardinalidade e duplicatas. Esse diagnóstico embasa os tratamentos das próximas seções.

# COMMAND ----------

df_bronze = spark.table(TBL_BRONZE)
source_columns = [c for c in df_bronze.columns if not c.startswith("_")]

profile = df_bronze.select(
    *[F.sum(F.when(F.col(quote_ident(c)).isNull() | (F.trim(F.col(quote_ident(c))) == ""), 1).otherwise(0)).alias(f"{c}__nulos") for c in source_columns],
    *[F.countDistinct(F.col(quote_ident(c))).alias(f"{c}__distintos") for c in source_columns],
    *[F.sum(F.when(F.col(quote_ident(c)) == "unknown", 1).otherwise(0)).alias(f"{c}__unknown") for c in source_columns],
).first().asDict()

display(spark.createDataFrame(
    [(c, profile[f"{c}__nulos"], profile[f"{c}__distintos"], profile[f"{c}__unknown"]) for c in source_columns],
    "coluna string, nulos_ou_vazios long, valores_distintos long, qtd_unknown long",
))

# COMMAND ----------

bronze_rows = df_bronze.count()
duplicate_groups = (
    df_bronze.groupBy(*[quote_ident(c) for c in source_columns])
    .agg(F.count("*").alias("ocorrencias"), F.sort_array(F.collect_list("_source_row_number")).alias("linhas_no_arquivo"))
    .filter("ocorrencias > 1")
)
print(f"Linhas no bronze: {bronze_rows:,} | grupos duplicados: {duplicate_groups.count()}")
display(duplicate_groups.select("linhas_no_arquivo", "ocorrencias", "age", "job", "month", "day_of_week", "duration", "y"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tipagem
# MAGIC
# MAGIC `try_cast` devolve `null` quando a conversão falha, em vez de abortar. Depois comparamos os nulos gerados com os nulos de origem,
# MAGIC e qualquer diferença vira falha de qualidade (seção 5). Colunas categóricas recebem `trim` + `lower` por segurança.

# COMMAND ----------

df_typed = df_bronze.select(
    F.col("_source_row_number").alias("contact_id"),
    *[F.expr(f"try_cast({c} AS INT)").alias(c) for c in INT_COLUMNS],
    *[F.expr(f"try_cast({quote_ident(src)} AS DOUBLE)").alias(dst) for src, dst in DOUBLE_COLUMNS.items()],
    *[F.lower(F.trim(F.col(c))).alias(c) for c in CATEGORICAL_COLUMNS],
)

cast_failures = df_bronze.select(
    *[F.sum(F.when(F.col(c).isNotNull() & F.expr(f"try_cast({c} AS INT)").isNull(), 1).otherwise(0)).alias(c) for c in INT_COLUMNS],
    *[F.sum(F.when(F.col(quote_ident(s)).isNotNull() & F.expr(f"try_cast({quote_ident(s)} AS DOUBLE)").isNull(), 1).otherwise(0)).alias(d)
      for s, d in DOUBLE_COLUMNS.items()],
).first().asDict()
print("Falhas de conversão por coluna:", cast_failures)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Perfil de qualidade por atributo
# MAGIC
# MAGIC Cada um dos 21 atributos é avaliado **no dado capturado** (já tipado, antes de qualquer tratamento), nas cinco dimensões de qualidade:
# MAGIC
# MAGIC | Dimensão | Como é medida |
# MAGIC |---|---|
# MAGIC | Completude | nulos/vazios e % de `unknown` (ausência de informação codificada como categoria) |
# MAGIC | Consistência | valores fora do domínio documentado (categóricos) ou que falharam na conversão numérica |
# MAGIC | Unicidade | nº de valores distintos (a unicidade de linha, ou seja, as duplicatas, é tratada na seção 3) |
# MAGIC | Acurácia | valores fora de uma faixa plausível para o contexto (ex.: idade entre 17 e 100) |
# MAGIC | Outliers | numéricos: regra do IQR, valores fora de [Q1 − 1,5·IQR, Q3 + 1,5·IQR]; categóricos: categorias raras (< 0,5% das linhas) |
# MAGIC
# MAGIC Em `pdays`, o valor 999 é um código sentinela ("sem registro"), não uma medida, e fica fora das estatísticas numéricas.

# COMMAND ----------

PLAUSIBLE_RANGES = {  # acurácia: faixa plausível no contexto do negócio
    "age": (17, 100), "duration": (0, 7200), "campaign": (1, 100), "pdays": (0, 999), "previous": (0, 50),
    "emp_var_rate": (-10, 10), "cons_price_idx": (80, 110), "cons_conf_idx": (-100, 0), "euribor3m": (0, 10), "nr_employed": (4000, 6000),
}
RARE_SHARE = 0.005

numeric_cols = INT_COLUMNS + list(DOUBLE_COLUMNS.values())
source_name = {**{c: c for c in INT_COLUMNS + CATEGORICAL_COLUMNS}, **{dst: src for src, dst in DOUBLE_COLUMNS.items()}}
raw_domains = {("default" if c == "credit_default" else c): dom for c, dom in DOMAINS.items()} | {"y": ["yes", "no"]}
measure = {c: F.when(F.col(c) != 999, F.col(c)) if c == "pdays" else F.col(c) for c in numeric_cols}

# 1ª passada: mínimos, máximos e quartis dos numéricos
qp_stats = df_typed.select(
    *[F.min(e).alias(f"{c}__min") for c, e in measure.items()],
    *[F.max(e).alias(f"{c}__max") for c, e in measure.items()],
    *[F.percentile_approx(e, [0.25, 0.75], 10000).alias(f"{c}__q") for c, e in measure.items()],
).first().asDict()
bounds = {}
for c in numeric_cols:
    q1, q3 = qp_stats[f"{c}__q"]
    bounds[c] = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))

# 2ª passada: contagens de outliers, acurácia, distintos, unknown e domínio
qp_counts = df_typed.select(
    *[F.sum(F.when((e < bounds[c][0]) | (e > bounds[c][1]), 1).otherwise(0)).alias(f"{c}__out") for c, e in measure.items()],
    *[F.sum(F.when(~F.col(c).between(lo, hi), 1).otherwise(0)).alias(f"{c}__acc") for c, (lo, hi) in PLAUSIBLE_RANGES.items()],
    *[F.countDistinct(c).alias(f"{c}__dist") for c in numeric_cols + CATEGORICAL_COLUMNS],
    *[F.sum(F.when(F.col(c) == "unknown", 1).otherwise(0)).alias(f"{c}__unk") for c in CATEGORICAL_COLUMNS],
    *[F.sum(F.when(~F.col(c).isin(dom), 1).otherwise(0)).alias(f"{c}__dom") for c, dom in raw_domains.items()],
).first().asDict()


def fmt_num(x) -> str:
    return f"{x:g}" if x is not None else None


rows = []
for c in numeric_cols:
    lo, hi = bounds[c]
    rows.append((
        source_name[c], "numérico", profile[f"{source_name[c]}__nulos"], 0.0, cast_failures[c], qp_counts[f"{c}__dist"], qp_counts[f"{c}__acc"],
        f"{fmt_num(qp_stats[f'{c}__min'])} a {fmt_num(qp_stats[f'{c}__max'])}", f"[{lo:.4g} ; {hi:.4g}]",
        qp_counts[f"{c}__out"], round(100 * qp_counts[f"{c}__out"] / bronze_rows, 2), None,
    ))
for c in CATEGORICAL_COLUMNS:
    counts = {r[0]: r[1] for r in df_typed.groupBy(c).count().collect()}
    rare = {k: n for k, n in counts.items() if n < RARE_SHARE * bronze_rows}
    rows.append((
        source_name[c], "categórico", profile[f"{c}__nulos"], round(100 * qp_counts[f"{c}__unk"] / bronze_rows, 2), qp_counts[f"{c}__dom"], qp_counts[f"{c}__dist"], 0,
        None, None, sum(rare.values()), round(100 * sum(rare.values()) / bronze_rows, 2),
        ", ".join(f"{k} ({n})" for k, n in sorted(rare.items(), key=lambda kv: kv[1])) or None,
    ))

quality_profile = spark.createDataFrame(rows, """
    atributo string, tipo string, nulos long, unknown_pct double, fora_dominio_ou_conversao long, distintos long,
    fora_faixa_plausivel long, min_max string, limites_iqr string, outliers long, outliers_pct double, categorias_raras string
""")
display(quality_profile)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Enriquecimento temporal e remoção de duplicatas
# MAGIC
# MAGIC **`contact_year` (inferido).** O dataset não traz o ano, mas a UCI informa que as linhas estão **ordenadas por data, de maio/2008 a novembro/2010**.
# MAGIC Percorrendo o arquivo em ordem (`contact_id`), cada vez que o número do mês **diminui** (ex.: dez → mar) começa um novo ano.
# MAGIC Isso acontece exatamente 2 vezes, o que produz os anos 2008, 2009 e 2010, consistente com a documentação.
# MAGIC O ano é calculado **antes** da deduplicação, para usar a sequência completa.
# MAGIC
# MAGIC **Duplicatas.** 12 linhas são idênticas a outra linha em todas as 21 colunas de origem. Como o dataset não tem identificador de cliente, não há como
# MAGIC distinguir dois contatos legítimos de um registro repetido; como os pares são idênticos até na duração da ligação em segundos, tratamos como duplicidade
# MAGIC e mantemos a **primeira ocorrência** (menor `contact_id`).

# COMMAND ----------

month_map = F.create_map(*[F.lit(x) for kv in MONTH_NUM.items() for x in kv])
day_map = F.create_map(*[F.lit(x) for kv in DAY_NUM.items() for x in kv])
file_order = Window.orderBy("contact_id")

df_time = (
    df_typed
    .withColumn("month_num", F.element_at(month_map, F.col("month")))
    .withColumn("day_of_week_num", F.element_at(day_map, F.col("day_of_week")))
    .withColumn("_year_turn", F.when(F.col("month_num") < F.lag("month_num").over(file_order), 1).otherwise(0))
    .withColumn("contact_year", (F.lit(FIRST_YEAR) + F.sum("_year_turn").over(file_order.rowsBetween(Window.unboundedPreceding, 0))).cast("int"))
    .drop("_year_turn")
)

business_columns = INT_COLUMNS + list(DOUBLE_COLUMNS.values()) + CATEGORICAL_COLUMNS
first_occurrence = Window.partitionBy(*business_columns).orderBy("contact_id")

df_dedup = (
    df_time
    .withColumn("_occurrence", F.row_number().over(first_occurrence))
    .filter("_occurrence = 1")
    .drop("_occurrence")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Limpeza, padronização e colunas derivadas

# COMMAND ----------

df_silver = (
    df_dedup
    .withColumnRenamed("default", "credit_default")
    # pdays = 999 significa "sem registro de dias desde o contato anterior" -> null (evita distorcer médias)
    .withColumn("pdays", F.when(F.col("pdays") == 999, F.lit(None).cast("int")).otherwise(F.col("pdays")))
    # contato prévio definido por `previous` (consistente com poutcome); ver seção 6
    .withColumn("is_previously_contacted", F.col("previous") > 0)
    .withColumn("duration_minutes", F.round(F.col("duration") / 60.0, 2))
    .withColumn(
        "age_group",
        F.when(F.col("age") <= 24, "17-24")
        .when(F.col("age") <= 34, "25-34")
        .when(F.col("age") <= 44, "35-44")
        .when(F.col("age") <= 54, "45-54")
        .when(F.col("age") <= 64, "55-64")
        .otherwise("65+"),
    )
    .withColumn("is_subscribed", F.col("y") == "yes")
    .select(
        "contact_id", "contact_year",
        # perfil do cliente
        "age", "age_group", "job", "marital", "education", "credit_default", "housing", "loan",
        # contato atual
        "contact", "month", "month_num", "day_of_week", "day_of_week_num", "duration", "duration_minutes", "campaign",
        # histórico de campanhas anteriores
        "pdays", "previous", "poutcome", "is_previously_contacted",
        # contexto econômico
        "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
        # alvo
        "is_subscribed",
    )
)

df_silver.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validação de qualidade de dados
# MAGIC
# MAGIC - **Críticas** (`FALHA` interrompe o pipeline): integridade, tipagem, nulos, domínios, faixas e regras de consistência.
# MAGIC - **Problemas conhecidos** (`ALERTA` se o volume mudar): anomalias documentadas da fonte, tratadas ou aceitas conscientemente. O valor esperado é o volume
# MAGIC   observado na versão atual do arquivo, então qualquer mudança na fonte aparece no relatório.

# COMMAND ----------

required_columns = [c for c in df_silver.columns if c != "pdays"]  # pdays é a única coluna que pode ser nula, por design

# Cada regra é uma condição SQL que identifica uma VIOLAÇÃO. Todas são contadas numa única passada sobre o DataFrame.
RULES = {
    "nulls": " OR ".join(f"{c} IS NULL" for c in required_columns),
    **{f"domain_{c}": f"NOT ({c} IN ({', '.join(repr(v) for v in dom)}))" for c, dom in DOMAINS.items()},
    "age_range": "age NOT BETWEEN 17 AND 100",
    "duration_negative": "duration < 0",
    "campaign_lt_1": "campaign < 1",
    "previous_negative": "previous < 0",
    "pdays_range": "pdays IS NOT NULL AND pdays NOT BETWEEN 0 AND 30",
    "euribor_range": "euribor3m NOT BETWEEN 0 AND 10",
    "previous_vs_poutcome": "(previous = 0) <> (poutcome = 'nonexistent')",
    "duration0_subscribed": "duration = 0 AND is_subscribed",
    "pdays_conflict": "pdays IS NULL AND previous > 0",
    "duration0": "duration = 0",
    "age_17": "age < 18",
    "default_yes": "credit_default = 'yes'",
    "any_unknown": "array_contains(array(job, marital, education, credit_default, housing, loan), 'unknown')",
}

v = df_silver.select(
    F.count("*").alias("rows"),
    F.countDistinct("contact_id").alias("distinct_ids"),
    F.countDistinct("contact_year").alias("distinct_years"),
    *[F.sum(F.when(F.expr(cond), 1).otherwise(0)).alias(name) for name, cond in RULES.items()],
).first().asDict()

silver_rows = v["rows"]
dq = DQReport("silver")

# Integridade e volume
dq.check("Duplicatas removidas", bronze_rows - silver_rows, 12, critical=False, treatment="Mantida a 1ª ocorrência (menor contact_id)")
dq.check("contact_id repetido", silver_rows - v["distinct_ids"], 0)

# Tipagem e completude
for col_name, failures in cast_failures.items():
    dq.check(f"Falha de conversão numérica: {col_name}", failures, 0)
dq.check("Linhas com nulo em coluna obrigatória", v["nulls"], 0)

# Domínios categóricos
for col_name in DOMAINS:
    dq.check(f"Fora do domínio: {col_name}", v[f"domain_{col_name}"], 0)

# Faixas numéricas
dq.check("age fora de [17, 100]", v["age_range"], 0)
dq.check("duration negativa", v["duration_negative"], 0)
dq.check("campaign < 1", v["campaign_lt_1"], 0)
dq.check("previous negativo", v["previous_negative"], 0)
dq.check("pdays (preenchido) fora de [0, 30]", v["pdays_range"], 0)
dq.check("euribor3m fora de [0, 10]", v["euribor_range"], 0)

# Consistência entre colunas
dq.check("previous = 0 divergente de poutcome = nonexistent", v["previous_vs_poutcome"], 0)
dq.check("duration = 0 com adesão (impossível segundo a UCI)", v["duration0_subscribed"], 0)
dq.check("Viradas de ano na sequência de meses", v["distinct_years"] - 1, 2)

# Problemas conhecidos da fonte (documentados)
dq.check("pdays sem registro (999) mas previous > 0", v["pdays_conflict"], 4_110, critical=False,
         treatment="pdays -> null; flag de contato prévio usa previous > 0")
dq.check("duration = 0 (ligação não completada)", v["duration0"], 4, critical=False,
         treatment="Mantido; duration não é usada para prever")
dq.check("Clientes com 17 anos (abaixo da faixa 18-24)", v["age_17"], 5, critical=False,
         treatment="Primeira faixa etária definida como 17-24")
dq.check("credit_default = yes (classe quase vazia)", v["default_yes"], 3, critical=False,
         treatment="Mantido; baixa variância documentada")
dq.check("Linhas com algum atributo 'unknown'", v["any_unknown"], 10_698, critical=False,
         treatment="unknown mantido como categoria válida")

display(dq.to_df())
dq.assert_ok()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Decisões de tratamento (documentação)
# MAGIC
# MAGIC | Tema | Decisão | Justificativa |
# MAGIC |---|---|---|
# MAGIC | **`unknown`** | Mantido como categoria válida em `job`, `marital`, `education`, `credit_default`, `housing` e `loan` | A própria UCI sugere tratá-lo como classe. Em `credit_default` o `unknown` (≈21%) é informativo: clientes que não declaram inadimplência convertem de forma diferente. Excluir as 10.698 linhas (26%) enviesaria a análise, e imputar criaria informação inexistente. |
# MAGIC | **`pdays = 999`** | Convertido para `null` | 999 é um código sentinela, não uma medida; mantê-lo distorceria médias e correlações. |
# MAGIC | **`is_previously_contacted`** | `previous > 0` (e não `pdays IS NOT NULL`) | Em 4.110 linhas `pdays = 999` mas `previous > 0` e `poutcome = failure`: o cliente **foi** contatado antes, só não tem a data registrada. No `bank-full.csv` original essa inconsistência não existe, e `previous > 0` coincide exatamente com `poutcome <> 'nonexistent'`. Por isso, na silver `pdays = null` significa "sem registro de dias", não "nunca contatado". |
# MAGIC | **Duplicatas** | 12 removidas | Linhas idênticas em todas as 21 colunas, inclusive a duração em segundos. |
# MAGIC | **`age_group`** | 17-24, 25-34, 35-44, 45-54, 55-64, 65+ | A menor idade observada é 17 (5 clientes), então a primeira faixa começa em 17. |
# MAGIC | **`default` → `credit_default`** | Renomeada | `DEFAULT` é palavra reservada em SQL; o novo nome também é mais descritivo. |
# MAGIC | **`duration`** | Mantida, com `duration_minutes` | **Data leakage:** só é conhecida depois da ligação, e ligações longas quase sempre indicam adesão. Serve para análise descritiva, **nunca** como variável de um modelo preditivo. |
# MAGIC | **`contact_year`** | Inferido pela ordem do arquivo | Suposição documentada: depende da ordenação cronológica declarada pela UCI (validada: exatamente 2 viradas de ano). |
# MAGIC | **Outliers** (`campaign` até 56, `duration` até 4.918 s) | Mantidos | São valores plausíveis de operação, não erros; a análise usa faixas. |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Catálogo de dados e escrita da tabela Delta

# COMMAND ----------

SILVER_TABLE_COMMENT = (
    "Camada silver: contatos de telemarketing do banco (UCI Bank Marketing, 2008-2010) limpos, tipados, deduplicados e enriquecidos. "
    "Relação 1:1 com o bronze (menos 12 duplicatas). Grão: 1 linha por cliente contatado na campanha. "
    "Categoria unknown mantida como válida. duration tem data leakage e não deve ser usada em modelos preditivos."
)

SILVER_COLUMN_COMMENTS = {
    "contact_id": "Identificador do contato = posição da linha no arquivo de origem (linhagem: bronze._source_row_number). Chave única.",
    "contact_year": "Ano do contato (2008-2010), INFERIDO pela ordem cronológica do arquivo: cada vez que o mês diminui, começa um novo ano.",
    "age": "Idade do cliente em anos. Faixa observada: 17 a 98.",
    "age_group": "Faixa etária derivada de age: 17-24, 25-34, 35-44, 45-54, 55-64, 65+.",
    "job": "Profissão do cliente. unknown mantido como categoria válida.",
    "marital": "Estado civil: divorced (inclui viúvos), married, single, unknown.",
    "education": "Escolaridade: basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown.",
    "credit_default": "Possui crédito em inadimplência? (origem: default, renomeada por ser palavra reservada em SQL). Domínio: no, yes, unknown.",
    "housing": "Possui financiamento imobiliário? no, yes, unknown.",
    "loan": "Possui empréstimo pessoal? no, yes, unknown.",
    "contact": "Canal do último contato: cellular ou telephone.",
    "month": "Mês do último contato (abreviado, jan a dec). Não há contatos em jan e fev.",
    "month_num": "Número do mês (1-12), derivado de month.",
    "day_of_week": "Dia da semana do último contato: mon a fri.",
    "day_of_week_num": "Número do dia da semana (1 = mon ... 5 = fri), derivado de day_of_week.",
    "duration": "Duração do último contato em segundos. DATA LEAKAGE: só é conhecida após a ligação; usar apenas em análise descritiva.",
    "duration_minutes": "Duração do último contato em minutos (duration / 60, 2 casas). Mesma ressalva de data leakage.",
    "campaign": "Número de contatos com o cliente nesta campanha, incluindo o último (mínimo 1).",
    "pdays": "Dias desde o último contato de uma campanha anterior. NULL = sem registro (origem 999); inclui 4.110 clientes contatados antes cuja data não foi registrada.",
    "previous": "Número de contatos com o cliente antes desta campanha.",
    "poutcome": "Resultado da campanha anterior: failure, nonexistent (sem campanha anterior), success.",
    "is_previously_contacted": "True se o cliente foi contatado em campanha anterior (previous > 0, equivalente a poutcome <> nonexistent).",
    "emp_var_rate": "Taxa de variação do emprego, trimestral (origem: emp.var.rate).",
    "cons_price_idx": "Índice de preços ao consumidor, mensal (origem: cons.price.idx).",
    "cons_conf_idx": "Índice de confiança do consumidor, mensal (origem: cons.conf.idx).",
    "euribor3m": "Taxa Euribor de 3 meses (%), diária.",
    "nr_employed": "Número de empregados (milhares), trimestral (origem: nr.employed).",
    "is_subscribed": "Variável-alvo: True se o cliente aderiu ao depósito a prazo (origem: y = yes).",
}

(
    with_comments(df_silver, SILVER_COLUMN_COMMENTS)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TBL_SILVER)
)

apply_comments(TBL_SILVER, SILVER_TABLE_COMMENT, SILVER_COLUMN_COMMENTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constraints Delta (CHECK)
# MAGIC Regras de domínio gravadas na própria tabela: qualquer escrita futura que as viole é rejeitada pelo Delta Lake.
# MAGIC Só são criadas se ainda não existirem, para manter o histórico limpo.

# COMMAND ----------

SILVER_CONSTRAINTS = {
    "chk_age": "age BETWEEN 17 AND 120",
    "chk_campaign": "campaign >= 1",
    "chk_duration": "duration >= 0",
    "chk_previous": "previous >= 0",
    "chk_pdays": "pdays IS NULL OR pdays >= 0",
    "chk_month_num": "month_num BETWEEN 1 AND 12",
    "chk_day_of_week_num": "day_of_week_num BETWEEN 1 AND 5",
}

existing = {r.key for r in spark.sql(f"SHOW TBLPROPERTIES {TBL_SILVER}").collect() if r.key.startswith("delta.constraints.")}
for name, expr in SILVER_CONSTRAINTS.items():
    if f"delta.constraints.{name}" not in existing:
        spark.sql(f"ALTER TABLE {TBL_SILVER} ADD CONSTRAINT {name} CHECK ({expr})")

display(spark.sql(f"SHOW TBLPROPERTIES {TBL_SILVER}").filter("key LIKE 'delta.constraints.%'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Evidências

# COMMAND ----------

display(spark.table(TBL_SILVER).orderBy("contact_id").limit(10))

# COMMAND ----------

display(spark.sql(f"""
    SELECT contact_year, COUNT(*) AS contatos, SUM(CAST(is_subscribed AS INT)) AS adesoes,
           ROUND(100 * AVG(CAST(is_subscribed AS INT)), 2) AS taxa_conversao_pct
    FROM {TBL_SILVER}
    GROUP BY contact_year ORDER BY contact_year
"""))

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {TBL_SILVER}").select("version", "timestamp", "operation", "operationMetrics"))

# COMMAND ----------

print(f"✔ Silver gravada: {spark.table(TBL_SILVER).count():,} linhas em {TBL_SILVER}")
