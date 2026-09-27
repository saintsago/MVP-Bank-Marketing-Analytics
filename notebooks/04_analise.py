# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Análise de Dados — respondendo às perguntas de negócio
# MAGIC
# MAGIC **Problema:** identificar os fatores que mais influenciam a conversão das campanhas de marketing direto de um banco português,
# MAGIC para otimizar a alocação do esforço de contato e aumentar a adesão ao depósito a prazo.
# MAGIC
# MAGIC Todas as consultas usam o **modelo estrela da camada gold** (`fato_contato` + 3 dimensões). Para cada pergunta: consulta SQL → tabela → gráfico → discussão.
# MAGIC
# MAGIC > **Nota sobre `duration`:** a duração da ligação só é conhecida depois que ela termina (*data leakage*). Aqui ela aparece apenas de forma descritiva
# MAGIC > e nunca como fator "explicativo" da conversão.

# COMMAND ----------

# MAGIC %run ./00_utils

# COMMAND ----------

import itertools
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Gráficos também são salvos em um volume, para uso no README
REPORTS_VOLUME = "relatorios"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_GOLD}.{REPORTS_VOLUME} COMMENT 'Artefatos gerados pela análise (gráficos PNG)'")
CHARTS_PATH = f"/Volumes/{CATALOG}/{SCHEMA_GOLD}/{REPORTS_VOLUME}/graficos"
os.makedirs(CHARTS_PATH, exist_ok=True)

# Paleta (validada para daltonismo) e estilo: marcas finas, grade discreta, texto em tons neutros
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#c4c3bd"
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "font.size": 10, "legend.frameon": False, "figure.dpi": 110,
})


def pct(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}%".replace(".", ",")


def num(v: int) -> str:
    return f"{int(v):,}".replace(",", ".")


def run(sql: str) -> pd.DataFrame:
    """Executa a consulta, exibe o resultado no notebook e devolve um pandas DataFrame para o gráfico."""
    df = spark.sql(sql)
    display(df)
    return df.toPandas()


def save(fig, name: str) -> None:
    fig.savefig(f"{CHARTS_PATH}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def rate_bars(ax, labels, rates, reference, title, horizontal=False, color=BLUE,
              fmt=pct, ref_label="média geral", value_label="Taxa de conversão", min_slots=0):
    """Barras de uma única medida (uma série, uma cor) com rótulo de valor e linha de referência (ex.: média geral).

    A linha de referência cobre só a área das barras; o rótulo dela fica num espaço reservado (à direita nas barras
    verticais, abaixo nas horizontais) para nunca colidir com barras ou valores. `min_slots` mantém a espessura das
    barras constante entre painéis com quantidades diferentes de categorias.
    """
    n = len(labels)
    pos = np.arange(n)
    axis_fmt = mpl.ticker.FuncFormatter(lambda v, _: fmt(v).replace(",0", ""))
    value_box = dict(facecolor=SURFACE, edgecolor="none", pad=0.6)
    if horizontal:
        ax.barh(pos, rates, height=0.66, color=color, zorder=2)
        ax.set_yticks(pos, labels)
        ax.set_ylim(max(n, min_slots) - 0.5 + 0.9, -0.5)  # eixo invertido + faixa livre abaixo das barras
        ax.vlines(reference, -0.5, n - 0.5, color=INK_2, linewidth=1, zorder=3)
        ax.text(reference, n - 0.5 + 0.15, f" {ref_label} {fmt(reference)}", ha="left", va="top", fontsize=8.5, color=INK_2)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlim(0, max(rates) * 1.18)
        for p, r in zip(pos, rates):
            ax.text(r + max(rates) * 0.015, p, fmt(r), va="center", fontsize=8.5, color=INK_2, bbox=value_box, zorder=4)
        ax.set_xlabel(value_label)
        ax.xaxis.set_major_formatter(axis_fmt)
        ax.tick_params(axis="y", length=0)
    else:
        ax.bar(pos, rates, width=0.66, color=color, zorder=2)
        ax.set_xticks(pos, labels)
        ax.set_xlim(-0.5, n - 0.5 + 1.3)  # faixa livre à direita para o rótulo da referência
        ax.hlines(reference, -0.5, n - 0.5, color=INK_2, linewidth=1, zorder=3)
        ax.text(n - 0.5 + 0.1, reference, f"{ref_label}\n{fmt(reference)}", ha="left", va="center", fontsize=8.5, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylim(0, max(rates) * 1.15)
        for p, r in zip(pos, rates):
            ax.text(p, r + max(rates) * 0.015, fmt(r), ha="center", va="bottom", fontsize=8.5, color=INK_2, bbox=value_box, zorder=4)
        ax.set_ylabel(value_label)
        ax.yaxis.set_major_formatter(axis_fmt)
        ax.tick_params(axis="x", length=0)
    ax.set_title(title)


def volume_bars(ax, labels, counts, title):
    """Barras de volume (contexto), em cinza neutro, para não competir com a taxa."""
    pos = np.arange(len(labels))
    ax.bar(pos, counts, width=0.66, color=NEUTRAL, zorder=2)
    ax.set_xticks(pos, labels)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: num(v)))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Contatos")
    ax.set_title(title)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base analítica: o modelo estrela
# MAGIC A view temporária abaixo junta a fato com as três dimensões. Ela é o ponto de partida de todas as consultas.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW vw_contatos AS
SELECT
    f.contact_id, f.duration, f.duration_minutes, f.campaign, f.pdays, f.previous, f.is_previously_contacted,
    CAST(f.is_subscribed AS INT) AS subscribed,
    cl.age, cl.age_group, cl.job, cl.marital, cl.education, cl.credit_default, cl.housing, cl.loan,
    ca.contact_year, ca.month, ca.month_num, ca.day_of_week, ca.day_of_week_num, ca.contact, ca.poutcome,
    ce.emp_var_rate, ce.cons_price_idx, ce.cons_conf_idx, ce.euribor3m, ce.nr_employed
FROM {TBL_FATO} f
JOIN {TBL_DIM_CLIENTE}  cl ON f.sk_cliente            = cl.sk_cliente
JOIN {TBL_DIM_CAMPANHA} ca ON f.sk_campanha           = ca.sk_campanha
JOIN {TBL_DIM_CONTEXTO} ce ON f.sk_contexto_economico = ce.sk_contexto_economico
""")

OVERALL = spark.sql("SELECT 100 * AVG(subscribed) FROM vw_contatos").first()[0]
print(f"Taxa geral de conversão: {pct(OVERALL, 2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## P1 · Qual é a taxa geral de conversão e como ela varia por canal de contato?

# COMMAND ----------

p1 = run("""
SELECT 'Total' AS canal, COUNT(*) AS contatos, 100.0 AS participacao_pct, SUM(subscribed) AS adesoes,
       ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos
UNION ALL
SELECT contact, COUNT(*), ROUND(100 * COUNT(*) / (SELECT COUNT(*) FROM vw_contatos), 1), SUM(subscribed),
       ROUND(100 * AVG(subscribed), 2)
FROM vw_contatos
GROUP BY contact
ORDER BY contatos DESC
""")

# COMMAND ----------

# Controle de confusão: a vantagem do celular se mantém dentro de cada ano?
p1_ano = run("""
SELECT contact_year AS ano, contact AS canal, COUNT(*) AS contatos, ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos
GROUP BY contact_year, contact
ORDER BY ano, canal
""")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
canal = p1[p1.canal != "Total"].sort_values("taxa_conversao_pct", ascending=False)
rate_bars(axes[0], [f"{c}\n({num(n)} contatos)" for c, n in zip(canal.canal, canal.contatos)],
          canal.taxa_conversao_pct.tolist(), OVERALL, "Taxa de conversão por canal")

anos = sorted(p1_ano.ano.unique())
x = np.arange(len(anos))
for i, (c, color) in enumerate([("cellular", BLUE), ("telephone", ORANGE)]):
    serie = p1_ano[p1_ano.canal == c].set_index("ano").reindex(anos)
    axes[1].bar(x + (i - 0.5) * 0.36, serie.taxa_conversao_pct, width=0.34, color=color, label=c, zorder=2)
    for xi, v in zip(x + (i - 0.5) * 0.36, serie.taxa_conversao_pct):
        axes[1].text(xi, v + 0.8, pct(v), ha="center", fontsize=8.5, color=INK_2)
axes[1].set_xticks(x, [str(a) for a in anos])
axes[1].grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
axes[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: pct(v, 0)))
axes[1].tick_params(axis="x", length=0)
axes[1].legend(loc="upper left")
axes[1].set_title("Taxa de conversão por canal dentro de cada ano")
fig.tight_layout()
save(fig, "p1_canal")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussão — P1
# MAGIC - **Taxa geral de 11,27%** (4.639 adesões em 41.176 contatos): cerca de 1 adesão a cada 9 clientes contatados. É uma base desbalanceada, e toda comparação a seguir usa essa taxa como referência (*lift* = taxa do grupo ÷ 11,27%).
# MAGIC - **O celular converte 2,8× mais que o telefone fixo** (14,74% × 5,23%) e responde por 63,5% dos contatos.
# MAGIC - **Controle de confusão:** como a conversão muda muito entre os anos (P6), verificamos se a diferença não é só efeito do período. Ela se mantém **dentro de todos os anos**
# MAGIC   (2008: 5,7% × 3,9%; 2009: 19,9% × 14,7%; 2010: 57,7% × 28,4%). O banco também mudou o mix: o fixo era 49,5% dos contatos em 2008 e caiu para 8,2% em 2009.
# MAGIC - **Implicação:** tornar o celular o canal padrão e usar o fixo só como alternativa. *Ressalva:* é uma associação observacional; o tipo de telefone cadastrado pode refletir o perfil do cliente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## P2 · Qual perfil demográfico tem maior propensão a aderir?
# MAGIC Taxa de conversão e *lift* (taxa do grupo ÷ taxa geral) por faixa etária, profissão, escolaridade e estado civil.
# MAGIC Grupos com menos de 100 contatos são omitidos por falta de suporte estatístico (ex.: `illiterate`, 18 contatos).

# COMMAND ----------


def profile_query(column: str, order: str) -> str:
    return f"""
    SELECT {column} AS categoria, COUNT(*) AS contatos, SUM(subscribed) AS adesoes,
           ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct,
           ROUND(100 * AVG(subscribed) / {OVERALL}, 2) AS lift
    FROM vw_contatos
    GROUP BY {column}
    HAVING COUNT(*) >= 100
    ORDER BY {order}
    """


p2 = {
    "age_group": run(profile_query("age_group", "categoria")),
    "job": run(profile_query("job", "taxa_conversao_pct DESC")),
    "education": run(profile_query("education", "taxa_conversao_pct DESC")),
    "marital": run(profile_query("marital", "taxa_conversao_pct DESC")),
}

# COMMAND ----------

# Painéis com altura proporcional ao nº de categorias, para manter a mesma espessura de barra em todos
fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[len(p2[c]) + 2.5 for c in ["age_group", "marital", "education"]])
panel_axes = {
    "age_group": fig.add_subplot(gs[0, 0]),
    "marital": fig.add_subplot(gs[1, 0]),
    "education": fig.add_subplot(gs[2, 0]),
    "job": fig.add_subplot(gs[:, 1]),
}
titles = {"age_group": "Faixa etária", "job": "Profissão", "education": "Escolaridade", "marital": "Estado civil"}
for col, df in p2.items():
    rate_bars(panel_axes[col], [f"{c} ({num(n)})" for c, n in zip(df.categoria, df.contatos)], df.taxa_conversao_pct.tolist(),
              OVERALL, titles[col], horizontal=True)
fig.suptitle("Taxa de conversão por perfil demográfico (entre parênteses: nº de contatos)", x=0.01, ha="left", fontsize=12.5, fontweight="semibold")
fig.tight_layout()
save(fig, "p2_perfil_demografico")

# COMMAND ----------

# Controle por período: o padrão por idade e profissão se mantém dentro de cada ano?
# (a conversão média muda muito de um ano para outro: 4,8% em 2008, 19,5% em 2009, 52,1% em 2010)
for col in ["age_group", "job"]:
    display(spark.sql(f"""
    SELECT {col} AS categoria,
           ROUND(100 * AVG(CASE WHEN contact_year = 2008 THEN subscribed END), 1) AS taxa_2008,
           ROUND(100 * AVG(CASE WHEN contact_year = 2009 THEN subscribed END), 1) AS taxa_2009,
           ROUND(100 * AVG(CASE WHEN contact_year = 2010 THEN subscribed END), 1) AS taxa_2010,
           ROUND(100 * AVG(CASE WHEN contact_year >= 2009 THEN 1 ELSE 0 END), 1) AS pct_contatos_2009_2010
    FROM vw_contatos
    GROUP BY {col}
    ORDER BY {'categoria' if col == 'age_group' else 'taxa_2008 DESC'}
    """))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussão — P2
# MAGIC - **A idade é o atributo demográfico mais discriminante**, com padrão em "U": 17-24 anos (24,0%) e 65+ (47,3%) convertem muito acima da média, enquanto 35-54 anos
# MAGIC   (8,6%–8,7%) ficam abaixo. É justamente nessa faixa central que estão 54% dos contatos.
# MAGIC - **Profissão** reforça o padrão (student 31,4%, retired 25,3%). **Blue-collar** tem a pior taxa (6,9%) apesar de ser o 2º maior volume (9.253 contatos), e services vem logo acima (8,1%).
# MAGIC - **Escolaridade e estado civil** têm efeito moderado: curso superior 13,7% × basic.9y 7,8%; solteiros 14,0% × casados 10,2%. A diferença por estado civil em parte reflete a idade.
# MAGIC - **Controle por período (importante):** nenhum cliente 65+ foi contatado em 2008, o pior ano, e 78% dos contatos com estudantes ocorreram em 2009-2010. Parte da vantagem bruta
# MAGIC   é efeito de período. Mesmo assim ela persiste **dentro dos anos**: em 2009 (média de 19,5%), 65+ converteu 40,7%, aposentados 36,5% e estudantes 32,1%,
# MAGIC   contra 10,1% de blue-collar. O *lift* real desses grupos fica em torno de **1,5–2×**, não os 3–4× da visão bruta.
# MAGIC - **Implicação:** deslocar esforço da faixa 35-54 e dos perfis blue-collar/services para estudantes/jovens, aposentados/65+ e clientes com curso superior.

# COMMAND ----------

# MAGIC %md
# MAGIC ## P3 · O número de contatos na campanha atual impacta a conversão?
# MAGIC `campaign` = nº de ligações feitas ao mesmo cliente nesta campanha (inclui a última). Além da taxa por cliente, medimos a **eficiência do esforço**:
# MAGIC adesões obtidas a cada 100 ligações.

# COMMAND ----------

p3 = run("""
SELECT CASE WHEN campaign <= 5 THEN CAST(campaign AS STRING) WHEN campaign <= 10 THEN '6-10' ELSE '11+' END AS contatos_na_campanha,
       MIN(campaign) AS ordem,
       COUNT(*) AS clientes,
       SUM(campaign) AS ligacoes,
       SUM(subscribed) AS adesoes,
       ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct,
       ROUND(100 * SUM(subscribed) / SUM(campaign), 2) AS adesoes_por_100_ligacoes
FROM vw_contatos
GROUP BY 1
ORDER BY ordem
""")
p3["adesoes_acumuladas_pct"] = (100 * p3.adesoes.cumsum() / p3.adesoes.sum()).round(1)
p3["ligacoes_acumuladas_pct"] = (100 * p3.ligacoes.cumsum() / p3.ligacoes.sum()).round(1)
display(p3)

# COMMAND ----------

# Descritivo (data leakage): duração média da última ligação por resultado
display(spark.sql("""
SELECT CASE WHEN subscribed = 1 THEN 'aderiu' ELSE 'não aderiu' END AS resultado,
       COUNT(*) AS contatos,
       ROUND(AVG(duration_minutes), 2) AS duracao_media_min,
       ROUND(PERCENTILE(duration_minutes, 0.5), 2) AS duracao_mediana_min
FROM vw_contatos GROUP BY 1
"""))

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
rate_bars(axes[0], p3.contatos_na_campanha.tolist(), p3.taxa_conversao_pct.tolist(), OVERALL,
          "Taxa de conversão por nº de contatos na campanha")
axes[0].set_xlabel("Contatos com o mesmo cliente nesta campanha")
overall_eff = 100 * p3.adesoes.sum() / p3.ligacoes.sum()
rate_bars(axes[1], p3.contatos_na_campanha.tolist(), p3.adesoes_por_100_ligacoes.tolist(), overall_eff,
          "Adesões a cada 100 ligações", fmt=lambda v: f"{v:.1f}".replace(".", ","), value_label="Adesões / 100 ligações")
axes[1].set_xlabel("Contatos com o mesmo cliente nesta campanha")
fig.tight_layout()
save(fig, "p3_numero_contatos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussão — P3
# MAGIC - **Impacto negativo e monotônico:** a taxa cai de 13,0% (1 contato) para 7,5% (5) e 3,1% (11 ou mais).
# MAGIC - **A eficiência cai ainda mais rápido:** a 1ª ligação rende 13,0 adesões a cada 100 ligações; a partir da 6ª, menos de 1. **88% das adesões acontecem até o 3º contato**,
# MAGIC   consumindo 52% das ligações. Clientes com 6 ou mais contatos consumiram **30,6% das ligações** para gerar só **4,0% das adesões**.
# MAGIC - **Leitura causal:** parte do efeito é seleção (quem precisa de muitas ligações tem menos interesse), mas para a operação o que importa é o retorno marginal, e ele despenca.
# MAGIC - **Implicação:** limitar a **3 tentativas por cliente por campanha**. O corte libera cerca de 28 mil ligações (26,5% do total), abrindo mão de no máximo 555 adesões (12%).
# MAGIC   Reaplicadas em primeiros contatos com clientes de perfil prioritário (P7), essas ligações tendem a render bem mais.
# MAGIC - **Duração (só descritivo, com data leakage):** quem aderiu teve ligações bem mais longas (média de 9,2 min × 3,7 min). Isso é consequência da conversa, não um fator conhecido antes da ligação.

# COMMAND ----------

# MAGIC %md
# MAGIC ## P4 · Clientes contatados em campanhas anteriores convertem mais do que novos contatos?

# COMMAND ----------

p4 = run("""
SELECT CASE WHEN NOT is_previously_contacted THEN '1. Nunca contatado'
            WHEN poutcome = 'failure'      THEN '2. Contatado antes, anterior sem sucesso'
            ELSE                                '3. Contatado antes, anterior com sucesso' END AS grupo,
       COUNT(*) AS contatos, SUM(subscribed) AS adesoes,
       ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct,
       ROUND(AVG(subscribed) / (SELECT AVG(subscribed) FROM vw_contatos), 2) AS lift
FROM vw_contatos
GROUP BY 1 ORDER BY 1
""")

# COMMAND ----------

# Visão agregada (flag da silver) e o papel do registro de data (pdays) entre os já contatados
display(spark.sql("""
SELECT is_previously_contacted, COUNT(*) AS contatos, ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos GROUP BY 1 ORDER BY 1
"""))
display(spark.sql("""
SELECT poutcome, pdays IS NOT NULL AS tem_data_do_contato_anterior, COUNT(*) AS contatos,
       ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos WHERE is_previously_contacted
GROUP BY 1, 2 ORDER BY 1, 2
"""))

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 3.4))
rate_bars(ax, [f"{g[3:]} ({num(n)})" for g, n in zip(p4.grupo, p4.contatos)], p4.taxa_conversao_pct.tolist(), OVERALL,
          "Taxa de conversão por histórico de contato (entre parênteses: nº de contatos)", horizontal=True)
fig.tight_layout()
save(fig, "p4_historico_contato")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussão — P4
# MAGIC - **Sim, e é o fator isolado mais forte.** Clientes nunca contatados (86% da base) convertem **8,8%**; os já contatados convertem **26,7%**, 3× mais.
# MAGIC - O **resultado anterior** é decisivo: quem aderiu na campanha passada converte **65,1%** (*lift* 5,8). Mesmo quem recusou antes converte 14,2%, 1,6× acima dos novos.
# MAGIC - **Ligação com a qualidade de dados:** entre os que recusaram antes, os 142 com data do contato anterior registrada (`pdays`) converteram 51,4%, contra 12,9% dos 4.110 sem registro.
# MAGIC   `pdays` só existe para contatos recentes (até 27 dias), um sinal de relacionamento ativo. Se a flag de contato prévio tivesse sido baseada em `pdays`,
# MAGIC   esses 4.110 clientes seriam tratados como "novos" e a comparação ficaria distorcida.
# MAGIC - **Implicação:** manter uma **lista quente** (ex-aderentes e contatos recentes) e ligar para ela primeiro. Como ela é pequena (1.373 ex-aderentes), o ganho de escala vem de
# MAGIC   priorizar bem os clientes novos (P7).

# COMMAND ----------

# MAGIC %md
# MAGIC ## P5 · Qual é o melhor mês e dia da semana para realizar contatos?

# COMMAND ----------

p5_mes = run("""
SELECT month_num, month AS mes, COUNT(*) AS contatos, SUM(subscribed) AS adesoes,
       ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos GROUP BY month_num, month ORDER BY month_num
""")
p5_dia = run("""
SELECT day_of_week_num, day_of_week AS dia, COUNT(*) AS contatos, SUM(subscribed) AS adesoes,
       ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos GROUP BY day_of_week_num, day_of_week ORDER BY day_of_week_num
""")
p5_ano_mes = run("""
SELECT contact_year AS ano, month_num, month AS mes, COUNT(*) AS contatos, ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos GROUP BY contact_year, month_num, month ORDER BY ano, month_num
""")

# COMMAND ----------

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
rate_bars(axes[0], p5_mes.mes.tolist(), p5_mes.taxa_conversao_pct.tolist(), OVERALL, "Taxa de conversão por mês")
volume_bars(axes[1], p5_mes.mes.tolist(), p5_mes.contatos.tolist(), "Volume de contatos por mês")
fig.tight_layout()
save(fig, "p5_mes")

fig, ax = plt.subplots(figsize=(7, 3.6))
rate_bars(ax, p5_dia.dia.tolist(), p5_dia.taxa_conversao_pct.tolist(), OVERALL, "Taxa de conversão por dia da semana")
fig.tight_layout()
save(fig, "p5_dia_semana")

# COMMAND ----------

# Mapa de calor ano x mês: separa efeito de calendário de efeito de período
meses = p5_mes[["month_num", "mes"]].values.tolist()
anos = sorted(p5_ano_mes.ano.unique())
grid_rate = p5_ano_mes.pivot(index="ano", columns="month_num", values="taxa_conversao_pct").reindex(index=anos, columns=[m for m, _ in meses])
grid_n = p5_ano_mes.pivot(index="ano", columns="month_num", values="contatos").reindex(index=anos, columns=[m for m, _ in meses])

fig, ax = plt.subplots(figsize=(11, 3.3))
im = ax.imshow(grid_rate.values, cmap=SEQUENTIAL, vmin=0, vmax=np.nanmax(grid_rate.values), aspect="auto")
for i in range(len(anos)):
    for j in range(len(meses)):
        r, n = grid_rate.values[i, j], grid_n.values[i, j]
        if not np.isnan(r):
            ink = "white" if r > 0.55 * np.nanmax(grid_rate.values) else INK
            ax.text(j, i - 0.12, pct(r, 0), ha="center", va="center", fontsize=9, color=ink, fontweight="semibold")
            ax.text(j, i + 0.22, f"n={num(n)}", ha="center", va="center", fontsize=7, color=ink)
ax.set_xticks(range(len(meses)), [m for _, m in meses])
ax.set_yticks(range(len(anos)), [str(a) for a in anos])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: pct(v, 0)))
ax.set_title("Taxa de conversão por ano e mês (células vazias = sem contatos)")
fig.tight_layout()
save(fig, "p5_ano_mes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussão — P5
# MAGIC - **Visão bruta:** março (50,5%), dezembro (48,9%), setembro (44,9%) e outubro (43,9%) têm as maiores taxas, e maio a pior (6,4%). Mas os meses de alta taxa são os de
# MAGIC   **menor volume** (182 a 717 contatos), e maio concentra 33% de todos os contatos.
# MAGIC - **O "efeito mês" é, na verdade, efeito de período:** no mapa ano × mês, quase todos os meses de 2008 ficam entre 3% e 6%, e de meados de 2009 em diante praticamente todos
# MAGIC   passam de 30%. Os meses "bons" são os que só tiveram campanhas em 2009-2010.
# MAGIC - O padrão consistente é outro: os **meses de campanha massiva** (maio/2008 com 7.762 contatos e 3,1%; maio/2009 com 5.793 e 9,1%) têm as piores taxas do seu ano.
# MAGIC   Volume alto e pouco seletivo derruba a conversão.
# MAGIC - **Dia da semana:** diferenças pequenas. Quinta (12,1%), terça (11,8%) e quarta (11,7%) ficam acima da média; segunda é o pior dia (9,9%).
# MAGIC - **Implicação:** os dados não mostram um "melhor mês" robusto. A recomendação é evitar ondas massivas e concentrar as ligações de terça a quinta, evitando a segunda-feira.
# MAGIC   É um ganho marginal (cerca de 2 p.p.) perto de canal e histórico.

# COMMAND ----------

# MAGIC %md
# MAGIC ## P6 · Os indicadores econômicos influenciam a decisão do cliente?

# COMMAND ----------

p6_serie = run("""
SELECT contact_year AS ano, month_num, month AS mes, COUNT(*) AS contatos,
       ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct,
       ROUND(AVG(euribor3m), 3) AS euribor3m_media,
       ROUND(AVG(nr_employed), 1) AS nr_employed_medio,
       ROUND(AVG(cons_conf_idx), 1) AS cons_conf_idx_medio
FROM vw_contatos GROUP BY contact_year, month_num, month ORDER BY ano, month_num
""")

p6_euribor = run("""
SELECT CASE WHEN euribor3m < 1 THEN '1. abaixo de 1%' WHEN euribor3m < 4 THEN '2. de 1% a 4%' ELSE '3. 4% ou mais' END AS faixa_euribor3m,
       COUNT(*) AS contatos, ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos GROUP BY 1 ORDER BY 1
""")

p6_emprego = run("""
SELECT nr_employed, COUNT(*) AS contatos, ROUND(100 * AVG(subscribed), 2) AS taxa_conversao_pct
FROM vw_contatos GROUP BY nr_employed ORDER BY nr_employed
""")

# COMMAND ----------

# Correlação (Pearson) entre os indicadores e a adesão (0/1 -> correlação ponto-bisserial)
indicadores = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]
pdf_econ = spark.table("vw_contatos").select(*indicadores, "subscribed").toPandas()
corr = pdf_econ.corr().round(2)
display(corr.reset_index().rename(columns={"index": "variavel"}))

# COMMAND ----------

labels_serie = [f"{m}/{str(a)[2:]}" for a, m in zip(p6_serie.ano, p6_serie.mes)]
x = np.arange(len(labels_serie))
fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
axes[0].plot(x, p6_serie.taxa_conversao_pct, color=BLUE, linewidth=2, marker="o", markersize=4, zorder=3)
axes[0].axhline(OVERALL, color=INK_2, linewidth=1)
axes[0].annotate(f"média geral {pct(OVERALL)}", xy=(0, OVERALL), xycoords=("axes fraction", "data"),
                 xytext=(2, 3), textcoords="offset points", fontsize=8.5, color=INK_2)
axes[0].set_title("Taxa de conversão mensal")
axes[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: pct(v, 0)))
axes[1].plot(x, p6_serie.euribor3m_media, color=ORANGE, linewidth=2, marker="o", markersize=4, zorder=3)
axes[1].set_title("Euribor 3 meses (média do mês, %)")
axes[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: pct(v, 1)))
for ax in axes:
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="x", length=0)
axes[1].set_xticks(x, labels_serie, rotation=60, ha="right")
fig.tight_layout()
save(fig, "p6_serie_temporal")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
rate_bars(axes[0], [f"{v:,.0f}".replace(",", ".") for v in p6_emprego.nr_employed], p6_emprego.taxa_conversao_pct.tolist(), OVERALL,
          "Taxa de conversão por nº de empregados (nr_employed, milhares)")
axes[0].tick_params(axis="x", rotation=45)

labels_corr = indicadores + ["adesão"]
im = axes[1].imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
for i in range(len(labels_corr)):
    for j in range(len(labels_corr)):
        v = corr.values[i, j]
        axes[1].text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                     color="white" if abs(v) > 0.6 else INK)
axes[1].set_xticks(range(len(labels_corr)), labels_corr, rotation=45, ha="right")
axes[1].set_yticks(range(len(labels_corr)), labels_corr)
axes[1].tick_params(length=0)
for s in axes[1].spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
cb.outline.set_visible(False)
axes[1].set_title("Correlação de Pearson")
fig.tight_layout()
save(fig, "p6_indicadores")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussão — P6
# MAGIC - **Associação forte.** Com Euribor abaixo de 1%, a conversão foi de **45,7%**; entre 1% e 4%, 15,8%; com 4% ou mais, **4,8%**. O mesmo aparece no emprego: com até 5.076 mil
# MAGIC   empregados, taxas de 36% a 57%; com 5.191 mil ou mais, de 3% a 6%.
# MAGIC - **Correlação com a adesão:** negativa e moderada para `nr_employed` (−0,35), `euribor3m` (−0,31) e `emp_var_rate` (−0,30); fraca para `cons_price_idx` (−0,14);
# MAGIC   praticamente nula para `cons_conf_idx` (0,05).
# MAGIC - **Os indicadores medem o mesmo ciclo** (Euribor × nr_employed = 0,95; Euribor × emp_var_rate = 0,97). A série temporal conta a história: com a crise de 2008 os juros
# MAGIC   despencaram (Euribor de ~5% para menos de 1%) e a conversão subiu de ~5% para ~50%.
# MAGIC - **Cautela causal:** no mesmo período o banco mudou a operação. O volume caiu de 27,7 mil contatos (2008) para 2,1 mil (2010), o uso de celular aumentou e a base quente passou a ser
# MAGIC   recontatada. Com dados observacionais não dá para separar o efeito macro da mudança de estratégia; o que se afirma é **associação, não causalidade**.
# MAGIC - **Implicação:** os indicadores não são controláveis, mas servem para **calibrar meta e volume**. Com juros baixos a propensão é maior e vale ampliar a campanha; com juros altos,
# MAGIC   é preciso ser muito mais seletivo. Num modelo preditivo, usar um único indicador (ex.: `nr_employed`) em vez dos cinco, por causa da colinearidade.

# COMMAND ----------

# MAGIC %md
# MAGIC ## P7 · Qual combinação de características forma o "perfil ideal" a priorizar?
# MAGIC
# MAGIC Duas abordagens complementares, ambas **sem `duration`** (leakage) e sem as variáveis de período/indicadores macro, que o banco não controla:
# MAGIC 1. **Ranking de segmentos:** todas as combinações de 3 atributos, com suporte mínimo de 200 contatos, ordenadas pela taxa de conversão.
# MAGIC    Rodamos para a base toda e, separadamente, para **clientes nunca contatados** (86% da base), onde o histórico não ajuda a priorizar.
# MAGIC 2. **Árvore de decisão rasa** (profundidade 3), que gera regras legíveis de priorização e a importância de cada atributo.

# COMMAND ----------

PERFIL = ["age_group", "job", "education", "marital", "credit_default", "housing", "loan", "contact", "poutcome"]
pdf = spark.table("vw_contatos").select(*PERFIL, "campaign", "subscribed").toPandas()


def top_segments(df: pd.DataFrame, features: list, k: int = 3, min_n: int = 200, top: int = 10) -> pd.DataFrame:
    rows = []
    for combo in itertools.combinations(features, k):
        g = df.groupby(list(combo))["subscribed"].agg(["count", "sum", "mean"]).reset_index()
        for _, r in g[g["count"] >= min_n].iterrows():
            rows.append({
                "segmento": " · ".join(f"{c}={r[c]}" for c in combo),
                "contatos": int(r["count"]), "adesoes": int(r["sum"]), "taxa_conversao_pct": round(100 * r["mean"], 2),
            })
    out = pd.DataFrame(rows).sort_values(["taxa_conversao_pct", "contatos"], ascending=[False, False]).head(top)
    out["lift"] = (out.taxa_conversao_pct / OVERALL).round(2)
    return out.reset_index(drop=True)


seg_todos = top_segments(pdf, PERFIL)
display(seg_todos)

novos = pdf[pdf.poutcome == "nonexistent"]
taxa_novos = 100 * novos.subscribed.mean()
seg_novos = top_segments(novos, [c for c in PERFIL if c != "poutcome"], min_n=200)
print(f"Clientes nunca contatados: {num(len(novos))} contatos, taxa base {pct(taxa_novos, 2)}")
display(seg_novos)

# COMMAND ----------

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = pd.get_dummies(pdf[PERFIL], prefix_sep="=").astype(int)
X["campaign"] = pdf["campaign"]
y = pdf["subscribed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=300, random_state=42).fit(X_train, y_train)
auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
print(f"AUC no conjunto de teste (30%): {auc:.3f}")

importancia = (
    pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False).head(6).round(3)
    .rename("importancia").reset_index().rename(columns={"index": "atributo"})
)
display(importancia)


def leaf_rules(model, feature_names: list) -> dict:
    """Converte cada folha da árvore numa regra legível (ex.: 'poutcome = success · contact ≠ cellular')."""
    t = model.tree_
    rules = {}

    def walk(node, conditions):
        if t.children_left[node] == -1:
            rules[node] = " · ".join(conditions)
            return
        name, threshold = feature_names[t.feature[node]], t.threshold[node]
        if "=" in name:  # variável dummy (0/1)
            col, val = name.split("=", 1)
            walk(t.children_left[node], conditions + [f"{col} ≠ {val}"])
            walk(t.children_right[node], conditions + [f"{col} = {val}"])
        else:
            walk(t.children_left[node], conditions + [f"{name} ≤ {threshold:.0f}"])
            walk(t.children_right[node], conditions + [f"{name} > {threshold:.0f}"])

    walk(0, [])
    return rules


# Regras (folhas da árvore) com a taxa de conversão observada na base completa
regras = (
    pd.DataFrame({"folha": tree.apply(X), "subscribed": y})
    .groupby("folha")["subscribed"].agg(contatos="count", adesoes="sum", taxa_conversao_pct="mean").reset_index()
)
regras["regra"] = regras.folha.map(leaf_rules(tree, list(X.columns)))
regras["taxa_conversao_pct"] = (100 * regras.taxa_conversao_pct).round(2)
regras["lift"] = (regras.taxa_conversao_pct / OVERALL).round(2)
regras = regras.sort_values("taxa_conversao_pct", ascending=False)[["regra", "contatos", "adesoes", "taxa_conversao_pct", "lift"]].reset_index(drop=True)
display(regras)

# COMMAND ----------

fig, axes = plt.subplots(2, 1, figsize=(12, 9.5), gridspec_kw={"height_ratios": [0.8, 1]})
rate_bars(axes[0], [f"{r}  (n={num(n)})" for r, n in zip(regras.regra, regras.contatos)], regras.taxa_conversao_pct.tolist(),
          OVERALL, "Regras de priorização (árvore de decisão, base completa)", horizontal=True)
rate_bars(axes[1], [f"{s}  (n={num(n)})" for s, n in zip(seg_novos.segmento, seg_novos.contatos)], seg_novos.taxa_conversao_pct.tolist(),
          taxa_novos, "Top 10 segmentos entre clientes nunca contatados", horizontal=True, ref_label="média dos nunca contatados")
fig.tight_layout()
save(fig, "p7_perfil_ideal")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussão — P7
# MAGIC **Regras da árvore** (sem `duration` e sem variáveis de período), em ordem de prioridade:
# MAGIC
# MAGIC | Prioridade | Regra | Contatos | Conversão | Lift |
# MAGIC |---|---|---|---|---|
# MAGIC | 1 | Aderiu na campanha anterior (`poutcome = success`) | 1.373 | ~65% | 5,8 |
# MAGIC | 2 | Sem sucesso anterior + celular + 65 anos ou mais | 468 | 41,2% | 3,7 |
# MAGIC | 3 | Sem sucesso anterior + celular + menos de 65 anos | 24.397 | 11,6% | 1,0 |
# MAGIC | 4 | Sem sucesso anterior + telefone fixo | 10.707 | 5,3% | 0,5 |
# MAGIC | 5 | Idem, com situação de crédito desconhecida (`credit_default = unknown`) | 4.231 | 3,5% | 0,3 |
# MAGIC
# MAGIC - O histórico (`poutcome = success`) responde por 81% da importância da árvore; celular (10%) e 65+ (8,5%) completam. A **AUC de 0,68** mostra que o perfil sozinho discrimina de
# MAGIC   forma moderada: boa parte da variação vem do contexto/período (P6), que ficou fora do modelo de propósito, por não ser controlável.
# MAGIC - **Entre os nunca contatados** (taxa base de 8,8%), os 10 melhores segmentos são todos de **65+**, em especial **aposentados com celular (45,4%, 5,1× a base)**, sem inadimplência
# MAGIC   ou empréstimo.
# MAGIC
# MAGIC **Perfil ideal a priorizar**
# MAGIC 1. **1ª onda (lista quente):** ex-aderentes e clientes recontatados recentemente.
# MAGIC 2. **2ª onda (novos com maior propensão):** 65+/aposentados com celular, seguidos de estudantes/17-24 anos e clientes com curso superior, todos por celular.
# MAGIC 3. **Regras operacionais:** celular como canal padrão, até 3 tentativas por cliente, preferência de terça a quinta, sem ondas massivas.
# MAGIC 4. **Baixa prioridade:** contatos só por telefone fixo (principalmente com `credit_default = unknown`) e perfis blue-collar/services de 35-54 anos.
# MAGIC
# MAGIC *Ressalva:* o grupo 65+ só foi contatado em 2009-2010 (ver P2), então parte do seu desempenho é efeito de período. A vantagem persiste dentro dos anos, porém menor.
# MAGIC Antes de escalar a estratégia, o ideal é validá-la com um **teste A/B** numa campanha real.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Síntese
# MAGIC
# MAGIC | # | Pergunta | Resposta curta | Alavanca |
# MAGIC |---|---|---|---|
# MAGIC | P1 | Taxa geral e canal | 11,27%; celular 14,7% × fixo 5,2% (vantagem mantida em todos os anos) | Celular como padrão |
# MAGIC | P2 | Perfil demográfico | Idade em "U": 17-24 e 65+ acima, 35-54 abaixo; student/retired acima, blue-collar abaixo (lift real ~1,5–2× após controlar o período) | Redirecionar esforço de segmento |
# MAGIC | P3 | Nº de contatos | Impacto negativo: 13,0% → 3,1%; 88% das adesões até o 3º contato | Teto de 3 tentativas |
# MAGIC | P4 | Contato anterior | Já contatados 26,7% × novos 8,8%; ex-aderentes 65,1% | Lista quente primeiro |
# MAGIC | P5 | Mês e dia | "Efeito mês" é efeito de período; ondas massivas convertem pior; ter–qui levemente melhores | Campanhas menores e seletivas |
# MAGIC | P6 | Indicadores econômicos | Forte associação (juros baixos ↔ conversão alta), indicadores colineares, sem prova de causalidade | Calibrar volume ao ciclo |
# MAGIC | P7 | Perfil ideal | Ex-aderente → 65+/aposentado com celular → jovens/universitários com celular | Ranking de priorização |
# MAGIC
# MAGIC **Conclusão:** os fatores **controláveis** que mais influenciam a conversão são o **histórico de relacionamento**, o **canal** e o **esforço por cliente**. O perfil demográfico
# MAGIC ajuda a priorizar os clientes novos. O contexto macro altera o nível geral da conversão, mas não é controlável: serve para calibrar metas.
