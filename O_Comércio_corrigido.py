# Databricks notebook source
# MAGIC %md
# MAGIC # 📱 O Comércio — Análise de Vendas de iPhones
# MAGIC
# MAGIC **Objetivo:** explorar o dataset de vendas de iPhones (`O_Phone_sales_dataset.csv`)
# MAGIC para responder perguntas de negócio sobre:
# MAGIC
# MAGIC - Distribuição de vendas por país e modelo
# MAGIC - Faturamento total e ticket médio por período
# MAGIC - Modelos mais caros e mais vendidos
# MAGIC - Sazonalidade mensal das vendas
# MAGIC
# MAGIC **Stack:** Databricks + pandas + matplotlib
# MAGIC **Autor:** Renato Lima
# MAGIC **Fonte dos dados:** `/Volumes/workspace/default/o_phone/O_Phone_sales_dataset.csv`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carga dos Dados
# MAGIC
# MAGIC Versão alternativa em PySpark (comentada) — usamos pandas por simplicidade,
# MAGIC já que o dataset cabe em memória.

# COMMAND ----------

# Versão PySpark (alternativa — não utilizada):
#df = spark.read.format("csv") \
#    .option("header", "true") \
#    .option("inferSchema", "true") \
#    .load("/Volumes/workspace/default/o_phone/O_Phone_sales_dataset.csv")
#display(df)

# COMMAND ----------

# Leitura do CSV com pandas e visualização da primeira linha para conferência do schema
import pandas as pd

df = pd.read_csv("/Volumes/workspace/default/o_phone/O_Phone_sales_dataset.csv")
display(df.head(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Análise Exploratória (EDA)
# MAGIC
# MAGIC Verificações iniciais: tipos de dados, estatísticas descritivas,
# MAGIC dimensões e valores nulos.

# COMMAND ----------

# Tipos das colunas, contagem de não-nulos e uso de memória
df.info()

# COMMAND ----------

# Estatísticas descritivas das colunas numéricas (média, desvio, min, max, quartis)
df.describe()

# COMMAND ----------

# Dimensões do dataset: (linhas, colunas)
df.shape

# COMMAND ----------

# Contagem de valores nulos por coluna — confirma a qualidade do dataset
df.isnull().sum()

# COMMAND ----------

# Visualização completa do DataFrame
display(df)

# COMMAND ----------

# Primeira linha em formato Series (transposto), útil para inspeção de campos
df.iloc[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Análises por Dimensão
# MAGIC
# MAGIC ### 3.1 Distribuição por País e Modelo

# COMMAND ----------

# Quantidade de registros por país — identifica os mercados mais ativos
df["Country"].value_counts()

# COMMAND ----------

# Quantidade de registros por modelo de iPhone — identifica os mais vendidos em volume
df["iPhone_Model"].value_counts()

# COMMAND ----------

# Filtro: somente linhas cujo modelo contém "iPhone" (defesa contra rótulos inconsistentes)
df[df["iPhone_Model"].str.contains("iPhone")]["iPhone_Model"]

# COMMAND ----------

# Mesma análise, agora com display() para renderização em tabela do Databricks
display(df[df["iPhone_Model"].str.contains("iPhone")]["iPhone_Model"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Top 10 iPhones mais caros

# COMMAND ----------

# Top 10 modelos de iPhone ordenados do mais caro para o mais barato (modelo + preço).
# IMPORTANTE: ordenar ANTES de aplicar head(10), para garantir que sejam os 10 maiores.
display(
    df[df["iPhone_Model"].str.contains("iPhone")][["iPhone_Model", "Price"]]
    .sort_values(by="Price", ascending=False)
    .head(10)
)

# COMMAND ----------

# Soma dos preços dos 10 iPhones mais caros (apenas coluna numérica para evitar concatenar strings)
display(
    df[df["iPhone_Model"].str.contains("iPhone")][["Price"]]
    .sort_values(by="Price", ascending=False)
    .head(10)
    .sum()
)

# COMMAND ----------

# Mesma soma, mas usando .agg(["sum"]) — retorna um DataFrame em vez de Series
display(
    df[df["iPhone_Model"].str.contains("iPhone")][["Price"]]
    .sort_values(by="Price", ascending=False)
    .head(10)
    .agg(["sum"])
)

# COMMAND ----------

# Mínimo dentre os 10 mais caros — funciona como "preço de corte" do top 10
display(
    df[df["iPhone_Model"].str.contains("iPhone")][["Price"]]
    .sort_values(by="Price", ascending=False)
    .head(10)
    .agg(["min"])
)

# COMMAND ----------

# Painel de estatísticas (soma, média, máximo, mínimo) para os 10 primeiros registros de iPhone
display(
    df[df["iPhone_Model"].str.contains("iPhone")][["Price"]]
    .head(10)
    .agg(["sum", "mean", "max", "min"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Faturamento por País

# COMMAND ----------

# Faturamento total por país, ordenado do maior para o menor
display(
    df.groupby("Country")["Price"]
    .sum()
    .reset_index()
    .sort_values(by="Price", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Conceitos: Series vs DataFrame
# MAGIC
# MAGIC Demonstração rápida das duas estruturas principais do pandas.

# COMMAND ----------

# df["Price"] retorna uma Series (uma única coluna, 1-D)
type(df["Price"])

# COMMAND ----------

# df[["Price"]] retorna um DataFrame (mesmo com uma só coluna, 2-D)
type(df[["Price"]])

# COMMAND ----------

# MAGIC %md
# MAGIC **Series** — usado para:
# MAGIC - cálculos
# MAGIC - médias
# MAGIC - soma
# MAGIC - filtros simples
# MAGIC
# MAGIC **DataFrame** — usado para:
# MAGIC - dashboards
# MAGIC - joins
# MAGIC - visualizações
# MAGIC - exportação
# MAGIC - análises completas

# COMMAND ----------

# Amostra das 10 primeiras linhas para conferência visual
df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Maior Preço por País × Modelo

# COMMAND ----------

# Preço máximo de cada combinação (País, Modelo), ordenado do mais caro para o mais barato
display(
    df.groupby(["Country", "iPhone_Model"])["Price"]
    .max()
    .reset_index()
    .sort_values(by="Price", ascending=False)
)

# COMMAND ----------

# Confirmação do tipo do objeto df
type(df)

# COMMAND ----------

# Top 10 combinações (País, Modelo) com maior preço de venda
display(
    df.groupby(["Country", "iPhone_Model"])["Price"]
    .max()
    .reset_index()
    .sort_values(by="Price", ascending=False)
    .head(10)
)

# COMMAND ----------

# Preço máximo registrado por modelo, ordenado do mais caro ao mais barato
display(
    df.groupby(["iPhone_Model"])["Price"]
    .max()
    .reset_index()
    .sort_values(by="Price", ascending=False)
)

# COMMAND ----------

# Maior quantidade vendida em uma única transação por modelo
display(
    df.groupby(["iPhone_Model"])["Quantity"]
    .max()
    .reset_index()
    .sort_values(by="Quantity", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Análise Temporal (Sazonalidade)
# MAGIC
# MAGIC Conversão da coluna `Sale_Date` para `datetime` e extração de mês/ano-mês
# MAGIC para analisar a evolução das vendas ao longo do tempo.

# COMMAND ----------

# Converte Sale_Date para datetime, permitindo extrair componentes (mês, ano, etc.)
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])

# COMMAND ----------

# Cria coluna Month com o número do mês (1–12), agregando vendas independente do ano
df["Month"] = df["Sale_Date"].dt.month

# COMMAND ----------

# Faturamento total por mês — identifica picos sazonais
display(
    df.groupby("Month")[["Price"]]
    .sum()
    .sort_values(by="Price", ascending=False)
    .reset_index()
)

# COMMAND ----------

# Para cada Price observado, a data mais recente de venda — top 10 datas mais recentes
# (assign formata a data como string YYYY-MM-DD para exibição limpa)
display(
    df.groupby(["Price"])["Sale_Date"]
    .max()
    .reset_index()
    .sort_values(by="Sale_Date", ascending=False)
    .head(10)
    .assign(Sale_Date=lambda x: x["Sale_Date"].dt.strftime('%Y-%m-%d'))
)

# COMMAND ----------

# Cria coluna "Nome" no formato YYYY-MM (ano-mês) para agrupamentos mensais com ano
df["Nome"] = df["Sale_Date"].dt.strftime('%Y-%m')

# COMMAND ----------

# Faturamento por ano-mês, com colunas renomeadas para português (Data, Preço)
display(
    df.rename(columns={"Nome": "Data", "Price": "Preço"})
    .groupby("Data")[["Preço"]]
    .sum()
    .sort_values(by="Preço", ascending=False)
    .reset_index()
)

# COMMAND ----------

# Faturamento por ano-mês mantendo o nome original "Nome"
display(
    df.groupby("Nome")[["Price"]]
    .sum()
    .sort_values(by="Price", ascending=False)
    .reset_index()
)

# COMMAND ----------

# Quantidade total vendida por ano-mês
display(
    df.groupby("Nome")[["Quantity"]]
    .sum()
    .sort_values(by="Quantity", ascending=False)
    .reset_index()
)

# COMMAND ----------

# Quantidade + Faturamento por mês, ordenado por Quantidade
display(
    df.rename(columns={"Nome": "Mês"})
    .groupby("Mês")[["Quantity", "Price"]]
    .sum()
    .sort_values(by="Quantity", ascending=False)
    .reset_index()
)

# COMMAND ----------

# Quantidade + Faturamento por mês, ordenado por Preço (faturamento)
display(
    df.rename(columns={"Nome": "Mês"})
    .groupby("Mês")[["Quantity", "Price"]]
    .sum()
    .sort_values(by="Price", ascending=False)
    .reset_index()
)

# COMMAND ----------

# Lista todas as colunas do DataFrame — útil antes de seleções/renames
print(df.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ticket Médio Mensal
# MAGIC
# MAGIC **Ticket médio = Preço total / Quantidade vendida** — indicador-chave
# MAGIC para entender se a receita vem de volume ou de valor unitário.

# COMMAND ----------

# Pipeline completo: agrupa por mês, soma Quantity e Price, calcula Ticket Médio
# e renomeia colunas para versão final em português.
resultado = (
    df.rename(columns={"Nome": "Data"})
    .groupby("Data")[["Quantity", "Price"]]
    .sum()
)
resultado["Ticket_Medio"] = resultado["Price"] / resultado["Quantity"]

# Renomear para versão final em português (apresentação)
resultado = resultado.rename(columns={"Quantity": "Quantidade", "Price": "Preço"})

display(resultado.reset_index())

# COMMAND ----------

# Mesma análise mantendo o nome original "Nome" (sem renomear para "Data")
resultado = (
    df.groupby("Nome")[["Quantity", "Price"]]
    .sum()
)
resultado["Ticket_Medio"] = resultado["Price"] / resultado["Quantity"]

display(resultado.reset_index())

# COMMAND ----------

# Versão final do DataFrame `resultado` (usado na visualização da próxima célula)
resultado = (
    df.rename(columns={"Nome": "Data"})
    .groupby("Data")[["Quantity", "Price"]]
    .sum()
)

resultado["Ticket_Medio"] = resultado["Price"] / resultado["Quantity"]

display(resultado.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualização — Faturamento Mensal
# MAGIC
# MAGIC Gráfico de barras do faturamento (`Price`) por mês (`Data`).

# COMMAND ----------

import matplotlib.pyplot as plt

resultado.reset_index().plot(
    x="Data",
    y="Price",
    kind="bar",
    figsize=(10, 5),
    title="Faturamento Mensal (Price por Ano-Mês)"
)
plt.xlabel("Ano-Mês")
plt.ylabel("Faturamento (Price)")
plt.tight_layout()
plt.show()
