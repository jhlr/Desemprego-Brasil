# Empregabilidade e Educação no Brasil

Análise exploratória da relação entre **desempenho educacional** (IDEB) e **desemprego**
(PNAD Contínua) no Brasil, com notebook, dashboard interativo e relatório em PDF.

## Perguntas

1. Quem tem mais escolaridade enfrenta menos desemprego?
2. Estados com melhor desempenho escolar (IDEB) têm menor desemprego?
3. Há desigualdade regional relevante no desempenho educacional?

## Dados

| Fonte | Conteúdo | Coleta |
|---|---|---|
| INEP — IDEB por escola | Edições 2017/2019/2021/2023, 3 etapas, 27 UFs | planilhas oficiais |
| IBGE — PNAD Contínua | Taxa de desocupação e mercado de trabalho, 2016–2026 | API SIDRA |

Os dois foram unidos por UF e ano em `data/educacao_emprego_brasil.csv`.

## Estrutura

```
data/                              datasets (IDEB, desemprego, base unida)
analise_educacao_desemprego.ipynb  notebook de EDA
dashboard.py                       dashboard Streamlit
analysis.py                        funcoes de carga e analise
dashboard_educacao_desemprego.pdf  versao estatica do dashboard
```

A base unida e gerada por `analysis.build_base_unida(salvar=True)` e ja vem pronta
em `data/educacao_emprego_brasil.csv`.

## Como executar

```bash
pip install -r requirements.txt
streamlit run dashboard.py
jupyter notebook analise_educacao_desemprego.ipynb
```

## Principais resultados

- Escolaridade reduz o desemprego: superior completo ~5% contra ~12% de quem tem
  so o medio completo (Wilcoxon pareado, p < 0,05).
- O IDEB medio por UF tem correlacao negativa com o desemprego estadual
  (Spearman rho ~ -0,45).
- O Sul tem IDEB de Ensino Medio significativamente maior que o Nordeste
  (Mann-Whitney, p < 0,05).
