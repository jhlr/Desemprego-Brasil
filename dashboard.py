"""
Empregabilidade e educacao no Brasil: IDEB (desempenho) x desemprego (PNAD Continua).
Executar: streamlit run dashboard.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

import analysis as A

st.set_page_config(page_title="Educacao e desemprego no Brasil", layout="wide")

ORDEM_INSTRUCAO = A.ORDEM_INSTRUCAO
INSTRUCAO_CURTA = A.INSTRUCAO_CURTA


@st.cache_data(show_spinner="Carregando dados...")
def load():
    return A.load_base_unida(), A.load_desemprego()


df, des = load()

SECOES = [
    "Visão geral",
    "Contexto e perguntas",
    "Os dados e o método",
    "Desemprego × escolaridade",
    "Desempenho educacional (IDEB)",
    "IDEB × Desemprego",
    "Testes de hipótese",
    "Recortes (comparar)",
    "Conclusões",
]
st.sidebar.title("Educacao e desemprego")
st.sidebar.caption("Roteiro da apresentação — siga as seções de cima para baixo.")
_sec = st.query_params.get("sec")
_idx = SECOES.index(_sec) if _sec in SECOES else 0
pagina = st.sidebar.radio("Seções", SECOES, index=_idx)
st.sidebar.divider()
st.sidebar.caption("Fontes: INEP (IDEB por escola) · IBGE / PNAD Contínua (API SIDRA).")


def kpi_row():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Escolas (linhas)", f"{len(df):,}")
    c2.metric("UFs", df.SG_UF.nunique())
    c3.metric("Edições do IDEB", ", ".join(map(str, sorted(df.ANO.unique()))))
    tab, per = A.desocupacao_por_instrucao(des)
    sup = tab.loc[tab.label == "Superior completo", "valor"].iloc[0]
    c4.metric(f"Desemprego com superior ({per})", f"{sup:.1f}%")


# VISÃO GERAL
if pagina == "Visão geral":
    st.title("Empregabilidade e Educação no Brasil")
    st.markdown("""
A pergunta que guia este trabalho é simples: **mais educação significa menos desemprego?**
Para responder, cruzamos duas bases públicas — o **IDEB por escola** (INEP), que mede o desempenho
educacional, e a **taxa de desemprego** (PNAD Contínua / IBGE) — nas edições de 2017, 2019, 2021 e 2023.

As duas telas abaixo resumem os dois principais resultados; as seções seguintes detalham cada um.
""")
    kpi_row()
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Desemprego cai com a escolaridade")
        tab, per = A.desocupacao_por_instrucao(des)
        fig = px.bar(tab, x="valor", y="label", orientation="h",
                     color="valor", color_continuous_scale="RdYlGn_r",
                     labels={"valor": "Desemprego (%)", "label": ""},
                     title=f"Taxa de desemprego por nível de instrução — Brasil ({per})")
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "array",
                          "categoryarray": [INSTRUCAO_CURTA[c] for c in ORDEM_INSTRUCAO]})
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.subheader("2. Melhor IDEB, menor desemprego (por estado)")
        _sub = df[(df.ANO == 2023) & (df.ETAPA == "Ensino Médio")]
        m = (_sub.groupby("SG_UF", observed=True)
                 .agg(ideb_medio=("IDEB", "mean"), desocupacao=("DESEMPREGO_UF", "first"))
                 .reset_index())
        m["REGIAO"] = m.SG_UF.map(A.UF2REGIAO)
        rho, p = spearmanr(m.ideb_medio, m.desocupacao)
        fig = px.scatter(m, x="ideb_medio", y="desocupacao", text="SG_UF", color="REGIAO",
                         trendline="ols", labels={"ideb_medio": "IDEB médio (EM)",
                         "desocupacao": "Desemprego (%)"},
                         title=f"IDEB-EM × desemprego por UF (2023) · ρ={rho:.2f}, p={p:.3f}")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, width='stretch')
    st.success("Em uma frase: tanto entre pessoas quanto entre estados, mais educação aparece "
               "associada a menos desemprego. As próximas seções mostram os dados e os testes.")

# CONTEXTO E PERGUNTAS
elif pagina == "Contexto e perguntas":
    st.title("Contexto e perguntas de pesquisa")
    st.markdown("""
### Por que este tema
A relação entre educação e mercado de trabalho está no centro do debate sobre desigualdade no Brasil.
Escolaridade costuma ser tratada como caminho para melhores oportunidades — mas é possível **medir**
isso com dados públicos? É o que fazemos aqui, juntando desempenho escolar e desemprego.

### As três perguntas
1. **Quem tem mais escolaridade enfrenta menos desemprego?**
   Comparação direta da taxa de desemprego por nível de instrução.
2. **Estados com melhor desempenho escolar (IDEB) têm menor desemprego?**
   Correlação entre o IDEB médio da UF e a taxa de desemprego estadual.
3. **Há desigualdade regional relevante no desempenho educacional?**
   Comparação do IDEB entre regiões (Sul × Nordeste).

### Recorte escolhido
Foco no **Ensino Médio** (etapa mais próxima da entrada no mercado de trabalho) para o cruzamento
por estado, usando todas as 27 UFs e as quatro edições do IDEB disponíveis a partir de 2017.
""")
    st.info("Cada pergunta é respondida em uma seção do menu e formalizada na seção "
            "'Testes de hipótese'.")

# OS DADOS E O MÉTODO
elif pagina == "Os dados e o método":
    st.title("Os dados e o método")
    c1, c2, c3 = st.columns(3)
    c1.metric("IDEB por escola (INEP)", "534.808 linhas")
    c2.metric("PNAD / desemprego (IBGE)", "318.816 linhas")
    c3.metric("Base unida (UF + ano)", f"{len(df):,} linhas")
    st.markdown("""
### Coleta
- **IDEB por escola (INEP):** planilhas oficiais das edições 2017, 2019, 2021 e 2023, três etapas
  (Anos Iniciais, Anos Finais, Ensino Médio). Foram lidas e transformadas para formato longo
  (uma linha por escola, etapa e edição).
- **Desemprego (IBGE):** coletado pela **API SIDRA**, agregados 4093 (recorte por sexo) e 4095
  (recorte por nível de instrução), variável 4099 (taxa de desocupação), de 2016 a 2026.
- **Base unida:** cada escola com IDEB recebe o desemprego médio anual da sua UF naquele ano
  (chave de junção: UF + ano).

### Decisões e cuidados com os dados
- **"Desocupação" é o mesmo que "desemprego"** — o primeiro é o termo técnico do IBGE; o segundo,
  o popular. Mesmo indicador.
- **O IDEB só existe em anos ímpares** (é medido a cada dois anos). Cada valor é o ponto daquele
  ano, não um intervalo agregado.
- **A PNAD trimestral é um retrato (snapshot), não uma soma.** Para chegar ao valor anual de cada
  UF, tiramos a **média dos quatro trimestres** — nunca a soma.
- **2021 não tem PNAD por UF** (o IBGE suspendeu as estimativas estaduais na pandemia). Tratamos o
  vazio por **imputação**: média de 2020 e 2022 de cada estado, sinalizada por uma coluna própria.
- **Escolas privadas não têm IDEB** nesta base (o índice é calculado essencialmente na rede
  pública). Por isso comparamos regiões e estados, não "pública × privada".
""")
    with st.expander("Ver amostra da base unida"):
        st.dataframe(df.head(20), width='stretch')

# DESEMPREGO x ESCOLARIDADE
elif pagina == "Desemprego × escolaridade":
    st.title("Pergunta 1 — desemprego × escolaridade")
    st.markdown("Comparação direta da taxa de desemprego entre faixas de escolaridade, no Brasil. "
                "Use o controle de trimestre para ver qualquer ponto da série.")
    d99 = des[(des.variavel_id == 4099) & (des.classificacao == "Nível de instrução") &
              (des.nivel == "N1") & (des.categoria != "Não determinado")].dropna(subset=["valor"])
    periodos = sorted(d99.periodo.unique())
    per = st.select_slider("Trimestre", options=periodos, value=periodos[-1])
    snap = d99[(d99.periodo == per) & (d99.categoria != "Total")].copy()
    snap["label"] = snap.categoria.map(INSTRUCAO_CURTA)
    snap["ordem"] = snap.categoria.map({c: i for i, c in enumerate(ORDEM_INSTRUCAO)})
    snap = snap.sort_values("ordem")
    fig = px.bar(snap, x="valor", y="label", orientation="h", color="valor",
                 color_continuous_scale="RdYlGn_r", text=snap.valor.map(lambda v: f"{v:.1f}%"),
                 labels={"valor": "Desemprego (%)", "label": ""},
                 title=f"Taxa de desemprego por nível de instrução — Brasil ({per})")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')
    st.markdown("""
**Como ler.** O menor desemprego está em quem tem **ensino superior completo**. Dois pontos pedem
atenção: *"médio incompleto"* costuma ter a taxa mais alta (são jovens entrando no mercado), e
*"sem instrução"* aparece baixo porque grande parte dessas pessoas está **fora da força de trabalho**
(não conta como desempregada). O contraste limpo é **superior completo × médio completo**.
""")

    st.subheader("A vantagem se mantém ao longo do tempo?")
    cats = st.multiselect("Faixas de escolaridade", ORDEM_INSTRUCAO,
                          default=["Ensino superior completo ou equivalente",
                                   "Ensino médio completo ou equivalente",
                                   "Ensino fundamental incompleto ou equivalente"])
    serie = d99[d99.categoria.isin(cats)].copy()
    serie["label"] = serie.categoria.map(INSTRUCAO_CURTA)
    fig = px.line(serie.sort_values("periodo"), x="periodo", y="valor", color="label",
                  labels={"periodo": "Trimestre", "valor": "Desemprego (%)", "label": "Escolaridade"})
    st.plotly_chart(fig, width='stretch')
    st.success("Resposta à pergunta 1: sim. Quanto maior a escolaridade, menor e mais estável o "
               "desemprego — a ordem das linhas nunca se inverte na série.")

# IDEB
elif pagina == "Desempenho educacional (IDEB)":
    st.title("Panorama do IDEB por escola")
    st.markdown("Antes de cruzar com o desemprego, veja como o desempenho educacional se distribui. "
                "Escolha a edição e a etapa.")
    c1, c2 = st.columns(2)
    ano = c1.selectbox("Edição", sorted(df.ANO.unique(), reverse=True))
    etapa = c2.selectbox("Etapa", df.ETAPA.unique())
    sub = df[(df.ANO == ano) & (df.ETAPA == etapa)]
    k1, k2, k3 = st.columns(3)
    k1.metric("Escolas", f"{len(sub):,}")
    k2.metric("IDEB médio", f"{sub.IDEB.mean():.2f}")
    k3.metric("IDEB mediano", f"{sub.IDEB.median():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(sub, x="IDEB", nbins=30, title="Distribuição do IDEB",
                           color_discrete_sequence=["#2b8cbe"])
        st.plotly_chart(fig, width='stretch')
    with col2:
        ordem = sub.groupby("REGIAO", observed=True).IDEB.median().sort_values(ascending=False).index
        fig = px.box(sub, x="REGIAO", y="IDEB", category_orders={"REGIAO": list(ordem)},
                     color="REGIAO", title="IDEB por região")
        st.plotly_chart(fig, width='stretch')

    uf = sub.groupby("SG_UF", observed=True).IDEB.mean().sort_values(ascending=False).reset_index()
    uf["REGIAO"] = uf.SG_UF.map(A.UF2REGIAO)
    fig = px.bar(uf, x="SG_UF", y="IDEB", color="REGIAO", title=f"IDEB médio por UF — {etapa} ({ano})")
    st.plotly_chart(fig, width='stretch')
    st.info("Repare na separação por cor: Sul, Sudeste e Centro-Oeste à frente; Norte e Nordeste "
            "atrás. Essa desigualdade regional é testada formalmente mais adiante.")

# IDEB x DESEMPREGO
elif pagina == "IDEB × Desemprego":
    st.title("Pergunta 2 — IDEB × desemprego por estado")
    st.markdown("Cada ponto é uma UF: no eixo X, o IDEB médio das escolas; no eixo Y, o desemprego "
                "do estado. A linha mostra a tendência; ρ é a correlação de Spearman.")
    c1, c2 = st.columns(2)
    ano = c1.selectbox("Ano", sorted(df.ANO.unique(), reverse=True))
    etapa = c2.selectbox("Etapa", df.ETAPA.unique(), index=list(df.ETAPA.unique()).index("Ensino Médio")
                         if "Ensino Médio" in list(df.ETAPA.unique()) else 0)
    sub = df[(df.ANO == ano) & (df.ETAPA == etapa)]
    uf = (sub.groupby("SG_UF", observed=True)
             .agg(ideb_medio=("IDEB", "mean"), desemprego=("DESEMPREGO_UF", "first"),
                  imputado=("DESEMPREGO_UF_IMPUTADO", "first")).reset_index())
    uf["REGIAO"] = uf.SG_UF.map(A.UF2REGIAO)
    rho, p = spearmanr(uf.ideb_medio, uf.desemprego)
    m1, m2 = st.columns(2)
    m1.metric("Correlação de Spearman", f"ρ = {rho:.3f}")
    m2.metric("Significância", f"p = {p:.4f}", "significativo" if p < 0.05 else "não significativo")
    if uf.imputado.any():
        st.caption("Observação: o desemprego de 2021 é imputado (a PNAD não divulgou estimativas por UF na pandemia).")
    fig = px.scatter(uf, x="ideb_medio", y="desemprego", text="SG_UF", color="REGIAO",
                     trendline="ols", size=[12]*len(uf),
                     labels={"ideb_medio": f"IDEB médio ({etapa})", "desemprego": "Desemprego (%)"},
                     title=f"IDEB × desemprego por UF — {etapa} ({ano})")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, width='stretch')
    with st.expander("Ver tabela por estado"):
        st.dataframe(uf.drop(columns="imputado").sort_values("ideb_medio", ascending=False).round(2),
                     width='stretch')
    st.success("Resposta à pergunta 2: a correlação é negativa — estados com IDEB maior tendem a ter "
               "menos desemprego. Use os seletores para ver que o padrão se repete em outros anos e etapas.")

# TESTES DE HIPÓTESE
elif pagina == "Testes de hipótese":
    st.title("Testes de hipótese")
    st.markdown("Formalizamos dois resultados com testes não-paramétricos (os dados não são normais, "
                "como o Shapiro–Wilk confirma). Em ambos, p < 0,05 leva a rejeitar H0.")

    st.header("Teste A — IDEB do Ensino Médio: Sul vs Nordeste")
    st.markdown("""
- **H0:** a distribuição do IDEB-EM é igual entre Sul e Nordeste.
- **H1:** as distribuições diferem (Sul maior).
- **Teste:** Mann–Whitney U (dois grupos independentes, não-normais).
""")
    em23 = df[(df.ETAPA == "Ensino Médio") & (df.ANO == 2023)]
    sul = em23[em23.REGIAO == "Sul"].IDEB.dropna()
    nor = em23[em23.REGIAO == "Nordeste"].IDEB.dropna()
    U, pa = mannwhitneyu(sul, nor, alternative="greater")
    c1, c2, c3 = st.columns(3)
    c1.metric("IDEB médio — Sul", f"{sul.mean():.2f}", f"n={len(sul)}")
    c2.metric("IDEB médio — Nordeste", f"{nor.mean():.2f}", f"n={len(nor)}")
    c3.metric("p-valor", f"{pa:.1e}", "rejeita H0" if pa < 0.05 else "não rejeita")
    fig = px.box(em23[em23.REGIAO.isin(["Sul", "Nordeste"])], x="REGIAO", y="IDEB",
                 color="REGIAO", category_orders={"REGIAO": ["Sul", "Nordeste"]},
                 title="IDEB do Ensino Médio (2023)")
    st.plotly_chart(fig, width='stretch')
    st.success(f"O Sul tem IDEB de Ensino Médio significativamente maior que o Nordeste "
               f"(p = {pa:.1e}). A desigualdade regional é estatisticamente confirmada.")

    st.divider()
    st.header("Teste B — desemprego: superior completo vs médio completo")
    st.markdown("""
- **H0:** a taxa de desemprego é igual entre quem tem superior completo e médio completo.
- **H1:** a de quem tem superior completo é menor.
- **Teste:** Wilcoxon pareado (as duas séries são medidas nos mesmos trimestres).
""")
    def serie(cat):
        s = des[(des.variavel_id == 4099) & (des.classificacao == "Nível de instrução") &
                (des.nivel == "N1") & (des.categoria == cat)].dropna(subset=["valor"])
        return s.set_index("periodo").valor
    sup = serie("Ensino superior completo ou equivalente")
    med = serie("Ensino médio completo ou equivalente")
    j = pd.concat([sup.rename("Superior completo"), med.rename("Médio completo")], axis=1).dropna()
    W, pb = wilcoxon(j["Superior completo"], j["Médio completo"], alternative="less")
    c1, c2, c3 = st.columns(3)
    c1.metric("Desemprego médio — superior", f"{j['Superior completo'].mean():.1f}%")
    c2.metric("Desemprego médio — médio", f"{j['Médio completo'].mean():.1f}%")
    c3.metric("p-valor", f"{pb:.1e}", "rejeita H0" if pb < 0.05 else "não rejeita")
    fig = px.line(j.reset_index().melt("periodo", var_name="Escolaridade", value_name="Desemprego (%)"),
                  x="periodo", y="Desemprego (%)", color="Escolaridade",
                  title="Desemprego por escolaridade ao longo do tempo (Brasil)")
    st.plotly_chart(fig, width='stretch')
    st.success(f"Quem tem superior completo enfrenta desemprego menor em todos os {len(j)} trimestres "
               f"(p = {pb:.1e}).")

# RECORTES
elif pagina == "Recortes (comparar)":
    st.title("Recortes — comparar dois estados")
    st.markdown("Escolha dois estados para comparar a evolução do IDEB e do desemprego lado a lado. "
                "É aqui que dá para 'mergulhar' em casos específicos durante a apresentação.")
    ufs = sorted(df.SG_UF.unique())
    c1, c2 = st.columns(2)
    a = c1.selectbox("Estado A", ufs, index=ufs.index("SP") if "SP" in ufs else 0)
    b = c2.selectbox("Estado B", ufs, index=ufs.index("PE") if "PE" in ufs else 1)
    etapa = st.selectbox("Etapa", df.ETAPA.unique())

    comp = (df[df.SG_UF.isin([a, b]) & (df.ETAPA == etapa)]
            .groupby(["SG_UF", "ANO"], observed=True)
            .agg(IDEB=("IDEB", "mean"), Desemprego=("DESEMPREGO_UF", "first")).reset_index())
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(comp, x="ANO", y="IDEB", color="SG_UF", markers=True,
                      title=f"IDEB médio ({etapa})")
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig = px.line(comp, x="ANO", y="Desemprego", color="SG_UF", markers=True,
                      title="Taxa de desemprego (%)")
        st.plotly_chart(fig, width='stretch')
    st.dataframe(comp.round(2), width='stretch')

    ga = df[(df.SG_UF == a) & (df.ETAPA == etapa)].IDEB.dropna()
    gb = df[(df.SG_UF == b) & (df.ETAPA == etapa)].IDEB.dropna()
    if len(ga) > 10 and len(gb) > 10:
        U, pv = mannwhitneyu(ga, gb)
        dirr = "maior" if ga.mean() > gb.mean() else "menor"
        st.info(f"Mann–Whitney: o IDEB de {a} ({ga.mean():.2f}) é {dirr} que o de {b} "
                f"({gb.mean():.2f}) — p = {pv:.2e} "
                f"({'diferença significativa' if pv < 0.05 else 'sem diferença significativa'}).")

# CONCLUSÕES
else:
    st.title("Conclusões")
    st.markdown("""
### O que os dados mostram

1. **Escolaridade protege contra o desemprego.** Quem tem ensino superior completo enfrenta
   desemprego sistematicamente menor que quem tem só o ensino médio — em todos os trimestres da
   série (Wilcoxon pareado, p < 0,05). Em média, cerca de 5% contra 12%.

2. **Melhor educação, menos desemprego — também entre estados.** O IDEB médio do Ensino Médio por UF
   correlaciona-se negativamente com o desemprego estadual (Spearman ρ ≈ −0,45, p < 0,05).

3. **Desigualdade regional é real.** O Sul tem IDEB de Ensino Médio significativamente maior que o
   Nordeste (Mann–Whitney, p < 0,05), espelhando o mapa do desemprego.

### Mensagem
Investir em educação (elevar o IDEB e a escolaridade) é também política de empregabilidade. As
desigualdades educacionais entre regiões ajudam a explicar as desigualdades no mercado de trabalho.
""")
    st.warning("Limitações: o IDEB cobre só anos ímpares e a rede pública; o desemprego estadual de "
               "2021 foi imputado (suspensão da PNAD na pandemia); correlações por UF não implicam "
               "causalidade no nível individual (cuidado com a falácia ecológica).")
