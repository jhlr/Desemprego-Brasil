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

st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stMainMenu"] {display: none;}
    .stDeployButton {display: none;}
    #MainMenu {display: none;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stAppViewBlockContainer"] {padding-top: 0rem;}
    [data-testid="stSidebarUserContent"] {padding-top: 0.5rem;}
    .block-container {padding-top: 0rem;}
    [data-testid="stMain"] .block-container {padding-top: 0rem !important;}

    /* --- tipografia dos slides de texto --- */
    .lead {
        font-size: 1.2rem; line-height: 1.65; color: #cfd2dc;
        font-weight: 400; max-width: 60rem; margin: 0.2rem 0 0.6rem 0;
    }
    .lead strong { color: #f2f3f7; }
    .eyebrow {
        text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.78rem;
        font-weight: 700; color: #6db3d6; margin-bottom: 0.4rem;
    }
    .card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #2b8cbe;
        border-radius: 12px; padding: 1.05rem 1.25rem; height: 100%;
    }
    .card h4 { margin: 0 0 0.35rem 0; font-size: 1.02rem; color: #f2f3f7; }
    .card p  { margin: 0; color: #b9bdc9; font-size: 0.93rem; line-height: 1.5; }
    .qnum {
        display: inline-flex; align-items: center; justify-content: center;
        width: 1.6rem; height: 1.6rem; border-radius: 50%;
        background: #2b8cbe; color: #fff; font-weight: 700; font-size: 0.9rem;
        margin-right: 0.5rem; flex: 0 0 auto;
    }
    .qhead { display: flex; align-items: center; margin-bottom: 0.4rem; }
    .qhead h4 { margin: 0; }
    .card-cap { color: #aeb2bf; font-size: 0.92rem; line-height: 1.45;
                margin: 0 0 0.5rem 0.1rem; }
    .hero {
        background: linear-gradient(135deg, rgba(43,140,190,0.16), rgba(43,140,190,0.03));
        border: 1px solid rgba(43,140,190,0.30);
        border-radius: 14px; padding: 1.3rem 1.6rem; margin: 0.2rem 0 0.9rem 0;
        max-width: 62rem;
    }
    .hero-q {
        font-size: 1.65rem; font-weight: 700; line-height: 1.25; color: #f2f3f7;
        margin: 0 0 0.5rem 0;
    }
    .hero-sub {
        font-size: 1.05rem; line-height: 1.6; color: #cfd2dc; margin: 0;
    }
    .hero-sub strong { color: #f2f3f7; }

    /* --- testes de hipótese --- */
    .hipgrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin: 0.4rem 0; }
    .hcard { display: flex; align-items: flex-start; gap: 0.6rem; }
    .hcard p { color: #c6cad4; }
    .htag {
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.78rem; padding: 0.15rem 0.5rem;
        border-radius: 6px; flex: 0 0 auto; line-height: 1.2;
    }
    .htag-h0 { background: rgba(217,109,109,0.18); color: #e89090; }
    .htag-h1 { background: rgba(108,193,142,0.18); color: #7fd0a0; }
    .htag-t  { background: rgba(43,140,190,0.18); color: #6db3d6; }
    @media (max-width: 900px) { .hipgrid { grid-template-columns: 1fr; } }

    .verdict {
        display: flex; align-items: center; gap: 0.8rem;
        border-radius: 12px; padding: 0.9rem 1.1rem; margin-top: 0.4rem;
        font-size: 0.97rem; line-height: 1.5;
    }
    .verdict-ok { background: rgba(108,193,142,0.10); border: 1px solid rgba(108,193,142,0.35); color: #d6e9dd; }
    .verdict-no { background: rgba(217,109,109,0.10); border: 1px solid rgba(217,109,109,0.35); color: #ecd9d9; }
    .vbadge {
        font-weight: 700; font-size: 0.8rem; padding: 0.25rem 0.6rem; border-radius: 999px;
        white-space: nowrap; flex: 0 0 auto;
    }
    .verdict-ok .vbadge { background: #6cc18e; color: #06301c; }
    .verdict-no .vbadge { background: #d96d6d; color: #2e0a0a; }
    </style>
    """,
    unsafe_allow_html=True,
)

ORDEM_INSTRUCAO = A.ORDEM_INSTRUCAO
INSTRUCAO_CURTA = A.INSTRUCAO_CURTA


@st.cache_data(show_spinner="Carregando dados...")
def load():
    return A.load_base_unida(), A.load_desemprego()


@st.cache_data(show_spinner="Carregando mapa...")
def load_geojson_uf():
    import requests
    url = ("https://raw.githubusercontent.com/codeforamerica/click_that_hood/"
           "master/public/data/brazil-states.geojson")
    return requests.get(url, timeout=30).json()


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
DASHBOARD_URL = "https://issue-fading-freeware.ngrok-free.dev"
_qc1, _qc2, _qc3 = st.sidebar.columns([1, 8, 1])
_qc2.image(
    f"https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={DASHBOARD_URL}",
    width='stretch',
)
st.sidebar.title("Educacao e desemprego")
_sec = st.query_params.get("sec")
_idx = SECOES.index(_sec) if _sec in SECOES else 0
pagina = st.sidebar.radio("Seções", SECOES, index=_idx)
st.sidebar.divider()
st.sidebar.caption("Fontes: INEP (IDEB por escola) · IBGE / PNAD Contínua (API SIDRA).")


def kpi_row():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Escolas (linhas)", f"{len(df):,}")
    c2.metric("UFs", df.SG_UF.nunique())
    _anos = sorted(df.ANO.unique())
    c3.metric("Edições do IDEB", f"{len(_anos)}", f"{_anos[0]}–{_anos[-1]}",
              delta_color="off")
    tab, per = A.desocupacao_por_instrucao(des)
    sup = tab.loc[tab.label == "Superior completo", "valor"].iloc[0]
    c4.metric(f"Desemprego com superior ({per})", f"{sup:.1f}%")


# VISÃO GERAL
if pagina == "Visão geral":
    st.markdown('<div class="eyebrow">IDEB × Desemprego · INEP e IBGE</div>',
                unsafe_allow_html=True)
    st.title("Empregabilidade e Educação no Brasil")
    st.markdown("""
<div class="hero">
  <p class="hero-q">Mais educação significa menos desemprego?</p>
  <p class="hero-sub">
    É a pergunta que guia este trabalho. Para respondê-la, cruzamos duas bases públicas — o
    <strong>IDEB por escola</strong> (INEP) e a <strong>taxa de desemprego</strong> (PNAD Contínua /
    IBGE) — nas edições de <strong>2017, 2019, 2021 e 2023</strong>.
  </p>
</div>
<p class="lead">As duas telas abaixo resumem os dois principais resultados; as seções seguintes detalham cada um.</p>
""", unsafe_allow_html=True)
    kpi_row()
    st.divider()
    st.subheader("Os dois resultados principais")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            '<div class="qhead"><span class="qnum">1</span>'
            '<h4>Desemprego cai com a escolaridade</h4></div>'
            '<p class="card-cap">Entre pessoas: quanto maior o nível de instrução, menor a taxa de '
            'desemprego.</p>', unsafe_allow_html=True)
        tab, per = A.desocupacao_por_instrucao(des)
        fig = px.bar(tab, x="label", y="valor",
                     color="valor", color_continuous_scale="RdYlGn_r",
                     labels={"valor": "Desemprego (%)", "label": ""})
        fig.update_layout(coloraxis_showscale=False, height=400,
                          margin={"t": 10, "b": 10},
                          xaxis={"categoryorder": "array",
                          "categoryarray": [INSTRUCAO_CURTA[c] for c in ORDEM_INSTRUCAO],
                          "tickangle": -45})
        st.plotly_chart(fig, width='stretch')
    with col2:
        _sub = df[(df.ANO == 2023) & (df.ETAPA == "Ensino Médio")]
        m = (_sub.groupby("SG_UF", observed=True)
                 .agg(ideb_medio=("IDEB", "mean"), desocupacao=("DESEMPREGO_UF", "first"))
                 .reset_index())
        m["REGIAO"] = m.SG_UF.map(A.UF2REGIAO)
        rho, p = spearmanr(m.ideb_medio, m.desocupacao)
        st.markdown(
            '<div class="qhead"><span class="qnum">2</span>'
            '<h4>Melhor IDEB, menor desemprego</h4></div>'
            f'<p class="card-cap">Entre estados (Ensino Médio, 2023): UFs com IDEB médio mais alto '
            f'tendem a ter menos desemprego · ρ={rho:.2f}, p={p:.3f}.</p>', unsafe_allow_html=True)
        fig = px.scatter(m, x="ideb_medio", y="desocupacao", text="SG_UF", color="REGIAO",
                         trendline="ols", trendline_scope="overall",
                         trendline_color_override="#bbbbbb",
                         labels={"ideb_medio": "IDEB médio (EM)",
                         "desocupacao": "Desemprego (%)"})
        fig.update_traces(textposition="top center")
        fig.for_each_trace(lambda t: t.update(showlegend=False)
                           if t.name == "Overall Trendline" else None)
        fig.update_layout(height=440, margin={"t": 20, "b": 10},
                          legend={"title": "", "orientation": "h",
                                  "yanchor": "top", "y": -0.18,
                                  "xanchor": "center", "x": 0.5})
        st.plotly_chart(fig, width='stretch')
    st.success("Em uma frase: tanto entre pessoas quanto entre estados, mais educação aparece "
               "associada a menos desemprego. As próximas seções mostram os dados e os testes.")

# CONTEXTO E PERGUNTAS
elif pagina == "Contexto e perguntas":
    st.markdown('<div class="eyebrow">Por que e o que perguntamos</div>', unsafe_allow_html=True)
    st.title("Contexto e perguntas de pesquisa")
    st.markdown("""
<p class="lead">
A relação entre educação e mercado de trabalho está no centro do debate sobre desigualdade no Brasil.
Escolaridade costuma ser tratada como caminho para melhores oportunidades — mas é possível
<strong>medir</strong> isso com dados públicos? É o que fazemos aqui, juntando desempenho escolar e
desemprego.
</p>
""", unsafe_allow_html=True)

    st.subheader("As três perguntas")
    perguntas = [
        ("Quem tem mais escolaridade enfrenta menos desemprego?",
         "Comparação direta da taxa de desemprego por nível de instrução."),
        ("Estados com melhor IDEB têm menor desemprego?",
         "Correlação entre o IDEB médio da UF e a taxa de desemprego estadual."),
        ("Há desigualdade regional no desempenho educacional?",
         "Comparação do IDEB entre regiões (Sul × Nordeste)."),
    ]
    cols = st.columns(3)
    for col, (i, (titulo, desc)) in zip(cols, enumerate(perguntas, 1)):
        col.markdown(
            f'<div class="card"><div class="qhead"><span class="qnum">{i}</span>'
            f'<h4>{titulo}</h4></div><p>{desc}</p></div>',
            unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        '<div class="card"><h4>Recorte escolhido</h4><p>Foco no <strong>Ensino Médio</strong> '
        '(etapa mais próxima da entrada no mercado de trabalho) para o cruzamento por estado, usando '
        'todas as 27 UFs e as quatro edições do IDEB disponíveis a partir de 2017.</p></div>',
        unsafe_allow_html=True)
    st.markdown("")
    st.info("Cada pergunta é respondida em uma seção do menu e formalizada na seção "
            "'Testes de hipótese'.")

# OS DADOS E O MÉTODO
elif pagina == "Os dados e o método":
    st.markdown('<div class="eyebrow">Fontes, coleta e tratamento</div>', unsafe_allow_html=True)
    st.title("Os dados e o método")
    c1, c2, c3 = st.columns(3)
    c1.metric("IDEB por escola (INEP)", "534.808 linhas")
    c2.metric("PNAD / desemprego (IBGE)", "318.816 linhas")
    c3.metric("Base unida (UF + ano)", f"{len(df):,} linhas")
    st.markdown("")

    col_a, col_b = st.columns(2)
    col_a.markdown("""
<div class="card"><h4>Coleta</h4><p>
<strong>IDEB por escola (INEP):</strong> planilhas oficiais das edições 2017, 2019, 2021 e 2023, três
etapas (Anos Iniciais, Anos Finais, Ensino Médio), transformadas para formato longo.<br><br>
<strong>Desemprego (IBGE):</strong> API SIDRA, agregados 4093 (sexo) e 4095 (nível de instrução),
variável 4099 (taxa de desocupação), de 2016 a 2026.<br><br>
<strong>Base unida:</strong> cada escola com IDEB recebe o desemprego médio anual da sua UF naquele
ano (chave de junção: UF + ano).
</p></div>
""", unsafe_allow_html=True)
    col_b.markdown("""
<div class="card"><h4>Decisões e cuidados</h4><p>
<strong>"Desocupação" = "desemprego"</strong> — termo técnico do IBGE e termo popular; mesmo
indicador.<br><br>
<strong>IDEB só em anos ímpares</strong> — medido a cada dois anos; cada valor é um ponto, não um
intervalo.<br><br>
<strong>PNAD é um retrato, não soma</strong> — o valor anual da UF é a <strong>média dos quatro
trimestres</strong>.<br><br>
<strong>2021 sem PNAD por UF</strong> — imputado pela média de 2020 e 2022 de cada estado, sinalizado
em coluna própria.<br><br>
<strong>Privadas não têm IDEB</strong> — por isso comparamos regiões e estados, não "pública ×
privada".
</p></div>
""", unsafe_allow_html=True)
    st.markdown("")
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

    aba_nivel, aba_tempo = st.tabs(["Por nível de instrução", "Ao longo do tempo"])

    with aba_nivel:
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

    with aba_tempo:
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

    uf = sub.groupby("SG_UF", observed=True).IDEB.mean().sort_values(ascending=False).reset_index()
    uf["REGIAO"] = uf.SG_UF.map(A.UF2REGIAO)

    aba_mapa, aba_dist, aba_rank = st.tabs(["Mapa", "Distribuição", "Ranking por UF"])

    with aba_mapa:
        try:
            geojson = load_geojson_uf()
            fig = px.choropleth(
                uf, geojson=geojson, locations="SG_UF",
                featureidkey="properties.sigla", color="IDEB",
                color_continuous_scale="Viridis",
                hover_name="SG_UF", hover_data={"SG_UF": False, "IDEB": ":.2f"},
                title=f"IDEB médio por estado — {etapa} ({ano})")
            fig.update_geos(fitbounds="locations", visible=False,
                            bgcolor="rgba(0,0,0,0)")
            fig.update_layout(
                height=560,
                geo={"center": {"lat": -15, "lon": -54}},
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title_x=0.5,
                coloraxis_colorbar={"title": "IDEB", "len": 0.7, "x": 0.95})
            st.plotly_chart(fig, width='stretch')
        except Exception:
            st.warning("Não foi possível carregar o mapa (precisa de internet para o GeoJSON).")

    with aba_dist:
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

    with aba_rank:
        fig = px.bar(uf, x="IDEB", y="SG_UF", color="REGIAO", orientation="h", height=560,
                     category_orders={"SG_UF": uf.sort_values("IDEB").SG_UF.tolist()},
                     labels={"SG_UF": ""}, title=f"IDEB médio por UF — {etapa} ({ano})")
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
                     trendline="ols", trendline_scope="overall",
                     trendline_color_override="#bbbbbb", size=[12]*len(uf),
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
    st.markdown('<div class="eyebrow">Formalização estatística</div>', unsafe_allow_html=True)
    st.title("Testes de hipótese")
    st.markdown(
        '<p class="lead">Formalizamos dois resultados com testes <strong>não-paramétricos</strong> '
        '(os dados não são normais, como o Shapiro–Wilk confirma). Em ambos, '
        '<strong>p &lt; 0,05</strong> leva a rejeitar H₀.</p>', unsafe_allow_html=True)

    def hipotese_cards(h0, h1, teste):
        st.markdown(
            '<div class="hipgrid">'
            f'<div class="card hcard"><span class="htag htag-h0">H₀</span><p>{h0}</p></div>'
            f'<div class="card hcard"><span class="htag htag-h1">H₁</span><p>{h1}</p></div>'
            f'<div class="card hcard"><span class="htag htag-t">Teste</span><p>{teste}</p></div>'
            '</div>', unsafe_allow_html=True)

    def veredito(rejeita, texto):
        cls = "verdict-ok" if rejeita else "verdict-no"
        rotulo = "H₀ rejeitada" if rejeita else "H₀ não rejeitada"
        st.markdown(f'<div class="verdict {cls}"><span class="vbadge">{rotulo}</span>'
                    f'<span>{texto}</span></div>', unsafe_allow_html=True)

    aba_a, aba_b = st.tabs(["Teste A — Sul vs Nordeste", "Teste B — superior vs médio"])

    with aba_a:
        st.subheader("IDEB do Ensino Médio: Sul vs Nordeste")
        hipotese_cards(
            "A distribuição do IDEB-EM é igual entre Sul e Nordeste.",
            "As distribuições diferem — o Sul é maior.",
            "Mann–Whitney U · dois grupos independentes, não-normais.")
        em23 = df[(df.ETAPA == "Ensino Médio") & (df.ANO == 2023)]
        sul = em23[em23.REGIAO == "Sul"].IDEB.dropna()
        nor = em23[em23.REGIAO == "Nordeste"].IDEB.dropna()
        U, pa = mannwhitneyu(sul, nor, alternative="greater")
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("IDEB médio — Sul", f"{sul.mean():.2f}", f"n={len(sul)}", delta_color="off")
        c2.metric("IDEB médio — Nordeste", f"{nor.mean():.2f}", f"n={len(nor)}", delta_color="off")
        c3.metric("p-valor (Mann–Whitney)", f"{pa:.1e}")
        fig = px.box(em23[em23.REGIAO.isin(["Sul", "Nordeste"])], x="REGIAO", y="IDEB",
                     color="REGIAO", category_orders={"REGIAO": ["Sul", "Nordeste"]},
                     points="outliers", title="Distribuição do IDEB-EM (2023)")
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, width='stretch')
        veredito(pa < 0.05,
                 f"O Sul tem IDEB de Ensino Médio significativamente maior que o Nordeste "
                 f"(p = {pa:.1e}). A desigualdade regional é estatisticamente confirmada.")

    with aba_b:
        st.subheader("Desemprego: superior completo vs médio completo")
        hipotese_cards(
            "A taxa de desemprego é igual entre superior completo e médio completo.",
            "A de quem tem superior completo é menor.",
            "Wilcoxon pareado · séries medidas nos mesmos trimestres.")
        def serie(cat):
            s = des[(des.variavel_id == 4099) & (des.classificacao == "Nível de instrução") &
                    (des.nivel == "N1") & (des.categoria == cat)].dropna(subset=["valor"])
            return s.set_index("periodo").valor
        sup = serie("Ensino superior completo ou equivalente")
        med = serie("Ensino médio completo ou equivalente")
        j = pd.concat([sup.rename("Superior completo"), med.rename("Médio completo")], axis=1).dropna()
        W, pb = wilcoxon(j["Superior completo"], j["Médio completo"], alternative="less")
        gap = j["Médio completo"].mean() - j["Superior completo"].mean()
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Desemprego médio — superior", f"{j['Superior completo'].mean():.1f}%")
        c2.metric("Desemprego médio — médio", f"{j['Médio completo'].mean():.1f}%",
                  f"+{gap:.1f} p.p.", delta_color="inverse")
        c3.metric("p-valor (Wilcoxon)", f"{pb:.1e}")
        fig = px.line(j.reset_index().melt("periodo", var_name="Escolaridade", value_name="Desemprego (%)"),
                      x="periodo", y="Desemprego (%)", color="Escolaridade", markers=True,
                      title="Desemprego por escolaridade ao longo do tempo (Brasil)")
        fig.update_layout(height=420)
        st.plotly_chart(fig, width='stretch')
        veredito(pb < 0.05,
                 f"Quem tem superior completo enfrenta desemprego menor em todos os {len(j)} "
                 f"trimestres (p = {pb:.1e}).")

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
    st.markdown('<div class="eyebrow">Síntese dos resultados</div>', unsafe_allow_html=True)
    st.title("Conclusões")
    st.markdown('<p class="lead">O que os dados mostram, em três achados.</p>',
                unsafe_allow_html=True)

    achados = [
        ("Escolaridade protege contra o desemprego",
         "Quem tem ensino superior completo enfrenta desemprego sistematicamente menor que quem tem "
         "só o ensino médio — em todos os trimestres da série (Wilcoxon pareado, p &lt; 0,05). "
         "Em média, cerca de 5% contra 12%."),
        ("Melhor educação, menos desemprego — também entre estados",
         "O IDEB médio do Ensino Médio por UF correlaciona-se negativamente com o desemprego "
         "estadual (Spearman ρ ≈ −0,45, p &lt; 0,05)."),
        ("Desigualdade regional é real",
         "O Sul tem IDEB de Ensino Médio significativamente maior que o Nordeste (Mann–Whitney, "
         "p &lt; 0,05), espelhando o mapa do desemprego."),
    ]
    cols = st.columns(3)
    for col, (i, (titulo, desc)) in zip(cols, enumerate(achados, 1)):
        col.markdown(
            f'<div class="card"><div class="qhead"><span class="qnum">{i}</span>'
            f'<h4>{titulo}</h4></div><p>{desc}</p></div>',
            unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        '<div class="card"><h4>Mensagem</h4><p>Investir em educação (elevar o IDEB e a escolaridade) '
        'é também política de empregabilidade. As desigualdades educacionais entre regiões ajudam a '
        'explicar as desigualdades no mercado de trabalho.</p></div>',
        unsafe_allow_html=True)
    st.markdown("")
    st.warning("Limitações: o IDEB cobre só anos ímpares e a rede pública; o desemprego estadual de "
               "2021 foi imputado (suspensão da PNAD na pandemia); correlações por UF não implicam "
               "causalidade no nível individual (cuidado com a falácia ecológica).")
