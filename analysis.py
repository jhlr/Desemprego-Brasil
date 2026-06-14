"""
Funções de carga e análise usadas pelo dashboard.

Fontes: IDEB por escola (INEP, edições 2017/2019/2021/2023) e desemprego
(IBGE/PNAD Contínua, agregados 4093 e 4095, variável 4099).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
IDEB_CSV = os.path.join(DATA, "ideb_escola_2016_2023.csv")
DES_CSV = os.path.join(DATA, "desemprego_pnadc_2016_2026.csv")
JOINED_CSV = os.path.join(DATA, "educacao_emprego_brasil.csv")

# Código IBGE da UF (local_id no nível N3) para sigla e região
UF_COD2SIGLA = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF",
}
UF2REGIAO = {
    **{u: "Norte" for u in ["RO", "AC", "AM", "RR", "PA", "AP", "TO"]},
    **{u: "Nordeste" for u in ["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"]},
    **{u: "Sudeste" for u in ["MG", "ES", "RJ", "SP"]},
    **{u: "Sul" for u in ["PR", "SC", "RS"]},
    **{u: "Centro-Oeste" for u in ["MS", "MT", "GO", "DF"]},
}
# ordem didática das faixas de escolaridade (baixa -> alta)
ORDEM_INSTRUCAO = [
    "Sem instrução e menos de 1 ano de estudo",
    "Ensino fundamental incompleto ou equivalente",
    "Ensino fundamental completo ou equivalente",
    "Ensino médio incompleto ou equivalente",
    "Ensino médio completo ou equivalente",
    "Ensino superior incompleto ou equivalente",
    "Ensino superior completo ou equivalente",
]
INSTRUCAO_CURTA = {
    "Sem instrução e menos de 1 ano de estudo": "Sem instrução",
    "Ensino fundamental incompleto ou equivalente": "Fund. incompleto",
    "Ensino fundamental completo ou equivalente": "Fund. completo",
    "Ensino médio incompleto ou equivalente": "Médio incompleto",
    "Ensino médio completo ou equivalente": "Médio completo",
    "Ensino superior incompleto ou equivalente": "Superior incompleto",
    "Ensino superior completo ou equivalente": "Superior completo",
}


def load_ideb() -> pd.DataFrame:
    df = pd.read_csv(IDEB_CSV)
    df["ANO"] = df["ANO"].astype(int)
    df["REDE"] = df["REDE"].astype("category")
    df["ETAPA"] = df["ETAPA"].astype("category")
    df["SETOR"] = np.where(df["REDE"].eq("Privada"), "Privada", "Pública")
    df["REGIAO"] = df["SG_UF"].map(UF2REGIAO).astype("category")
    return df


def load_desemprego() -> pd.DataFrame:
    df = pd.read_csv(DES_CSV)
    df["ano"] = df["ano"].astype(int)
    # sigla da UF apenas para o nível estadual (N3)
    df["sigla_uf"] = np.where(df["nivel"].eq("N3"),
                              df["local_id"].map(UF_COD2SIGLA), np.nan)
    return df


def load_base_unida() -> pd.DataFrame:
    """Carrega a base unida (IDEB por escola + desemprego da UF)."""
    return pd.read_csv(JOINED_CSV)


def _desemprego_uf_anual_por_sexo(des: pd.DataFrame) -> pd.DataFrame:
    """Taxa de desemprego média anual por UF e sexo, em formato largo."""
    d = des[(des.variavel_id == 4099) & (des.classificacao == "Sexo") &
            (des.nivel == "N3")].dropna(subset=["valor"]).copy()
    g = (d.groupby(["sigla_uf", "ano", "categoria"], observed=True).valor.mean()
           .reset_index())
    wide = g.pivot_table(index=["sigla_uf", "ano"], columns="categoria",
                         values="valor").reset_index()
    wide = wide.rename(columns={
        "sigla_uf": "SG_UF", "ano": "ANO",
        "Total": "DESEMPREGO_UF", "Homens": "DESEMPREGO_UF_HOMENS",
        "Mulheres": "DESEMPREGO_UF_MULHERES"})

    # 2021 não tem PNAD ao nível de UF (suspensão na pandemia); imputa pela
    # média de 2020 e 2022 de cada estado, marcando como imputado.
    val_cols = ["DESEMPREGO_UF", "DESEMPREGO_UF_HOMENS", "DESEMPREGO_UF_MULHERES"]
    imp = (wide[wide.ANO.isin([2020, 2022])]
           .groupby("SG_UF")[val_cols].mean().reset_index())
    imp["ANO"] = 2021
    wide["DESEMPREGO_UF_IMPUTADO"] = False
    imp["DESEMPREGO_UF_IMPUTADO"] = True
    return pd.concat([wide, imp], ignore_index=True)


def build_base_unida(salvar: bool = False) -> pd.DataFrame:
    """Junta IDEB por escola e desemprego por UF (chave SG_UF + ANO).

    Mantém apenas escolas com IDEB e anexa a taxa de desemprego média anual da
    UF (total e por sexo). Com salvar=True, grava em JOINED_CSV.
    """
    base = load_ideb()
    base = base[base.IDEB.notna()].copy()
    desemp = _desemprego_uf_anual_por_sexo(load_desemprego())
    df = base.merge(desemp, on=["SG_UF", "ANO"], how="left")

    cols = ["ANO", "SG_UF", "REGIAO", "CO_MUNICIPIO", "NO_MUNICIPIO",
            "ID_ESCOLA", "NO_ESCOLA", "REDE", "SETOR", "ETAPA",
            "IDEB", "SAEB_MEDIA", "REND",
            "DESEMPREGO_UF", "DESEMPREGO_UF_HOMENS", "DESEMPREGO_UF_MULHERES",
            "DESEMPREGO_UF_IMPUTADO"]
    df = df[cols].sort_values(["ANO", "SG_UF", "ETAPA", "NO_MUNICIPIO", "NO_ESCOLA"])
    df[["DESEMPREGO_UF", "DESEMPREGO_UF_HOMENS", "DESEMPREGO_UF_MULHERES"]] = \
        df[["DESEMPREGO_UF", "DESEMPREGO_UF_HOMENS", "DESEMPREGO_UF_MULHERES"]].round(2)
    df["IDEB"] = df["IDEB"].round(2)
    df["SAEB_MEDIA"] = df["SAEB_MEDIA"].round(3)
    df["REND"] = df["REND"].round(4)

    if salvar:
        df.to_csv(JOINED_CSV, index=False, encoding="utf-8")
    return df


def remove_outliers_iqr(s: pd.Series, k: float = 1.5):
    """Retorna (máscara_de_não_outlier, limites). IQR clássico."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return s.between(low, high), (low, high)


def desocupacao_por_instrucao(des: pd.DataFrame, periodo: str | None = None) -> pd.DataFrame:
    """Taxa de desocupação no Brasil (N1) por nível de instrução, num trimestre."""
    d = des[(des.variavel_id == 4099) &
            (des.classificacao == "Nível de instrução") &
            (des.nivel == "N1") &
            (des.categoria != "Não determinado")].copy()
    if periodo is None:
        periodo = d.dropna(subset=["valor"]).periodo.max()
    out = d[d.periodo == periodo].copy()
    out = out[out.categoria != "Total"]
    out["ordem"] = out.categoria.map({c: i for i, c in enumerate(ORDEM_INSTRUCAO)})
    out["label"] = out.categoria.map(INSTRUCAO_CURTA)
    return out.sort_values("ordem"), periodo


def serie_instrucao(des: pd.DataFrame, categorias) -> pd.DataFrame:
    """Série trimestral (Brasil) da taxa de desocupação para faixas escolhidas."""
    d = des[(des.variavel_id == 4099) &
            (des.classificacao == "Nível de instrução") &
            (des.nivel == "N1") &
            (des.categoria.isin(categorias))].dropna(subset=["valor"]).copy()
    d["t"] = d.periodo.astype(str)
    return d


def ideb_uf_por_etapa(ideb: pd.DataFrame, etapa: str, ano: int) -> pd.DataFrame:
    d = ideb[(ideb.ETAPA == etapa) & (ideb.ANO == ano) & ideb.IDEB.notna()]
    g = d.groupby("SG_UF", observed=True).IDEB.mean().rename("ideb_medio").reset_index()
    g["REGIAO"] = g.SG_UF.map(UF2REGIAO)
    return g


def desemprego_uf_anual(des: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Média anual (4 trimestres) da taxa de desocupação por UF — snapshot, logo média."""
    d = des[(des.variavel_id == 4099) & (des.classificacao == "Sexo") &
            (des.categoria == "Total") & (des.nivel == "N3") & (des.ano == ano)]
    g = d.groupby("sigla_uf", observed=True).valor.mean().rename("desocupacao").reset_index()
    g = g.rename(columns={"sigla_uf": "SG_UF"})
    return g


def join_ideb_desemprego(ideb: pd.DataFrame, des: pd.DataFrame,
                         etapa: str = "Ensino Médio", ano: int = 2023) -> pd.DataFrame:
    a = ideb_uf_por_etapa(ideb, etapa, ano)
    b = desemprego_uf_anual(des, ano)
    m = a.merge(b, on="SG_UF", how="inner")
    return m


if __name__ == "__main__":
    from scipy.stats import spearmanr, mannwhitneyu, wilcoxon, shapiro
    ideb, des = load_ideb(), load_desemprego()
    print("IDEB", ideb.shape, "| DESEMPREGO", des.shape)

    tab, per = desocupacao_por_instrucao(des)
    print(f"\n[Desocupação por instrução — Brasil {per}]")
    print(tab[["label", "valor"]].to_string(index=False))

    m = join_ideb_desemprego(ideb, des, "Ensino Médio", 2023)
    rho, p = spearmanr(m.ideb_medio, m.desocupacao)
    print(f"\n[IDEB-EM x desocupação por UF, 2023] n={len(m)} rho={rho:.3f} p={p:.4f}")

    em = ideb[(ideb.ETAPA == "Ensino Médio") & (ideb.ANO == 2023) & ideb.IDEB.notna()]
    sul = em[em.REGIAO == "Sul"].IDEB
    nor = em[em.REGIAO == "Nordeste"].IDEB
    u, pu = mannwhitneyu(sul, nor, alternative="greater")
    print(f"\n[IDEB-EM Sul vs Nordeste, 2023] sul={sul.mean():.2f} nordeste={nor.mean():.2f} "
          f"U={u:.0f} p={pu:.2e} (n_sul={len(sul)}, n_nor={len(nor)})")
    print("  Shapiro Sul:", shapiro(sul.sample(min(5000, len(sul)), random_state=1)).pvalue,
          "| Nordeste:", shapiro(nor.sample(min(5000, len(nor)), random_state=1)).pvalue)

    sup = serie_instrucao(des, ["Ensino superior completo ou equivalente"]).set_index("periodo").valor
    med = serie_instrucao(des, ["Ensino médio completo ou equivalente"]).set_index("periodo").valor
    j = pd.concat([sup.rename("sup"), med.rename("med")], axis=1).dropna()
    w, pw = wilcoxon(j.sup, j.med)
    print(f"\n[Wilcoxon pareado superior vs médio completo] n={len(j)} "
          f"sup_med={j.sup.mean():.2f} med_med={j.med.mean():.2f} W={w:.0f} p={pw:.2e}")
