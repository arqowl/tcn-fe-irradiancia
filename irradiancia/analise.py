"""Contagens usadas na investigação dos dados faltantes do INMET."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dados import Estacao, Serie
from .geometria_solar import irradiacao_horaria_topo

AMOSTRAS_POR_DIA_INMET = 13


def _maior_sequencia(marcas: pd.Series) -> int:
    grupos = (marcas != marcas.shift()).cumsum()
    return int(marcas.groupby(grupos).sum().max())


def faltantes_por_ano(tabelas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Para cada estação e ano: horas sem dado de radiação, separadas pela posição do Sol."""
    linhas = []
    for nome, tabela in tabelas.items():
        for ano, t in tabela.groupby(tabela.index.year):
            sem_dado, noite = t["radiacao"].isna(), t["topo"] == 0
            falha = sem_dado & ~noite
            linhas.append({"estação": nome, "ano": ano, "horas": len(t),
                           "sem dado (%)": 100 * sem_dado.mean(),
                           "Sol abaixo do horizonte (%)": 100 * (sem_dado & noite).mean(),
                           "falha com Sol (%)": 100 * falha.mean(),
                           "maior falha com Sol (dias)": _maior_sequencia(falha) / AMOSTRAS_POR_DIA_INMET})
    return pd.DataFrame(linhas).set_index(["estação", "ano"]).round(1)


def faltantes_por_hora(tabela: pd.DataFrame) -> pd.DataFrame:
    """Fração das horas sem dado em cada hora UTC, separando as horas com o Sol abaixo do horizonte."""
    sem_dado, noite = tabela["radiacao"].isna(), tabela["topo"] == 0
    hora = tabela.index.hour
    return pd.DataFrame({
        "Sol abaixo do horizonte": (sem_dado & noite).groupby(hora).mean() * 100,
        "falha com Sol": (sem_dado & ~noite).groupby(hora).mean() * 100,
    }).rename_axis("hora UTC")


def conferir_convencao_horaria(tabelas: dict[str, pd.DataFrame], estacoes: dict[str, Estacao]) -> pd.DataFrame:
    """A que intervalo de uma hora se refere o valor gravado às HH UTC?

    Cada hipótese desloca o fim do intervalo em relação a HH. Duas coisas valem na hipótese certa:
    - nenhuma hora com o Sol abaixo do horizonte o tempo todo tem radiação medida;
    - nos dias mais limpos (percentil 90 do índice de claridade k em cada hora), k não passa
      de 1. Um deslocamento errado deixa k alto de um lado do meio-dia e baixo do outro.
    """
    hipoteses = {-1.0: "HH−2 a HH−1", -0.5: "HH−1,5 a HH−0,5", 0.0: "HH−1 a HH",
                 0.5: "HH−0,5 a HH+0,5", 1.0: "HH a HH+1"}
    horas_de_k = (10, 12, 14, 16, 18, 20)
    linhas = []
    for nome, tabela in tabelas.items():
        e = estacoes[nome]
        radiacao = tabela["radiacao"].to_numpy()
        for deslocamento, rotulo in hipoteses.items():
            topo = irradiacao_horaria_topo(tabela.index + pd.Timedelta(hours=deslocamento), e.latitude, e.longitude)
            noite = topo == 0
            k = pd.Series(radiacao / np.where(noite, np.nan, topo), index=tabela.index)
            k_limpo = k.groupby(tabela.index.hour).quantile(0.9)
            linha = {"estação": nome, "intervalo": rotulo,
                     "horas com Sol abaixo do horizonte": int(noite.sum()),
                     "dessas, com radiação > 50 kJ/m²": int(np.sum(noite & (radiacao > 50)))}
            linha.update({f"k às {h} UTC": round(float(k_limpo.get(h, np.nan)), 2) for h in horas_de_k})
            linhas.append(linha)
    return pd.DataFrame(linhas).set_index(["estação", "intervalo"])


def vizinhanca_das_falhas(tabelas: dict[str, pd.DataFrame], horas=(13, 17)) -> pd.DataFrame:
    """Compara a radiação medida logo antes das falhas de meio do dia com a radiação típica dessas horas."""
    linhas = []
    for nome, tabela in tabelas.items():
        radiacao = tabela["radiacao"]
        meio_do_dia = (tabela.index.hour >= horas[0]) & (tabela.index.hour <= horas[1])
        falha = radiacao.isna() & meio_do_dia
        grupos = (radiacao.isna() != radiacao.isna().shift()).cumsum()
        duracao = radiacao.isna().groupby(grupos).transform("sum")[falha]
        linhas.append({"estação": nome, f"falhas entre {horas[0]} e {horas[1]} UTC": int(falha.sum()),
                       "mediana antes da falha": radiacao.shift(1)[falha].median(),
                       "mediana dessas horas": radiacao[meio_do_dia].median(),
                       "percentil 95 dessas horas": radiacao[meio_do_dia].quantile(0.95),
                       "falhas de até 3 h (%)": 100 * (duracao <= 3).mean() if len(duracao) else np.nan})
    return pd.DataFrame(linhas).set_index("estação").round(0)


def valores_altos(tabelas: dict[str, pd.DataFrame], limites=(3600, 3700, 3800, 3900, 4000, 4100, 4200, 5000)) -> pd.DataFrame:
    """Quantas horas medidas caem em cada faixa de radiação alta, por estação."""
    faixas = {nome: pd.cut(tabela["radiacao"], limites).value_counts().sort_index()
              for nome, tabela in tabelas.items()}
    contagem = pd.DataFrame(faixas).T
    contagem.columns = [f"{int(f.left)} a {int(f.right)}" for f in contagem.columns]
    contagem["maior valor medido"] = [tabela["radiacao"].max() for tabela in tabelas.values()]
    return contagem.rename_axis("estação")


def falhas_no_teste(series: dict[str, Serie]) -> pd.DataFrame:
    """Quantos alvos do teste não foram medidos. Com o preenchimento por zero, viram zeros falsos."""
    linhas = []
    for nome, serie in series.items():
        medido = serie.medido[-serie.divisao.teste:]
        horas = serie.indice_teste.hour
        linhas.append({"dataset": nome, "alvos no teste": len(medido), "sem medição": int((~medido).sum()),
                       "sem medição (%)": round(100 * (~medido).mean(), 1),
                       "sem medição entre 13 e 17 UTC": int((~medido & (horas >= 13) & (horas <= 17)).sum())})
    return pd.DataFrame(linhas).set_index("dataset")
