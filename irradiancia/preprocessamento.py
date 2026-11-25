"""Tratamento de dados faltantes e de valores extremos.

Os dois tratamentos de faltantes seguem a mesma interface (`aplicar`) e podem ser trocados
entre si: `PreenchimentoSimples` é a condição SEM pré-processamento e `ImputacaoCausal`,
a condição COM.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .dados import Serie


class TratamentoDeFaltantes(ABC):
    @abstractmethod
    def aplicar(self, serie: Serie) -> pd.Series:
        """Devolve a série sem NaN."""


class PreenchimentoSimples(TratamentoDeFaltantes):
    """O mínimo para os modelos rodarem.

    No INMET, falha vira zero, como no código de Santos Júnior et al. (2022). No Kaggle,
    a série nunca chega a zero, então a falha recebe o último valor medido.
    """

    def aplicar(self, serie: Serie) -> pd.Series:
        if serie.irradiacao_topo is not None:
            return serie.valores.fillna(0.0)
        return serie.valores.ffill().bfill()


class ImputacaoCausal(TratamentoDeFaltantes):
    """Estima cada falha usando apenas o passado, para não vazar informação do teste.

    O cálculo é feito sobre uma razão r = valor / referência:
    1. r da última hora medida no mesmo dia;
    2. se o dia ainda não tem medição, a mediana de r no mesmo horário dos `dias` dias
       anteriores. É a variação diurna média de Falge et al. (2001), com a mediana no lugar
       da média para não ser puxada por dias muito nublados ou muito limpos;
    3. se a falha dura mais que `dias` dias, a última mediana conhecida.
    Com `limitar=True`, a estimativa fica entre o menor e o maior valor medidos até aquele
    instante, para não criar valores que a estação nunca registrou.

    No INMET, a referência é a irradiação no topo da atmosfera, e r é o índice de claridade
    k_t (DUFFIE; BECKMAN, 2013). Assim a estimativa acompanha o ciclo do dia e o
    céu daquele dia. Sem referência (Kaggle e variáveis meteorológicas), r é o próprio valor.
    """

    def __init__(self, dias: int = 14, claridade_maxima: float = 1.0, limitar: bool = True):
        self.dias = dias
        self.claridade_maxima = claridade_maxima
        self.limitar = limitar

    def aplicar(self, serie: Serie) -> pd.Series:
        return self.imputar(serie.valores, serie.irradiacao_topo)

    def imputar(self, valores: pd.Series, referencia: pd.Series | None = None) -> pd.Series:
        if referencia is None:
            razao = valores.astype(float)
        else:
            razao = (valores / referencia.where(referencia > 0)).clip(0, self.claridade_maxima)
        tabela = _dias_por_horas(razao)
        mesmo_dia = tabela.ffill(axis=1)
        perfil = tabela.rolling(self.dias, min_periods=1).median().shift(1).ffill()
        estimativa = _de_volta_a_serie(mesmo_dia.fillna(perfil), valores.index)
        if referencia is not None:
            estimativa = estimativa * referencia
        if self.limitar:
            menor_ate_agora = valores.cummin().ffill().fillna(-np.inf)
            maior_ate_agora = valores.cummax().ffill().fillna(np.inf)
            estimativa = estimativa.clip(menor_ate_agora, maior_ate_agora)
        # Só o começo do treino pode ficar sem histórico; ali o primeiro valor seguinte resolve.
        return valores.fillna(estimativa).ffill().bfill()


class CorteIQR:
    """Limita cada coluna a [Q1 − 1,5·IQR, Q3 + 1,5·IQR], com quartis do treino.

    Regra de Tukey (1977), usada por Honnurvali et al. (2026, Seção III).
    """

    def __init__(self, n_treino: int):
        self.n_treino = n_treino

    def aplicar(self, tabela: pd.DataFrame) -> pd.DataFrame:
        cortada = tabela.copy()
        for coluna in cortada.columns:
            q1, q3 = cortada[coluna].iloc[:self.n_treino].quantile([0.25, 0.75])
            amplitude = q3 - q1
            cortada[coluna] = cortada[coluna].clip(q1 - 1.5 * amplitude, q3 + 1.5 * amplitude)
        return cortada


def _dias_por_horas(serie: pd.Series) -> pd.DataFrame:
    """Reorganiza a série horária numa tabela: uma linha por dia, uma coluna por hora."""
    dias = serie.index.normalize()
    tabela = pd.DataFrame({"dia": dias, "hora": serie.index.hour, "valor": serie.to_numpy()})
    tabela = tabela.pivot(index="dia", columns="hora", values="valor")
    return tabela.reindex(pd.date_range(dias.min(), dias.max(), freq="D"))


def _de_volta_a_serie(tabela: pd.DataFrame, indice: pd.DatetimeIndex) -> pd.Series:
    linhas = tabela.index.get_indexer(indice.normalize())
    colunas = tabela.columns.get_indexer(indice.hour)
    return pd.Series(tabela.to_numpy()[linhas, colunas], index=indice)
