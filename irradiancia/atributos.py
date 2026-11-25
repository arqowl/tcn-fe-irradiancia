"""Engenharia de atributos de Honnurvali et al. (2026, Seção IV.A, Fig. 2 e Tabela 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FAIXAS_DE_INTENSIDADE = ("very_low", "low", "medium", "high", "very_high")


class EngenhariaDeAtributos:
    """Acrescenta à tabela os atributos derivados do alvo e do calendário.

    O artigo trabalha com 24 amostras por dia. Aqui "1 dia" vale `amostras_por_dia` passos:
    24 no Kaggle e 13 no INMET. Cada atributo da linha t usa só valores até t.
    """

    def __init__(self, amostras_por_dia: int, n_treino: int):
        self.dia = amostras_por_dia
        self.n_treino = n_treino

    def gerar(self, tabela: pd.DataFrame) -> pd.DataFrame:
        """`tabela` tem a coluna `alvo` e, se houver, as variáveis meteorológicas."""
        alvo = tabela["alvo"]
        partes = [tabela, self._defasagens(alvo), self._estatisticas_moveis(alvo)]
        if "temp" in tabela:
            partes.append((alvo * tabela["temp"]).rename("interacao_temp"))
        partes += [self._intensidade(alvo), self._calendario(tabela.index)]
        return pd.concat(partes, axis=1)

    def _defasagens(self, alvo: pd.Series) -> pd.DataFrame:
        passos = [1, 2, 3, 4, self.dia, 2 * self.dia, 7 * self.dia]       # 1–4 h, 1, 2 e 7 dias
        return pd.DataFrame({f"lag_{k}": alvo.shift(k) for k in passos})

    def _estatisticas_moveis(self, alvo: pd.Series) -> pd.DataFrame:
        janela = alvo.rolling(self.dia)
        return pd.DataFrame({
            "rolling_mean_dia": janela.mean(),
            "rolling_max_dia": janela.max(),
            "rolling_min_dia": janela.min(),
            "ewm_meio_dia": alvo.ewm(span=max(2, self.dia // 2), adjust=False).mean(),
            "ewm_dia": alvo.ewm(span=self.dia, adjust=False).mean(),
        })

    def _intensidade(self, alvo: pd.Series) -> pd.DataFrame:
        """Cinco faixas pelos quintis do treino (Seção IV.D), em one-hot."""
        limites = alvo.iloc[:self.n_treino].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        faixa = np.digitize(alvo, limites)
        return pd.DataFrame({f"intensidade_{nome}": (faixa == i).astype(float)
                             for i, nome in enumerate(FAIXAS_DE_INTENSIDADE)}, index=alvo.index)

    @staticmethod
    def _calendario(indice: pd.DatetimeIndex) -> pd.DataFrame:
        """Hora, dia da semana e mês em one-hot (Tabela 1)."""
        calendario = pd.DataFrame({"hora": indice.hour, "dia_semana": indice.day_of_week,
                                   "mes": indice.month}, index=indice).astype(str)
        return pd.get_dummies(calendario, dtype=float)
