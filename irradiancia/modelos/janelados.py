"""SVR e MLP sobre janelas de defasagens, com busca em grade na validação.

Reproduz o procedimento do código de Santos Júnior et al. (2022): Min-Max ajustado em
treino e validação, janelas de 12 defasagens e escolha da configuração de menor RMSE
médio na validação.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler

from ..dados import Divisao, Serie
from ..metricas import rmse_medido
from ..preprocessamento import PreenchimentoSimples
from .base import Previsor


def criar_janelas(valores: np.ndarray, defasagens: int) -> pd.DataFrame:
    """Colunas lag_12 … lag_1 e `alvo`. A primeira janela começa na 13ª amostra."""
    serie = pd.Series(valores)
    colunas = {f"lag_{i}": serie.shift(i) for i in range(defasagens, 0, -1)}
    return pd.concat([pd.DataFrame(colunas), serie.rename("alvo")], axis=1).dropna()


@dataclass
class Ajuste:
    """Previsões de todas as janelas (treino, validação e teste) de um ajuste."""
    previsto: np.ndarray
    n_treino: int           # janelas usadas no `fit`

    def validacao(self, divisao: Divisao) -> slice:
        return slice(self.n_treino, len(self.previsto) - divisao.teste)


def ajustar_e_prever(y: np.ndarray, estimador: RegressorMixin, defasagens: int, divisao: Divisao) -> Ajuste:
    escala = MinMaxScaler().fit(y[:len(y) - divisao.teste].reshape(-1, 1))
    janelas = criar_janelas(escala.transform(y.reshape(-1, 1)).ravel(), defasagens)
    X, alvo = janelas.drop(columns="alvo").to_numpy(), janelas["alvo"].to_numpy()
    n_treino = len(janelas) - divisao.teste - divisao.validacao
    estimador.fit(X[:n_treino], alvo[:n_treino])
    previsto = escala.inverse_transform(estimador.predict(X).reshape(-1, 1)).ravel()
    return Ajuste(previsto, n_treino)


class PrevisorJanelado(Previsor):
    """Estimador do scikit-learn (PEDREGOSA et al., 2011) com busca em grade.

    A busca roda uma vez por série e é guardada. `execucoes_na_busca` é quantas vezes
    cada configuração é ajustada para tirar a média do RMSE de validação.
    """

    def __init__(self, nome: str, estimador: RegressorMixin, grade: dict, execucoes_na_busca: int = 1,
                 deterministico: bool = False):
        self.nome = nome
        self.estimador = estimador
        self.grade = grade
        self.execucoes_na_busca = execucoes_na_busca
        self.deterministico = deterministico
        self._escolhas: dict[str, tuple[RegressorMixin, int]] = {}

    def prever(self, serie: Serie, semente: int) -> np.ndarray:
        if serie.nome not in self._escolhas:
            self._escolhas[serie.nome] = self._buscar(serie)
        estimador, defasagens = self._escolhas[serie.nome]
        np.random.seed(semente)                       # na MLP, muda só os pesos iniciais
        ajuste = ajustar_e_prever(self._entrada(serie), estimador, defasagens, serie.divisao)
        return ajuste.previsto[-serie.divisao.teste:]

    def descrever(self, serie: Serie) -> str:
        if serie.nome not in self._escolhas:
            return ""
        estimador, _ = self._escolhas[serie.nome]
        return str({k: v for k, v in estimador.get_params().items() if k in self.grade})

    @staticmethod
    def _entrada(serie: Serie) -> np.ndarray:
        return PreenchimentoSimples().aplicar(serie).to_numpy(float)

    def _buscar(self, serie: Serie) -> tuple[RegressorMixin, int]:
        """Configuração com o menor RMSE médio na validação, só nos instantes medidos."""
        y, divisao = self._entrada(serie), serie.divisao
        np.random.seed(0)
        melhor, melhor_rmse = None, np.inf
        for parametros in ParameterGrid(self.grade):
            parametros = dict(parametros)
            defasagens = parametros.pop("time_window")
            candidato = clone(self.estimador).set_params(**parametros)
            rmses = []
            for _ in range(self.execucoes_na_busca):
                ajuste = ajustar_e_prever(y, candidato, defasagens, divisao)
                validacao = ajuste.validacao(divisao)
                real = y[-len(ajuste.previsto):]
                medido = serie.medido[-len(ajuste.previsto):]
                rmses.append(rmse_medido(real[validacao], ajuste.previsto[validacao], medido[validacao]))
            if np.mean(rmses) < melhor_rmse:
                melhor, melhor_rmse = (candidato, defasagens), np.mean(rmses)
        return melhor
