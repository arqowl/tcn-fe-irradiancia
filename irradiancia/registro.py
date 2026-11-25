"""Gravação incremental dos resultados, para retomar o experimento se a execução cair."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dados import Serie
from .metricas import METRICAS, avaliar
from .modelos.base import Previsor

BASELINES = ("ARIMA", "SVR", "MLP")


class RegistroDeResultados:
    """`metricas.csv` tem uma linha por execução; `previsoes/` guarda as previsões do teste."""

    def __init__(self, pasta: Path):
        self.pasta = Path(pasta)
        (self.pasta / "previsoes").mkdir(parents=True, exist_ok=True)
        self.arquivo = self.pasta / "metricas.csv"
        colunas = ["dataset", "modelo", "condicao", "execucao", *METRICAS]
        self.tabela = pd.read_csv(self.arquivo) if self.arquivo.exists() else pd.DataFrame(columns=colunas)

    def feitas(self, dataset: str, previsor: Previsor) -> set[int]:
        linhas = self._linhas(dataset, previsor.nome, previsor.condicao)
        return set(linhas.execucao.astype(int))

    def registrar(self, serie: Serie, previsor: Previsor, execucao: int, previsto: np.ndarray) -> None:
        previsto = np.clip(np.asarray(previsto, float), 0, None)     # irradiância e potência não são negativas
        if len(previsto) != serie.divisao.teste:
            raise ValueError(f"{previsor.rotulo}: {len(previsto)} previsões, esperado {serie.divisao.teste}")
        np.save(self._arquivo_previsao(serie.nome, previsor.nome, previsor.condicao, execucao), previsto)
        linha = {"dataset": serie.nome, "modelo": previsor.nome, "condicao": previsor.condicao,
                 "execucao": execucao, **avaliar(serie, previsto)}
        self.tabela = pd.concat([self.tabela, pd.DataFrame([linha])], ignore_index=True)
        self.tabela.to_csv(self.arquivo, index=False)

    def previsao(self, dataset: str, modelo: str, condicao: str, execucao: int) -> np.ndarray:
        return np.load(self._arquivo_previsao(dataset, modelo, condicao, execucao))

    def rmse(self, dataset: str, previsor: Previsor) -> pd.Series:
        return self._linhas(dataset, previsor.nome, previsor.condicao).RMSE

    def com_rotulos(self) -> pd.DataFrame:
        """Tabela com a coluna `modelo_cond`: o nome dos baselines ou "TCN (COM FE)" etc."""
        tabela = self.tabela.copy()
        tabela["modelo_cond"] = np.where(tabela.modelo.isin(BASELINES), tabela.modelo,
                                         tabela.modelo + " (" + tabela.condicao + " FE)")
        return tabela

    def _linhas(self, dataset: str, modelo: str, condicao: str) -> pd.DataFrame:
        t = self.tabela
        return t[(t.dataset == dataset) & (t.modelo == modelo) & (t.condicao == condicao)]

    def _arquivo_previsao(self, dataset: str, modelo: str, condicao: str, execucao: int) -> Path:
        return self.pasta / "previsoes" / f"{dataset}__{modelo}__{condicao}__{execucao}.npy"


def tabela_media_dp(tabela: pd.DataFrame, dataset: str, ordem: list[str]) -> pd.DataFrame:
    """Média ± desvio-padrão das execuções, no formato da Tabela 3 de Santos Júnior et al. (2022)."""
    casas = {"RMSE": 4, "MAPE": 2, "MAE": 4, "ARV": 4, "IA": 4}
    grupos = tabela[tabela.dataset == dataset].groupby("modelo_cond")[list(METRICAS)]
    media, dp = grupos.mean(), grupos.std().fillna(0)
    return pd.DataFrame({m: [f"{media.loc[i, m]:.{casas[m]}f} ± {dp.loc[i, m]:.{casas[m]}f}" for i in media.index]
                         for m in METRICAS}, index=media.index).reindex(ordem)
