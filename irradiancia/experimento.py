"""Laço do experimento: cada previsor em cada série, várias execuções."""
from __future__ import annotations

from typing import Iterable

from .dados import Serie
from .modelos.base import Previsor
from .registro import RegistroDeResultados


class Experimento:
    """Roda os previsores e grava cada execução assim que ela termina.

    Execuções já gravadas são puladas. Se o previsor for determinístico, ele roda uma vez
    e o resultado é repetido nas demais execuções (SANTOS JÚNIOR et al., 2022, Seção 5.4).
    """

    def __init__(self, series: dict[str, Serie], registro: RegistroDeResultados, n_execucoes: int):
        self.series = series
        self.registro = registro
        self.n_execucoes = n_execucoes

    def rodar(self, previsores: Iterable[Previsor]) -> None:
        for previsor in previsores:
            for serie in self.series.values():
                self._rodar_um(previsor, serie)

    def _rodar_um(self, previsor: Previsor, serie: Serie) -> None:
        feitas = self.registro.feitas(serie.nome, previsor)
        faltam = [e for e in range(self.n_execucoes) if e not in feitas]
        if not faltam:
            return
        if previsor.deterministico:
            previsto = previsor.prever(serie, semente=0)
            for execucao in faltam:
                self.registro.registrar(serie, previsor, execucao, previsto)
        else:
            for execucao in faltam:
                self.registro.registrar(serie, previsor, execucao, previsor.prever(serie, semente=execucao))
        rmse = self.registro.rmse(serie.nome, previsor)
        print(f"{serie.nome:14s} {previsor.rotulo:24s} RMSE {rmse.mean():.4f} ± {rmse.std():.4f}  "
              f"{previsor.descrever(serie)}")
