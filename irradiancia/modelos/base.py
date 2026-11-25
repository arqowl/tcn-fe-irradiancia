"""Interface comum dos modelos (padrão Strategy).

O experimento só conhece `Previsor.prever`. Trocar de modelo é trocar o objeto, sem
mudar o laço que treina, avalia e grava os resultados.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..dados import Serie


class Previsor(ABC):
    nome: str = ""
    condicao: str = "SEM"          # SEM ou COM pré-processamento/FE
    deterministico: bool = False   # se verdadeiro, as execuções repetidas dão o mesmo resultado

    @property
    def rotulo(self) -> str:
        """Nome usado em tabelas e figuras."""
        return self.nome

    @abstractmethod
    def prever(self, serie: Serie, semente: int) -> np.ndarray:
        """Treina com treino e validação e devolve a previsão 1 passo à frente de cada
        instante do teste, na escala original da série."""

    def descrever(self, serie: Serie) -> str:
        """Texto curto sobre o ajuste (ordem escolhida, hiperparâmetros etc.)."""
        return ""
