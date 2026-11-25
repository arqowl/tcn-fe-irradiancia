"""Teste de Friedman (1937) com pós-teste de Nemenyi (1963), como recomenda Demšar (2006)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

# Valores críticos q_α de Nemenyi para α = 0,05 (DEMŠAR, 2006, Tabela 5a), por número de métodos.
Q_NEMENYI_5 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


@dataclass
class ResultadoFriedmanNemenyi:
    postos: pd.Series                 # posto médio de cada método (1 = melhor)
    qui_quadrado: float
    p: float
    distancia_critica: float
    n_blocos: int
    grupos: list[tuple[int, int]]     # intervalos de métodos sem diferença significativa, na ordem dos postos

    @property
    def ordem(self) -> list[str]:
        return self.postos.sort_values().index.tolist()


def friedman_nemenyi(erros: pd.DataFrame) -> ResultadoFriedmanNemenyi:
    """`erros`: uma linha por dataset (bloco) e uma coluna por método; menor é melhor."""
    n_blocos, k = erros.shape
    postos = pd.Series(np.apply_along_axis(rankdata, 1, erros.to_numpy()).mean(axis=0), index=erros.columns)
    qui_quadrado, p = friedmanchisquare(*[erros[c] for c in erros.columns])
    cd = Q_NEMENYI_5[k] * np.sqrt(k * (k + 1) / (6 * n_blocos))
    return ResultadoFriedmanNemenyi(postos, float(qui_quadrado), float(p), float(cd), n_blocos,
                                    _grupos_sem_diferenca(postos, cd))


def _grupos_sem_diferenca(postos: pd.Series, cd: float) -> list[tuple[int, int]]:
    """Maiores sequências de métodos, na ordem dos postos, cuja distância é menor que a CD."""
    ordem = postos.sort_values()
    valores, k = ordem.to_numpy(), len(ordem)
    grupos: list[tuple[int, int]] = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if valores[jj] - valores[i] < cd)
        if j > i and not any(gi <= i and j <= gj for gi, gj in grupos):
            grupos.append((i, j))
    return grupos
