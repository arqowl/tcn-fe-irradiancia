"""TCN e TCN-Attention como estratégias do experimento."""
from __future__ import annotations

import numpy as np
import torch

from ..dados import Serie
from ..preparacao import PreparadorTCN
from .base import Previsor
from .treino import Treinador


class PrevisorTCN(Previsor):
    """Junta uma rede (`RedeTCN` ou `RedeTCNAtencao`), uma condição e o treinador.

    O mesmo `PreparadorTCN` pode ser dividido entre vários previsores: os dados de cada
    (série, condição) são montados uma vez só.
    """

    def __init__(self, classe_rede, condicao: str, preparador: PreparadorTCN, treinador: Treinador):
        self.classe_rede = classe_rede
        self.nome = classe_rede.NOME
        self.condicao = condicao
        self.preparador = preparador
        self.treinador = treinador

    @property
    def rotulo(self) -> str:
        return f"{self.nome} ({self.condicao} FE)"

    def prever(self, serie: Serie, semente: int) -> np.ndarray:
        dados = self.preparador.preparar(serie, self.condicao)
        torch.manual_seed(semente)                                    # pesos iniciais
        rede = self.classe_rede(dados.n_variaveis)
        return dados.para_escala_original(self.treinador.treinar(rede, dados, semente))
