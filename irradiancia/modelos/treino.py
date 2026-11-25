"""Treino das redes com parada antecipada."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ..preparacao import DadosTCN


class Treinador:
    """Adam (KINGMA; BA, 2015) minimizando o MSE, como no Algoritmo 1 de Honnurvali et al. (2026).

    Ao fim de cada época, mede o MSE na validação, só nos alvos medidos, e guarda os pesos
    da melhor época. Para depois de `paciencia` épocas sem melhora.
    """

    def __init__(self, max_epocas: int = 200, taxa: float = 1e-3, lote: int = 64, paciencia: int = 10,
                 dispositivo: str = "cpu"):
        self.max_epocas = max_epocas
        self.taxa = taxa
        self.lote = lote
        self.paciencia = paciencia
        self.dispositivo = dispositivo

    def treinar(self, rede: nn.Module, dados: DadosTCN, semente: int) -> np.ndarray:
        """Devolve a previsão do teste na escala [0, 1]."""
        torch.manual_seed(semente)
        X, y = dados.treino.X.to(self.dispositivo), dados.treino.y.to(self.dispositivo)
        medido = dados.validacao.medido
        Xv = dados.validacao.X[medido].to(self.dispositivo)
        yv = dados.validacao.y[medido].to(self.dispositivo)
        rede = rede.to(self.dispositivo)
        perda, otimizador = nn.MSELoss(), torch.optim.Adam(rede.parameters(), lr=self.taxa)

        melhor_perda, melhores_pesos, epocas_sem_melhora = float("inf"), None, 0
        for _ in range(self.max_epocas):
            rede.train()
            for indices in torch.randperm(len(X)).split(self.lote):
                otimizador.zero_grad()
                perda(rede(X[indices]), y[indices]).backward()
                otimizador.step()
            rede.eval()
            with torch.no_grad():
                perda_validacao = perda(rede(Xv), yv).item()
            if perda_validacao < melhor_perda:
                melhor_perda, melhores_pesos, epocas_sem_melhora = perda_validacao, copy.deepcopy(rede.state_dict()), 0
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora == self.paciencia:
                    break

        rede.load_state_dict(melhores_pesos)
        rede.eval()
        with torch.no_grad():
            return rede(dados.teste.X.to(self.dispositivo)).cpu().numpy()
