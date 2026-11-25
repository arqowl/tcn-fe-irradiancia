"""Redes TCN e TCN-Attention (HONNURVALI et al., 2026, Seção II e Algoritmo 1).

A TCN segue Bai, Kolter e Koltun (2018): convoluções causais dilatadas com conexão residual.
A atenção é a aditiva de Bahdanau, Cho e Bengio (2015).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BlocoResidualCausal(nn.Module):
    """h⁽ˡ⁾ = ReLU(W⁽ˡ⁾ ∗_δ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾) + h⁽ˡ⁻¹⁾  (Eq. 2 e 4).

    O preenchimento com (k − 1)·δ zeros fica só à esquerda, então a saída em t não vê o futuro.
    """

    def __init__(self, canais: int, k: int, dilatacao: int):
        super().__init__()
        self.preenchimento = (k - 1) * dilatacao
        self.conv = nn.Conv1d(canais, canais, kernel_size=k, dilation=dilatacao)

    def forward(self, h: torch.Tensor) -> torch.Tensor:           # h: (lote, canais, tempo)
        z = self.conv(nn.functional.pad(h, (self.preenchimento, 0)))
        return torch.relu(z) + h


class CodificadorTCN(nn.Module):
    """X (lote, tempo, variáveis) → H_TCN (lote, tempo, filtros)  (Eq. 5).

    Campo receptivo R = 1 + (k − 1)·Σδ (Eq. 3): 63 passos com k = 3 e δ = 1, 2, 4, 8, 16.
    """

    def __init__(self, n_variaveis: int, filtros: int = 32, k: int = 3, dilatacoes=(1, 2, 4, 8, 16)):
        super().__init__()
        self.entrada = nn.Conv1d(n_variaveis, filtros, 1)
        self.camadas = nn.Sequential(*[BlocoResidualCausal(filtros, k, d) for d in dilatacoes])
        self.campo_receptivo = 1 + (k - 1) * sum(dilatacoes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.entrada(x.transpose(1, 2))                          # Conv1d espera (lote, canais, tempo)
        return self.camadas(h).transpose(1, 2)


class AtencaoAditiva(nn.Module):
    """e_t = vᵀ tanh(W_a h_t + b_a),  α = softmax_t(e),  c = Σ_t α_t h_t  (Eq. 6 a 8)."""

    def __init__(self, filtros: int):
        super().__init__()
        self.W_a = nn.Linear(filtros, filtros)
        self.v = nn.Linear(filtros, 1, bias=False)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.v(torch.tanh(self.W_a(H)))                          # (lote, tempo, 1)
        alfa = torch.softmax(e, dim=1)
        return (alfa * H).sum(dim=1), alfa


class RedeTCN(nn.Module):
    """TCN sem atenção: ŷ = σ(wᵀ h_L + b), com h_L o estado do último passo."""
    NOME = "TCN"

    def __init__(self, n_variaveis: int, filtros: int = 32):
        super().__init__()
        self.tcn = CodificadorTCN(n_variaveis, filtros)
        self.saida = nn.Linear(filtros, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.saida(self.tcn(x)[:, -1])).squeeze(-1)


class RedeTCNAtencao(nn.Module):
    """TCN-Attention: ŷ = σ(wᵀ c + b), com c o vetor de contexto da atenção (Alg. 1, linha 10)."""
    NOME = "TCN-Attention"

    def __init__(self, n_variaveis: int, filtros: int = 32):
        super().__init__()
        self.tcn = CodificadorTCN(n_variaveis, filtros)
        self.atencao = AtencaoAditiva(filtros)
        self.saida = nn.Linear(filtros, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contexto, _ = self.atencao(self.tcn(x))
        return torch.sigmoid(self.saida(contexto)).squeeze(-1)
