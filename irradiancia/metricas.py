"""Métricas de erro de Santos Júnior et al. (2022, Seção 5.2), na escala usada pelo artigo."""
from __future__ import annotations

import numpy as np

from .dados import Serie

METRICAS = ("RMSE", "MAPE", "MAE", "ARV", "IA")


def calcular_metricas(real, previsto) -> dict[str, float]:
    """RMSE, MAPE, MAE, ARV e IA. O MAPE ignora os instantes em que o valor real é zero."""
    real, previsto = np.asarray(real, float).ravel(), np.asarray(previsto, float).ravel()
    erro = real - previsto
    media = real.mean()
    nao_zero = real != 0
    return {
        "RMSE": float(np.sqrt(np.mean(erro ** 2))),
        "MAPE": float(np.mean(np.abs(erro[nao_zero] / real[nao_zero])) * 100),
        "MAE": float(np.mean(np.abs(erro))),
        "ARV": float(np.sum(erro ** 2) / np.sum((previsto - media) ** 2)),
        "IA": float(1 - np.sum(erro ** 2) / np.sum((np.abs(previsto - media) + np.abs(real - media)) ** 2)),
    }


class EscalaDoArtigo:
    """Leva os valores para [0,1; 0,9] com o mínimo e o máximo medidos na série inteira
    (SANTOS JÚNIOR et al., 2022, Seção 5). Só muda a unidade do erro, não o modelo."""

    def __init__(self, serie: Serie):
        self.minimo = float(np.nanmin(serie.valores))
        self.maximo = float(np.nanmax(serie.valores))

    def __call__(self, valores) -> np.ndarray:
        return 0.1 + 0.8 * (np.asarray(valores, float) - self.minimo) / (self.maximo - self.minimo)


def avaliar(serie: Serie, previsto: np.ndarray) -> dict[str, float]:
    """Métricas no teste, só nos instantes em que houve medição."""
    real = serie.real_teste()
    medido = ~np.isnan(real)
    escala = EscalaDoArtigo(serie)
    return calcular_metricas(escala(real[medido]), escala(np.asarray(previsto)[medido]))


def rmse_medido(real: np.ndarray, previsto: np.ndarray, medido: np.ndarray) -> float:
    """RMSE na escala original, só nos instantes medidos. Usado na validação."""
    erro = np.asarray(real, float)[medido] - np.asarray(previsto, float)[medido]
    return float(np.sqrt(np.mean(erro ** 2)))
