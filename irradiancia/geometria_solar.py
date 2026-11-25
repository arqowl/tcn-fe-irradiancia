"""Posição do Sol e irradiação no topo da atmosfera.

As fórmulas seguem Duffie e Beckman (2013, cap. 1). A declinação e a equação do tempo
são as séries de Fourier de Spencer (1971). Os instantes estão em UTC e as longitudes
são negativas a oeste de Greenwich.
"""
import numpy as np
import pandas as pd

CONSTANTE_SOLAR = 1361.0  # W/m², valor adotado por Duffie e Beckman (2013)


def _angulo_do_dia(instantes: pd.DatetimeIndex) -> np.ndarray:
    """B = 2π(n − 1)/365, em radianos, com n = dia do ano."""
    return 2 * np.pi * (instantes.dayofyear.to_numpy() - 1) / 365


def declinacao(instantes: pd.DatetimeIndex) -> np.ndarray:
    """Declinação solar em radianos (SPENCER, 1971)."""
    b = _angulo_do_dia(instantes)
    return (0.006918 - 0.399912 * np.cos(b) + 0.070257 * np.sin(b) - 0.006758 * np.cos(2 * b)
            + 0.000907 * np.sin(2 * b) - 0.002697 * np.cos(3 * b) + 0.00148 * np.sin(3 * b))


def equacao_do_tempo(instantes: pd.DatetimeIndex) -> np.ndarray:
    """Diferença entre o tempo solar aparente e o tempo solar médio, em minutos (SPENCER, 1971)."""
    b = _angulo_do_dia(instantes)
    return 229.2 * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b)
                    - 0.014615 * np.cos(2 * b) - 0.04089 * np.sin(2 * b))


def angulo_horario(instantes: pd.DatetimeIndex, longitude: float) -> np.ndarray:
    """Ângulo horário em radianos: zero ao meio-dia solar, 15° por hora, positivo à tarde."""
    minutos_utc = instantes.hour.to_numpy() * 60 + instantes.minute.to_numpy()
    tempo_solar = minutos_utc + 4 * longitude + equacao_do_tempo(instantes)
    return np.radians((tempo_solar / 60 - 12) * 15)


def irradiacao_horaria_topo(fim_da_hora: pd.DatetimeIndex, latitude: float, longitude: float) -> np.ndarray:
    """Irradiação no topo da atmosfera, sobre superfície horizontal, na hora que termina em `fim_da_hora`.

    Resultado em kJ/m² (DUFFIE; BECKMAN, 2013, Eq. 1.10.4). Vale zero quando o Sol fica abaixo
    do horizonte durante a hora inteira.
    """
    meio_da_hora = fim_da_hora - pd.Timedelta(minutes=30)
    delta = declinacao(meio_da_hora)
    phi = np.radians(latitude)
    w2 = angulo_horario(fim_da_hora, longitude)
    w1 = w2 - np.radians(15)
    por_do_sol = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    w1, w2 = np.clip(w1, -por_do_sol, por_do_sol), np.clip(w2, -por_do_sol, por_do_sol)
    distancia = 1 + 0.033 * np.cos(2 * np.pi * meio_da_hora.dayofyear.to_numpy() / 365)
    joules = (12 * 3600 / np.pi) * CONSTANTE_SOLAR * distancia * (
        np.cos(phi) * np.cos(delta) * (np.sin(w2) - np.sin(w1)) + (w2 - w1) * np.sin(phi) * np.sin(delta))
    return np.maximum(joules, 0) / 1000
