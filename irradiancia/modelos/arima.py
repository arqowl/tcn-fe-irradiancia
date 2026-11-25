"""ARIMA com escolha automática da ordem."""
from __future__ import annotations

import numpy as np
from statsforecast.models import AutoARIMA

from ..dados import Serie
from ..preprocessamento import PreenchimentoSimples
from .base import Previsor


class PrevisorARIMA(Previsor):
    """AutoARIMA *stepwise* de Hyndman e Khandakar (2008), ajustado no treino.

    No teste, os parâmetros ficam fixos e cada previsão usa as observações até t − 1.
    É o mesmo procedimento de Santos Júnior et al. (2022), que usam o `auto.arima` do R.
    """
    nome = "ARIMA"
    deterministico = True

    def __init__(self) -> None:
        self._ordens: dict[str, tuple[int, int, int]] = {}

    def prever(self, serie: Serie, semente: int) -> np.ndarray:
        y = PreenchimentoSimples().aplicar(serie).to_numpy(float)
        modelo = AutoARIMA(season_length=1).fit(y[:serie.divisao.treino])
        previsto = modelo.forward(y=y, h=1, fitted=True)["fitted"][-serie.divisao.teste:]
        p, q, _, _, _, d, _ = modelo.model_["arma"]
        self._ordens[serie.nome] = (p, d, q)
        return previsto

    def descrever(self, serie: Serie) -> str:
        return f"ordem (p, d, q) = {self._ordens.get(serie.nome)}"
