"""Montagem das janelas de entrada das TCNs nas condições SEM e COM pré-processamento/FE."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from .atributos import EngenhariaDeAtributos
from .dados import Serie
from .preprocessamento import CorteIQR, ImputacaoCausal, PreenchimentoSimples

CONDICOES = ("SEM", "COM")


@dataclass
class Particao:
    """Janelas de uma parte da série. `medido` marca os alvos que foram medidos de fato."""
    X: torch.Tensor          # (janelas, passos, variáveis)
    y: torch.Tensor          # (janelas,)
    medido: torch.Tensor     # (janelas,) booleano

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class DadosTCN:
    treino: Particao
    validacao: Particao
    teste: Particao
    n_variaveis: int
    escala_alvo: MinMaxScaler

    def para_escala_original(self, previsto: np.ndarray) -> np.ndarray:
        return self.escala_alvo.inverse_transform(np.asarray(previsto).reshape(-1, 1)).ravel()


class PreparadorTCN:
    """Monta e guarda em memória os dados de cada (série, condição).

    SEM: o alvo com `PreenchimentoSimples` e Min-Max. Uma variável por passo de tempo.
    COM: imputação causal do alvo e das variáveis meteorológicas, corte por IQR,
    engenharia de atributos e Min-Max (HONNURVALI et al., 2026, Seções III e IV).
    Em ambas, a escala e os quartis são ajustados só no treino.
    """

    def __init__(self) -> None:
        self._guardados: dict[tuple[str, str], DadosTCN] = {}

    def preparar(self, serie: Serie, condicao: str) -> DadosTCN:
        chave = (serie.nome, condicao)
        if chave not in self._guardados:
            self._guardados[chave] = self._montar(serie, condicao)
        return self._guardados[chave]

    def _montar(self, serie: Serie, condicao: str) -> DadosTCN:
        if condicao == "SEM":
            alvo = PreenchimentoSimples().aplicar(serie)
            tabela = alvo.to_frame("alvo")
        elif condicao == "COM":
            alvo, tabela = self._tabela_com_atributos(serie)
        else:
            raise ValueError(f"condição desconhecida: {condicao}")
        return self._janelas(serie, alvo, tabela)

    @staticmethod
    def _tabela_com_atributos(serie: Serie) -> tuple[pd.Series, pd.DataFrame]:
        imputacao, n_treino = ImputacaoCausal(), serie.divisao.treino
        alvo = imputacao.aplicar(serie)
        meteorologia = serie.exogenas.apply(imputacao.imputar)
        tabela = CorteIQR(n_treino).aplicar(pd.concat([alvo.rename("alvo"), meteorologia], axis=1))
        tabela = EngenhariaDeAtributos(serie.amostras_por_dia, n_treino).gerar(tabela)
        return alvo, tabela

    @staticmethod
    def _janelas(serie: Serie, alvo: pd.Series, tabela: pd.DataFrame) -> DadosTCN:
        d, passos = serie.divisao, serie.janela_tcn
        tabela = tabela.copy()
        continuas = [c for c in tabela.columns if tabela[c].nunique() > 2]
        tabela[continuas] = MinMaxScaler().fit(tabela[continuas].iloc[:d.treino]).transform(tabela[continuas])
        escala_alvo = MinMaxScaler().fit(alvo.iloc[:d.treino].to_frame())

        matriz = tabela.to_numpy(np.float32)
        y = escala_alvo.transform(alvo.to_frame()).ravel().astype(np.float32)
        medido = serie.medido
        primeira_linha_completa = int(np.argmax(~np.isnan(matriz).any(axis=1)))
        primeiro_alvo = primeira_linha_completa + passos          # precisa de `passos` linhas completas antes

        def particao(inicio: int, fim: int) -> Particao:
            alvos = np.arange(max(inicio, primeiro_alvo), fim)
            janelas = np.stack([matriz[t - passos:t] for t in alvos])
            return Particao(torch.from_numpy(janelas), torch.from_numpy(y[alvos]), torch.from_numpy(medido[alvos]))

        n = len(alvo)
        return DadosTCN(treino=particao(0, d.treino),
                        validacao=particao(d.treino, d.treino + d.validacao),
                        teste=particao(n - d.teste, n),
                        n_variaveis=matriz.shape[1], escala_alvo=escala_alvo)
