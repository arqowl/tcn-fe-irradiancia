"""Figuras do experimento."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dados import Serie
from .estatistica import ResultadoFriedmanNemenyi
from .registro import RegistroDeResultados

COR_EIXO, COR_MEDIANA = "#52514e", "#eb6834"


def _salvar(fig, caminho: Path | None) -> None:
    fig.tight_layout()
    if caminho is not None:
        fig.savefig(caminho, dpi=150)


def faltantes_por_hora(por_estacao: dict[str, pd.DataFrame], caminho: Path | None = None):
    """Barras empilhadas: horas sem dado em cada hora UTC, separadas pela posição do Sol."""
    fig, eixos = plt.subplots(1, len(por_estacao), figsize=(4 * len(por_estacao), 3.2), sharey=True)
    for ax, (nome, tabela) in zip(np.atleast_1d(eixos), por_estacao.items()):
        tabela.plot.bar(stacked=True, ax=ax, color=["#9aa5b1", COR_MEDIANA], width=0.8, legend=False)
        ax.set_title(nome, loc="left")
        ax.set_xlabel("hora UTC")
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(eixos)[0].set_ylabel("horas sem dado (%)")
    np.atleast_1d(eixos)[0].legend(fontsize=8)
    _salvar(fig, caminho)
    return fig


def efeito_do_tratamento(serie: Serie, sem: pd.Series, com: pd.Series, inicio: str, fim: str,
                         sem_limite: pd.Series | None = None, caminho: Path | None = None):
    """Um trecho da série: medido, preenchido com zero (SEM) e imputado (COM).

    `sem_limite`, se dado, é a imputação sem o teto do maior valor já medido.
    """
    trecho = slice(inicio, fim)
    fig, ax = plt.subplots(figsize=(12, 3.4))
    if serie.irradiacao_topo is not None:
        ax.plot(serie.irradiacao_topo[trecho], color="#9aa5b1", lw=1, label="topo da atmosfera")
    ax.plot(sem[trecho], color=COR_MEDIANA, ls=":", lw=1.4, label="SEM: falha vira zero")
    if sem_limite is not None:
        ax.plot(sem_limite[trecho], color="#7fb3e0", ls="-.", lw=1.2, label="imputação sem limite")
    ax.plot(com[trecho], color="#2a6fb0", ls="--", lw=1.4, label="COM: imputação causal")
    ax.plot(serie.valores[trecho], color="k", lw=1.8, label="medido")
    ax.set_ylabel(serie.unidade)
    ax.set_title(f"{serie.nome}, {inicio} a {fim}", loc="left")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    _salvar(fig, caminho)
    return fig


def boxplot_rmse(tabela: pd.DataFrame, datasets: list[str], ordem: list[str], caminho: Path | None = None):
    fig, eixos = plt.subplots(len(datasets), 1, figsize=(12, 3.2 * len(datasets)))
    for ax, nome in zip(np.atleast_1d(eixos), datasets):
        t = tabela[tabela.dataset == nome]
        ax.boxplot([t[t.modelo_cond == m].RMSE.to_numpy() for m in ordem],
                   tick_labels=[m.replace(" (", "\n(") for m in ordem], widths=0.5,
                   medianprops=dict(color=COR_MEDIANA, lw=1.5))
        ax.set_title(nome, loc="left")
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.3)
    _salvar(fig, caminho)
    return fig


def diagrama_distancia_critica(resultado: ResultadoFriedmanNemenyi, caminho: Path | None = None):
    """Diagrama de Demšar (2006): postos médios e barras ligando métodos sem diferença significativa."""
    postos, cd, ordem = resultado.postos, resultado.distancia_critica, resultado.ordem
    k, y0 = len(ordem), 0.75
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(-1.8, k + 2.8)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([1, k], [y0, y0], color=COR_EIXO)
    for t in range(1, k + 1):
        ax.plot([t, t], [y0, y0 + 0.03], color=COR_EIXO)
        ax.text(t, y0 + 0.05, str(t), ha="center")
    ax.plot([1, 1 + cd], [0.95, 0.95], color="k", lw=1.6)
    ax.text(1 + cd + 0.1, 0.95, f"CD = {cd:.2f}", va="center")
    for i, metodo in enumerate(ordem):
        posto, esquerda = postos[metodo], i < (k + 1) // 2
        y_rotulo = 0.52 - 0.11 * (i if esquerda else k - 1 - i)
        x_rotulo = 0.6 if esquerda else k + 0.4
        ax.plot([posto, posto, x_rotulo], [y0, y_rotulo, y_rotulo], color=COR_EIXO, lw=0.8)
        ax.scatter([posto], [y0], s=50, color="k", zorder=5)
        ax.text(x_rotulo + (-0.08 if esquerda else 0.08), y_rotulo, f"{metodo} ({posto:.2f})",
                ha="right" if esquerda else "left", va="center", fontsize=9.5)
    for g, (i, j) in enumerate(resultado.grupos):
        ax.plot([postos[ordem[i]] - 0.05, postos[ordem[j]] + 0.05], [y0 - 0.06 - 0.035 * g] * 2, color="k", lw=4)
    ax.set_title(f"Distância crítica: Friedman p = {resultado.p:.3g}, Nemenyi α = 0,05 "
                 f"(RMSE médio, {resultado.n_blocos} datasets)", loc="left")
    _salvar(fig, caminho)
    return fig


def real_vs_previsto(series: dict[str, Serie], registro: RegistroDeResultados, modelo: str,
                     pontos: int = 150, caminho: Path | None = None):
    """Primeiros `pontos` instantes do teste. Para cada condição, a execução de RMSE mediano."""
    tabela = registro.tabela
    fig, eixos = plt.subplots(len(series), 1, figsize=(13, 3 * len(series)))
    for ax, (nome, serie) in zip(np.atleast_1d(eixos), series.items()):
        indice = serie.indice_teste[:pontos]
        ax.plot(indice, serie.real_teste()[:pontos], color="k", lw=1.8, label="real")
        for condicao, estilo in [("SEM", ":"), ("COM", "--")]:
            t = tabela[(tabela.dataset == nome) & (tabela.modelo == modelo) & (tabela.condicao == condicao)]
            if t.empty:
                continue
            execucao = int(t.sort_values("RMSE").execucao.iloc[len(t) // 2])
            previsto = registro.previsao(nome, modelo, condicao, execucao)
            ax.plot(indice, previsto[:pontos], ls=estilo, lw=1.4, label=f"{modelo} {condicao} FE (exec. {execucao})")
        ax.set_title(nome, loc="left")
        ax.set_ylabel(serie.unidade)
        ax.legend(loc="upper right", fontsize=8)
    _salvar(fig, caminho)
    return fig
