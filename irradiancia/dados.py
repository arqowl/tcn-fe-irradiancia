"""Leitura das séries do INMET e do Kaggle e divisão cronológica em treino, validação e teste."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .geometria_solar import irradiacao_horaria_topo

COLUNA_RADIACAO = "RADIACAO GLOBAL (Kj/m)"
COLUNAS_METEOROLOGICAS = {
    "temp": "TEMPERATURA DO AR - BULBO SECO",
    "umid": "UMIDADE RELATIVA DO AR",
    "pres": "PRESSAO ATMOSFERICA AO NIVEL",
    "vento": "VENTO, VELOCIDADE",
    "chuva": "PRECIPITACAO TOTAL",
}
URL_KAGGLE = ("https://www.kaggle.com/api/v1/datasets/download/"
              "chaitanyakumar12/time-series-forecasting-of-solar-energy")


@dataclass(frozen=True)
class Estacao:
    """Estação automática do INMET. As coordenadas vêm do cabeçalho dos arquivos de 2020."""
    nome: str
    prefixo_arquivo: str
    latitude: float
    longitude: float


ESTACOES_INMET = (
    Estacao("Sao Paulo", "INMET_SE_SP_A701_SAO PAULO - MIRANTE", -23.496, -46.620),
    Estacao("Florianopolis", "INMET_S_SC_A806_FLORIANOPOLIS", -27.603, -48.620),
    Estacao("Salvador", "INMET_NE_BA_A401_SALVADOR", -13.006, -38.506),
    Estacao("Fortaleza", "INMET_NE_CE_A305_FORTALEZA", -3.816, -38.538),
)


@dataclass(frozen=True)
class Divisao:
    """Tamanhos de treino, validação e teste, em ordem cronológica (SANTOS JÚNIOR et al., 2022, Seção 5)."""
    treino: int
    validacao: int
    teste: int

    @classmethod
    def cronologica(cls, n: int, fracao_validacao: float = 0.2, fracao_teste: float = 0.2) -> Divisao:
        teste, validacao = int(fracao_teste * n), int(fracao_validacao * n)
        return cls(n - validacao - teste, validacao, teste)


@dataclass
class Serie:
    """Uma série do experimento.

    `valores` guarda NaN onde não houve medição. No INMET, as horas com o Sol abaixo do
    horizonte já chegam com zero, porque nelas a radiação é zero de fato.
    """
    nome: str
    valores: pd.Series
    exogenas: pd.DataFrame
    amostras_por_dia: int
    janela_tcn: int
    unidade: str
    irradiacao_topo: pd.Series | None = None
    divisao: Divisao = field(init=False)

    def __post_init__(self) -> None:
        self.divisao = Divisao.cronologica(len(self.valores))

    @property
    def medido(self) -> np.ndarray:
        return self.valores.notna().to_numpy()

    @property
    def indice_teste(self) -> pd.DatetimeIndex:
        return self.valores.index[-self.divisao.teste:]

    def real_teste(self) -> np.ndarray:
        """Valores do teste, com NaN onde não houve medição."""
        return self.valores.to_numpy(float)[-self.divisao.teste:]

    def resumo(self) -> str:
        idx, d = self.valores.index, self.divisao
        fim_treino, inicio_teste = idx[d.treino - 1], idx[len(idx) - d.teste]
        meses = idx[:d.treino].month.nunique()
        falhas_teste = int((~self.medido[-d.teste:]).sum())
        return (f"{self.nome:14s} n = {len(idx):6d} | treino {idx[0]:%d/%m/%Y} a {fim_treino:%d/%m/%Y} "
                f"({d.treino}, {meses}/12 meses) | validação {d.validacao} | teste {inicio_teste:%d/%m/%Y} a "
                f"{idx[-1]:%d/%m/%Y} ({d.teste}, {falhas_teste} sem medição) | variáveis meteorológicas: "
                f"{self.exogenas.shape[1]}")


class CarregadorINMET:
    """Lê os arquivos anuais de uma estação e mantém as horas de 09 a 21 UTC (06 às 18 h locais).

    É o mesmo recorte de Santos Júnior et al. (2022, Seção 5.1): 13 amostras por dia.
    O valor de cada hora é a radiação acumulada na hora anterior.
    """
    PRIMEIRA_HORA, ULTIMA_HORA = 9, 21

    def __init__(self, pasta: Path):
        self.pasta = Path(pasta)

    def arquivo(self, estacao: Estacao, ano: int) -> Path:
        caminho = self.pasta / f"{estacao.prefixo_arquivo}_01-01-{ano}_A_31-12-{ano}.csv"
        if not caminho.exists():
            raise FileNotFoundError(f"{caminho} não existe. Gere-o com: python ferramentas/baixar_inmet.py {ano}")
        return caminho

    def ler(self, estacao: Estacao, anos) -> pd.DataFrame:
        """Tabela horária com a radiação como está no arquivo, as variáveis meteorológicas e a
        irradiação no topo da atmosfera (`topo`)."""
        tabela = pd.concat([self._ler_ano(self.arquivo(estacao, ano)) for ano in anos])
        assert tabela.index.is_monotonic_increasing and not tabela.index.duplicated().any()
        tabela["topo"] = irradiacao_horaria_topo(tabela.index, estacao.latitude, estacao.longitude)
        return tabela

    def carregar(self, estacao: Estacao, anos) -> Serie:
        tabela = self.ler(estacao, anos)
        radiacao = tabela["radiacao"].copy()
        noite = (tabela["topo"] == 0) & radiacao.isna()
        radiacao[noite] = 0.0
        return Serie(nome=estacao.nome, valores=radiacao, exogenas=tabela[list(COLUNAS_METEOROLOGICAS)],
                     amostras_por_dia=13, janela_tcn=12, unidade="kJ/m²", irradiacao_topo=tabela["topo"])

    def _ler_ano(self, caminho: Path) -> pd.DataFrame:
        bruto = pd.read_csv(caminho, sep=";", decimal=",", encoding="utf8")
        bruto.index = pd.to_datetime(bruto["Data"] + " " + bruto["Hora UTC"], format="%d/%m/%Y %H%M UTC")
        horas = bruto.index.hour
        bruto = bruto[(horas >= self.PRIMEIRA_HORA) & (horas <= self.ULTIMA_HORA)]
        colunas = {"radiacao": COLUNA_RADIACAO}
        for curto, prefixo in COLUNAS_METEOROLOGICAS.items():
            colunas[curto] = next(c for c in bruto.columns if c.startswith(prefixo))
        tabela = pd.DataFrame({curto: pd.to_numeric(bruto[original], errors="coerce")
                               for curto, original in colunas.items()}, index=bruto.index)
        return tabela.where(tabela > -9000)       # -9999 era o código de faltante até 2018


class CarregadorKaggle:
    """Série `solar_mw` do dataset do Kaggle usado por Honnurvali et al. (2026)."""

    def __init__(self, pasta: Path):
        self.arquivo = Path(pasta) / "solarenergy.csv"

    def baixar_se_preciso(self) -> None:
        if self.arquivo.exists():
            return
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL_KAGGLE).read())) as pacote:
            self.arquivo.write_bytes(pacote.read("solarenergy.csv"))

    def carregar(self, inicio: str) -> Serie:
        self.baixar_se_preciso()
        tabela = pd.read_csv(self.arquivo, low_memory=False)
        tabela["Datetime"] = pd.to_datetime(tabela["Datetime"], format="%d/%m/%Y %H:%M")
        tabela = tabela.sort_values("Datetime").drop_duplicates("Datetime").set_index("Datetime").asfreq("h")
        valores = pd.to_numeric(tabela.loc[inicio:, "solar_mw"], errors="coerce").rename("solar_mw")
        # A meteorologia só existe nas últimas 2920 horas da série e não cobre o período usado.
        return Serie(nome="Kaggle", valores=valores, exogenas=pd.DataFrame(index=valores.index),
                     amostras_por_dia=24, janela_tcn=48, unidade="MW")
