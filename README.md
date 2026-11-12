# TCN-FE Irradiância

Este repositório avalia o efeito do **pré-processamento e do *feature engineering* (FE)** na previsão de irradiância e de geração solar com **TCN** e **TCN-Attention**. A comparação é com três *baselines*: **ARIMA**, **SVR** e **MLP**.

O experimento combina dois trabalhos:

- **[TCN]** Honnurvali et al., *A TCN-Attention Model for High-Accuracy Solar Energy Generation Forecasting*, IEEE Open Access Journal of Power and Energy, v. 13, pp. 690–701, 2026. DOI [10.1109/OAJPE.2026.3723147](https://doi.org/10.1109/OAJPE.2026.3723147). Daqui vêm a arquitetura, o pré-processamento e o FE.
- **[MDPI]** de O. Santos Jr. et al., *Solar Irradiance Forecasting Using Dynamic Ensemble Selection*, Applied Sciences, 12(7), 3510, 2022. DOI [10.3390/app12073510](https://doi.org/10.3390/app12073510). Daqui vêm as séries do INMET, os *baselines*, as grades de hiperparâmetros, o *split*, as métricas e as figuras de comparação.

## Datasets

| Dataset | Origem | Amostras | Frequência |
|---|---|---|---|
| Kaggle | [*Time Series Forecasting of Solar Energy*](https://www.kaggle.com/datasets/chaitanyakumar12/time-series-forecasting-of-solar-energy), base do [TCN]. Só as 2920 h com meteorologia ([TCN] Sec. IV.D) | 2920 | horária, 24 h/dia |
| São Paulo, Florianópolis, Salvador, Fortaleza | INMET 2020 (estações A701, A806, A401, A305), repositório [domingos108/solar_forecasting](https://github.com/domingos108/solar_forecasting) | 4758 cada | horária, 06h–18h locais (13 h/dia) |

- **INMET:** os CSVs estão em `notebooks/dados/`.
- **Kaggle:** é baixado da API pública do Kaggle quando o notebook roda, sem precisar de credenciais.

## Modelos e protocolo

| Item | Escolha | Fonte |
|---|---|---|
| *Split* | 60/20/20 cronológico em todos os datasets | [MDPI] Sec. 5, p. 8 |
| ARIMA | `AutoARIMA` (statsforecast), método *stepwise* de Hyndman-Khandakar | [MDPI] Tabela 1 e ref. [65] |
| SVR | RBF; C ∈ {10, 100, 1000}, γ ∈ {0.1, 0.01, 0.001}, ε ∈ {0.1, 0.01, 0.001}; 12 lags | [MDPI] Tabela 1 |
| MLP | 1 camada oculta, sigmoide, L-BFGS; {20, 50, 100} neurônios; tol ∈ {1e-3, 1e-4, 1e-5}; 12 lags | [MDPI] Tabela 1 |
| TCN / TCN-Attention | Conv. causal dilatada, ReLU, resíduo, atenção aditiva, saída σ, MSE, Adam | [TCN] Eq. 2–8 e Alg. 1 |
| Janela das TCNs | 48 no Kaggle; 12 nas cidades (12 lags + o 13º como alvo, como nos *baselines*) | [TCN] Sec. IV.F; [MDPI] Sec. 5.1 |
| SEM pré-proc./FE | Série bruta, só Min-Max | — |
| COM pré-proc./FE | Faltantes, IQR, lags (1, 2, 3, 4, 1, 2 e 7 dias), médias móveis, ewm, interação com temperatura, *binning* de intensidade, calendário *one-hot* e variáveis meteorológicas | [TCN] Sec. III–IV, Tabela 1 e Fig. 2 |
| Métricas | RMSE, MAPE, MAE, ARV e IA na escala [0.1, 0.9] | [MDPI] Sec. 5.2 e Tabela 3 |
| Repetições | 10 execuções independentes por modelo, condição e dataset | [MDPI] Sec. 5.3 |
| Estatística | Friedman–Nemenyi, com diagrama de distância crítica sobre os 5 datasets | [MDPI] Fig. 8; Demšar (2006) |

- **Pré-processamento e FE (com e sem):** aplicados só às TCNs. Os *baselines* rodam como no [MDPI]: o ARIMA é univariado por definição, e SVR e MLP foram ajustados para entrada de 12 lags.
- **Código dos *baselines*:** as funções do repositório do professor (`src/time_series_functions.py` e `src/fit_predict_models.py`) estão **reimplementadas dentro do notebook**, cada uma indicando a original que reproduz. Comparadas com o código original nas mesmas entradas, deram resultados idênticos: carregamento, métricas, previsões do SVR e da MLP com a mesma semente, e a escolha da busca em grade.

## Como rodar

Tudo está no notebook [`notebooks/experimento_tcn_fe_baselines.ipynb`](notebooks/experimento_tcn_fe_baselines.ipynb).

### Opção 1: VS Code + Dev Container (recomendado)

1. Instale o Docker e, no VS Code, a extensão *Dev Containers*.
2. Abra a pasta do repositório e escolha **"Reopen in Container"**. O container é construído a partir do `Dockerfile`.
3. Abra o notebook e use **Run All**.

O `devcontainer.json` pede GPU NVIDIA (`--gpus all`). Em máquina sem GPU, remova o bloco `runArgs`; o notebook passa a usar a CPU sozinho.

### Opção 2: Docker direto

```bash
docker build -t tcn-fe-irradiancia .
docker run --gpus all -p 8888:8888 -v "$(pwd)":/workspace tcn-fe-irradiancia
```

Abra o endereço com *token* que aparece no terminal e rode `notebooks/experimento_tcn_fe_baselines.ipynb`.

### Opção 3: Google Colab

Suba o notebook, escolha **T4 GPU** em *Ambiente de execução* e use *Executar tudo*. Na primeira célula, descomente as linhas do Google Drive para não perder os resultados se a sessão cair.

### Configuração e retomada

- **Teste rápido:** na primeira célula, `MODO_RAPIDO = True` roda tudo com 2 execuções e 3 épocas, só para verificar o ambiente. O experimento real usa `False`.
- **Retomada:** cada execução é gravada em `notebooks/resultados/` assim que termina. Se a máquina cair, rode o notebook de novo e ele continua de onde parou.

## Saídas (`notebooks/resultados/`)

| Arquivo | Conteúdo |
|---|---|
| `metricas.csv` | As 5 métricas de cada execução (dataset × modelo × condição × execução) |
| `tabela_<dataset>.csv` | Média ± DP das 10 execuções, no formato da Tabela 3 do [MDPI] |
| `boxplot_rmse.png` | Dispersão do RMSE nas 10 execuções, um painel por dataset |
| `distancia_critica.png` | Diagrama de distância crítica (Friedman–Nemenyi, α = 0,05) |
| `forecast_tcn_attention.png` | Real × TCN-Attention (SEM e COM FE) nos 150 primeiros pontos do teste |
| `previsoes/*.npy` | Previsões de teste de cada execução |

## Estrutura

```text
.devcontainer/         container do VS Code (constrói a partir do Dockerfile)
.vscode/               configurações do editor
notebooks/
  experimento_tcn_fe_baselines.ipynb   experimento completo
  dados/                               CSVs do INMET 2020 (repositório do professor)
  resultados/                          saídas geradas pelo notebook
zfiles/                artigos de referência ([TCN] e [MDPI], ambos CC BY 4.0)
Dockerfile             imagem pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime + dependências
requirements.txt       dependências (o PyTorch vem na imagem)
```

## Limitações conhecidas

- **Hiperparâmetros das TCNs:** o [TCN] não informa kernel, filtros, camadas, dilatações, *learning rate*, lote nem épocas. Usamos k = 3, dilatações 1–16, 32 filtros, lr = 0,001, lote 64 e até 200 épocas com *early stopping* (paciência 10).
- **Dataset do Kaggle:** a variável `solar_mw` nunca chega a zero de noite e tem pico às 19h, comportamento mais típico de consumo do que de geração solar. A meteorologia só existe em 2920 horas, embora o [TCN] cite cerca de 8700. A temperatura está em °F e a pressão em inHg.
- ***Baselines* determinísticos:** ARIMA e SVR dão o mesmo resultado nas 10 execuções ([MDPI] Sec. 5.4).
- **Friedman–Nemenyi:** com N = 5 datasets e k = 7 concorrentes, a distância crítica fica grande (≈ 4,03 postos). Só diferenças muito consistentes aparecem como significativas.

## Licença dos dados e artigos

- **INMET:** os dados são públicos ([portal.inmet.gov.br](https://portal.inmet.gov.br/dadoshistoricos)).
- **Kaggle:** o dataset não é redistribuído aqui; ele é baixado da fonte quando o notebook roda.
- **Artigos em `zfiles/`:** os dois estão sob licença Creative Commons Attribution 4.0 e são redistribuídos com a citação completa acima.
