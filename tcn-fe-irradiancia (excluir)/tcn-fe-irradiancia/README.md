# TCN-FE Irradiância

Este repositório avalia o efeito do **pré-processamento e do *feature engineering* (FE)** na previsão de irradiância e de geração solar com **TCN** e **TCN-Attention**. A comparação é com três *baselines*: **ARIMA**, **SVR** e **MLP**.

O experimento combina dois trabalhos:

- **[TCN]** Honnurvali et al., *A TCN-Attention Model for High-Accuracy Solar Energy Generation Forecasting*, IEEE Open Access Journal of Power and Energy, v. 13, pp. 690–701, 2026. DOI [10.1109/OAJPE.2026.3723147](https://doi.org/10.1109/OAJPE.2026.3723147). Daqui vêm a arquitetura, o pré-processamento e o FE.
- **[MDPI]** de O. Santos Jr. et al., *Solar Irradiance Forecasting Using Dynamic Ensemble Selection*, Applied Sciences, 12(7), 3510, 2022. DOI [10.3390/app12073510](https://doi.org/10.3390/app12073510). Daqui vêm as séries do INMET, os *baselines*, as grades de hiperparâmetros, o *split*, as métricas e as figuras de comparação.

## Datasets

Todos os datasets usam o **mesmo período, 2016–2020 (5 anos)**, com os faltantes mantidos. Com o *split* 60/20/20, o treino cobre 2016–2018, com os 12 meses repetidos 3 vezes, e o notebook confere isso. Nas cidades, a validação é 2019 e o teste é 2020 inteiro.

| Dataset | Origem | Período | Amostras | Treino | Validação | Teste |
|---|---|---|---|---|---|---|
| Kaggle | [*Time Series Forecasting of Solar Energy*](https://www.kaggle.com/datasets/chaitanyakumar12/time-series-forecasting-of-solar-energy), base do [TCN]; série `solar_mw`, horária (24 h/dia) | 01/01/2016–03/08/2020 (a série termina aí) | 40.225 | 01/01/2016–02/10/2018 | out/2018–ago/2019 | 02/09/2019–03/08/2020 |
| São Paulo (A701) | [INMET](https://portal.inmet.gov.br/dadoshistoricos), horária, 06h–18h locais (13 h/dia) | 01/01/2016–31/12/2020 | 23.751 | 2016–2018 | 2019 | 2020 |
| Florianópolis (A806) | idem | idem | 23.751 | idem | idem | idem |
| Salvador (A401) | idem | idem | 23.751 | idem | idem | idem |
| Fortaleza (A305) | idem | idem | 23.751 | idem | idem | idem |

**Faltantes:** fração das horas de luz sem dado de radiação. Viram 0, como no carregador do professor.

| Estação | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|
| São Paulo | 8,2% | 8,4% | 6,4% | 7,7% | 7,1% |
| Florianópolis | 7,4% | 8,5% | 9,1% | 10,1% | 7,2% |
| Salvador | 7,3% (falha de 21 dias) | 1,7% | 1,7% | 1,3% | 1,4% |
| Fortaleza | 13,2% | 1,5% | 4,3% | 26,7% (falha de 73 dias) | 1,0% |

Há duas falhas longas: Salvador 2016 (21 dias, no treino) e Fortaleza 2019 (73 dias, na validação). O resto são buracos de poucas horas, quase todos ao amanhecer. O Kaggle tem 13,3 anos (05/2007–08/2020), mas só o trecho de 2016–2020 é usado, para ficar no mesmo período das cidades.

**Arquivos (`notebooks/dados/`):**
- **INMET:** 4 estações × 2016–2025 (o experimento usa 2016–2020; os outros anos ficam disponíveis para trocar o período), no formato do repositório do professor ([domingos108/solar_forecasting](https://github.com/domingos108/solar_forecasting)). O notebook **só lê esses arquivos**; não acessa o portal do INMET. Eles foram gerados por [`ferramentas/baixar_inmet.py`](ferramentas/baixar_inmet.py) a partir dos pacotes anuais do portal. Conferido: o pacote de 2020 convertido fica idêntico, valor a valor, aos arquivos de 2020 do professor. Até 2018, o INMET usava outro cabeçalho e marcava faltante como `-9999`; o script converte esses anos para o mesmo formato.
- **Kaggle:** `solarenergy.csv` não vai no repositório, porque a licença não é informada. Na primeira execução, o notebook o baixa da API pública do Kaggle (sem credenciais) e o guarda em `notebooks/dados/`. Nas seguintes, lê o arquivo local.
- **Meteorologia do Kaggle:** só existe nas últimas 2920 h da série (abr–ago/2020), então o Kaggle de 2016–2020 entra **sem variáveis exógenas** e sem a interação com temperatura.

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
| COM pré-proc./FE | Faltantes, IQR, lags (1, 2, 3, 4, 1, 2 e 7 dias), médias móveis, ewm, *binning* de intensidade e calendário *one-hot*; nas cidades, também variáveis meteorológicas e interação com temperatura | [TCN] Sec. III–IV, Tabela 1 e Fig. 2 |
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

1. Suba o notebook **e a pasta `dados/` ao lado dele**, ou clone o repositório no Colab.
2. Escolha **T4 GPU** em *Ambiente de execução*.
3. Use *Executar tudo*.

Na primeira célula, descomente as linhas do Google Drive para não perder os resultados se a sessão cair.

### Configuração e retomada

- **Períodos:** na primeira célula, `ANOS_INMET` (anos por estação) e `INICIO_KAGGLE`. Para usar um ano que não está em `notebooks/dados/`, gere o arquivo antes, da raiz do repositório: `python ferramentas/baixar_inmet.py 2026`.
- **Teste rápido:** `MODO_RAPIDO = True` roda tudo com 2 execuções e 3 épocas, só para verificar o ambiente. O experimento real usa `False`.
- **Retomada:** cada execução é gravada em `notebooks/resultados_2016_2020/` assim que termina. Se a máquina cair, rode o notebook de novo e ele continua de onde parou. A pasta é nova para não misturar com os resultados do experimento anterior, que usava só 2020.
- **Duração:** medida numa CPU de notebook (Core Ultra 7, sem GPU), num teste com 5 anos de dados, do mesmo tamanho que 2016–2020. Somando os 5 datasets:
  - **ARIMA:** cerca de 3 min.
  - **SVR:** cerca de 1h20 (27 configurações, determinístico). Só usa CPU.
  - **MLP:** cerca de 1h45 (9 configurações × 10 execuções + 10 finais). Só usa CPU.
  - **TCNs:** cerca de 9 s por época na CPU. As 200 execuções (5 datasets × 2 condições × 2 modelos × 10), com *early stopping*, podem passar de 20 h na CPU. Use GPU.
  - A retomada permite dividir o experimento em várias sessões.

## Saídas (`notebooks/resultados_2016_2020/`)

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
ferramentas/
  baixar_inmet.py      baixa os pacotes anuais do INMET e grava as 4 estações em notebooks/dados/
notebooks/
  experimento_tcn_fe_baselines.ipynb   experimento completo
  dados/                               CSVs do INMET 2016–2025 (formato do professor); solarenergy.csv é baixado na 1ª execução
  resultados_2016_2020/                saídas geradas pelo notebook
zfiles/                artigos de referência ([TCN] e [MDPI], ambos CC BY 4.0)
Dockerfile             imagem pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime + dependências
requirements.txt       dependências (o PyTorch vem na imagem)
```

## Limitações conhecidas

- **Hiperparâmetros das TCNs:** o [TCN] não informa kernel, filtros, camadas, dilatações, *learning rate*, lote nem épocas. Usamos k = 3, dilatações 1–16, 32 filtros, lr = 0,001, lote 64 e até 200 épocas com *early stopping* (paciência 10).
- **Dataset do Kaggle:** a variável `solar_mw` nunca chega a zero de noite e tem pico às 19h, comportamento mais típico de consumo do que de geração solar. A meteorologia só existe nas últimas 2920 horas, embora o [TCN] cite cerca de 8700, e por isso não é usada em 2016–2020.
- **Comparação com o [MDPI]:** o artigo do professor usa só 2020, com teste em out–dez/2020. Aqui o teste das cidades é 2020 inteiro e o treino é 2016–2018, então os números não se comparam diretamente com a Tabela 3 dele. A comparação válida é entre os modelos deste repositório, todos no mesmo teste.
- **Faltantes do INMET:** o carregador do professor troca radiação faltante por 0, e o teste mantém esses zeros. Isso afeta todos os modelos igualmente.
- ***Baselines* determinísticos:** ARIMA e SVR dão o mesmo resultado nas 10 execuções ([MDPI] Sec. 5.4).
- **Friedman–Nemenyi:** com N = 5 datasets e k = 7 concorrentes, a distância crítica fica grande (≈ 4,03 postos). Só diferenças muito consistentes aparecem como significativas.

## Licença dos dados e artigos

- **INMET:** os dados são públicos ([portal.inmet.gov.br](https://portal.inmet.gov.br/dadoshistoricos)).
- **Kaggle:** o dataset não é redistribuído aqui; ele é baixado da fonte na primeira execução e fica fora do Git (`.gitignore`).
- **Artigos em `zfiles/`:** os dois estão sob licença Creative Commons Attribution 4.0 e são redistribuídos com a citação completa acima.
