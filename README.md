# TCN-FE Irradiância

Este repositório mede o efeito do pré-processamento e da engenharia de atributos (FE) na previsão de irradiância e de geração solar com TCN e TCN-Attention. A comparação é feita com ARIMA, SVR e MLP. Todas as previsões são de 1 passo à frente.

A arquitetura, o pré-processamento e os atributos vêm de Honnurvali et al. (2026). As séries do INMET, os modelos de referência, as grades de hiperparâmetros, a divisão dos dados, as métricas e as figuras de comparação vêm de Santos Júnior et al. (2022).

## Dados

Todas as séries cobrem 2016 a 2020. Com a divisão cronológica 60/20/20, as cidades treinam em 2016 a 2018, validam em 2019 e testam em 2020. O treino tem os 12 meses do ano repetidos três vezes.

| Série | Origem | Período | Amostras | Teste |
|---|---|---|---|---|
| Kaggle | `solar_mw`, horária, 24 por dia | 01/01/2016 a 03/08/2020 (fim da série) | 40.225 | 02/09/2019 a 03/08/2020 |
| São Paulo (A701) | INMET, 09 a 21 UTC, 13 por dia | 01/01/2016 a 31/12/2020 | 23.751 | 2020 |
| Florianópolis (A806) | INMET | idem | 23.751 | 2020 |
| Salvador (A401) | INMET | idem | 23.751 | 2020 |
| Fortaleza (A305) | INMET | idem | 23.751 | 2020 |

Os arquivos do INMET ficam em `notebooks/dados/`, de 2016 a 2025, no formato do repositório de Santos Júnior et al. (2022). O script `ferramentas/baixar_inmet.py` os gera a partir dos pacotes anuais do portal; o notebook só lê os arquivos locais. O pacote de 2020 convertido fica igual, valor a valor, aos arquivos de 2020 daquele repositório.

O `solarenergy.csv` do Kaggle não é redistribuído, porque o dataset não informa a licença. Na primeira execução, o notebook o baixa da API pública do Kaggle e o guarda em `notebooks/dados/`. As variáveis meteorológicas desse arquivo só existem nas últimas 2920 horas e não entram no experimento.

## Dados faltantes

A Seção 2 do notebook investiga os faltantes do INMET. Os resultados que definem o tratamento:

1. O valor das HH UTC é a radiação acumulada entre HH − 1 e HH. Com essa convenção, nenhuma hora com o Sol abaixo do horizonte tem radiação medida.
2. A maior parte dos brancos é noite, ou seja, uma hora em que o Sol ficou abaixo do horizonte o tempo todo. Ali a radiação é zero de fato.
3. As falhas com Sol são poucas, mas em São Paulo caem em horas de radiação alta. A estação não tem nenhum valor acima de 4.000 kJ/m², embora tenha quase duzentos entre 3.900 e 4.000; tudo indica que valores acima desse limite foram descartados. Preencher com zero cria, no teste de 2020, mais de cem horas de "zero ao meio-dia".

Tratamento adotado:

| Caso | SEM pré-processamento | COM pré-processamento |
|---|---|---|
| Sol abaixo do horizonte a hora toda | zero | zero |
| Falha com Sol (INMET) | zero, como em Santos Júnior et al. (2022) | imputação causal pelo índice de claridade |
| Falha (Kaggle) | último valor medido | imputação causal |
| Avaliação | só horas medidas | só horas medidas |

A imputação causal estima o índice de claridade $k_t = G_t/G_{0,t}$ da hora que falta (DUFFIE; BECKMAN, 2013). Primeiro tenta o $k$ da última hora medida no mesmo dia. Se não houver, usa a mediana de $k$ no mesmo horário dos 14 dias anteriores, a variação diurna média de Falge et al. (2001). A estimativa fica entre o menor e o maior valor medidos até aquele instante. Tudo usa só o passado, então não há vazamento de informação para o teste.

## Modelos e protocolo

| Item | Escolha | Fonte |
|---|---|---|
| Divisão | 60/20/20 cronológica | Santos Júnior et al. (2022, Seção 5) |
| ARIMA | AutoARIMA *stepwise* | Hyndman e Khandakar (2008) |
| SVR | RBF; C ∈ {10, 100, 1000}, γ ∈ {0,1; 0,01; 0,001}, ε ∈ {0,1; 0,01; 0,001}; 12 defasagens | Santos Júnior et al. (2022, Tabela 1) |
| MLP | 1 camada oculta sigmoide, L-BFGS; {20, 50, 100} neurônios; tol ∈ {1e-3, 1e-4, 1e-5}; 12 defasagens | Santos Júnior et al. (2022, Tabela 1) |
| TCN e TCN-Attention | convolução causal dilatada, ReLU, resíduo, atenção aditiva, saída sigmoide, MSE, Adam | Honnurvali et al. (2026, Eq. 2 a 8 e Alg. 1) |
| Janela das TCNs | 48 no Kaggle, 12 nas cidades | Honnurvali et al. (2026, Seção IV.F); Santos Júnior et al. (2022, Seção 5.1) |
| COM pré-processamento e FE | imputação causal, IQR, defasagens de 1 a 4 passos e de 1, 2 e 7 dias, estatísticas móveis, médias exponenciais, faixas de intensidade, calendário em one-hot; nas cidades, também meteorologia e interação com a temperatura | Honnurvali et al. (2026, Seções III e IV) |
| Métricas | RMSE, MAPE, MAE, ARV e IA em [0,1; 0,9], só nas horas medidas | Santos Júnior et al. (2022, Seção 5.2) |
| Repetições | 10 execuções por modelo, condição e série | Santos Júnior et al. (2022, Seção 5.3) |
| Estatística | Friedman e Nemenyi sobre as 5 séries, com diagrama de distância crítica | Demšar (2006) |

Os modelos de referência rodam só na condição SEM, como no artigo de origem. O pré-processamento e o FE entram só nas TCNs.

## Código

O notebook `notebooks/experimento_tcn_fe_baselines.ipynb` contém a configuração, a análise dos dados, as chamadas de treino e os resultados. O resto está no pacote `irradiancia/`:

```text
irradiancia/
  dados.py              Serie, Divisao, CarregadorINMET, CarregadorKaggle
  geometria_solar.py    posição do Sol e irradiação no topo da atmosfera
  analise.py            contagens da investigação dos faltantes
  preprocessamento.py   PreenchimentoSimples e ImputacaoCausal (mesma interface), CorteIQR
  atributos.py          EngenhariaDeAtributos
  preparacao.py         PreparadorTCN: janelas das TCNs nas condições SEM e COM
  metricas.py           métricas e escala de avaliação
  modelos/
    base.py             Previsor, a interface comum
    arima.py            PrevisorARIMA
    janelados.py        PrevisorJanelado (SVR e MLP com busca em grade)
    redes.py            BlocoResidualCausal, CodificadorTCN, AtencaoAditiva, RedeTCN, RedeTCNAtencao
    treino.py           Treinador (Adam, parada antecipada)
    tcn.py              PrevisorTCN
  experimento.py        Experimento: roda cada previsor em cada série e grava as execuções
  registro.py           RegistroDeResultados: metricas.csv, previsões e retomada
  estatistica.py        Friedman e Nemenyi
  graficos.py           figuras
```

Cada modelo é uma estratégia (padrão Strategy) com o método `prever(serie, semente)`. O `Experimento` não sabe qual modelo está rodando: recebe uma lista de previsores e trata todos do mesmo jeito. Para incluir um modelo, basta escrever uma classe nova que herde de `Previsor`.

## Como rodar

### VS Code com Dev Container

1. Instale o Docker e, no VS Code, a extensão Dev Containers.
2. Abra a pasta do repositório e escolha "Reopen in Container".
3. Abra o notebook e use Run All.

O `devcontainer.json` pede GPU NVIDIA (`--gpus all`). Em máquina sem GPU, remova o bloco `runArgs`, e o notebook passa a usar a CPU.

### Docker

```bash
docker build -t tcn-fe-irradiancia .
docker run --gpus all -p 8888:8888 -v "$(pwd)":/workspace tcn-fe-irradiancia
```

Abra o endereço com token que aparece no terminal e rode o notebook.

### Google Colab

1. Escolha a GPU T4 em Ambiente de execução.
2. Na segunda célula, descomente o `git clone` e o `%cd`, para que o notebook encontre o pacote `irradiancia/` e a pasta `dados/`.
3. Use Executar tudo. Para não perder resultados se a sessão cair, aponte `PASTA_RESULTADOS` para o Google Drive, como indicado na primeira célula.

### Configuração

- `MODO_RAPIDO = True` roda 2 execuções, 3 épocas e grades com uma configuração. Serve só para testar o código. O experimento usa `False`.
- `ANOS_INMET` e `INICIO_KAGGLE` definem o período. Para um ano que não esteja em `notebooks/dados/`, gere o arquivo antes: `python ferramentas/baixar_inmet.py 2026`.
- Cada execução é gravada em `notebooks/resultados_2016_2020_v2/` assim que termina. Se a máquina cair, rode o notebook de novo e ele continua de onde parou. O nome da pasta é novo para não misturar com resultados de versões anteriores, que avaliavam as falhas como zero.

Tempo medido numa CPU de notebook (Core Ultra 7, sem GPU), com dados do mesmo tamanho: ARIMA, cerca de 3 min; SVR, cerca de 1h20; MLP, cerca de 1h45. SVR e MLP só usam CPU. As 200 execuções das TCNs passam de 20 h na CPU, então use GPU.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `metricas.csv` | as 5 métricas de cada execução |
| `tabela_<série>.csv` | média ± desvio-padrão das 10 execuções |
| `faltantes_por_hora.png` | horas sem dado em cada hora UTC, por estação |
| `efeito_do_tratamento.png` | três dias de São Paulo: medido, falha em zero e imputação |
| `boxplot_rmse.png` | RMSE nas 10 execuções, um painel por série |
| `distancia_critica.png` | diagrama de Friedman e Nemenyi, α = 0,05 |
| `forecast_tcn_attention.png` | real × TCN-Attention nos 150 primeiros instantes do teste |
| `previsoes/*.npy` | previsões do teste de cada execução |

## Limitações

- Honnurvali et al. (2026) não informam kernel, filtros, dilatações, taxa de aprendizado, lote nem épocas. Usamos k = 3, dilatações de 1 a 16, 32 filtros, taxa 0,001, lote 64 e até 200 épocas com parada após 10 épocas sem melhora.
- A série `solar_mw` do Kaggle nunca chega a zero de noite e tem pico às 19 h. Esse comportamento é mais típico de consumo do que de geração solar.
- Santos Júnior et al. (2022) usam só 2020, com teste de outubro a dezembro. Aqui o treino é 2016 a 2018 e o teste é 2020 inteiro, então os números não se comparam diretamente com a Tabela 3 daquele artigo. A comparação válida é entre os modelos deste repositório, todos no mesmo teste.
- No corte por IQR, a chuva tem $Q_1 = Q_3 = 0$, e toda hora com chuva vira zero na condição COM. O procedimento do artigo foi mantido como está.
- Com 5 séries e 7 concorrentes, a distância crítica de Nemenyi é de cerca de 4 postos. Só diferenças muito consistentes aparecem como significativas.

## Licenças

Os dados do INMET são públicos. O dataset do Kaggle é baixado da fonte e fica fora do Git (`.gitignore`). Os dois artigos em `zfiles/` estão sob licença Creative Commons Attribution 4.0.

## Referências

DEMŠAR, J. Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, v. 7, p. 1-30, 2006.

DUFFIE, J. A.; BECKMAN, W. A. *Solar engineering of thermal processes*. 4. ed. Hoboken: John Wiley & Sons, 2013.

FALGE, E. et al. Gap filling strategies for defensible annual sums of net ecosystem exchange. *Agricultural and Forest Meteorology*, v. 107, n. 1, p. 43-69, 2001.

HONNURVALI, M. S. et al. A TCN-Attention model for high-accuracy solar energy generation forecasting. *IEEE Open Access Journal of Power and Energy*, v. 13, p. 690-701, 2026. DOI: 10.1109/OAJPE.2026.3723147.

HYNDMAN, R. J.; KHANDAKAR, Y. Automatic time series forecasting: the forecast package for R. *Journal of Statistical Software*, v. 27, n. 3, p. 1-22, 2008.

SANTOS JÚNIOR, D. S. O. et al. Solar irradiance forecasting using dynamic ensemble selection. *Applied Sciences*, v. 12, n. 7, 3510, 2022. DOI: 10.3390/app12073510.

As demais referências estão no fim do notebook.
