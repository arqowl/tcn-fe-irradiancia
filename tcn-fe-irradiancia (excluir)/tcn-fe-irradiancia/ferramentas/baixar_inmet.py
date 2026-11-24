"""Baixa os dados históricos do INMET e grava as 4 estações no formato do repositório do professor.

O notebook NÃO usa este script: ele lê os CSVs já gravados em notebooks/dados/.
O script existe para refazer ou ampliar esses arquivos.

Uso (da raiz do repositório):
    python ferramentas/baixar_inmet.py 2019 2020 2021 2022 2023 2024 2025
    python ferramentas/baixar_inmet.py 2025 --zips pasta_com_zips   # usa zips já baixados

Formato de saída (igual a data/*.csv de github.com/domingos108/solar_forecasting):
- sem as 8 linhas de metadados do INMET;
- nomes de coluna sem acentos, sem "°" e "²", e radiação sempre "RADIACAO GLOBAL (Kj/m)";
- data em dd/mm/aaaa; separador ";" e vírgula decimal mantidos.
Conferido: o pacote 2020 do INMET convertido assim fica idêntico, valor a valor,
aos arquivos de 2020 do professor nas 4 estações.
"""
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

ESTACOES = ["A701", "A806", "A401", "A305"]          # São Paulo, Florianópolis, Salvador, Fortaleza
URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
DESTINO = Path(__file__).resolve().parents[1] / "notebooks" / "dados"


def normalizar(nome):
    nome = nome.replace("²", "").replace("°", "")
    nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode().strip()
    if nome.upper().startswith("RADIACAO GLOBAL"):
        nome = "RADIACAO GLOBAL (Kj/m)"                               # 2019 usa "KJ/m²"
    return nome


def bruto_para_professor(texto):
    linhas = texto.splitlines()[8:]                                     # tira os 8 metadados
    cab = [normalizar(c) for c in linhas[0].split(";")]
    if cab and cab[-1] == "":
        cab = cab[:-1]
    cab[0], cab[1] = "Data", "Hora UTC"                                 # até 2018: "DATA (YYYY-MM-DD)", "HORA (UTC)"
    saida = [";".join(cab)]
    for linha in linhas[1:]:
        if not linha.strip():
            continue
        campos = linha.split(";")[:len(cab)]
        campos = ["" if c.strip().startswith("-9999") else c for c in campos]   # até 2018, faltante = -9999; depois, vazio
        a, m, d = campos[0].split("/") if "/" in campos[0] else campos[0].split("-")
        campos[0] = f"{d}/{m}/{a}"
        if ":" in campos[1]:                                            # anos antigos: "12:00"
            campos[1] = campos[1].replace(":", "")[:4] + " UTC"
        saida.append(";".join(campos))
    return "\n".join(saida) + "\n"


def obter_zip(ano, pasta_zips):
    if pasta_zips and (Path(pasta_zips) / f"{ano}.zip").exists():
        return zipfile.ZipFile(Path(pasta_zips) / f"{ano}.zip")
    destino = Path(pasta_zips or ".") / f"{ano}.zip"
    print(f"baixando {URL.format(ano=ano)} …")
    req = urllib.request.Request(URL.format(ano=ano), headers={"User-Agent": UA})   # o portal recusa o user-agent padrão
    with urllib.request.urlopen(req) as r, open(destino, "wb") as f:
        while bloco := r.read(1 << 20):
            f.write(bloco)
    return zipfile.ZipFile(destino)


def main(args):
    pasta_zips = None
    if "--zips" in args:
        i = args.index("--zips"); pasta_zips = args[i + 1]; args = args[:i] + args[i + 2:]
    DESTINO.mkdir(parents=True, exist_ok=True)
    for ano in [int(a) for a in args]:
        z = obter_zip(ano, pasta_zips)
        for cod in ESTACOES:
            nome = [n for n in z.namelist() if f"_{cod}_" in n][0]
            saida = DESTINO / (Path(nome).stem + ".csv")
            if saida.exists():
                print("já existe:", saida.name); continue
            saida.write_text(bruto_para_professor(z.read(nome).decode("latin-1")), encoding="utf8")
            print("gravado:", saida.name)


if __name__ == "__main__":
    main(sys.argv[1:])
